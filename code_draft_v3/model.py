import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from diffusers import StableUnCLIPImg2ImgPipeline

import vec2text
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

from tasnet import ConvTasNet as tasnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vec2Text(nn.Module):
    def __init__(self, mode='embed'):
        super(Vec2Text, self).__init__()

        self.mode = mode

        if mode == 'embed':
            self.text_encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(device)
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        elif mode == 'invert':
            self.corrector = vec2text.load_pretrained_corrector("gtr_base")

    def invert_embedding(self, embedding):
        if self.mode == 'embed':
            return 0
        
        with torch.no_grad():
            inverted_text = vec2text(
                embeddings = embedding,
                corrector = self.corrector
            )
        return inverted_text
    
    def text_embedding(self, text_list) -> torch.Tensor:
        if self.mode == 'invert':
            return 0

        inputs = self.tokenizer(text_list,
                          return_tensors="pt",
                          max_length=128,
                          truncation=True,
                          padding="max_length",).to(device)

        with torch.no_grad():
            model_output = self.text_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            hidden_state = model_output.last_hidden_state
            embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])
        
        return embeddings

from diffusers import AutoencoderKL
from torchvision.transforms import Resize
class VAE_Img(nn.Module):
    def __init__(self, args):
        super(VAE_Img, self).__init__()

        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")


    def encode_img(self, input_img):
        # Single image -> single latent in a batch (so size b, 4, 64, 64)
        if input_img.size(-1) == 3:
            input_img = Resize(size=(512,512))(input_img.permute(0, -1, 1, 2)).float()
        else:
            input_img = Resize(size=(512,512))(input_img).float()
        if len(input_img.shape)<4:
            input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            latent_list = []
            for img_idx in range(input_img.size(0)):
                img_item = input_img[img_idx].unsqueeze(0)
                latent = self.vae.encode(img_item*2 - 1) # Note scaling
                latent_list.append(0.18215 * latent.latent_dist.sample())
            
        return torch.stack(latent_list, dim=0)

    def decode_img(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach()
        return image


class FrozenCLIP(nn.Module):
    def __init__(self, args):
        super(FrozenCLIP, self).__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(args.clip_arch
                                                                               , device = device
                                                                               , pretrained=args.clip_pretrain)
        self.tokenizer = open_clip.get_tokenizer(args.clip_arch)

        self.model.eval()
        for param in self.model.parameters():
            param.require_grads = False
        
    def text_encode(self, caption):
        feature_store = []

        for item_capt in caption:
            capt_token = self.tokenizer(item_capt).to(device)
            capt_features = self.model.encode_text(capt_token)
            feature_store.append(capt_features)
        
        return torch.stack(feature_store)
    
    def image_encode(self, image):
        pre_image = self.preprocess(image).to(device).unsqueeze(0)
        image_features = self.model.encode_image(pre_image)

        return image_features

class Decoder_Img(nn.Module):
    def __init__(self, args):
        super(Decoder_Img, self).__init__()

        self.args = args

        self.decoder_pip = StableUnCLIPImg2ImgPipeline.from_pretrained(args.decoder_img, device=device, torch_dtype=torch.float16, variation="fp16")

        # for param in self.decoder_pip.parameters():
        #     param.require_grads = False

    def forward(self, text_embedding, img_embedding):
        img_recover = self.decoder_pip(image=img_embedding, prompt_embeds=text_embedding)

        return img_recover[0]

    
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()

        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class Brain_Modelling(nn.Module):
    def __init__(self, args):
        super(Brain_Modelling, self).__init__()

        self.args = args
        self.brain_arch = args.brain_arch
        self.n_state = args.n_state
        self.regions = args.x_slice*args.y_slice*args.z_slice


        # self.projection_layer = nn.Linear(self.original_regions, regions)
        if args.brain_arch == 'MLP':
            self.brain_extractor = nn.ModuleList()
            for i in range(1+args.n_state):
                self.brain_module = [res_MLP(self.regions) for i_ in range(args.R)]
                self.brain_extractor.append(nn.Sequential(*self.brain_module))
        elif args.brain_arch == 'tasnet':
            self.brain_extractor = tasnet(N=self.regions, L=self.regions, B=args.B, H=args.H
                                          , P=args.P, C=1+args.n_state, R=args.R, X=args.X)

        if args.brain_arch == 'MLP':
            self.alpha_est = nn.Sequential(nn.Conv1d(1, args.n_state, kernel_size=3, padding=1)
                                                , nn.Softmax(dim=1))
            self.la_est = nn.Sequential(nn.Linear(self.regions, self.regions)
                                    , nn.Sigmoid())
        
        self.high_MLP = nn.Sequential(*[res_MLP(self.regions, activation=args.res_activation) for i in range(args.n_layer)])
        self.low_MLP_module = nn.Sequential(*[res_MLP(self.regions, activation=args.res_activation) for i in range(args.n_layer)])
        self.low_MLP = nn.ModuleList([self.low_MLP_module for i in range(args.n_state)])

        gtr_high = args.token_embed_size_gtr
        clip_high = args.token_embed_size_clip
        r_low = args.img_embed_size
        self.high_prior = nn.Sequential(nn.Linear(self.regions, gtr_high*2)
                                        , nn.PReLU()
                                        , nn.Linear(gtr_high*2, gtr_high)
                                        , nn.LayerNorm(gtr_high))
        self.high_prior_clip = nn.Linear(gtr_high, clip_high)
        # self.low_prior = nn.Sequential(nn.Linear(self.regions, r_low*2)
        #                               , nn.PReLU()
        #                               , nn.Linear(r_low*2, r_low)
        #                               , nn.LayerNorm(r_low))
        self.low_prior = Voxel2StableDiffusionModel(in_dim=self.regions)
    
    def forward(self, x):
        
        if self.brain_arch == 'MLP':

            # alpha = self.alpha_est(x.unsqueeze(1)) # alpha->R^{B*T*J}
            la = self.la_est(x)

            # low_states, r_loss = self.low_extractor(q=x, k=x, v=x) # L->R^{B*J*T*N}
            # high_states, _ = self.high_extractor(q=x, k=x, v=x) # H->R^{B*T*N}
            high_states = self.brain_extractor[0](x)
            low_states_list = []
            for i in range(1, len(self.brain_extractor)):
                low_states_list.append(self.brain_extractor[i](x))
            low_states = torch.stack(low_states_list, dim=1)
            del low_states_list

            # theo_brain = torch.mul(high_states, la) + torch.mul(torch.mul(low_states, alpha).sum(dim=1), 1-la)
            theo_brain = torch.mul(high_states, la) + torch.mul(low_states.sum(dim=1), 1-la)

        elif self.brain_arch == 'tasnet':

            brain_sep, sep_weight = self.brain_extractor(x)
            sep_weight = sep_weight.squeeze(-1)

            high_states = brain_sep[:,0,:]
            low_states = brain_sep[:,1:,:]

            theo_brain = torch.sum(brain_sep, dim=1).squeeze()

            # high_states = torch.div(high_states, la)
            # low_states = torch.div(low_states, 1-la)
        
        high_proc = self.high_MLP(high_states)
        proc_low_list = []
        for state_idx in range(low_states.size(1)):
            proc_low_list.append(self.low_MLP[state_idx](low_states[:,state_idx,:]))
        low_proc = torch.stack(proc_low_list, dim=1)
        del proc_low_list

        if self.brain_arch == 'MLP':
            # low_proc = torch.sum(torch.mul(alpha, low_proc), dim=1).squeeze(1) # R^{N*B*T*J}
            low_proc = low_proc.sum(dim=1)
        elif self.n_state > 1:
            low_proc = low_proc.sum(1)

        feature_low = self.low_prior(low_proc)
        feature_high = self.high_prior(high_states)
        feature_clip = self.high_prior_clip(feature_high)

        return feature_high, feature_low, theo_brain, feature_clip
    
    def n_reparameter(self, mu, sigma):
        return mu + sigma*torch.randn(sigma.size(), device=device)

class res_MLP(nn.Module):
    def __init__(self, in_channels, activation='prelu'):
        super(res_MLP, self).__init__()

        self.in_channels = in_channels
        self.mlp = nn.Sequential(nn.Linear(in_channels, 2*in_channels)
                                 , nn.PReLU()
                                 , nn.Linear(2*in_channels, in_channels))
        if activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise Exception('Unknown Activation Function in Residual MLP Module')
    
    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.act, 'reset_parameters'):
            self.act.reset_parameters()
    
    def forward(self, x):
        x_res = self.mlp(x)
        x = self.act(x_res+x)

        return x

from diffusers.models.autoencoders.vae import Decoder
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=False, ups_mode='4x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    def forward(self, x, return_transformer_feats=False):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)



def wasserstain(mu1, mu2, sigma1, sigma2):
    mean_distance = F.mse_loss(mu1, mu2, reduction='mean')
    b_distance_step1 = sigma1 + sigma2 - 2*sigma1.sqrt()*sigma2.sqrt()
    b_distance_step2 = b_distance_step1.sum(dim=-1)
    w_distance = mean_distance + torch.mean(b_distance_step2)

    return w_distance.to(device)
        