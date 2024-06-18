import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

from tasnet import ConvTasNet as tasnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        r_high = args.token_embed_size
        r_low = args.img_embed_size
        self.high_prior = nn.Linear(self.regions, r_high)
        self.low_prior = nn.Linear(self.regions, r_low)
    
    def forward(self, x):
        
        if self.brain_arch == 'MLP':

            alpha = self.alpha_est(x.unsqueeze(1)) # alpha->R^{B*T*J}
            la = self.la_est(x)

            # low_states, r_loss = self.low_extractor(q=x, k=x, v=x) # L->R^{B*J*T*N}
            # high_states, _ = self.high_extractor(q=x, k=x, v=x) # H->R^{B*T*N}
            high_states = self.brain_extractor[0](x)
            low_states_list = []
            for i in range(1, len(self.brain_extractor)):
                low_states_list.append(self.brain_extractor[i](x))
            low_states = torch.stack(low_states_list, dim=1)
            del low_states_list

            theo_brain = torch.mul(high_states, la) + torch.mul(torch.mul(low_states, alpha), 1-la)
            
        elif self.brain_arch == 'tasnet':

            brain_sep, sep_weight = self.brain_extractor(x)
            la = sep_weight[:,0,:]
            alpha = torch.div(sep_weight[:,1:,:], 1-la.repeat(sep_weight[:,1:,:].size()))

            high_states = brain_sep[:,0,:]
            low_states = brain_sep[:,1:,:]

            theo_brain = torch.sum(brain_sep, dim=1).squeeze()

            high_states = torch.div(high_states, la)
            low_states = torch.div(low_states, 1-la.repeat(sep_weight[:,1:,:].size()))
        
        high_proc = self.high_MLP(high_states)
        proc_low_list = []
        for state_idx in range(low_states.size(1)):
            proc_low_list.append(self.low_MLP[state_idx](low_states[:,state_idx,:]))
        low_proc = torch.stack(proc_low_list, dim=1)
        del proc_low_list

        if self.brain_arch == 'MLP':
            low_proc = torch.sum(torch.mul(alpha, low_proc), dim=1).squeeze(1) # R^{N*B*T*J}

        feature_low = self.low_prior(low_proc)
        feature_high = self.high_prior(high_states)

        return feature_high, feature_low, theo_brain
    
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

class VCM_Encoder(nn.Module):
    def __init__(self, args):
        super(VCM_Encoder, self).__init__()

        self.args = args
        self.regions = args.regions
        self.time_points = args.time_points
        self.topk_ratio = args.topk_ratio
        self.projection = nn.Linear(self.time_points, 1)
        # self.process = nn.Sequential(nn.Linear(int(self.regions*self.topk_ratio), int(self.regions*self.topk_ratio))
        #                               , nn.ReLU()
        #                               , nn.BatchNorm1d(self.time_points)
        #                               , nn.Dropout(p=0.5))
        self.process = nn.Sequential(nn.Linear(int(self.regions*self.topk_ratio), int(self.regions*self.topk_ratio))
                                      , nn.BatchNorm1d(self.time_points))
        # self.compress = nn.Sequential(nn.Linear(int(self.regions*self.topk_ratio), int(self.regions*self.topk_ratio**2))
        #                               , nn.BatchNorm1d(self.time_points))
        self.compress = nn.Sequential(nn.Linear(int(self.regions*self.topk_ratio), int(self.regions*self.topk_ratio**2)))
        
    def forward(self, x):
        border_mask = torch.isnan(x)
        x[torch.isnan(x)] = 0
        norm_dim = 1 if len(x.size()) == 3 else 0
        x = F.normalize(x, dim=norm_dim)
        score = self.projection(x.transpose(-1,-2)).squeeze()
        border_limit = torch.mean(border_mask.float(), dim=norm_dim)
        score[border_limit.bool()] = -10000
        # print(score.size())
        # print(x.size())
        if len(x.size()) == 3:
            x_topk_list = []
            mask_list = []
            index_list = []
            for b in range(x.size(0)):
                score_ini = score[b].squeeze()
                # print(score_ini.size())
                topk_score, topk_index = torch.topk(score_ini, int(x.size(-1)*self.topk_ratio))
                x_ini = x[b].squeeze()
                x_topk_list.append(x_ini[:, topk_index])
                un_x = torch.zeros(x_ini.size(), device=device)
                # un_x[:] = torch.Tensor([0], device=device)
                un_x = un_x.index_fill(1, topk_index, 1)
                mask_list.append(un_x)
                index_list.append(topk_index)
            x_topk = torch.stack(x_topk_list)
            mask = torch.stack(mask_list)
            index = torch.stack(index_list)
        elif len(x.size()) == 2:
            topk_score, topk_index = torch.topk(score_ini, int(x.size(-1)*self.topk_ratio))
            x_topk = x[:, topk_index]
            mask = torch.zeros(x_ini.size(), device=device)
            mask = mask.index_fill(1, topk_index, 1)
            index = topk_index
        x = self.process(x_topk)
        x = self.compress(x)

        return x, mask, border_mask, index

class VCM_Decoder(nn.Module):
    def __init__(self, args):
        super(VCM_Decoder, self).__init__()

        self.args = args
        self.topk_ratio = args.topk_ratio
        self.regions = args.regions
        self.unzip = nn.Sequential(nn.Linear(int(self.regions*self.topk_ratio**2), int(self.regions*self.topk_ratio)))
        self.unprocess = nn.Linear(int(self.regions*self.topk_ratio), int(self.regions*self.topk_ratio))
        self.rest = nn.Linear(int(self.regions*self.topk_ratio), args.rest_limitation)


    def forward(self, x, border_mask, index):
        x = self.unzip(x)
        x = self.unprocess(x)
        x_rest = self.rest(x)
        x_recon = torch.zeros(border_mask.size()).to(device)
        # print(mask)
        for b in range(x_recon.size(0)):
            x_recon[b, :, index[b]] = x[b]
            rest_num = int(torch.sum(border_mask[b].sum(dim=0)==False)-self.regions*self.topk_ratio)
            rest_index = border_mask[b].sum(dim=0)
            rest_index[index[b]] = True
            if rest_num > self.args.rest_limitation:
                print('adjust')
                rest_num = self.args.rest_limitation
                rest_index_res = rest_index[rest_index==False]
                rest_index_res[rest_num:] = True
                rest_index[rest_index==False] = rest_index_res
            x_recon[b, :, rest_index==False] = x_rest[b, :, :rest_num]
        # x_recon = self.recon(x_recon)
        # x_recon[border_mask] = torch.nan

        return x_recon

class VCM(nn.Module):
    def __init__(self, args):
        super(VCM, self).__init__()

        self.args = args
        self.encoder = VCM_Encoder(args)
        self.decoder = VCM_Decoder(args)

    def forward(self, x, mask=None, border_mask=None, index=None, mode='full'):

        if mode == 'full':
            x_compress, mask, border_mask, index = self.encoder(x)
            x_recon = self.decoder(x_compress, border_mask, index)
            if torch.sum(torch.isnan(x_compress)) != 0:
                print('Compression is NAN')
            if torch.sum(torch.isnan(x_recon)) != 0:
                print('Reconstruction is NAN')

            return x_recon, x_compress, mask, border_mask, index
        elif mode == 'encode':
            x_compress, mask, border_mask, index = self.encoder(x)

            return x_compress, mask, border_mask, index
        elif mode == 'decode':
            x_recon = self.decoder(x, border_mask, index)

            return x_recon



def wasserstain(mu1, mu2, sigma1, sigma2):
    mean_distance = F.mse_loss(mu1, mu2, reduction='mean')
    b_distance_step1 = sigma1 + sigma2 - 2*sigma1.sqrt()*sigma2.sqrt()
    b_distance_step2 = b_distance_step1.sum(dim=-1)
    w_distance = mean_distance + torch.mean(b_distance_step2)

    return w_distance.to(device)
        