import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        # print(k.size())
        batch_size, time, regions = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(-1, -2)  # transpose
        score = (q @ k_t) / math.sqrt(regions)  # scaled dot product

        score = torch.tril(score)

        # 2. apply masking (opt)
        score = score.masked_fill(score == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        if v.size(1) > 1:
            v_copy = v.clone()
            for b in range(v.size(0)):
                v_copy[b] = torch.matmul(score[b], v[b])
            v = v_copy
        else:
            v = score @ v

        return v, score

class State_Encoding(nn.Module):

    def __init__(self, regions, n_state=1, level='low'):
        super(State_Encoding, self).__init__()
        self.n_state = n_state
        self.level = level
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(regions, regions)
        self.w_k = nn.Linear(regions, regions)
        self.w_v_list = nn.ModuleList()
        if self.level == 'low':
            for i in range(n_state):
                self.w_v_list.append(nn.Linear(regions, regions))
            self.w_project = nn.Linear(regions, 1)
        elif self.level == 'high':
            self.w_v = nn.Linear(regions, regions)
        else:
            raise Exception('Unknown Brain State Setting')
    
    def reset_parameters(self):
        self.w_q.reset_parameters()
        self.w_k.reset_parameters()
        if self.level == 'low':
            for layer in self.w_v_list:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        elif self.level == 'high':
            self.w_v.reset_parameters()

    def forward(self, q, k, v):
        # dimension q->R^{B*T*N}, k->R^{B*T*N}, v->R^{B*J*T*N}
        # 1. dot product with weight matrices
        q, k = self.w_q(q), self.w_k(k)

        if self.level == 'low':
            v_list = []
            for i in range(self.n_state):
                v_list.append(self.w_v_list[i](v))
            v = torch.stack(v_list, dim=1)
        elif self.level == 'high':
            v = self.w_v(v)

        # 2. split tensor by number of heads
        # q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v)

        # 4. concat and pass to linear layer
        # out = self.concat(out)
        # out = self.w_concat(out)
        if self.level == 'low':
            proj_score = self.w_project(v).squeeze(-1) # proj_score->R^{B*J*T}
            sim_loss_list = []
            for i in range(proj_score.size(0)):
                sim_loss_list.append(1/2*torch.sum(torch.abs(torch.cov(proj_score[i]))))
            sim_loss = torch.mean(torch.stack(sim_loss_list))
        else:
            sim_loss = 0

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out, sim_loss
    
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()

        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class Brain_Modelling(nn.Module):
    def __init__(self, args, regions, time_points):
        super(Brain_Modelling, self).__init__()

        self.args = args
        self.n_state = args.n_state
        self.time_points = time_points
        self.original_regions = args.original_regions
        self.regions = regions
        self.semantic = args.semantic
        self.beta = args.beta

        # self.projection_layer = nn.Linear(self.original_regions, regions)

        self.dist_estimator = nn.Linear(regions, self.n_state*2) # R^{B*T*2J}
        self.low_extractor = State_Encoding(regions, self.n_state)
        self.high_extractor = State_Encoding(regions, self.n_state, level='high')

        self.la = nn.Linear(time_points, 1)
        
        self.dist_high_theo = nn.Linear(regions, regions*2)
        self.dist_high = nn.Linear(regions, regions*2)

        self.classifier = nn.Sequential(nn.Linear(regions, regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(regions, 1)
                                        , Transpose(-1, -2)
                                        , nn.Linear(time_points, args.classes))
        self.regresser = nn.Sequential(nn.Linear(regions, regions)
                                        , nn.ReLU()
                                        , nn.Dropout(p=0.1)
                                        , nn.Linear(regions, self.semantic))

    def reset_parameters(self):
        self.dist_estimator.reset_parameters()
        self.low_extractor.reset_parameters()
        self.high_extractor.reset_parameters()
        self.la.reset_parameters()
        self.dist_high_theo.reset_parameters()
        self.dist_high.reset_parameters()

        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.regresser:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, x): # x->R^{B*T*N}
        # print(x.size())
        dist_theta = self.dist_estimator(x)
        # print(x.size())

        mu_the, sigma_the = dist_theta[:,:,:self.n_state], F.softplus(dist_theta[:,:,self.n_state:])

        theta = self.n_reparameter(mu_the, sigma_the)

        alpha = F.softmax(theta, dim=-1) # alpha->R^{B*T*J}

        low_states, r_loss = self.low_extractor(q=x, k=x, v=x) # L->R^{B*J*T*N}
        high_states, _ = self.high_extractor(q=x, k=x, v=x) # H->R^{B*T*N}

        weighted_low = low_states.view(low_states.size(-1), low_states.size(0), low_states.size(2), low_states.size(1)) * alpha # R^{N*B*T*J}
        weighted_low = weighted_low.mean(dim=-1).view(high_states.size()) # R^{B*T*N}

        weight_la = F.sigmoid(self.la(torch.eye(self.time_points, device=device))) # R^{T*1}
        expand_weight_la = weight_la.expand(weight_la.size(0), weighted_low.size(-1))
        theo_low = weighted_low * (1-expand_weight_la)
        theo_high = 1/expand_weight_la * high_states
        theo_brain = theo_low + theo_high

        dist_high_theo = self.dist_high_theo(theo_high)
        dist_high = self.dist_high(high_states)

        # mu_high_theo, sigma_high_theo = dist_high_theo[:,:,:self.regions], F.softplus(dist_high_theo[:,:,self.regions:])
        # mu_high, sigma_high = dist_high[:,:,:self.regions], F.softplus(dist_high[:,:,self.regions:])

        # kl_loss = wasserstain(mu_high_theo, mu_high, sigma_high_theo, sigma_high)

        logits1 = self.classifier(weighted_low)
        logits1_high = self.classifier(high_states)
        expand_pred_la = weight_la.expand(weight_la.size(0), self.semantic)
        logits2 = (1-expand_pred_la)*self.regresser(weighted_low) + expand_pred_la*self.regresser(high_states)
        # logits2 = self.regresser(high_states)
        # print(r_loss)
        # print(kl_loss)
        # print(logits2)

        return F.softmax(logits1), F.softmax(logits1_high), logits2, r_loss, theo_brain, high_states, low_states, weight_la, alpha
    
    def n_reparameter(self, mu, sigma):
        return mu + sigma*torch.randn(sigma.size(), device=device)

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
        