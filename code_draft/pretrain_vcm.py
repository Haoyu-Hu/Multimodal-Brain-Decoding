import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torch.nn.functional as F

from dataset import integrated_dataset
from model import VCM

from param_parser import parameter_parser

import numpy as np
import os
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(6789)
np.random.seed(6789)
torch.cuda.manual_seed_all(6789)
os.environ['PYTHONHASHSEED'] = str(6789)

def train_vcm():
    args = parameter_parser()
    data_path = '/content/drive/MyDrive/Cam_proj/Multimodal/Multimodal_data/integrated_data_new'
    vcm_path = '/content/drive/MyDrive/Cam_proj/Multimodal/VCM'
    # print(args)
    dataset = integrated_dataset(args)
    data_loader = DataLoader(dataset, batch_size=1)
    # print('go dataset')
    # args(666)

    model = VCM(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    train_epoch = 100
    flag = 1
    best_loss = 0
    for epoch in np.arange(train_epoch):
        data_comp_list = []
        data_mask_list = []
        data_border_list = []
        data_index_list = []

        path_comp_list_1 = []
        path_comp_list_2 = []
        path_mask_list = []
        path_border_list = []
        path_index_list = []

        sum_loss = 0
        sum_loss2 = 0
        for batch in data_loader:
            data = batch['data'].to(device).squeeze()
            data_recon, data_comp, mask, border_mask, index = model(data, mode='full')
            # print(data_recon.size())
            # print(data_comp.size())

            # print(data_recon.size())
            # print(data.size())

            file_name = batch['files']
            subject = batch['subfolder'][0]
            sub_folder = os.path.join(data_path, subject)
            print(file_name)
            if not os.path.isdir(sub_folder):
                os.mkdir(sub_folder)

            mask_file = os.path.join('mask', subject+'.npy')
            border_file = os.path.join('border', subject+'.npy')
            index_file = os.path.join('index', subject+'.npy')

            data_comp_list.append(data_comp.detach().cpu().numpy())
            data_mask_list.append(mask.detach().cpu().numpy())
            data_border_list.append(border_mask.detach().cpu().numpy())
            data_index_list.append(index.detach().cpu().numpy())

            path_comp_list_1.append(os.path.join(sub_folder, file_name[0][0]))
            path_comp_list_2.append(os.path.join(sub_folder, file_name[1][0]))
            path_mask_list.append(os.path.join(vcm_path, mask_file))
            path_border_list.append(os.path.join(vcm_path, border_file))
            path_index_list.append(os.path.join(vcm_path, index_file))

            # data_recon[torch.isnan(data_recon)] = 0
            data[torch.isnan(data)] = 0

            data = F.normalize(data, dim=1 if len(data.size()) == 3 else 0)
            # data_recon = F.normalize(data_recon)

            # print(torch.max(data))
            # print(torch.min(data))
            # print(torch.max(data_comp))
            # print(torch.min(data_comp))

            data_recon_outborder = data_recon[border_mask==False]
            data_outborder = data[border_mask==False]

            loss = F.mse_loss(data_recon, data)
            loss2 = F.mse_loss(data_recon_outborder, data_outborder, reduction='sum')
            sum_loss += loss.item()
            sum_loss2 += loss2.item()
            print('loss1:{:.3f}, loss2:{:.3f}'.format(loss.item(), loss2.item()))

            loss2.backward()
            print(model)
            for name, param in model.named_parameters():
                print(param.size())
                print(name)
                print(param)
        model(1,1)
        
        if flag == 1:
            best_loss = sum_loss
            flag = 0
            torch.save(model.state_dict(), os.path.join(vcm_path, 'pth/vcm.pt'))

            for i in range(len(path_comp_list_1)):
                np.save(path_comp_list_1[i], data_comp_list[i][0])
                np.save(path_comp_list_2[i], data_comp_list[i][1])
                np.save(path_mask_list[i], data_mask_list[i])
                np.save(path_border_list[i], data_border_list[i])
                np.save(path_index_list[i], data_index_list[i])

            # file_name = batch['files']
            # subject = batch['subfolder'][0]
            # sub_folder = os.path.join(data_path, subject)
            # print(file_name)
            # if not os.path.isdir(sub_folder):
            #     os.mkdir(sub_folder)
            # np.save(os.path.join(sub_folder, file_name[0][0]), data_comp[0].detach().cpu().numpy())
            # np.save(os.path.join(sub_folder, file_name[1][0]), data_comp[1].detach().cpu().numpy())
            # mask_file = os.path.join('mask', subject+'.npy')
            # border_file = os.path.join('border', subject+'.npy')
            # index_file = os.path.join('index', subject+'.npy')
            # np.save(os.path.join(vcm_path, mask_file), mask.detach().cpu().numpy())
            # np.save(os.path.join(vcm_path, border_file), border_mask.detach().cpu().numpy())
            # np.save(os.path.join(vcm_path, index_file), index.detach().cpu().numpy())
        else:
            if best_loss > sum_loss:
                best_loss = sum_loss
                torch.save(model.state_dict(), os.path.join(vcm_path, 'pth/vcm.pt'))

                for i in range(len(path_comp_list_1)):
                    np.save(path_comp_list_1[i], data_comp_list[i][0])
                    np.save(path_comp_list_2[i], data_comp_list[i][1])
                    np.save(path_mask_list[i], data_mask_list[i])
                    np.save(path_border_list[i], data_border_list[i])
                    np.save(path_index_list[i], data_index_list[i])

        
        print('Epoch: {:d}, train loss: {:.3f}, match loss: {:.3f}'.format(epoch, sum_loss, sum_loss2))



    # print('hello model')
    # print(model.__repr__())
    
    

if __name__ == '__main__':
  train_vcm()