import os
import numpy as np
import pickle
# from NeuroGraph.datasets import NeuroGraphDataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from pycocotools.coco import COCO

class test_neurograph_dataset(Dataset):
    def __init__(self, args):
        super(test_neurograph_dataset).__init__

        self.args = args
        self.data_path = args.data_path
        self.ng_data = args.ng_data
        self.data_org = NeuroGraphDataset(root = self.data_path, name = self.ng_data)
        self.data = self.data_org.x.view(-1, self.data_org.x.size(-1), self.data_org.x.size(-1))[:200]
        self.y = self.data_org.y[:200]

    def __getitem__(self, index):
        item = {'data': self.data[index]
              , 'yse': self.data[index]
              , 'yta': self.y[index]}
        return item
    
    def __len__(self):
        return self.data.size(0)

class integrated_dataset(Dataset):
    def __init__(self, args):
        super(integrated_dataset).__init__()

        self.args = args
        self.slices = [args.z_slice, args.x_slice, args.y_slice]
        self.dataset_path = args.data_path
        folders = os.listdir(self.dataset_path)
        folders.remove('stim')
        self.folders = folders
        self.sub_path = []
        for sub_folder in folders:
            self.sub_path.append(os.path.join(self.dataset_path, sub_folder))
        self.stim_path = os.path.join(self.dataset_path, 'stim')
        # self.__getitem__(111)
    def __getitem__(self, index):
        sub_folder = self.sub_path[index]
        data_list = []
        yse_list = []
        yta_list = []
        for files in os.listdir(sub_folder):
            data_temp = torch.Tensor(np.load(os.path.join(sub_folder, files))).view(-1,30,100,100)
            # import ipdb; ipdb.set_trace()
            if files == 'movie.npy':
                label_temp = torch.Tensor([0])
                sem_temp = np.load(os.path.join(self.stim_path, 'movie.npy'))
            else:
                label_temp = torch.Tensor([1])
                sem_temp = np.load(os.path.join(self.stim_path, 'story.npy'))
            for dim_index in range(1,4):
                data_slice = data_temp.chunk(self.slices[dim_index-1], dim=dim_index)
                data_list_temp = []
                for slices in data_slice:
                    data_list_temp.append(slices.mean(dim=dim_index))
                data_temp = torch.stack(data_list_temp)
            patch_num = self.slices[0]*self.slices[1]*self.slices[2]
            data_list.append(data_temp.view(-1, patch_num))
            del data_temp
            yse_list.append(torch.Tensor(sem_temp))
            yta_list.append(label_temp)
        data = torch.stack(data_list)
        del data_list
        yse = torch.stack(yse_list)
        del yse_list
        yta = torch.stack(yta_list)
        item = {'data': data,
                'yse': yse,
                'yta': yta
                ,'subfolder': self.folders[index]
                ,'files': os.listdir(sub_folder)}
        return item
    
    def __len__(self):
        return len(self.sub_path)

class coco_brain(Dataset):
    def __init__(self, data_path, slice_set):
        super(coco_brain).__init__()

        self.data_path = data_path
        self.slices = slice_set

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
    def __getitem__(self, index):
        fmri = torch.Tensor(self.data[index]['fmri'])
        img = self.data[index]['image']
        capt_dict = self.data[index]['caption']
        capt = []
        rand_capt_idx = np.random.randint(0, len(capt_dict))
        capt = capt_dict[rand_capt_idx]['caption']
        
        for dim_index in range(3):
            data_slice = fmri.tensor_split(self.slices[dim_index], dim=dim_index)
            data_list_temp = []
            for slices in data_slice:
                data_list_temp.append(slices.mean(dim=dim_index))
            fmri = torch.stack(data_list_temp, dim=dim_index)
        fmri_patch = fmri.view(-1)
        del fmri

        return fmri_patch, img, capt
    
    def __len__(self):
        return len(self.data)

                    