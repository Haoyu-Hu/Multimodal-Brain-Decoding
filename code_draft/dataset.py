import os
import numpy as np
# from NeuroGraph.datasets import NeuroGraphDataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
            data_temp = np.load(os.path.join(sub_folder, files))
            # import ipdb; ipdb.set_trace()
            if files == 'movie.npy':
                label_temp = torch.Tensor([0])
                sem_temp = np.load(os.path.join(self.stim_path, 'movie.npy'))
            else:
                label_temp = torch.Tensor([1])
                sem_temp = np.load(os.path.join(self.stim_path, 'story.npy'))
            data_list.append((torch.Tensor(data_temp)))
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
                
                    