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

def main():
    args = parameter_parser()

    model = VCM(args).to(device)
    print('loading model...')
    model.load_state_dict(torch.load(args.vcm_path))
    print('Done!')

    subject = args.subject

    data_dict = torch.load(args.data_dict)

    index_data = np.load(args.index_path)

    border_mask = torch.Tensor(np.load(args.border_path)).to(device)

    for data in data_dict:
        if data['subject'] == subject:
            break

    theo_latent = data['theo_brain']
    high_latent = data['high_states']
    low_latent = data['low_states']

    root_path, file_path = os.path.split(args.data_dict)
    root_path = os.path.join(root_path, subject)

    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    print('processing and saving theoretical brain activity...')
    theo_recon = model(theo_latent, index=index_data, border_mask=border_mask, mode='decode')
    np.save(os.path.join(root_path, 'theo_recon.npy'), theo_recon.detach().cpu().numpy())
    print('processing and saving high-level brain activities...')
    high_recon = model(high_latent, index=index_data, border_mask=border_mask, mode='decode')
    np.save(os.path.join(root_path, 'high_recon.npy'), high_recon.detach().cpu().numpy())
    print('processing and saving low-level brain activities...')
    for i in range(low_latent.size(1)):
        low_recon = model(low_latent[:,i,:,:], index=index_data, border_mask=border_mask, mode='decode')
        np.save(os.path.join(root_path, str(i)+'_low_recon.npy'), low_recon.detach().cpu().numpy())
    print('Done!')

    print('saving others...')
    np.save(os.path.join(root_path, 'lambda.npy'), data['lambda'].detach().cpu().numpy())
    np.save(os.path.join(root_path, 'alpha.npy'), data['alpha'].detach().cpu().numpy())

    print('All done!')

    return 1

if __name__ == '__main__':
    main()

