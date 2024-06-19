import wandb
import numpy as np
import math
import os
import tqdm

from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from dataset import integrated_dataset, coco_brain
from model import Brain_Modelling, VAE_Img, Decoder_Img, Vec2Text

from param_parser import parameter_parser
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruct():
   args = parameter_parser()

   wandb.init(project=args.project_name, name = args.run_name, config = args)

   slice_set = [args.z_slice, args.x_slice, args.y_slice]

   main_model = Brain_Modelling(args).to(device)
   main_model.load_state_dict(torch.load(os.path.join(args.save_path, 'best.pt')))
   main_model.eval()

   wandb.watch(main_model, log='all')

   optimizer = torch.optim.Adam(main_model.parameters(), args.lr
                      , weight_decay=args.weight_decay)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   
   test_dataset = coco_brain(args.test, slice_set)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

   img_encoder = VAE_Img(args).to(device)
  #  decoder = Decoder_Img(args).to(device)
   v2t = Vec2Text(mode='invert').to(device)

   test(args, test_loader, main_model, img_encoder, v2t)

def test(args, test_loader, main_model, img_encoder, text_encoder):

  optim_loss = 0
  optim_loss_flag = 1

      # adjust_learning_rate(optimizer, epoch, args)
      
  loss_epoch = 0
  test_store = []
  for count, (fmri, img, caption) in enumerate(tqdm(test_loader)):
      test_temp = dict()

      fmri = fmri.to(device)

      with torch.no_grad():
          feature_txt, feature_img, theo_brain, _ = main_model(fmri)

      theo_loss = F.mse_loss(fmri, theo_brain)

      embeded_text = text_encoder.text_embedding(caption)
      embeded_img = img_encoder.img_embedding(img)
      
      inverted_text = text_encoder.invert_embedding(feature_txt)
      inverted_img = img_encoder.decode_img(feature_img)

      loss_text = soft_clip_loss(feature_txt, embeded_text)
      loss_img = soft_clip_loss(feature_img.view(feature_img.size(0),-1), embeded_img.view(feature_img.size(0),-1))

      loss = loss_text + loss_img
      
      wandb.log({'test_loss': loss.item()
                , 'text_loss': loss_text
                , 'img_loss': loss_img
                , 'brain_loss': theo_loss})
      
      for idx in range(len(inverted_text)):
          test_temp['text'] = inverted_text[idx]
          test_temp['image'] = inverted_img[idx]
          test_store.append(test_temp)
      
  if args.save_test:
      import pickle
      with open(os.path.join(args.save_test, 'results.pkl', 'wb')) as f:
          pickle.dump(test_store, f)

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


if __name__ == '__main__':
    reconstruct()