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

def main():
   args = parameter_parser()

   accelerator = Accelerator()

   wandb.init(project=args.project_name, name = args.run_name, config = args)

   slice_set = [args.z_slice, args.x_slice, args.y_slice]

   main_model = Brain_Modelling(args).to(device)

   wandb.watch(main_model, log='all')

   optimizer = torch.optim.Adam(main_model.parameters(), args.lr
                      , weight_decay=args.weight_decay)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   
   train_dataset = coco_brain(args.train_path, slice_set)
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

   main_model, optimizer, train_loader = accelerator.prepare(main_model, optimizer, train_loader)

   img_encoder = VAE_Img(args).to(device)
  #  decoder = Decoder_Img(args).to(device)
   v2t = Vec2Text(mode='embed').to(device)

   train(args, train_loader, main_model, img_encoder, v2t, optimizer, scheduler, accelerator)


def train(args, train_loader, main_model, img_encoder, text_encoder, optimizer, scheduler=None, accelerator=None):

   optim_loss = 0
   optim_loss_flag = 1
   for epoch in range(args.epochs):
      # adjust_learning_rate(optimizer, epoch, args)
      
      loss_epoch = 0
      for count, (fmri, img, caption) in enumerate(train_loader):

         fmri = fmri.to(device)

         feature_txt, feature_img, theo_brain, _ = main_model(fmri)

         theo_loss = F.mse_loss(fmri, theo_brain)

         embeded_text = text_encoder.text_embedding(caption)
         embeded_img = img_encoder.img_embedding(img)

         loss_text = soft_clip_loss(feature_txt, embeded_text)
         loss_img = soft_clip_loss(feature_img.view(feature_img.size(0),-1), embeded_img.view(feature_img.size(0),-1))

         loss = loss_text + loss_img + args.beta*theo_loss
         loss_epoch += loss.item()

         optimizer.zero_grad()
         
         accelerator.backward(loss)

         optimizer.step()
         
         wandb.log({'train_loss': loss.item()
                    , 'text_loss': loss_text
                    , 'img_loss': loss_img
                    , 'brain_loss': theo_loss
                    , 'epoch': epoch
                    , 'learning_rate': get_lr(optimizer)})
      
      if args.save_model:
          if optim_loss_flag:
            optim_loss = loss_epoch/(count+1)
            torch.save(main_model.state_dict(), os.path.join(args.save_model, 'best_model.pt'))
            optim_loss_flag = 0
          elif optim_loss > loss_epoch/(count+1):
            optim_loss = loss_epoch/(count+1)
            torch.save(main_model.state_dict(), os.path.join(args.save_model, 'best_model.pt'))

      scheduler.step()

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
  main()