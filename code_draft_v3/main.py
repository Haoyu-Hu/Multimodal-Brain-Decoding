import wandb
import numpy as np
import math

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

   train_dataset = coco_brain(args.train_path, slice_set)

   main_model = Brain_Modelling(args).to(device)

   wandb.watch(main_model, log='all')

   optimizer = torch.optim.Adam(main_model.parameters(), args.lr
                      , weight_decay=args.weight_decay)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

   main_model, optimizer, train_loader = accelerator.prepare(main_model, optimizer, train_loader)

   img_encoder = VAE_Img(args).to(device)
  #  decoder = Decoder_Img(args).to(device)
   v2t = Vec2Text(mode='embed').to(device)

   for epoch in range(args.epochs):
      # adjust_learning_rate(optimizer, epoch, args)
      
      for fmri, img, caption in train_loader:

        #  print(caption)
        #  print(caption[0][:3])
        #  print(caption[0][3:6])

         fmri = fmri.to(device)
        #  embeded_img_list = []
        #  for img_idx in range(img.size(0)):
        #     img_sig = img[img_idx].cpu().numpy()
        #     img_sig = Image.fromarray(img_sig.astype(np.uint8))
        #     embeded_img_list.append(img_encoder.image_encode(img_sig).squeeze())
        #  embeded_img = torch.stack(embeded_img_list)
        #  del embeded_img_list
         embeded_img = img_encoder.encode_img(img)

         feature_txt, feature_img, theo_brain, _ = main_model(fmri)

         theo_loss = F.mse_loss(fmri, theo_brain)

         embeded_text = v2t.text_embedding(caption)

         loss_text = soft_clip_loss(feature_txt, embeded_text)
         loss_img = soft_clip_loss(feature_img.view(feature_img.size(0),-1), embeded_img.view(feature_img.size(0),-1))

         loss = loss_text + loss_img

         optimizer.zero_grad()
         
         accelerator.backward(loss)

         optimizer.step()
         
         wandb.log({'train_loss': loss.item()
                    , 'text_loss': loss_text
                    , 'img_loss': loss_img
                    , 'brain_loss': theo_loss
                    , 'epoch': epoch
                    , 'learning_rate': get_lr(optimizer)})

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