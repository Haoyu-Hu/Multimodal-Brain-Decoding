import wandb
import numpy as np

from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from dataset import integrated_dataset, coco_brain
from model import Brain_Modelling, FrozenCLIP
from train_eval import cross_validation_with_val_set

from param_parser import parameter_parser
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main_classification():
    args = parameter_parser()
    # print(args)
    dataset = integrated_dataset(args)
    # print('go dataset')
    # args(666)

    model = Brain_Modelling(args, args.regions, args.time_points)
    # print('hello model')
    model = model.to(device)
    # print(model.__repr__())
    cross_validation_with_val_set(dataset, model, args.folds, args.epochs, args.batch_size, args.lr,
                                  args, args.weight_decay, logger=None)

def main():
   args = parameter_parser()

   accelerator = Accelerator()

   slice_set = [args.z_slice, args.x_slice, args.y_slice]

   train_dataset = coco_brain(args.train_path, slice_set)

   main_model = Brain_Modelling(args).to(device)

   wandb.init(project=args.project_name, name = args.run_name, config = args)

   wandb.watch(main_model, log='all')

   optimizer = torch.optim.Adam(main_model.parameters(), args.lr
                      , weight_decay=args.weight_decay)

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

   main_model, optimizer, train_loader = accelerator.prepare(main_model, optimizer, train_loader)

   clip = FrozenCLIP(args).to(device)

   for epoch in range(args.epochs):
      for fmri, img, caption in train_loader:
         fmri = fmri.to(device)
         embeded_img_list = []
         for img_idx in range(img.size(0)):
            img_sig = img[img_idx].cpu().numpy()
            img_sig = Image.fromarray(img_sig.astype(np.uint8))
            embeded_img_list.append(clip.image_encode(img_sig).squeeze())
         embeded_img = torch.stack(embeded_img_list)
         del embeded_img_list

         feature_txt, feature_img, theo_brain = main_model(fmri)

         theo_loss = F.mse_loss(fmri, theo_brain)

         embeded_text = clip.text_encode(caption)

         loss_text = 0
         loss_text_flag = 1
         for idx in range(embeded_text.size(0)):
            loss_text_temp = soft_clip_loss(feature_txt, embeded_text[idx])
            if loss_text_flag:
                loss_text = loss_text_temp
                loss_text_flag = 0
            elif loss_text > loss_text_temp:
                loss_text = loss_text_temp
         loss_img = soft_clip_loss(feature_img, embeded_img)

         loss = loss_text + loss_img + theo_loss

         optimizer.zero_grad()
         
         accelerator.backward(loss)

         optimizer.step()

         wandb.log({'train_loss': loss.item()
                    , 'text_loss': loss_text
                    , 'img_loss': loss_img
                    , 'brain_loss': theo_loss})

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

if __name__ == '__main__':
  main()