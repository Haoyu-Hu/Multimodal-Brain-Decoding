import torch

from dataset import integrated_dataset
from model import Brain_Modelling
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

if __name__ == '__main__':
  main_classification()