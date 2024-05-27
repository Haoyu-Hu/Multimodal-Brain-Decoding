import time
import math
import json
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(6789)
np.random.seed(6789)
torch.cuda.manual_seed_all(6789)
os.environ['PYTHONHASHSEED'] = str(6789)

def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size, lr,
                                  args, weight_decay, logger=None):
    val_losses, val_accs, test_accs, durations = [], [], [], []
    test_maes, test_rmses = [], []

    save_path = '/content/drive/MyDrive/Cam_proj/Multimodal/Results/state_' + str(args.n_state)
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        # print(len(train_idx))
        # print(dataset[1])
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        val_dataset = Subset(dataset, val_idx)

        # print(len(train_idx))
        # print(len(train_dataset))
        # print(len(dataset))
        # print(len(Subset(dataset, [0,1])))

        fold_val_losses = []
        fold_val_accs = []
        fold_test_accs = []
        fold_test_maes = []
        fold_test_rmses = []

        infos = dict()

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = 0
        val_flag = 0

        for epoch in range(1, epochs + 1):

            train_loss, _, _, _, _, train_datalist = train_DR(model, optimizer, train_loader)
            val_loss, val_acc, val_mae, val_rmse, val_datalist = eval_DR(model, val_loader)
            
            val_losses.append(val_loss)
            fold_val_losses.append(val_loss)

            test_loss, test_acc, test_mae, test_rmse, test_datadict = eval_DR(model, test_loader)
            test_maes.append(test_mae)
            test_rmses.append(test_rmse)
            test_accs.append(test_acc)
            fold_test_maes.append(test_mae)
            fold_test_rmses.append(test_rmse)
            fold_test_accs.append(test_acc)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'test_acc': test_acc,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
            infos[epoch] = eval_info

            if logger is not None:
                logger(eval_info)

            # if epoch % lr_decay_step_size == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_decay_factor * param_group['lr']

            adjust_learning_rate(optimizer, epoch, args)
            if val_flag == 0:
                best_val_loss = val_loss
                val_flag = 1
                datalist_file = os.path.join(save_path, 'data_list.pt')
                torch.save(test_datadict, datalist_file)
            else:
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    datalist_file = os.path.join(save_path, 'data_list.pt')
                    torch.save(test_datadict, datalist_file)
            
            if epoch % 10 == 0:
                print('Epoch: {:d}, train loss: {:.3f}, val loss: {:.5f},, val_acc: {:.3f}, val mae: {:.3f}, test_acc: {:.3f}, test mae: {:.3f}'
                      .format(epoch, eval_info["train_loss"], eval_info["val_loss"], eval_info["val_acc"], eval_info["val_mae"], eval_info["test_acc"], eval_info["test_mae"]))
                # model(1,1)

        fold_val_loss, argmin = tensor(fold_val_losses).min(dim=0)
        fold_test_mae = fold_test_maes[argmin]
        fold_test_rmse = fold_test_rmses[argmin]
        fold_test_acc = fold_test_accs[argmin]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        log_path = os.path.join(save_path, 'log')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        log_file = os.path.join(log_path, str(fold)+'.json')
        with open(log_file, 'w') as font:
            json.dump(infos, font)
        
        print('Fold: {:d}, Val loss: {:.3f}, Test acc:{:.3f}, Test mae: {:.3f}, Test rmse: {:.3f}'
              .format(eval_info["fold"], fold_val_loss, fold_test_acc, fold_test_mae, fold_test_rmse))


    val_losses, duration = tensor(val_losses), tensor(durations)
    val_losses = val_losses.view(folds, epochs)
    test_maes, test_rmses, test_accs = tensor(test_maes), tensor(test_rmses), tensor(test_accs)
    test_maes, test_rmses, test_accs = test_maes.view(folds, epochs), test_rmses.view(folds, epochs), test_accs.view(folds, epochs)


    min_val_loss, argmin = val_losses.min(dim=1)
    test_mae = test_maes[torch.arange(folds, dtype=torch.long), argmin]
    test_rmse = test_rmses[torch.arange(folds, dtype=torch.long), argmin]
    test_acc = test_accs[torch.arange(folds, dtype=torch.long), argmin]

    val_loss_mean = min_val_loss.mean().item()
    duration_mean = duration.mean().item()

    test_mae_mean = test_mae.mean().item()
    test_mae_std = test_mae.std().item()
    test_rmse_mean = test_rmse.mean().item()
    test_rmse_std = test_rmse.std().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    print('Val Loss: {:.4f}, Test ACC: {:.3f}+{:.3f}, Test MAE: {:.3f}+{:.3f}, Test RMSE: {:.3f}+{:.3f}, Duration: {:.3f}'
          .format(val_loss_mean, test_acc_mean, test_acc_std, test_mae_mean, test_mae_std, test_rmse_mean, test_rmse_std, duration_mean))

    return 1

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=6789)

    test_indices, train_indices = [], []

    for _, idx in skf.split(torch.zeros(len(dataset)), [0]*len(dataset)):
            test_indices.append(torch.from_numpy(idx).to(torch.long))
    # print(len(train_indices))
    # print(len(dataset))
    # print(dataset.y)
    # print(len(test_indices[0]))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    # print(len(train_indices[0]))
    # print(len(val_indices[0]))
    # print(len(dataset))

    return train_indices, test_indices, val_indices

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

def train_DR(model, optimizer, loader):
    model.train()

    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_loss_ta = 0
    total_loss_se = 0

    data_list = []
    r_dict = {}
    
    for data_batch in loader:
        data_dict = {}
        optimizer.zero_grad()
        data = data_batch['data'].to(device).squeeze()
        data[torch.isnan(data)] = 0
        # data = data[~torch.isnan(data)].reshape(data.shape[0], data.shape[1], -1)
        yse = data_batch['yse'].to(device).squeeze()
        yta = data_batch['yta'].to(device).squeeze().long()
        # print(data.size())
        logits_ta, logits_tah, logits_se, lr_loss, theo_brain, high_states, low_states, w_lambda, w_alpha = model(data)

        data_dict['theo_brain'] = theo_brain
        data_dict['high_states'] = high_states
        data_dict['low_states'] = low_states
        data_dict['lambda'] = w_lambda
        data_dict['alpha'] = w_alpha
        data_dict['subject'] = data_batch['subfolder'][0]
        data_list.append(data_dict)

        # print(beta*w_loss)
        loss1 = nn.CrossEntropyLoss()(logits_ta.squeeze(), yta)
        loss1_h = nn.CrossEntropyLoss()(logits_tah.squeeze(), yta)
        loss2 = nn.MSELoss()(logits_se.squeeze(), yse)
        theo_loss = nn.MSELoss()(theo_brain, data)
        mae_loss = F.l1_loss(logits_se.squeeze(), yse)
        rmse_loss = torch.sqrt(loss2)

        # loss =  1000*loss1 + loss2 + lr_loss + model.beta*kl_loss
        loss =  loss1 + loss2 + lr_loss - loss1_h + theo_loss

        loss.backward()
        total_loss += loss.item() * data.size(0)
        total_loss_ta += loss1.item()*data.size(0)
        total_loss_se += loss2.item()*data.size(0)

        total_mae += mae_loss.item() * data.size(0)
        total_rmse += rmse_loss.item() * data.size(0)
        optimizer.step()

        len_brain_series = len(loader.dataset)*2

    return total_loss / len_brain_series, total_loss_ta / len_brain_series \
        , total_loss_se / len_brain_series, total_mae / len_brain_series, total_rmse / len_brain_series \
        , data_list

def eval_DR(model, loader):
    model.eval()

    total_loss = 0
    total_mae = 0
    total_rmse = 0
    correct = 0
    data_list = []
    # print('val_or_test')
    for data_batch in loader:
        data_dict = {}
        data = data_batch['data'].to(device).squeeze()
        data[torch.isnan(data)] = 0
        yse = data_batch['yse'].to(device).squeeze()
        yta = data_batch['yta'].to(device).squeeze().long()
        with torch.no_grad():
            logits_ta, logits_tah, logits_se, lr_loss, theo_brain, high_states, low_states, w_lambda, w_alpha = model(data)
            pred = logits_ta.squeeze().max(1)[1]

            data_dict['theo_brain'] = theo_brain
            data_dict['high_states'] = high_states
            data_dict['low_states'] = low_states
            data_dict['lambda'] = w_lambda
            data_dict['alpha'] = w_alpha
            data_dict['subject'] = data_batch['subfolder'][0]
            data_list.append(data_dict)
        # print(w_loss*beta)
        # print(beta)
        correct += pred.squeeze().eq(yta.view(-1)).sum().item()
        loss1 = nn.CrossEntropyLoss()(logits_ta.squeeze(), yta)
        loss1_h = nn.CrossEntropyLoss()(logits_tah.squeeze(), yta)
        loss2 = nn.MSELoss()(logits_se.squeeze(), yse)
        theo_loss = nn.MSELoss()(theo_brain, data)

        mae_loss = F.l1_loss(logits_se.squeeze(), yse)
        rmse_loss = torch.sqrt(loss2)

        loss =  loss1 + loss2 + lr_loss - loss1_h + theo_loss
        total_loss += loss.item() * data.size(0)
        total_mae += mae_loss.item() * data.size(0)
        total_rmse += rmse_loss.item() * data.size(0)

        len_brain_series = len(loader.dataset)*2

    return total_loss / len_brain_series, correct / len_brain_series, total_mae / len_brain_series, total_rmse / len_brain_series \
            , data_list