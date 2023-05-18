from statistics import mode
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import random

class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=2):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    def forward(self,model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list,self.weight_decay,p=self.p)
        return reg_loss

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'extra' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss

# def get_extra_edge_index(x, pair_num=2 , connect_num=5):
#     x = x.data.cpu()
#     res = [[],[]]
#     corrs = []
#     num1 = num_max = x.shape[0]
#     for i in range(num_max):
#         for j in range(connect_num):
#             rand_choice = random.sample(range(x.shape[0]), pair_num)
#             xx = (x[rand_choice[0], :] + x[rand_choice[1], :])/ pair_num
#             coef = np.corrcoef(x[i], xx)
#             if abs(coef[0][1]) > 0.8:
#                 corrs.append(coef[0][1])
#                 x = torch.cat((x, xx.unsqueeze(0)), 0)
#                 res[0].append(i)
#                 res[1].append(num1)
#                 num1 += 1
#     return x, torch.tensor(res), torch.tensor(corrs)



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    reg_loss = Regularization(model, 1e-8, p=2).to(config['device'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 5  # early stop

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            loss = loss + reg_loss(model)
            
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1

        # extra_edge_index = get_extra_edge_index(xx[-1,:,:])
        
        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss



    return train_loss_list
