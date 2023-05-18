import gc
from platform import node
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from math import sqrt
from .graph_layer import GraphLayer, GraphLayer1
import random

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

def get_inter_index(tmp, cos, batch_num, node_num):
    index_i = []
    index_j = []
    cos_n = []
    for i in range(node_num):
        k = len(tmp[i])
        index_i.extend([i]*k)
        index_j.extend(tmp[i])
        cos_index = torch.tensor(tmp[i]).to('cuda')
        cos_n.extend(torch.index_select(cos[i], 0, cos_index).tolist())

    inter_index = torch.tensor([index_i, index_j])
    inter_index = get_batch_edge_index(inter_index, batch_num, node_num)

    return inter_index, torch.tensor(cos_n*batch_num)

def get_corr(fake_Y, Y):#计算两个向量person相关系数
        fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
        fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
        corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
                    torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
        return corr

def get_extra_edge_index(x, pair_num=2 , connect_num=5):

    res = [[],[]]
    corrs = []
    num1 = num_max = x.shape[0]
    for i in range(num_max):
        for j in range(connect_num):
            rand_choice = random.sample(range(x.shape[0]), pair_num)
            xx = (x[rand_choice[0], :] + x[rand_choice[1], :])/ pair_num
            coef = get_corr(x[i], xx)
            # coef = np.corrcoef(x[i], xx)
            if abs(coef) > 0.8:
                corrs.append(coef)
                x = torch.cat((x, xx.unsqueeze(0)), 0)
                res[0].append(i)
                res[1].append(num1)
                num1 += 1
    return x, torch.tensor(res), torch.tensor(corrs)




class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, cos_num, inter_dim=0, heads=1, node_num=100 ):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, cos_num, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, cos_topk, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, cos_topk, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)

class GNNLayer1(nn.Module):
    def __init__(self, in_channel, out_channel, cos_num, inter_dim=0, heads=1 ):
        super(GNNLayer1, self).__init__()


        self.gnn = GraphLayer1(in_channel, out_channel, cos_num, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, corrs):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, corrs , return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = out.type(torch.FloatTensor)
        
        return self.relu(out)


class GGNN(nn.Module):
    def __init__(self, edge_index_sets, node_num, batch_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, heads=1):

        super(GGNN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)  # .from_pretrained(weight, freeze=False)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, node_num, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])
        self.gnn_layer = GNNLayer1(dim, dim, node_num, inter_dim=dim+embed_dim, heads=1)

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.extra_pram = Parameter(torch.Tensor(node_num*batch_num, node_num*batch_num))

        self.init_params()
    
    def init_params(self):
        nn.init.sparse_(self.extra_pram, 0.8, 1)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))


    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()


        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num* batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            # # #######
            # # # implicit
            learned_graph = torch.mm(weights, weights.transpose(0, 1))
            norm = torch.norm(weights, p=2, dim=1, keepdim=True)
            norm = torch.mm(norm, norm.transpose(0, 1))
            learned_graph = learned_graph / norm
            learned_graph = (learned_graph + 1) / 2.
            # learned_graph = F.sigmoid(learned_graph)
            learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
            adj = F.gumbel_softmax(learned_graph, tau=1, hard=True)
            adj = adj[:, :, 0].clone().reshape(node_num, -1)
            # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
            mask = torch.eye(node_num, node_num).bool().cuda()
            adj.masked_fill_(mask, 0)
            adj = adj.repeat(batch_num, batch_num)
            # tmp_list = []
            # for iii in range(node_num):
            #     tmp = []
            #     for jjj in range(node_num):
            #         if adj[iii][jjj] > 0:
            #             tmp.append(jjj)
            #     tmp_list.append(tmp)
            # final_list = []
            # for iii in range(node_num):
            #     final_list.append(list(set((topk_indices_ji.tolist())[iii]+tmp_list[iii])))
            
            # inter_edge_index, cos_topk = get_inter_index(final_list, cos_ji_mat, batch_num, node_num)
            # inter_edge_index = inter_edge_index.to(device)
            # # ###

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_mat, topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)
            
            # cos
            cos_topk = topk_mat.flatten().repeat(batch_num).unsqueeze(1)
            
            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)

            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, cos_topk, node_num=node_num* batch_num, embedding=all_embeddings)


            # # multi-connect
            # # gcn_out = gcn_out.clone().detach()
            # gcn_out, extra_edge_index, corrs = get_extra_edge_index(gcn_out, pair_num = 2, connect_num = 2)
            # gcn_out = gcn_out.to(device)
            # extra_edge_index = extra_edge_index.to(device)
            # self_loop = torch.tensor([1]*gcn_out.shape[0], dtype= torch.float32)
            # corrs = torch.cat((corrs, self_loop),0)
            # corrs = corrs.to(device)
            # gcn_out = self.gnn_layer(gcn_out, extra_edge_index, corrs)
            # gcn_out = gcn_out[:gcn_out.shape[0]-extra_edge_index.shape[1],:]
            

            extra_num = 25
            len_x = int(gcn_out.shape[0])
            
            if self.extra_pram.shape[1] == gcn_out.shape[0]:
                extra_features = torch.matmul(gcn_out.T, self.extra_pram).T
                # extra_features = torch.cat((extra_features, torch.matmul(gcn_out.T, adj).T), 0)
                gcn_out = torch.cat((gcn_out, extra_features), 0)
                cos_ji_mat = torch.matmul(gcn_out, gcn_out.T)
                normed_mat = torch.matmul(gcn_out.norm(dim=-1).view(-1,1), gcn_out.norm(dim=-1).view(1,-1))
                cos_ji_mat = cos_ji_mat / normed_mat

                topk_mat, topk_indices_ji = torch.topk(cos_ji_mat, extra_num, dim=-1)

                # cos coefficient
                corrs = topk_mat[0:len_x,:].flatten().unsqueeze(1)

                extra_i = torch.arange(0, len_x).T.unsqueeze(1).repeat(1, extra_num).flatten().to(device).unsqueeze(0)
                extra_j = topk_indices_ji[0:len_x,:].flatten().unsqueeze(0)
                extra_edge_index = torch.cat((extra_i,extra_j), dim=0)

                gcn_out = self.gnn_layer(gcn_out, extra_edge_index, corrs)
                gcn_out = gcn_out[0:len_x,:]

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1).to(device)

        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        return out
        


class Model(nn.Module):
    def __init__(self, edge_index_sets, node_num, batch_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, heads=1):

        super(Model, self).__init__()

        self.heads = heads

        # self.dim_out = dim*heads
        # self.Q = Linear(input_dim, dim*heads)
        # self.K = Linear(input_dim, dim*heads)
            
        # self.relu = nn.ReLU()
        self.ggnn = GGNN(edge_index_sets, node_num, batch_num, dim, out_layer_inter_dim, input_dim, out_layer_num, topk=topk, heads=heads)
        
        self.init_params()

    def init_params(self):
        pass

    def forward(self, data, edge_index):

        # batch, node_num, dim_in = data.shape
        # Q_R = self.Q(data).reshape(batch, node_num, self.heads, self.dim_out//self.heads).transpose(1,2)
        # K_R = self.K(data).reshape(batch, node_num, self.heads, self.dim_out//self.heads).transpose(1,2)
        
        # R = torch.matmul(Q_R, K_R.transpose(2, 3))*(1 / sqrt(self.dim_out // self.heads))
        # R = torch.softmax(R, dim=-1)

        # R = F.gumbel_softmax(R, tau=0.8, hard=False)

        out = self.ggnn(data, edge_index)

        return out
