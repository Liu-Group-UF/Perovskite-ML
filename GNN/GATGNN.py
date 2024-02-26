from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

from torch_scatter import scatter_add

class Composition_Attention(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.node_layer = nn.Linear(neurons+103, 32) # 103 is the number of elements encoded in the element composition vector
        self.atten_layer = nn.Linear(32, 1)

    def forward(self, x, batch, global_feat):
        counts = torch.unique(batch, return_counts=True)[-1]
        graph_feat = torch.repeat_interleave(global_feat, counts, dim=0)
        x = torch.cat([x, graph_feat], dim=-1)
        x = F.softplus(self.node_layer(x))
        x = self.atten_layer(x)
        weights = softmax(x, batch)
        return weights

class Cluster_Attention(nn.Module):
    def __init__(self, neurons_1, neurons_2, num_clusters, cluster_method='random'):
        '''
        Global-Attention Mechanism based on clusters (position grouping) of crystals' elements
        > Defined in paper as *GI-M2*, *GI-M3*, *GI-M4*
        ======================================================================================
        neurons_1       : number of neurons to use for layer_1
        neurons_2       : number of neurons to use for the attention-layer
        num_cluster     : number of clusters to use 
        cluster_method  : unpooling method to use 
            - fixed     : (GI-M2)
            - random    : (GI-M3)
            - learnable : (GI-M4)
        '''          
        super().__init__()
        self.learn_unpool = nn.Linear(neurons_1*2+3, num_clusters)
        self.layer_1 = nn.Linear(neurons_1, neurons_2)
        self.negative_slope = 0.45
        self.attention = nn.Linear(neurons_2, 1)
        self.cluster_method = cluster_method
    
    def forward(self, x, cls, batch):
        r_x = self.unpooling_feature(x, cls, batch)
        r_x = F.leaky_relu(self.layer_1(r_x), self.negative_slope)
        r_x = self.attention(r_x)
        weights = softmax(r_x, batch)
        return weights
    
    def unpooling_feature(self, x, cls, batch):
        g_counts = torch.unique(batch, return_counts=True)[-1].tolist()
        split_x = torch.split(x, g_counts)
        split_cls = torch.split(cls, g_counts)
        new_x = torch.tensor([])

        # break batch into individual graphs
        for i in range(len(split_x)):
            graph_feat = split_x[i]
            cluster_t = split_cls[i].view(-1)
            cluster_sum = scatter_add(graph_feat, cluster_t, dim=0)
            zero_sum = torch.zeros_like(cluster_sum)
            if len(graph_feat) == 1:
                new_x = torch.cat([new_x, cluster_sum],dim=0)
            elif len(graph_feat) == 2:
                new_x = torch.cat([new_x, cluster_sum, cluster_sum],dim=0)
            else:
                region_arr = np.array(cluster_t.tolist())
                # choose unpooling method
                if self.cluster_method == 'fixed': # GI-M2
                    # fixed unpooling
                    random_sets = cluster_t.tolist()
                elif self.cluster_method == 'random': # GI-M3
                    # random unpooling
                    random_sets = [np.random.choice(np.setdiff1d(region_arr, i)) for i in region_arr]
                elif self.cluster_method == 'learnable':   # GI-M4
                    # learnable unpooling
                    total_feat = graph_feat.sum(dim=0).unsqueeze(0)
                    region_input = torch.cat([graph_feat,total_feat,cluster_t.unsqueeze(0).float()], dim=-1)
                    random_sets = torch.argmax(F.softmax(self.learn_unpool(region_input)), dim=1).tolist()

                # normalized regions
                unique, counts = np.unique(region_arr, return_counts=True)
                counts = counts/counts.sum()
                set_dict = dict(zip(unique, counts))
                random_ratio = torch.tensor([set_dict[i] for i in random_sets])
                random_ratio = (random_ratio/random_ratio.sum()).view(-1,1)

                cluster_sum =cluster_sum[random_sets]
                cluster_sum = cluster_sum*random_ratio
                new_x = torch.cat([new_x, cluster_sum],dim=0)

        return new_x
    
class AGATConv(MessagePassing):
    def __init__(self, in_features: int, out_features: int, edge_dim: int, heads: int = 1, aggr: str = 'add',
                 concat: bool = True, bias: bool = True, dropout: float = 0., **kwargs):
        """
        =======================================================================
        in_features    : input-features
        out_features   : output-features
        edge_dim       : edge-features
        heads          : attention-heads
        aggr           : agregation method
        concat         : to concatenate the attention-heads or sum them
        bias           : True 
        dropout        : 0
        =======================================================================
        """
        super().__init__(aggr=aggr, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = 0.2   
        self.prelu = nn.PReLU() 
        self.bn = nn.BatchNorm1d(heads)
        self.W = nn.Parameter(torch.Tensor(in_features+edge_dim, heads * out_features))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

        def reset_parameters(self):
            glorot(self.W)
            glorot(self.att)
            zeros(self.bias)
        
        def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        def message(self, edge_index_i , x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
            x_i = torch.cat([x_i, edge_attr], dim=-1)
            x_j = torch.cat([x_j, edge_attr], dim=-1)

            x_i = F.softplus(torch.matmul(x_i, self.W).view(-1, self.heads, self.out_features))
            x_j = F.softplus(torch.matmul(x_j, self.W).view(-1, self.heads, self.out_features))

            alpha = F.softplus((torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1))
            alpha = F.softplus(self.bn(alpha))
            alpha = softmax(alpha, edge_index_i)

            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            out = (x_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)
            return out

        def update(self, aggr_out: Tensor) -> Tensor:
            if self.concat:
                aggr_out = aggr_out.view(-1, self.heads * self.out_features)
            else:
                aggr_out = aggr_out.mean(dim=0)

            if self.bias is not None:
                aggr_out = aggr_out + self.bias
            return aggr_out


class GATGNN(nn.Module):
    def __init__(self, heads, classification: bool = False, neurons: int = 64, extra_layer: bool = False, global_attention: str = 'composition',
                 num_conv_layer: int =3, unpooling_technique: str = 'random', concat_comp: bool = False, edge_foramt: str = 'CGCNN'):
        super().__init__()
        self.heads = heads
        self.classification = classification
        self.neurons = neurons
        self.addition = extra_layer 
        self.global_attention = global_attention
        self.unpooling = unpooling_technique
        self.concat_comp = concat_comp
        self.negative_slope = 0.2
        
        n_h, n_h_2 = neurons, neurons*2

        self.embedding_n = nn.Linear(92, n_h)
        self.embedding_e = nn.Linear(41, n_h) if edge_foramt == 'CGCNN' else nn.Linear(9, n_h)
        self.embedding_comp = nn.Linear(103, n_h)

        self.node_conv = nn.ModuleList([AGATConv(n_h, n_h, n_h, heads=self.heads) for _ in range(num_conv_layer)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(n_h) for _ in range(num_conv_layer)])
        self.cluster_attention = Cluster_Attention(n_h, n_h, 3, cluster_method=self.unpooling)
        self.composition_attention = Composition_Attention(n_h)

        if self.concat_comp:
            reg_h = n_h_2
        else:
            reg_h = n_h
        
        if self.addition:
            self.linear1 = nn.Linear(reg_h, reg_h)
            self.linear2 = nn.Linear(reg_h, reg_h)
        
        if self.classification:
            self.fc_out = nn.Linear(reg_h, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(reg_h, 1)

    def forward(self, data:Tensor) -> Tensor:
        x, edge_index, edge_attr, batch, global_feat, cluster = data.x, data.edge_index, data.edge_attr, data.batch, data.global_feat, data.cluster

        x = self.embedding_n(x)
        edge_attr = self.embedding_e(edge_attr) 
        edge_attr = F.leaky_relu(edge_attr, self.negative_slope)
        
        for i, conv_func in enumerate(self.node_conv):
            x = conv_func(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = x if self.batch_norm[i] is None else self.batch_norm[i](x)
            x = F.softplus(x)
        
        if self.global_attention == 'composition':
            weights = self.composition_attention(x, batch, global_feat)
        elif self.global_attention in ['cluster', 'unpooling', 'clustering']:
            weights = self.cluster_attention(x, cluster, batch)

        x = x * weights

        # Pooling layer for graph-level prediction
        crystal_fea = global_add_pool(x, batch)

        if self.concat_comp:
            crystal_fea = torch.cat([crystal_fea, F.leaky_relu(self.embedding_comp(global_feat),self.negative_slope)], dim=-1)
        
        if self.addition:
            crystal_fea = F.softplus(self.linear1(crystal_fea))
            crystal_fea = F.softplus(self.linear2(crystal_fea))
        
        if self.classification:
            crystal_fea = self.softmax(self.fc_out(crystal_fea))
        else:
            crystal_fea = self.fc_out(crystal_fea)
        
        return crystal_fea
        
        
   












