from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.typing import (Adj, # Adj = Union[Tensor, SparseTensor] 
                            OptTensor, # OptTensor = Optional[Tensor] 
                            PairTensor) # Tuple[Tensor, Tensor]

class CGConv(MessagePassing):
    """
    Args: 
        channels (int or tuple): Size of each input sample. A tuple
                corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (str, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`. 
   
   Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F_{t})` if bipartite
    """
    def __init__(self, channels:int|Tuple[int,int], dim:int = 0, aggr:str = "add", 
                 batch_norm:bool = True, bias:bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.aggr = aggr
        
        if isinstance(channels, int):
            channels = (channels, channels) # this for add dimension of xi and xj
        
        self.linear_f = Linear(sum(channels) + dim, channels[1], bias =  bias)
        self.linear_s = Linear(sum(channels) + dim, channels[1], bias =  bias)
        
        if batch_norm:
            self.batch_norm = BatchNorm1d(channels[1])
        else:
            self.batch_norm = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        self.linear_f.reset_parameters()
        self.linear_s.reset_parameters()
        if self.batch_norm is not None:
            self.batch_norm.reset_parameters()
    
    def forward(self, x: Tensor|PairTensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.batch_norm is None else self.batch_norm(out)
        out = out + x[1]
        
        return out
    
    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor: # args pass to message will also pass to 
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return F.sigmoid(self.linear_f(z)) * F.softplus(self.linear_s(z))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'

    
class CGCNN(nn.Module):
    '''
    Args: 
        node_fea_len: length of node feature
        edge_fea_len: length of edge feature
        orig_atom_fea_len: length of intial input node feature
        num_conv_layer: number of how many CGConv layer will be used 
        hidden_layer_len: number of hidden features after pooling
    '''
    def __init__(self, node_fea_len:int, edge_fea_len:int, orig_node_fea_len:int, 
                 num_conv_layer:int = 3, hidden_layer_len:int = 128, classification:bool = False):
        super().__init__()
        self.classification = classification
        self.embedding = Linear(orig_node_fea_len, node_fea_len)
        self.convs = nn.ModuleList([CGConv(channels=node_fea_len, dim=edge_fea_len) 
                                   for _ in range(num_conv_layer)])
        self.conv_to_fc = Linear(node_fea_len, hidden_layer_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if self.classification:
            self.fc_out = Linear(hidden_layer_len, 2) # True or False
            self.dropout = nn.Dropout()
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(hidden_layer_len, 1)        
    
    def forward(self, data:Tensor) -> Tensor:
        node_fea = data.x
        edge_fea = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch
        
        node_fea = self.embedding(node_fea)
        for conv_func in self.convs:
            node_fea = conv_func(x=node_fea, edge_index=edge_index, edge_attr=edge_fea)        
        
        #pooling layer
        crystal_fea = global_mean_pool(node_fea, batch)
        
        crystal_fea = self.conv_to_fc(self.conv_to_fc_softplus(crystal_fea))
        crystal_fea = self.conv_to_fc_softplus(crystal_fea)
        if self.classification:
            crystal_fea = self.dropout(crystal_fea)
            
        out = self.fc_out(crystal_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out    