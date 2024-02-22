import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor


class MatformerConv(MessagePassing):
    r"""Periodic Graph Transformers for Crystal Material Property Prediction" <https://arxiv.org/abs/2209.11807>`_ paper
    math:
    Args:
        in_channels (int or Tuple[int, int]): Size of each input sample (node feature dimension).
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of attention heads in multi-head attention.
        concat (bool, optional): If True, concatenate multi-head attentions; otherwise, average them.
        dropout (float, optional): Dropout probability for attention coefficients.
        edge_dim (int or None, optional): Dimensionality of edge features.
        bias (bool, optional): If True, layers will include a learnable bias.
    """

    def __init__(self, in_channels: int | Tuple[int, int], out_channels: int, heads: int = 1, aggr: str = 'add',
                 concat: bool = True, dropout: float = 0.0, edge_dim: int | None = None, bias: bool = True, **kwargs):
        # node_dim: The axis along which to propagate.(default: :obj:`-2`), but in here we have extra dimension for head, so first dimension 
        #  is no more -2, we should 0 to indicate it. 
        super().__init__(node_dim=0,aggr=aggr, **kwargs) 
        
        # Initialize class variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        # Define layers for query, key, value, and edge attribute transformations
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)
        # Edge transformation is conditional on the presence of edge features
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=bias) if edge_dim is not None else self.register_parameter('lin_edge', None)

        # Additional linear transformations and batch normalization
        self.lin_fea = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_concat = nn.Linear(heads * out_channels, out_channels, bias=bias)
        self.line_update = nn.Linear(out_channels * 3, out_channels * 3, bias=bias)

        # Sequential layer for message passing
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels, bias=bias), nn.LayerNorm(out_channels))

        # Batch normalization and layer normalization
        self.bn = nn.BatchNorm1d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)

        # Reset parameters of the model
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the model to their default initialization.
        """
        super().reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_fea.reset_parameters()
        self.lin_concat.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.line_update.reset_parameters()
        self.msg_layer[0].reset_parameters()
        self.msg_layer[1].reset_parameters()
        self.bn.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, x: Tensor | PairTensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        """
        Forward pass of the MatformerConv layer.

        Args:
            x (Tensor or PairTensor): Node feature tensor(s).
            edge_index (Adj): Graph connectivity information.
            edge_attr (OptTensor, optional): Edge feature tensor.

        Returns:
            Tensor: Output tensor after applying the MatformerConv layer.
        """
        # Ensure x is a PairTensor for source and target node features
        if isinstance(x, Tensor):
            x = (x, x)

        # Propagate messages using the defined message and aggregation scheme
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Process the output based on the concatenation flag
        if self.concat:  # Concatenate multi-head attention outputs
            out = out.view(-1, self.heads * self.out_channels)
            out = self.lin_concat(out)
        else:  # Average multi-head attention outputs
            out = out.mean(dim=1)

        # Apply batch normalization and a skip connection
        out = F.silu(self.bn(out))
        out += self.lin_fea(x[0])
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Defines the computation performed at every message passing step.

        Args:
            x_i (Tensor): Source node features.
            x_j (Tensor): Target node features.
            edge_attr (Tensor): Edge attribute features.

        Returns:
            Tensor: Message tensor for each edge.
        """
        # Compute queries, keys, values, and edge attributes for each head
        q_i = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        k_i = self.lin_key(x_i).view(-1, self.heads, self.out_channels)
        k_j = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        e_ij = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels) if edge_attr is not None else None
        v_i = self.lin_value(x_i).view(-1, self.heads, self.out_channels)
        v_j = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        
        # Concatenate Q, K, V for attention computation
        q_ij = torch.concat((q_i, q_i, q_i), dim=-1)
        k_ij = torch.concat((k_i, k_j, e_ij), dim=-1) if e_ij is not None else torch.concat((k_i, k_j), dim=-1)
        v_ij = torch.concat((v_i, v_j, e_ij), dim=-1) if e_ij is not None else torch.concat((v_i, v_j), dim=-1)

        # Compute attention coefficients and apply dropout
        alpha_ij = q_ij * k_ij / math.sqrt(self.out_channels * 3)
        alpha_ij = F.dropout(alpha_ij, p=self.dropout)

        # Update messages and apply message layer
        m_ij = self.sigmoid(self.layer_norm(alpha_ij)) * self.line_update(v_ij)
        m_ij = self.msg_layer(m_ij)
        return m_ij

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class Matformer(nn.Module):
    """
    Matformer model for crystal material property prediction.

    Args:
        atom_input_features (int): Number of features for each atom in the input.
        node_fea (int): Number of features for each node (atom) after embedding.
        edge_fea (int): Number of features for each edge in the input.
        conv_layers (int): Number of convolutional layers in the model.
        hidden_layer (int): Number of features in the hidden layer of the fully connected network.
        heads (int): Number of attention heads in each MatformerConv layer.
        classification (bool): If True, the model will be configured for a classification task.
    """

    def __init__(self, atom_input_features: int = 4, node_fea: int = 64, edge_fea: int = 9, conv_layers: int = 5, 
                hidden_layer: int = 128, heads: int = 4, classification: bool = False):
        super().__init__()
        self.classification = classification

        # Embedding layers for atoms and edges
        self.atom_embedding = nn.Linear(atom_input_features, node_fea)
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_fea, node_fea),
            nn.Softplus(node_fea, node_fea),
            nn.Linear(node_fea, node_fea)
        )

        # Attention layers
        self.att_layers = nn.ModuleList(
            [MatformerConv(in_channels=node_fea, out_channels=node_fea, heads=heads, edge_dim=node_fea)
             for _ in range(conv_layers)]
        )

        # Fully connected layer
        self.fc = nn.Sequential(nn.Linear(node_fea, hidden_layer), nn.SiLU())

        # Output layer
        if self.classification:
            self.fc_out = nn.Linear(hidden_layer, 2)  # Binary classification
            self.softmax = nn.LogSoftmax(dim=1)
        else: 
            self.fc_out = nn.Linear(hidden_layer, 1)  # Regression

    def forward(self, data:Tensor) -> Tensor:
        """
        Forward pass of the Matformer model.

        Args:
            data (Tensor): Input data containing node features, edge features, edge indices, and batch information.

        Returns:
            Tensor: The output of the model, either class logits or regression values.
        """
        node_fea = data.x
        edge_fea = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch

        # Embedding transformations
        node_fea = self.atom_embedding(node_fea)
        edge_fea = self.edge_embedding(edge_fea)

        # Apply attention layers
        for conv_func in self.att_layers:
            node_fea = conv_func(x=node_fea, edge_index=edge_index, edge_attr=edge_fea)

        # Pooling layer for graph-level prediction
        crystal_fea = global_mean_pool(node_fea, batch)

        # Fully connected layer
        crystal_fea = self.fc(crystal_fea)
        out = self.fc_out(crystal_fea)

        # Apply softmax for classification
        if self.classification:
            out = self.softmax(out)

        return out
