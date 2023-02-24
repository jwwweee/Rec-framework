import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from base.BPR_MF import *


class NGCFConv(nn.Module):
    def __init__(self, layer_size):
        super(NGCFConv, self).__init__()
        
        self.layer_size = layer_size

        # weights for different types of messages
        self.W1 = nn.Linear(layer_size, layer_size, bias=True)
        self.W2 = nn.Linear(layer_size, layer_size, bias=True)

        # leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        # initialize weights
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

    def forward(self, L, I, E):
        
        # left side of the equation
        ego_messages = torch.matmul((L + I), E)
        ego_embeddings = self.W1(ego_messages)

        # right side of the equation
        side_messages = torch.matmul(L, E)
        side_embeddings = self.W2(side_messages)

        return ego_embeddings + side_embeddings
    

class NGCF(BPR_MF):
    def __init__(self, num_users, num_items, args):
        super().__init__(args)
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(args.layer_size)
        self.device = args.device
        self.embed_size = args.embed_size
        self.layer_size = eval(args.layer_size)

        self.message_dropout = eval(args.mess_dropout)
        self.node_dropout = eval(args.node_dropout)[0] 

        # initialize the parameters of embeddings
        initializer = nn.init.xavier_uniform_

        # initial embeddings layers
        self.parameter_list = nn.ParameterDict({
            'embed_user': nn.Parameter(initializer(torch.empty(self.num_users,
                                                 self.embed_size))),
            'embed_item': nn.Parameter(initializer(torch.empty(self.num_items,
                                                 self.embed_size)))
        })

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layer):
            layer = NGCFConv(self.layer_size[i])
            self.conv_layers.append(layer)
        
        self = self.to(self.device)

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """ Model feedforward procedure
        """
        # node dropout
        self.L = self.sparse_dropout(self.L,
                                    self.node_dropout,
                                    self.L._nnz()) if self.node_dropout else self.L

        # initial concatenated embeddings (users and items)
        E = torch.cat([self.parameter_list['embed_user'], self.parameter_list['embed_item']], dim=0)
        
        # message propagation for each layer (both user and item phases)
        concat_g_embeddings = E.clone()

        for i in range(self.num_layer):
            E = self.conv_layers[i](self.L, self.I, E) 
            E = F.leaky_relu(E, negative_slope=0.2)

            # message dropout
            if self.message_dropout:
              E = nn.Dropout(self.message_dropout[i])(E)

              E = F.normalize(E, dim=1, p=2)

            concat_g_embeddings = torch.cat([concat_g_embeddings, E.clone()], dim=1)
        
        # separate users and items
        user_g_embeddings, item_g_embeddings = concat_g_embeddings[:self.num_users], concat_g_embeddings[self.num_users:]

        # retrieve batched users and items
        batch_user_g_embeddings = user_g_embeddings[batch_user, :]

        # get positive items representations
        batch_pos_items_repr = item_g_embeddings[batch_pos_item, :]

        # get negative items representations
        batch_neg_items_repr = item_g_embeddings[batch_neg_item, :]

        return batch_user_g_embeddings, batch_pos_items_repr, batch_neg_items_repr

    def model_initialize(self, sparse_graph):
        """ Initialize the model, create the saprse Laplacian matrix for user-item interaction matrix.
            
            interact_matrix size: interaction num x 2

        """

        # interaction adjacent matrices
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = sparse_graph.tolil()

        # construct A in Eq.(8)
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()
        
        # compute L
        rowsum = np.array(adj_mat.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        self.L = bi_lap.tocoo()

        # initial L and I
        self.L = self._convert_sp_mat_to_sp_tensor(self.L).to(self.device)
        self.I = torch.eye(self.num_users + self.num_items).to_sparse().to(self.device)

    def sparse_dropout(self, L, dropout_rate, noise_shape):
        """ Node dropout.
        """
        
        random_tensor = 1 - dropout_rate
        random_tensor += torch.rand(noise_shape).to(L.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        indices = L._indices()
        values = L._values()

        indices = indices[:, dropout_mask]
        values = values[dropout_mask]

        out = torch.sparse.FloatTensor(indices, values, L.shape).to(L.device)
        return out * (1. / (1 - dropout_rate))