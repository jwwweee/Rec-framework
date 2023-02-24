import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from base.BPR_MF import *

class LightGCNConv(nn.Module):
    def __init__(self):
        super(LightGCNConv, self).__init__()

    def forward(self, A_hat, E):
        # left side of the equation
        side_embeddings = torch.matmul(A_hat, E)

        return side_embeddings
        

class LightGCN(BPR_MF):
    def __init__(self, num_users, num_items, args):
        super().__init__(args)
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(eval(args.layer_size))
        self.embed_size = args.embed_size

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
            layer = LightGCNConv()
            self.conv_layers.append(layer)

        self.device = args.device
        self = self.to(self.device)

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """ Model feedforward procedure
        """
        # ----------------------- feed-forward process -----------------------
        # initial concatenated embeddings (users and items)
        E = torch.cat([self.parameter_list['embed_user'], self.parameter_list['embed_item']], dim=0)
        
        # message propagation for each layer (both user and item phases)
        sum_g_embeddings = E.clone()
        for i in range(self.num_layer):
            E = self.conv_layers[i](self.A_hat, E) 
            
            sum_g_embeddings = sum_g_embeddings + E.clone() # layer combination
        
        # average the sum of layers (a_k=1/K+1)
        out_embeddings = torch.div(sum_g_embeddings, (self.num_layer + 1))
        
        # ----------------------- retrieving target users and items -----------------------
        # separate users and items
        user_g_embeddings, item_g_embeddings = out_embeddings[:self.num_users], out_embeddings[self.num_users:]

        # retrieve batched users and items
        batch_user_g_embeddings = user_g_embeddings[batch_user,:]

        # get positive items representations
        batch_pos_items_repr = item_g_embeddings[batch_pos_item,:]

        # get negative items representations
        batch_neg_items_repr = item_g_embeddings[batch_neg_item,:]

        return batch_user_g_embeddings, batch_pos_items_repr, batch_neg_items_repr

    def initialize_graph(self, sparse_interact_graph):
        """ Initialize the graph, create the saprse Laplacian matrix for user-item interaction matrix.
            
            interact_matrix size: interaction num x 2

        """

        # interaction adjacent matrices
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = sparse_interact_graph.tolil()

        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        self.A_hat = bi_lap.tocoo()

        self.A_hat = self._convert_sp_mat_to_sp_tensor(self.A_hat).to(self.device)
        
    def sparse_dropout(self, A_hat, dropout_rate, noise_shape):
        """ Node dropout.
        """
        
        random_tensor = 1 - dropout_rate
        random_tensor += torch.rand(noise_shape).to(A_hat.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        indices = A_hat._indices()
        values = A_hat._values()

        indices = indices[:, dropout_mask]
        values = values[dropout_mask]

        out = torch.sparse.FloatTensor(indices, values, A_hat.shape).to(A_hat.device)
        return out * (1. / (1 - dropout_rate))