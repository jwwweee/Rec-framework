import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from base.BPR_MF import *


class DiffNet(BPR_MF):
    def __init__(self, num_users, num_items, args):
        super().__init__(args)
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(eval(args.layer_size))
        
        self.device = args.device
        self.embed_size = args.embed_size
        self.layer_size = eval(args.layer_size)

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
            self.Pool = nn.Linear(self.layer_size[i], self.layer_size[i], bias=True)
            self.conv_layers.append(self.Pool)
        
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)

        self.device = args.device
        self = self.to(self.device)

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """ Model feedforward procedure
        """
        # ----------------------- feed-forward process -----------------------
        # initial user and item embeddings (users and items)

        U = self.parameter_list['embed_user']
        V = self.parameter_list['embed_item']
        
        # message propagation for each layer (user social phase)
        for i in range(self.num_layer):
            U = torch.matmul(self.S, U)
            U = self.Pool(U)
            U = F.relu(U)

        user_g_embeddings = U + torch.matmul(self.R, V)
        item_g_embeddings = V

        # ----------------------- retrieving target users and items -----------------------
        # retrieve batched users and items
        batch_user_g_embeddings = user_g_embeddings[batch_user, :]

        # get positive items representations
        batch_pos_items_repr = item_g_embeddings[batch_pos_item, :]

        # get negative items representations
        batch_neg_items_repr = item_g_embeddings[batch_neg_item, :]

        return batch_user_g_embeddings, batch_pos_items_repr, batch_neg_items_repr

    def initialize_graph(self, sparse_interact_graph, sparse_social_graph):
        """ Initialize the graphs, create the saprse Laplacian matrix for user-item interaction graph and social graph.
        """

        # interaction adjacent matrices
        R = sparse_interact_graph.tolil()
        S = sparse_social_graph.tolil()

        # normalization
        interact_D = np.array(R.sum(1))
        interact_D = np.diag(np.power(interact_D, -1))
        norm_R = interact_D.dot(R)

        social_D = np.array(S.sum(1))
        social_D = np.diag(np.power(social_D, -1))
        norm_S = interact_D.dot(S)

        self.R = norm_R.tocoo()
        self.S = norm_S.tocoo()

        self.R = self._convert_sp_mat_to_sp_tensor(self.R).to(self.device)
        self.S = self._convert_sp_mat_to_sp_tensor(self.S).to(self.device)

