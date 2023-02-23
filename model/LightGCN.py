import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class LightGCNConv(nn.Module):
    def __init__(self):
        super(LightGCNConv, self).__init__()

    def forward(self, A_hat, E):
        # left side of the equation
        side_embeddings = torch.matmul(A_hat, E)

        return side_embeddings
        

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(args.layer_size)
        self.batch_size = args.batch_size
        self.device = args.device
        self.embed_size = args.embed_size
        self.reg_coef = eval(args.regs)[0]

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

        self.to(self.device)

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        
        # initial concatenated embeddings (users and items)
        E = torch.cat([self.parameter_list['embed_user'], self.parameter_list['embed_item']], dim=0)
        
        # message propagation for each layer (both user and item phases)
        sum_g_embeddings = E.clone()
        for i in range(self.num_layer):
            E = self.conv_layers[i](self.A_hat, E) 
            
            sum_g_embeddings = sum_g_embeddings + E.clone() # layer combination
        
        # average the sum of layers (a_k=1/K+1)
        out_embeddings = torch.div(sum_g_embeddings, (self.num_layer + 1))
        
        # separate users and items
        user_g_embeddings, item_g_embeddings = out_embeddings[:self.num_users], out_embeddings[self.num_users:]

        # retrieve batched users and items
        batch_user_g_embeddings = user_g_embeddings[batch_user,:]

        # get positive items representations
        batch_pos_items_repr = item_g_embeddings[batch_pos_item,:]

        # get negative items representations
        batch_neg_items_repr = item_g_embeddings[batch_neg_item,:]

        return batch_user_g_embeddings, batch_pos_items_repr, batch_neg_items_repr
    
    def loss_func(self, user_g_embeddings, pos_item_g_embeddings, neg_item_g_embeddings):
        """ BPR loss function, compute BPR loss for ranking task in recommendation.
        """

        # compute positive and negative scores
        pos_scores = torch.sum(torch.mul(user_g_embeddings, pos_item_g_embeddings), axis=1)
        neg_scores = torch.sum(torch.mul(user_g_embeddings, neg_item_g_embeddings), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # compute regularizer
        regularizer = (torch.norm(user_g_embeddings) ** 2
                       + torch.norm(pos_item_g_embeddings) ** 2
                       + torch.norm(neg_item_g_embeddings) ** 2) / 2

        emb_loss = self.reg_coef * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def predict_score(self, user_g_embeddings, all_item_g_embeddings):
        """ Predict the score of a pair of user-item interaction
        """
        score = torch.matmul(user_g_embeddings, all_item_g_embeddings.t())

        return score

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
        self.A_hat = bi_lap.tocoo()

        # initial L and I
        self.A_hat = self._convert_sp_mat_to_sp_tensor(self.A_hat).to(self.device)
        
    def _convert_sp_mat_to_sp_tensor(self, L):
        """ Convert sparse mat to sparse tensor.
        """
        coo = L.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(indices, values, coo.shape)