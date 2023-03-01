import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F


class GraphRec(nn.Module):
    def __init__(self, num_users, num_items, R, S, config, device):
        super(GraphRec, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = len(eval(config['layer_size']))
        self.embed_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.reg_coef = eval(config['regs'])[0]

        # initialize the parameters of embeddings
        initializer = nn.init.xavier_uniform_

        # initial embeddings layers
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.batch_size, embedding_dim=self.embed_size)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.batch_size, embedding_dim=self.embed_size)
        
        self.parameter_list = nn.ParameterDict({
            'embed_rate': nn.Parameter(initializer(torch.empty(self.num_users,
                                                 self.embed_size)))})

        self.item_conv_layers = nn.ModuleList()
        for i in range(self.num_layer):
            # self.W_c = tor
            self.item_conv_layers.append(item_layer)

        self.device = device
        self = self.to(self.device)

    def forward(self, batch_user, batch_item):
        """ Model feedforward procedure
        """
        # ----------------------- feed-forward process -----------------------
        # initial embeddings (users and items)
        U = self.user_embedding(batch_user)
        V = self.item_embedding(batch_item)
        e_r = self.parameter_list['embed_rate']

        # concatenate rating embeddings
        U = torch.cat([U, e_r], dim=0)
        V = torch.cat([V, e_r], dim=0)

        # ---------- user modeling ----------
        A = self.w2_A(F.relu(self.W1_A(torch.cat([V, U], dim=0)))) # user-item attention coef
        H_I = F.relu(self.W_I(torch.matmul(torch.mul(self.R, A), V)))

        B = self.w2_B(F.relu(self.W1_B(torch.cat([H_I, U], dim=0)))) # social attention coef
        H_S = F.relu(self.W_S(torch.matmul(torch.mul(self.S, B), V)))

        H = torch.cat([H_I, H_S], dim=0) # combine user_item and social embeddings

        # user domain combination
        for i in range(self.num_layer):
            H = self.W_c[i](H)
            H = F.relu(H)

        # ---------- item modeling ----------
        M = self.w2_M(F.relu(self.W1_M(torch.cat([U, V], dim=0)))) # social attention coef
        Z = F.relu(self.W_M(torch.matmul(torch.mul(self.R.t(), M), U)))

        # ----------------------- retrieving target users and items -----------------------
        # retrieve batched users and items
        batch_user_g_embeddings = H[batch_user,:]

        # get positive items representations
        batch_items_embeddings = Z[batch_item,:]

        return batch_user_g_embeddings, batch_items_embeddings
        
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
    
    def loss_func(self, user_g_embeddings, item_g_embeddings, batch_y):
        """ BPR loss function, compute BPR loss for ranking task in recommendation.
        """

        # ---------- batch rating prediction ----------
        G = torch.cat([user_g_embeddings, item_g_embeddings], dim=0)
        for i in range(self.num_layer):
            G = self.W_g[i](G)
            G = F.relu(G)

        pred_loss = (G - batch_y).norm(2).pow(2)

        # compute regularizer
        regularizer = (torch.norm(user_g_embeddings) ** 2
                       + torch.norm(item_g_embeddings) ** 2) / 2

        emb_loss = self.reg_coef * regularizer / self.batch_size
        
        batch_loss = pred_loss + emb_loss

        return batch_loss
    
    def predict_score(self, user_g_embeddings, all_item_g_embeddings):
        """ Predict the score of a pair of user-item interaction
        """
        score = torch.matmul(user_g_embeddings, all_item_g_embeddings.t())

        return score
    
    def train_epoch(self, train_set, optimizer, num_train_batch):
        """ Train each epoch, return total loss of the epoch
        """
        loss = 0.
        for idx in range(num_train_batch):
            users, pos_items, neg_items = self.data.pair_data_sampling(train_set, self.config['batch_size'])
            user_final_embeddings, pos_item_final_embeddings = self.model(users,
                                                                        pos_items,
                                                                        neg_items)

            batch_loss = self.loss_func(user_final_embeddings, pos_item_final_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        return loss

    def _convert_sp_mat_to_sp_tensor(self, L):
        """ Convert sparse mat to sparse tensor.
        """
        coo = L.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(indices, values, coo.shape)