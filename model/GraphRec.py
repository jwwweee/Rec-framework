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
        self.num_scores = config['num_scores']
        self.num_layer = config['num_layer']
        self.embed_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.reg_coef = eval(config['regs'])[0]
        self.config = config
        self.device = device

        # initialize graphs
        self.R = R.tocoo()
        self.S = S.tocoo()

        self.R = self._convert_sp_mat_to_sp_tensor(self.R).to(device)
        self.S = self._convert_sp_mat_to_sp_tensor(self.S).to(device)

        # initialize the parameters of embeddings
        initializer = nn.init.xavier_uniform_

        # initialize embeddings layers
        self.e_r = torch.Tensor(list(range(1, self.num_scores + 1))).to(device)
        self.rate_embedding = nn.Linear(self.num_scores, self.embed_size)
        
        self.parameter_list = nn.ParameterDict({
            'embed_user': nn.Parameter(initializer(torch.empty(self.num_users,
                                                 self.embed_size))),
            'embed_item': nn.Parameter(initializer(torch.empty(self.num_items,
                                                 self.embed_size)))})
        
        # initialize linear layers
        self.W_u = nn.ModuleList()
        self.W_v = nn.ModuleList()
        self.W_c = nn.ModuleList()
        self.W_g = nn.ModuleList()
        for i in range(self.num_layer):
            if i == 0:
                W_u = nn.Linear(self.embed_size*2, self.embed_size)
                W_v = nn.Linear(self.embed_size*2, self.embed_size)
            else:
                W_u = nn.Linear(self.embed_size, self.embed_size)
                W_v = nn.Linear(self.embed_size, self.embed_size)

            W_c = nn.Linear(self.embed_size, self.embed_size)
            W_g = nn.Linear(self.embed_size, self.embed_size)
            
            self.W_u.append(W_u)
            self.W_v.append(W_v)
            self.W_c.append(W_c)
            self.W_g.append(W_g)
        
        self.W_I = nn.Linear(self.embed_size, self.embed_size)
        self.W_S = nn.Linear(self.embed_size, self.embed_size)

        self.W1_A = nn.Linear(self.embed_size*2, self.embed_size)
        self.w2_A = nn.Linear(self.embed_size, 1)

        self.W1_B = nn.Linear(self.embed_size*2, self.embed_size)
        self.w2_B = nn.Linear(self.embed_size, 1)

        self.W1_M = nn.Linear(self.embed_size*2, self.embed_size)
        self.w2_M = nn.Linear(self.embed_size, 1)

        self = self.to(self.device)

    def forward(self, batch_user, batch_item):
        """ Model feedforward procedure
        """
        # ----------------------- feed-forward process -----------------------
        # initialize embeddings (users, items and scores)
        U = self.parameter_list['embed_user']
        V = self.parameter_list['embed_item']

        e_r = self.rate_embedding(self.e_r)

        user_ones = torch.full((self.num_users, self.embed_size), 1).to(self.device)
        e_r_user = user_ones * e_r

        item_ones = torch.full((self.num_items, self.embed_size), 1).to(self.device)
        e_r_item = item_ones * e_r


        U = torch.cat([U, e_r_user], dim=1)
        print(U.size())
        V = torch.cat([V, e_r_item], dim=1)
        print(V.size())

        for i in range(self.num_layer):
            U = self.W_u[i](U)
            U = F.relu(U)

            V = self.W_v[i](V)
            V = F.relu(V)
        
        print(U.size())
        print(V.size())
        # ---------- user modeling ----------
        A = self.w2_A(F.relu(self.W1_A(torch.cat([V[batch_item,:], U[batch_user,:]], dim=1)))) # user-item attention coef
        A = F.softmax(A, dim=0)
        H_I = F.relu(self.W_I(torch.matmul(torch.mul(self.R[batch_user,:], A), V[batch_item,:])))

        B = self.w2_B(F.relu(self.W1_B(torch.cat([H_I, U[batch_user,:]], dim=1)))) # social attention coef
        B = F.softmax(B, dim=0)
        H_S = F.relu(self.W_S(torch.matmul(torch.mul(self.S[batch_user,:], B), V)))

        H = torch.cat([H_I, H_S], dim=1) # combine user_item and social embeddings

        # user domain combination
        for i in range(self.num_layer):
            H = self.W_c[i](H)
            H = F.relu(H)

        # ---------- item modeling ----------
        M = self.w2_M(F.relu(self.W1_M(torch.cat([U[batch_user,:], V[batch_item,:]], dim=1)))) # social attention coef
        M = F.softmax(M, dim=0)
        Z = F.relu(self.W1_M(torch.matmul(torch.mul(self.R.t()[batch_item,:], M), U[batch_user,:])))

        # ----------------------- retrieving target users and items -----------------------
        # # retrieve batched users and items
        # batch_H = H[batch_user,:]

        # # get positive items representations
        # batch_H = Z[batch_item,:]

        # ---------- batch rating prediction ----------
        G = torch.cat([H, Z], dim=1)
        for i in range(self.num_layer):
            G = self.W_g[i](G)
            G = F.relu(G)

        return G
        
    
    def loss_func(self, G, batch_y):
        """ BPR loss function, compute BPR loss for ranking task in recommendation.
        """

        pred_loss = (G - batch_y).norm(2).pow(2)

        for param in self.parameters():
            regularizer += torch.sum(torch.square(param))

        batch_loss = pred_loss + regularizer

        return batch_loss
    
    def train_epoch(self, train_set, optimizer, num_train_batch, _):
        """ Train each epoch, return total loss of the epoch
        """
        loss = 0.
        for idx in range(num_train_batch):
            users = train_set[(idx+1)*self.batch_size : (idx+2)*self.batch_size, 0]
            items = train_set[(idx+1)*self.batch_size : (idx+2)*self.batch_size, 1]
            batch_y = train_set[(idx+1)*self.batch_size : (idx+2)*self.batch_size, 2]

            scores = self.forward(users, items)

            batch_loss = self.loss_func(scores, batch_y)
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