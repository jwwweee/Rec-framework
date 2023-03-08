import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffNet(nn.Module):
    def __init__(self, num_users, num_items, R, S, config, device):
        super(DiffNet, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layer = config['num_layer']
        self.embed_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.reg_coef = config['regs']
        self.config = config

        # initialize graphs
        self.R = R.tocoo()
        self.S = S.tocoo()

        self.R = self._convert_sp_mat_to_sp_tensor(self.R).to(device)
        self.S = self._convert_sp_mat_to_sp_tensor(self.S).to(device)

        # initialize the parameters of embeddings
        initializer = nn.init.xavier_uniform_
        
        # initialize embeddings layers
        self.parameter_list = nn.ParameterDict({
            'embed_user': nn.Parameter(initializer(torch.empty(self.num_users,
                                                 self.embed_size))),
            'embed_item': nn.Parameter(initializer(torch.empty(self.num_items,
                                                 self.embed_size)))
        })
        
        self.user_fusion = nn.Linear(self.embed_size, self.embed_size, bias=True)
        self.item_fusion = nn.Linear(self.embed_size, self.embed_size, bias=True)

        self.diff_layers = nn.ModuleList()
        for i in range(self.num_layer):
            self.s_layer = nn.Linear(self.embed_size*2, self.embed_size, bias=True)
            self.diff_layers.append(self.s_layer)
        
        torch.nn.init.xavier_uniform_(self.s_layer.weight)
        torch.nn.init.constant_(self.s_layer.bias, 0)

        self = self.to(device)

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """ Model feedforward procedure
        """
        # ----------------------- feed-forward process -----------------------
        # initialize user and item embeddings (users and items)

        # U = self.activation(self.user_fusion(self.parameter_list['embed_user']))
        # V = self.activation(self.item_fusion(self.parameter_list['embed_item']))
        U = self.parameter_list['embed_user']
        V = self.parameter_list['embed_item']
        
        # message propagation for each layer (user social phase)
        for i in range(self.num_layer):
            U = torch.concat([torch.matmul(self.S, U), U], dim=1)
            U = self.diff_layers[i](U)
            U = F.relu(U)

        user_g_embeddings = U + torch.matmul(self.R, V)
        item_g_embeddings = V

        # ----------------------- retrieving target users and items -----------------------
        # retrieve batched users and items
        batch_user_g_embeddings = user_g_embeddings[batch_user, :]

        # get positive items representations
        batch_pos_items_embeddings = item_g_embeddings[batch_pos_item, :]

        # get negative items representations
        batch_neg_items_embeddings = item_g_embeddings[batch_neg_item, :]

        return batch_user_g_embeddings, batch_pos_items_embeddings, batch_neg_items_embeddings

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

        for param in self.parameters():
            regularizer += torch.sum(torch.square(param))

        emb_loss = self.reg_coef * regularizer / self.batch_size
        
        batch_loss = mf_loss + emb_loss

        return batch_loss
    
    def predict_score(self, user_g_embeddings, all_item_g_embeddings):
        """ Predict the score of a pair of user-item interaction
        """
        score = torch.matmul(user_g_embeddings, all_item_g_embeddings.t())

        return score
    
    def train_epoch(self, train_set, optimizer, lr_scheduler, num_train_batch, data):
        """ Train each epoch, return total loss of the epoch
        """
        loss = 0.
        for idx in range(num_train_batch):
            users, pos_items, neg_items = data.pair_data_sampling(train_set, self.config['batch_size'])
            user_final_embeddings, pos_item_final_embeddings, neg_item_final_embeddings = self.forward(users,
                                                                        pos_items,
                                                                        neg_items)

            batch_loss = self.loss_func(user_final_embeddings, pos_item_final_embeddings, neg_item_final_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss += batch_loss

        return loss
        
    def _convert_sp_mat_to_sp_tensor(self, L):
        """ Convert sparse mat to sparse tensor.
        """
        coo = L.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(indices, values, coo.shape)