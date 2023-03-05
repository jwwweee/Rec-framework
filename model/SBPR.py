import torch
import torch.nn as nn
import numpy as np
import random


class SBPR(nn.Module):
    def __init__(self, num_users, num_items, _, S, config, device):
        super(SBPR, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embed_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.regs = config['regs']
        self.config = config
        self.device = device

        # initialize the parameters of embeddings
        initializer = nn.init.xavier_uniform_
        
        # initialize embeddings layers
        self.parameter_list = nn.ParameterDict({
            'embed_user': nn.Parameter(initializer(torch.empty(self.num_users,
                                                 self.embed_size))),
            'embed_item': nn.Parameter(initializer(torch.empty(self.num_items,
                                                 self.embed_size)))
        })

        self = self.to(device)

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """ Model feedforward procedure
        """
        U = self.parameter_list['embed_user']
        V = self.parameter_list['embed_item']

        # ----------------------- retrieving target users and items -----------------------
        # retrieve batched users and items
        batch_user_embeddings = U[batch_user, :]

        # get positive items representations
        batch_pos_items_embeddings = V[batch_pos_item, :]

        # get negative items representations
        batch_neg_items_embeddings = V[batch_neg_item, :]

        # get social items represesntations
        self.soc_item_embeddings = V[self.batch_soc_items, :]

        return batch_user_embeddings, batch_pos_items_embeddings, batch_neg_items_embeddings

    def _pair_data_sampling(self, user_items_dict, user_pool, batch_size: int, user_friends_dict):
        """ Sample a batch of pair-wise data.

            Params:
                dataset_dict: train_set_dict
                batch_size
            
            Return:
                users, positive_items, negative_items
        """
        
        users = random.sample(user_pool, batch_size)
         
        positive_items, negative_items, social_items, social_coeffs = [], [], [], []
        
        for user in users:
            # sampling a positive item for each user
            interacted_items = user_items_dict[user]
            friends = user_friends_dict[user]

            friends_items_repeat = []
            for f in friends:
                friends_items_repeat.extend(user_items_dict[f])

            friends_items = np.unique(friends_items_repeat)

            positive_item = random.choice(interacted_items)
            positive_items.append(positive_item)

            # sampling a negative item for each user            
            while True:
                # sample 1 neg-item from all items, if not exist in train set, 
                # to the train set, the item is negative
                
                negative_item = np.random.randint(low=0, high=self.num_items, size=1)[0]

                if (negative_item not in interacted_items) and (negative_item not in friends_items):
                    negative_items.append(negative_item)
                    break
            
            # sampling a social feedback item for each user            
            while True:
                # sample 1 social-item from all items
                
                social_item = random.choice(friends_items)

                if (social_item not in interacted_items):
                    social_items.append(social_item)
                    social_coeff = friends_items_repeat.count(social_item)
                    social_coeffs.append(social_coeff)
                    break

        return users, positive_items, negative_items, social_items, social_coeffs
    
    def loss_func(self, user_embeddings, pos_item_embeddings, neg_item_embeddings, soc_item_embeddings, social_coeff):
        """ BPR loss function, compute BPR loss for ranking task in recommendation.
        """
        social_coeff = torch.Tensor(social_coeff).to(self.device)
        # compute positive and negative scores
        pos_scores = torch.sum(torch.mul(user_embeddings, pos_item_embeddings), axis=1)
        neg_scores = torch.sum(torch.mul(user_embeddings, neg_item_embeddings), axis=1)
        soc_scores = torch.sum(torch.mul(user_embeddings, soc_item_embeddings), axis=1)

        loss_ik = -1 * torch.sum(nn.LogSigmoid()((pos_scores - soc_scores) / (social_coeff + 1)))
        loss_kj = -1 * torch.sum(nn.LogSigmoid()((soc_scores - neg_scores)))

        regularizer = (torch.norm(user_embeddings) ** 2
                       + torch.norm(pos_item_embeddings) ** 2
                       + torch.norm(neg_item_embeddings) ** 2
                       + torch.norm(soc_item_embeddings) ** 2) / 2

        emb_loss = regularizer

        batch_loss = loss_ik + loss_kj + self.regs * emb_loss

        return batch_loss
    
    def predict_score(self, user_embeddings, all_item_embeddings):
        """ Predict the score of a pair of user-item interaction
        """
        score = torch.matmul(user_embeddings, all_item_embeddings.t())

        return score
    
    def train_epoch(self, train_set, optimizer, num_train_batch, data):
        """ Train each epoch, return total loss of the epoch
        """
        """ Train each epoch, return total loss of the epoch
        """

        user_friends_dict = data.retrieve_user_interacts(data.social_graph)

        # 遍历 user_items_dict 的键
        user_items_dict = train_set
        for user in user_items_dict.keys():
            # 如果 user 在 user_friends_dict 中，则更新其朋友列表，只保留交互过商品的朋友
            if user in user_friends_dict:
                user_friends_dict[user] = [friend for friend in user_friends_dict[user] if friend in user_items_dict]
        
        train_users = user_items_dict.keys()
        user_pool = list(set(user_friends_dict.keys()).intersection(set(train_users)))

        loss = 0.
        for idx in range(num_train_batch):
            users, pos_items, neg_items, soc_items, social_coeff = self._pair_data_sampling(user_items_dict, user_pool, self.config['batch_size'], user_friends_dict)
            self.batch_soc_items = soc_items
            user_embeddings, pos_item_embeddings, neg_item_embeddings = self.forward(users,
                                                                        pos_items,
                                                                        neg_items)

            batch_loss = self.loss_func(user_embeddings, pos_item_embeddings, neg_item_embeddings, self.soc_item_embeddings, social_coeff)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        return loss
