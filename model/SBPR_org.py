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
        self.device = torch.device('cpu')

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

    def forward(self, user, pos_item, neg_item):
        """ Model feedforward procedure
        """
        U = self.parameter_list['embed_user']
        V = self.parameter_list['embed_item']

        # ----------------------- retrieving target users and items -----------------------
        # retrieve batched users and items
        user_embedding = U[user, :]

        # get positive items representations
        pos_item_embedding = V[pos_item, :]

        # get negative items representations
        neg_item_embedding = V[neg_item, :]

        # get social items represesntations
        if self.social_item == -1:
            pass
        else:
            self.soc_item_embedding = V[self.social_item, :]

        return user_embedding, pos_item_embedding, neg_item_embedding

    def _sampling_and_train_user(self, user_items_dict, user_friends_dict):
        """ Sample a batch of pair-wise data.

            Params:
                dataset_dict: train_set_dict
                batch_size
            
            Return:
                users, positive_items, negative_items
        """
        users = list(user_items_dict.keys())
        batch_loss = 0
        for i in range(self.batch_size):

            user = random.choice(users) # sample a user

            interacted_items = user_items_dict[user] # get her item
            
            positive_item = random.choice(interacted_items)

            # check user has friend. If yes, use SBPR, if not, use BPR
            if user in user_friends_dict.keys():
                user_friends_dict[user]
                friends = user_friends_dict[user] # get her friends
                
                friends_items_repeat = []
                for f in friends:
                    if f in users:
                        friends_items_repeat.extend(user_items_dict[f])

                # check user's friend has interacted history (yes: SBPR, no: BPR)
                if friends_items_repeat == []:
                    
                    friends_items = np.unique(friends_items_repeat)

                    # sampling a negative item for each user            
                    while True:
                        # sample 1 neg-item from all items, if not exist in train set, 
                        # to the train set, the item is negative
                        
                        negative_item_candidate = np.random.randint(low=0, high=self.num_items, size=1)[0]

                        if (negative_item_candidate not in interacted_items) and (negative_item_candidate not in friends_items):
                            negative_item = negative_item_candidate
                            break
                    
                    # sampling a social feedback item for each user            
                    while True:
                        # sample 1 social-item from all items
                        
                        social_item_candidate = random.choice(friends_items)

                        if (social_item_candidate not in interacted_items):
                            social_item = social_item_candidate
                            social_coeff = friends_items_repeat.count(social_item)
                            break
                        
                    self.social_item = social_item
                    user_embedding, pos_item_embedding, neg_item_embedding = self.forward(user, positive_item, negative_item)

                    loss = self.SBPR_loss_func(user_embedding, pos_item_embedding, neg_item_embedding, self.soc_item_embedding, social_coeff)
                    batch_loss += loss

                else: # user's friend has no history, use BPR loss
                    
                    # sampling a negative item for each user  
                    while True:
                        # sample 1 neg-item from all items, if not exist in train set, 
                        # to the train set, the item is negative
                        
                        negative_item_candidate = np.random.randint(low=0, high=self.num_items, size=1)[0]

                        if (negative_item_candidate not in interacted_items):
                            negative_item = negative_item_candidate
                            break
                    
                    self.social_item = -1 # social item not exist
                    user_embedding, pos_item_embedding, neg_item_embedding = self.forward(user, positive_item, negative_item)

                    loss = self.BPR_loss_func(user_embedding, pos_item_embedding, neg_item_embedding)
                    batch_loss += loss
            else: # user has no friends, use BPR loss
                
                while True:
                        # sample 1 neg-item from all items, if not exist in train set, 
                        # to the train set, the item is negative
                        
                        negative_item_candidate = np.random.randint(low=0, high=self.num_items, size=1)[0]

                        if (negative_item_candidate not in interacted_items):
                            negative_item = negative_item_candidate
                            break
                    
                self.social_item = -1 # social item not exist
                user_embedding, pos_item_embedding, neg_item_embedding = self.forward(user, positive_item, negative_item)

                loss = self.BPR_loss_func(user_embedding, pos_item_embedding, neg_item_embedding)
                batch_loss += loss
        
        return batch_loss

    
    def SBPR_loss_func(self, user_embeddings, pos_item_embeddings, neg_item_embeddings, soc_item_embeddings, social_coeff):
        """ BPR loss function, compute BPR loss for ranking task in recommendation.
        """
        social_coeff = torch.Tensor(social_coeff).to(self.device)
        # compute positive and negative scores
        pos_scores = torch.sum(torch.mul(user_embeddings, pos_item_embeddings), axis=0)
        neg_scores = torch.sum(torch.mul(user_embeddings, neg_item_embeddings), axis=0)
        soc_scores = torch.sum(torch.mul(user_embeddings, soc_item_embeddings), axis=0)

        loss_ik = -1 * nn.LogSigmoid()((pos_scores - soc_scores) / (social_coeff + 1))
        loss_kj = -1 * nn.LogSigmoid()((soc_scores - neg_scores))

        regularizer = (torch.norm(user_embeddings) ** 2
                       + torch.norm(pos_item_embeddings) ** 2
                       + torch.norm(neg_item_embeddings) ** 2
                       + torch.norm(soc_item_embeddings) ** 2) / 2

        emb_loss = regularizer

        loss = loss_ik + loss_kj + self.regs * emb_loss

        return loss
    
    def BPR_loss_func(self, user_embeddings, pos_item_embeddings, neg_item_embeddings):
        """ BPR loss function, compute BPR loss for ranking task in recommendation.
        """
        # compute positive and negative scores
        pos_scores = torch.sum(torch.mul(user_embeddings, pos_item_embeddings), axis=0)
        neg_scores = torch.sum(torch.mul(user_embeddings, neg_item_embeddings), axis=0)

        bpr_loss = -1 * nn.LogSigmoid()(pos_scores - neg_scores)

        regularizer = (torch.norm(user_embeddings) ** 2
                       + torch.norm(pos_item_embeddings) ** 2
                       + torch.norm(neg_item_embeddings) ** 2
                       ) / 2

        emb_loss = regularizer

        loss = bpr_loss + self.regs * emb_loss

        return loss
    
    def predict_score(self, user_embeddings, all_item_embeddings):
        """ Predict the score of a pair of user-item interaction
        """
        score = torch.matmul(user_embeddings, all_item_embeddings.t())

        return score
    
    def train_epoch(self, train_set, optimizer, lr_scheduler, num_train_batch, data):
        """ Train each epoch, return total loss of the epoch
        """
        """ Train each epoch, return total loss of the epoch
        """

        user_friends_dict = data.retrieve_user_interacts(data.social_graph)

        # 遍历 user_items_dict 的键
        user_items_dict = train_set
        
        loss = 0.
        for idx in range(num_train_batch):
            batch_loss = self._sampling_and_train_user(user_items_dict, user_friends_dict)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss += batch_loss

        return loss
