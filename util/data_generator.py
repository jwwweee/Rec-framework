import numpy as np
import random
import scipy.sparse as sp

class Data(object):
    def __init__(self, data_path: str, name_data: str, is_social: bool=False, social_weight: bool=False, interact_weight: bool=True) -> None:
    
        # init graphs, indeces start from 0
        interact_graph = np.loadtxt(data_path + name_data + '/rating.txt', dtype=np.int64, delimiter=',')
        if interact_weight:
            self.interact_weight = interact_graph[:, 2]

        self.interact_graph = interact_graph[:, :2]

        # initialize statistics
        self.num_users = max(self.interact_graph[:,0]) + 1
        self.num_items = max(self.interact_graph[:,1]) + 1

        self.num_train = 0
        self.num_valid = 0
        self.num_test = 0

        # social graph
        if is_social:
            
            social_graph = np.loadtxt(data_path + name_data + '/trustnetwork.txt', dtype=np.int64, delimiter=',')

            self.social_graph = social_graph[:, :2]

            if social_weight:
                self.social_weight = social_graph[:, 2]

        self.name_data = name_data

    def get_entire_dataset(self) -> list:
        """ Get entire dataset

            Return: entire user-item dataset list (interactions)
        """

        return self.interact_graph

    def get_social_graph(self) -> list:
        """ Get social graph.

            Return: egde list of social graph
                e.g. [[user_id, user_id],...]
        """

        return self.social_graph

    def get_interact_graph(self) -> list:
        """ Get interaction netowrk.
            
            Return: egde list of user-item interaction graph
                e.g. [[user_id, item_id],...]
        """
        
        return self.interact_graph

    def retrieve_user_interacts(self, interact_graph: list) -> dict:
        """ Retrieve the objects lists of each user to a dict.

            Return: dict of categaries and their items 
                e.g. {user_id: [object1, object2,...],...}

        """
        user_interacts_dict = {}

        for user, object in interact_graph:
            if user not in user_interacts_dict:
                user_interacts_dict[user] = [object]
            else:
                user_interacts_dict[user].append(object)

        return user_interacts_dict

    def get_interact_weight(self) -> list:
        """ Get interact graph weights
        """

        return self.interact_weight

    def get_sparse_graph(self, num_users, num_items, graph: list, graph_type: str, weight: list=[], is_weighted_graph: bool=False):
        """ Convert graph to torch sparse graph

            Params:
                graph: original social or interact graph list
                graph_type: 'interact' or 'social'
                weight: weight of graph edge list
                is_weighted_graph: a bool that whether the graph is weighted
            
            Return: sparse_graph: torch sparse graph

        """
        if graph_type == 'social':
            sparse_graph = sp.dok_matrix((num_users, num_users), dtype=np.float32)
        else:
            sparse_graph = sp.dok_matrix((num_users, num_items), dtype=np.float32)

        if is_weighted_graph:
            ind = 0
            for row in graph:
                sparse_graph[row[0], row[1]] = weight[ind]
                ind += 1
        else:
            for row in graph:
                sparse_graph[row[0], row[1]] = 1

        return sparse_graph.tocsr()

    def split_dataset(self, train_split_rate: float=0.8, valid_split_rate: float=0.1, seed: int=1):
        """ Split the interactions as training, validation and testing sets.

            Params: 
                train_split_rate: the rate of training set, default is 0.8.

                valid_split_rate: the rate of valid set from training set, default is 0.8.

            Return: train set, valid set and test set. 
        """

        dataset = np.concatenate((self.interact_graph, self.interact_weight.reshape(-1, 1)),axis=1)
        # dataset = interact_graph

        # set the random seed
        np.random.seed(seed)

        # shuffle the interacted items for each user
        np.random.shuffle(dataset)

        # split train and test sets
        train_split_ind = int(len(dataset) * train_split_rate)
        
        train_set_all = dataset[ : train_split_ind]
        test_set = dataset[train_split_ind : ]

        # split valid set from train set
        valid_split_ind = int(len(train_set_all) * valid_split_rate)
        
        valid_set = train_set_all[ : valid_split_ind]
        train_set = train_set_all[valid_split_ind : ]

        self.num_train = len(train_set)
        self.num_valid = len(valid_set)
        self.num_test = len(test_set)
        
        return train_set, valid_set, test_set
    
    def pair_data_sampling(self, train_set_dict: dict, batch_size: int):
        """ Sample a batch of pair-wise data.

            Params:
                dataset_dict: train_set_dict
                batch_size
            
            Return:
                users, positive_items, negative_items
        """
        
        user_items_dict = train_set_dict

        train_users = user_items_dict.keys()

        users = random.sample(train_users, batch_size)
        
        positive_items, negative_items = [], []
        for user in users:

            # sampling a positive item for each user
            interacted_items = user_items_dict[user]
            
            positive_item = random.choice(interacted_items)
            
            positive_items.append(positive_item)

            # sampling a negative item for each user            
            while True:
                # sample 1 neg-item from all items, if not exist in train set, 
                # to the train set, the item is negative
                negative_item = np.random.randint(low=0, high=self.num_items, size=1)[0]

                if (negative_item not in interacted_items):
                    negative_items.append(negative_item)
                    break

        return users, positive_items, negative_items
                
    def print_statistics(self):
        """ Print statistics of datasets.
        """
        print('Num of users=%d, num of items=%d' % (self.num_users, self.num_items))
        print('Num of interactions=%d' % (self.num_train + self.num_valid + self.num_test))
        print('Size of training set=%d, size of valid set=%d, size of test set=%d, sparsity=%.5f' % (self.num_train, self.num_valid, self.num_test, (self.num_train + self.num_valid + self.num_test)/(self.num_users * self.num_items)))

