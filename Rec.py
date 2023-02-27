import torch
import torch.optim as optim
import warnings
from util.data_generator import *
warnings.filterwarnings('ignore')
from time import time
import os
from util.evaluator import *
import yaml
from util.arg_parser import parse_args



class Rec(object):
    def __init__(self, data, name_data, name_model, K=10, is_social=False, task_type='ranking') -> None:
        
        self.name_model = name_model
        self.is_social = is_social
        self.task_type = task_type
        self.EPOCH = 1000
        self.K = K
        
        self.early_stopping_counter = 0
        if self.task_type == 'ranking':
            self.best_metric = 0
        else:
            self.best_metric = np.inf

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        self.data = data

        print('Initializing...')
        initial_start_time = time()
        # ----------------- initialze graphs -----------------
        # initialize interact graph
        interact_graph_path = 'data/' + name_data + '/graph/interact_sparse_graph.npz'
    
        if os.path.exists(interact_graph_path):
            self.sparse_interact_graph = sp.load_npz(interact_graph_path)
        else:
            interact_graph = self.data.get_interact_graph()
            self.sparse_interact_graph = self.data.get_sparse_graph(interact_graph, graph_type='interact')
            sp.save_npz(interact_graph_path, self.sparse_interact_graph)

        # initialize social graph if "is_social" is true
        if self.is_social:
            social_graph_path = 'data/' + name_data + '/graph/social_sparse_graph.npz'

            if os.path.exists(social_graph_path):
                self.sparse_social_graph = sp.load_npz(social_graph_path)
            else:
                social_graph = self.data.get_social_graph()
                self.sparse_social_graph = self.data.get_sparse_graph(social_graph, graph_type='social')
                sp.save_npz(social_graph_path, self.sparse_social_graph)

        # ----------------- initial model -----------------
        
        with open('config/' + name_model + '.yaml', "r") as file:
            self.config = yaml.load(file)

        model_import_state = 'from model.' + name_model + ' import ' + name_model
        exec(model_import_state) # import model

        self.model = eval(name_model + '(self.data.num_users, self.data.num_items, self.config, self.device)') # initial model
        
        self.model_path = 'model/pt/' + name_model + '.pt'

        if self.is_social:
            self.model.initialize_graph(self.sparse_interact_graph, self.sparse_social_graph)
        else:
            self.model.initialize_graph(self.sparse_interact_graph)
        
        initial_state =  'Initialize compeleted [%.1fs]' % (time() - initial_start_time)
        print(initial_state)

    def _ealry_stop(self, valid_results, stop_metric_type):
        """
        """
        stop_metric = valid_results[stop_metric_type]

        if self.task_type == 'ranking':
            if stop_metric > self.best_metric:
                self.best_metric = stop_metric
                self.early_stopping_counter = 0 # restart counter
                
                print("Saving model...")
                torch.save(self.model, self.model_path) # save model in terms of validation result
            else:
                self.early_stopping_counter += 1
        else: # rating
            if stop_metric < self.best_metric:
                self.best_metric = stop_metric
                self.early_stopping_counter = 0 # restart counter
                
                print("Saving model...")
                torch.save(self.model, self.model_path) # save model in terms of validation result
            else:
                self.early_stopping_counter += 1

    def _validate(self, epoch, evaluator_valid):
        """
        """
        valid_start_time = time()
        
        valid_results = evaluator_valid.evaluate(self.model)
        
        valid_finish_time = time()
        
        valid_stat = 'Epoch %d [%.1fs]: recall=[%.5f], ' \
                'precision=[%.5f], hit=[%.5f], ndcg=[%.5f]' % \
                (epoch, valid_finish_time - valid_start_time, valid_results['recall'],
                    valid_results['precision'], valid_results['hit_ratio'],
                    valid_results['ndcg'])
        print(valid_stat)

        return valid_results

    def train(self, train_set, valid_set, stop_metric_type='recall'):
        """
        """

        train_set = self.data.retrieve_user_interacts(train_set)
        valid_set = self.data.retrieve_user_interacts(valid_set)
        
        # ----------------- Train -----------------
        # initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        
        # initial evaluator for validation
        evaluator_valid = Evaluator(valid_set, self.K, self.data.num_items, self.config['batch_size'])
        
        for epoch in range(self.EPOCH):
            epoch_time = time()
            loss = 0.

            num_train_batch = self.data.num_train // self.config['batch_size'] + 1

            for idx in range(num_train_batch):
                users, pos_items, neg_items = self.data.pair_data_sampling(train_set, self.config['batch_size'])
                user_final_embeddings, pos_item_final_embeddings, neg_item_final_embeddings = self.model(users,
                                                                            pos_items,
                                                                            neg_items)

                batch_loss = self.model.loss_func(user_final_embeddings, pos_item_final_embeddings, neg_item_final_embeddings)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss

            # training statement
            train_stat = 'Epoch %d [%.1fs]: train loss==[%.5f]' % (
                epoch, time() - epoch_time, loss)
            print(train_stat)

            # ----------------- Validation -----------------
            if (epoch + 1) % 10 == 0:
                valid_results = self._validate(epoch, evaluator_valid)
        
                # ----------------- Early stopping -----------------
                self._ealry_stop(valid_results, stop_metric_type)

                if self.early_stopping_counter > 10: # validate 10 times (100 epoches), if the best metric is not updated, then early stop
                    print("Early stopping...")
                    break
            
    def test(self, test_set):
        """
        """
        # ----------------- Test -----------------
        test_set = self.data.retrieve_user_interacts(test_set)

        evaluator_test = Evaluator(test_set, self.K, self.data.num_items, self.config['batch_size'])

        model = torch.load(self.model_path) # load model

        test_results = evaluator_test.evaluate(model)

        test_stat = 'Test results with Top-%d: recall=[%.5f], ' \
                        'precision=[%.5f], hit=[%.5f], ndcg=[%.5f]' % \
                        (self.K, test_results['recall'],
                            test_results['precision'], test_results['hit_ratio'],
                            test_results['ndcg'])
        print(test_stat)

    def save_config(self, config):
        """
        """
        config = parse_args()
        if not os.path.exists("config"):
            os.mkdir("config")
        config = vars(config)
        config_file_name = self.name_model + '.yaml'
        with open(os.path.join("config", config_file_name), "w") as file:
            file.write(yaml.dump(config))