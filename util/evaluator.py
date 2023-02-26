import util.metrics as metrics
import itertools 
import numpy as np
import heapq
import multiprocessing
from util.arg_parser import parse_args
import torch

args = parse_args()

CORES = multiprocessing.cpu_count() // 2

class Evaluator():
    def __init__(self, evaluate_set_dict, K, num_items, batch_size) -> None:
        self.K = K
        self.num_items = num_items
        self.batch_size = batch_size * 4
        self.all_evaluate_items = list(set(itertools.chain.from_iterable(evaluate_set_dict.values())))
        self.evaluate_set_dict = evaluate_set_dict
        self.evaluate_users = list(evaluate_set_dict.keys())

    def get_performance(self, evaluate_positive_items: list, true_pos_items_ind: list, K: int):
        """ Get the performances of different metrics
                
            Params:
                evaluate_positive_items: a user's evaluate positive items list
                true_pos_items_ind: the true predicted positive item index list
                K: @K of items for evaluate

            Return:
                a dict of metrics for this user
        """

        precision, recall, ndcg, hit_ratio = [], [], [], []
        
        precision.append(metrics.precision_at_k(true_pos_items_ind, K))
        recall.append(metrics.recall_at_k(true_pos_items_ind, K, len(evaluate_positive_items)))
        ndcg.append(metrics.ndcg_at_k(true_pos_items_ind, K, evaluate_positive_items))
        hit_ratio.append(metrics.hit_at_k(true_pos_items_ind, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

    def rank_and_evaluate(self, evaluate_positive_items, all_evaluate_items, user_items_scores, K):
        """ Rank top-K items of a user and evaluate
            
            Params:
                evaluate_positive_items: the user's evaluate positive items
                all_evaluate_items: all the evaluate items list in evaluate (or valid) set
                user_items_scores: the user's the scores of different items
                K: @K

            Return: 
                true_pos_items_ind: the true predicted positive item index list
        """
        item_score = {}
        for i in all_evaluate_items:
            item_score[i] = user_items_scores[i]

        K_item_score = heapq.nlargest(K, item_score, key=item_score.get)

        true_pos_items_ind = []
        for i in K_item_score:
            if i in evaluate_positive_items:
                true_pos_items_ind.append(1)
            else:
                true_pos_items_ind.append(0)

        return true_pos_items_ind

    def evaluate_one_user(self, pack):
        """ evaluate each user

            Params:
                pack: pack[0] is the score list of the user (1 x num_item), pack[1] is user_id
            
            Return:
                the result dict of one user
        """

        # unpack params
        user_items_scores = pack[0]
        user = pack[1]

        # retrieve evaluate positive items
        evaluate_positive_items = self.evaluate_set_dict[user]

        true_pos_items_ind = self.rank_and_evaluate(evaluate_positive_items, self.all_evaluate_items, user_items_scores, self.K)
        
        evaluate_result_dict = self.get_performance(evaluate_positive_items, true_pos_items_ind, self.K)

        return evaluate_result_dict

    def evaluate(self, model):
        """ evaluateing (or validation)

            Param:
                model: the training model
            
            Return:
                The dict results of each user
        """

        result = {'precision': 0., 'recall': 0., 'ndcg': 0., 'hit_ratio': 0.}

        pool = multiprocessing.Pool(CORES)

        num_evaluate_users = len(self.evaluate_users)
        all_items = range(self.num_items)
        
        model.eval()  
        with torch.no_grad():

            num_user_batches = num_evaluate_users // self.batch_size + 1
            
            count = 0

            # evaluate by user batch
            for u_batch_id in range(num_user_batches):
                start = u_batch_id * self.batch_size
                end = (u_batch_id + 1) * self.batch_size
                
                user_batch = self.evaluate_users[start: end]

                user_final_embeddings, all_pos_item_final_embeddings, _ = model(user_batch, all_items, [])
                                                                
                all_items_scores = model.predict_score(user_final_embeddings, all_pos_item_final_embeddings).detach().cpu()

                pack = zip(all_items_scores.numpy(), user_batch)
                
                batch_result = pool.map(self.evaluate_one_user, pack) # evaluate each user by multiprocessing

                count += len(batch_result)
                
                # sum up the results of batch users
                for re_dict in batch_result:
                    result['precision'] += re_dict['precision']/num_evaluate_users
                    result['recall'] += re_dict['recall']/num_evaluate_users
                    result['ndcg'] += re_dict['ndcg']/num_evaluate_users
                    result['hit_ratio'] += re_dict['hit_ratio']/num_evaluate_users
        assert count == num_evaluate_users
        pool.close()

        return result