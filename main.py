import torch
import torch.optim as optim
from model.NGCF import NGCF
import warnings
from util.data_generator import *
warnings.filterwarnings('ignore')
from time import time
import os
from util.tester import *


if __name__ == '__main__':
    print('Start preparing...')
    prepare_start_time = time()

    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

    data = Data(data_path='/data/', name_data=args.dataset) # get data
    
    train_set, valid_set, test_set = data.split_dataset()
    
    train_set_dict = data.retrieve_user_interacts(train_set)
    valid_set_dict = data.retrieve_user_interacts(valid_set)
    test_set_dict = data.retrieve_user_interacts(test_set)

    data.print_statistics()
    
    graph_path = '/data/' + args.dataset + '/sparse_graph.npz'

    if os.path.exists(graph_path):
        sparse_graph = sp.load_npz(graph_path)
    else:
        graph = data.get_interact_graph()
        sparse_graph = data.get_sparse_graph(graph, weight=[], graph_type='interact')
        sp.save_npz(graph_path, sparse_graph)


    # initial model
    model = NGCF(data.num_users,
                 data.num_items,
                 args)

    model.model_initialize(sparse_graph)
    
    train_start_time = time()
    
    prepare_state = 'Preparation compeleted [%.1fs]' % (train_start_time - prepare_start_time)
    print(prepare_state)
    print('Start training...')

    # ----------------- Train -----------------

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # initial tester for validation
    tester_valid = Tester(valid_set_dict, args.K, data.num_items, args.batch_size)

    for epoch in range(args.epoch):
        epoch_time = time()
        loss = 0.

        num_train_batch = data.num_train // args.batch_size + 1

        for idx in range(num_train_batch):
            t1 = time()
            users, pos_items, neg_items = data.batch_sampling(train_set_dict, args.batch_size)
            user_final_embeddings, pos_item_final_embeddings, neg_item_final_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items)

            batch_loss = model.loss_func(user_final_embeddings, pos_item_final_embeddings, neg_item_final_embeddings)
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
            valid_start_time = time()
            
            valid_results = tester_valid.test(model)
            
            valid_finish_time = time()
            
            valid_stat = 'Epoch %d [%.1fs]: traininig loss==[%.5f], recall=[%.5f], ' \
                       'precision=[%.5f], hit=[%.5f], ndcg=[%.5f]' % \
                       (epoch, valid_finish_time - valid_start_time, loss, valid_results['recall'],
                        valid_results['precision'], valid_results['hit_ratio'],
                        valid_results['ndcg'])
            print(valid_stat)
    
    # ----------------- Test -----------------
    tester_test = Tester(test_set_dict, args.K, data.num_items, args.batch_size)
    test_results = tester_test.test(model)

    test_stat = 'Test results with Top-%d: traininig loss==[%.5f], recall=[%.5f], ' \
                       'precision=[%.5f], hit=[%.5f], ndcg=[%.5f]' % \
                       (args.K, loss, test_results['recall'],
                        test_results['precision'], test_results['hit_ratio'],
                        test_results['ndcg'])
    print(test_stat)

        

            
