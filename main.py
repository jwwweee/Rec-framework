import warnings
from util.data_generator import *
warnings.filterwarnings('ignore')
from Rec import *
from time import time

if __name__ == '__main__':

    # ----------------- Load data -----------------
    print('Loading data...')
    prepare_start_time = time()

    name_data = 'ciao'
    
    data = Data(data_path='data/', name_data=name_data, is_social=True) # get data

    data.print_statistics()

    train_set, valid_set, test_set = data.split_dataset() # split dataset


    # ----------------- Train model -----------------
    train_start_time = time()
    prepare_state = 'Preparation compeleted [%.1fs]' % (train_start_time - prepare_start_time)
    print(prepare_state)
    print('Start training...')

    rec = Rec(data, name_data, name_model='DiffNet', is_social=True)

    rec.train(train_set, valid_set) # train and valid

    rec.test(test_set) # test
