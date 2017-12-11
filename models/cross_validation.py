import pandas as pd
import numpy as np
from datetime import timedelta
from models import utils
from sklearn.model_selection import StratifiedKFold


class CrossValidation:
    """
        Cross Validation
    """
    def __init__(self):

        self.trained_cv = []

    def yield_data(self,x,y,data,index_nums,index_mt):
        bool_valid = np.in1d(data,index_nums)
        bool_train = ~np.in1d(data,index_nums)
        train_index = index_mt[bool_train]
        valid_index = index_mt[bool_valid]
        np.random.shuffle(train_index)
        np.random.shuffle(valid_index)
        x_train = x[train_index]
        y_train = y[train_index]
        x_valid = x[valid_index]
        y_valid = y[valid_index]
        return x_train,y_train,x_valid,y_valid

    def era_splid(self,x,y,n_splits=None,n_cv=10,cv_seed=None):
        if cv_seed is not None:
            np.random.seed(cv_seed)
        n_pack = 7
        days_pack = timedelta(days=(n_pack - 1))
        data_1 = pd.read_csv('./inputs/train_lea.csv', index_col='date', parse_dates=True)
        data_1['ID'] = np.arange(len(data_1))
        date_low = data_1.index[0]
        i = 1
        while True:
            date_higt = date_low + days_pack
            if len(data_1[date_low:date_higt]) != 0:
                data_1.loc[date_low:date_higt, 'era'] = i
                date_low = date_higt + timedelta(days=1)
                i += 1
            else:
                break
        era_list = data_1['era'].unique()
        era_mt = data_1['era'].as_matrix()
        index_mt = data_1['ID'].as_matrix()
        rest_list = era_list
        n_traverse = len(era_list) // n_cv
        if len(era_list) % n_cv != 0:
            n_traverse += 1
        for i in range(n_cv):
            list_lenth = len(rest_list)
            if list_lenth != 0:
                if list_lenth >= n_traverse:
                    index_now = np.random.choice(rest_list, n_traverse, replace=False)
                    rest_list = [i for i in rest_list if i not in index_now]
                    yield self.yield_data(x,y,era_mt,index_now,index_mt)
                else:
                    index_now = rest_list
                    rest_list = [i for i in rest_list if i not in index_now]
                    index_now = np.append(np.array(index_now),
                                      np.random.choice(era_list, n_traverse - list_lenth, replace=False))
                    yield self.yield_data(x,y,era_mt,index_now,index_mt)
            else:
                index_now = np.random.choice(era_list, n_traverse, replace=False)
                yield self.yield_data(x,y,era_mt,index_now,index_mt)


    @staticmethod
    def random_split(x,y,n_splits=None,n_cv=None,cv_seed=None):
        train_data = utils.load_pkl_to_data('./data/preprocessed_data/x_g_train.p')
        data_mt = np.array(train_data)
        index = data_mt[:,2]
        # station_list = index.tolist()
        # min_number = 10000
        # for i in np.unique(index):
        #     if min_number > station_list.count(i):
        #         min_number = station_list.count(i)
        # if n_splits > min_number:
        #     raise ValueError(
        #         '--The least populated station  has only %d members,please input new cv_number--' % min_number)
        cv_count = 0
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=cv_seed)
        for train_index, valid_index in skf.split(index, index):
            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            cv_count += 1
            utils.print_cv_info(cv_count, n_cv)
            yield x_train, y_train, x_valid, y_valid


    # @staticmethod
    # def sk_k_fold(x, y, n_splits=None, n_cv=None, cv_seed=None):
    #
    #     if cv_seed is not None:
    #         np.random.seed(cv_seed)
    #
    #     if n_cv % n_splits != 0:
    #         raise ValueError('n_cv must be an integer multiple of n_splits!')
    #
    #     n_repeats = int(n_cv / n_splits)
    #     era_k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed)
    #     cv_count = 0
    #
    #     for train_index, valid_index in era_k_fold.split(x, y):
    #
    #         np.random.shuffle(train_index)
    #         np.random.shuffle(valid_index)
    #
    #         # Training data
    #         x_train = x[train_index]
    #         y_train = y[train_index]
    #
    #         # Validation data
    #         x_valid = x[valid_index]
    #         y_valid = y[valid_index]
    #
    #         cv_count += 1
    #         utils.print_cv_info(cv_count, n_cv)
    #
    #         yield x_train, y_train, x_valid, y_valid
