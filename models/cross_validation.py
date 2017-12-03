import numpy as np
from models import utils
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold


class CrossValidation:
    """
        Cross Validation
    """
    def __init__(self):

        self.trained_cv = []

    @staticmethod
    def random_split(x, y, valid_rate=None, n_cv=None, cv_seed=None):

        ss_train = StratifiedShuffleSplit(n_splits=n_cv, test_size=valid_rate, random_state=cv_seed)
        cv_count = 0

        for train_index, valid_index in ss_train.split(x, y):

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]

            cv_count += 1
            utils.print_cv_info(cv_count, n_cv)

            yield x_train, y_train, x_valid, y_valid

    @staticmethod
    def sk_k_fold(x, y, n_splits=None, n_cv=None, cv_seed=None):

        if cv_seed is not None:
            np.random.seed(cv_seed)

        if n_cv % n_splits != 0:
            raise ValueError('n_cv must be an integer multiple of n_splits!')

        n_repeats = int(n_cv / n_splits)
        era_k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed)
        cv_count = 0

        for train_index, valid_index in era_k_fold.split(x, y):

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]

            cv_count += 1
            utils.print_cv_info(cv_count, n_cv)

            yield x_train, y_train, x_valid, y_valid
