import time
import numpy as np
import pandas as pd
from models import utils
from sklearn.preprocessing import LabelBinarizer

train_csv_path = './inputs/stock_train_data_20171125.csv'
test_csv_path = './inputs/stock_test_data_20171125.csv'
preprocessed_path = './data/preprocessed_data/'
gan_prob_path = './data/gan_outputs/'


class DataPreProcess:

    def __init__(self, train_path, test_path, preprocess_path, use_group_list=None,
                 use_global_valid=False, global_valid_rate=None):

        self.train_path = train_path
        self.test_path = test_path
        self.preprocess_path = preprocess_path
        self.x_train = pd.DataFrame()
        self.x_g_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.x_g_test = pd.DataFrame()
        self.id_test = pd.DataFrame()

        # Validation Set
        self.x_valid = np.array([])
        self.x_g_valid = np.array([])
        self.y_valid = np.array([])

        self.use_group_list = use_group_list
        self.drop_feature_list = []
        self.use_global_valid_ = use_global_valid
        self.global_valid_rate = global_valid_rate

        if use_group_list is not None:
            self.g_train = pd.DataFrame()
            self.g_test = pd.DataFrame()
            self.g_train_dict = {}
            self.g_test_dict = {}

    # Load CSV Files Using Pandas
    def load_csv(self):

        train_f = pd.read_csv(self.train_path, header=0, dtype=np.float64)
        test_f = pd.read_csv(self.test_path, header=0, dtype=np.float64)

        return train_f, test_f

    # Load Data Using Pandas
    def load_data(self):

        try:
            print('======================================================')
            print('Loading data...')
            train_f, test_f = self.load_csv()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        # Drop Unnecessary Columns
        self.x_train = train_f.drop(['number'], axis=1)
        self.y_train = train_f['number']
        self.x_test = test_f.drop(['id'], axis=1)
        self.id_test = test_f['id']

        print('------------------------------------------------------')
        print('Train Features: {}\n'.format(self.x_train.shape[1]),
              'Test Features: {}'.format(self.x_test.shape[1]))

        if self.use_group_list is not None:
            for i in self.use_group_list:
                self.g_train_dict[i] = train_f[i]
                self.g_test_dict[i] = test_f[i]
                self.x_g_train = self.x_train
                self.x_g_test = self.x_test

    # Convert pandas DataFrames to numpy arrays
    def convert_pd_to_np(self):

        print('======================================================')
        print('Converting pandas DataFrames to numpy arrays...')

        self.x_train = np.array(self.x_train, dtype=np.float64)
        self.y_train = np.array(self.y_train, dtype=np.float64)
        self.x_test = np.array(self.x_test, dtype=np.float64)
        self.id_test = np.array(self.id_test, dtype=int)

    # Convert Column 'group' to Dummies
    def convert_group_to_dummies(self, add_train_dummies=False):

        print('======================================================')
        print('Converting Groups of Train Set to Dummies...')

        lb_dict = {}
        for i in self.use_group_list:
            lb = LabelBinarizer()
            if add_train_dummies:
                print('------------------------------------------------------')
                print('Add Zero Dummies to Train Set of Group: {}'.format(i))
                add_list = list((set(self.g_train_dict[i]) ^ set(self.g_test_dict[i])) & set(self.g_test_dict[i]))
                lb.fit(np.append(self.g_train_dict[i], add_list))
            else:
                lb.fit(self.g_train_dict[i])
            lb_dict[i] = lb

        for i in self.use_group_list:

            print('------------------------------------------------------')
            print('Converting Group {} to Dummies...'.format(i))

            train_dummies = lb_dict[i].transform(self.g_train_dict[i])
            test_dummies = lb_dict[i].transform(self.g_test_dict[i])

            print('Train Dummies: {}\n'.format(train_dummies.shape[1]),
                  'Test Dummies: {}'.format(test_dummies.shape[1]))

            if self.x_train.shape[1] > 500:
                print('So Many Features!')
                for ii in range(len(self.x_train)):
                    self.x_train[ii] = np.concatenate((self.x_train[ii], train_dummies[ii]))
                for ii in range(len(self.x_test)):
                    self.x_test[i] = np.concatenate((self.x_test[ii], test_dummies[ii]))
            else:
                self.x_train = np.concatenate((self.x_train, train_dummies), axis=1)
                self.x_test = np.concatenate((self.x_test, test_dummies), axis=1)

            self.x_g_train = np.column_stack((self.x_g_train, self.g_train_dict[i]))
            self.x_g_test = np.column_stack((self.x_g_test, self.g_test_dict[i]))

        print('------------------------------------------------------')
        print('Total Features of x_train: {}\n'.format(self.x_train.shape[1]),
              'Total Features of x_test: {}\n'.format(self.x_test.shape[1]),
              'Total Features of x_g_train: {}\n'.format(self.x_g_train.shape[1]),
              'Total Features of x_g_test: {}'.format(self.x_g_test.shape[1]))

    # Spilt Validation Set by valid_rate
    def split_validation_set(self, valid_rate=None):

        print('======================================================')
        print('Splitting Validation Set by Valid Rate: {}'.format(valid_rate))

        train_index = []
        valid_index = []

        # Validation Set
        self.x_valid = self.x_train[valid_index]
        self.x_g_valid = self.x_g_train[valid_index]
        self.y_valid = self.y_train[valid_index]

        # Train Set
        self.x_train = self.x_train[train_index]
        self.x_g_train = self.x_g_train[train_index]
        self.y_train = self.y_train[train_index]

    # Save Data
    def save_data(self):

        print('======================================================')
        print('Saving Preprocessed Data...')
        utils.save_data_to_pkl(self.x_train, self.preprocess_path + 'x_train.p')
        utils.save_data_to_pkl(self.x_g_train, self.preprocess_path + 'x_g_train.p')
        utils.save_data_to_pkl(self.y_train, self.preprocess_path + 'y_train.p')
        utils.save_data_to_pkl(self.x_test, self.preprocess_path + 'x_test.p')
        utils.save_data_to_pkl(self.x_g_test, self.preprocess_path + 'x_g_test.p')
        utils.save_data_to_pkl(self.id_test, self.preprocess_path + 'id_test.p')

    # Save Validation Set
    def save_global_valid_set(self):

        print('======================================================')
        print('Saving Validation Set...')
        utils.save_data_to_pkl(self.x_valid, self.preprocess_path + 'x_global_valid.p')
        utils.save_data_to_pkl(self.x_g_valid, self.preprocess_path + 'x_g_global_valid.p')
        utils.save_data_to_pkl(self.y_valid, self.preprocess_path + 'y_global_valid.p')

    # Save Data Split by Era Distribution

    # Preprocess
    def preprocess(self):

        print('======================================================')
        print('Start Preprocessing...')

        start_time = time.time()

        # Load original data
        self.load_data()

        # Convert pandas DataFrames to numpy arrays
        self.convert_pd_to_np()

        # Spilt Validation Set by valid_rate
        if self.use_global_valid_:
            self.split_validation_set(valid_rate=self.global_valid_rate)
            self.save_global_valid_set()

        # Save Data to pickle files
        self.save_data()

        end_time = time.time()

        print('======================================================')
        print('Done!')
        print('Using {:.3}s'.format(end_time - start_time))
        print('======================================================')


if __name__ == '__main__':

    utils.check_dir(['./data/', preprocessed_path])

    preprocess_args = {'use_group_list': ['week', 'week_era', 'station', 'temperature', 'weather', 'holiday'],
                       'use_global_valid': False}

    DPP = DataPreProcess(train_csv_path, test_csv_path, preprocessed_path, **preprocess_args)
    DPP.preprocess()
