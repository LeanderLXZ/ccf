import random
import time
import scipy as sp
from models import utils, parameters
from models.training_mode import TrainingMode


class Training:

    def __init__(self):
        pass

    @staticmethod
    def rmse(y_prob, y):
        return sp.sqrt(sp.mean((y_prob - y) ** 2))

    @staticmethod
    def get_base_params(model_name=None):
        """
            Get Base Parameters
        """
        if model_name == 'xgb':
            """
                XGB
            """
            base_parameters = {'learning_rate': 0.003,
                               'gamma': 0.001,
                               'max_depth': 10,
                               'min_child_weight': 8,
                               'subsample': 0.92,
                               'colsample_bytree': 0.85,
                               'colsample_bylevel': 0.7,
                               'lambda': 0,
                               'alpha': 0,
                               'early_stopping_rounds': 10000,
                               'n_jobs': -1,
                               'objective': 'binary:logistic',
                               'eval_metric': 'logloss'}

        elif model_name == 'lgb':
            """
                LGB
            """
            base_parameters = {'application': 'binary',
                               'boosting': 'gbdt',
                               'learning_rate': 0.003,
                               'num_leaves': 88,
                               'max_depth': 9,
                               'min_data_in_leaf': 2500,
                               'min_sum_hessian_in_leaf': 1e-3,
                               'feature_fraction': 0.6,
                               'feature_fraction_seed': 19,
                               'bagging_fraction': 0.8,
                               'bagging_freq': 5,
                               'bagging_seed': 1,
                               'lambda_l1': 0,
                               'lambda_l2': 0,
                               'min_gain_to_split': 0,
                               'max_bin': 225,
                               'min_data_in_bin': 5,
                               'metric': 'binary_logloss',
                               'num_threads': -1,
                               'verbosity': 1,
                               'early_stopping_rounds': 10000}

        elif model_name == 'dnn':
            """
                DNN
            """
            base_parameters = {'version': '1.0',
                               'epochs': 100,
                               'unit_number': [64, 32],
                               'learning_rate': 0.01,
                               'keep_probability': 0.9,
                               'batch_size': 512,
                               'display_step': 100}

        else:
            print('------------------------------------------------------')
            print('[W] Training without Base Parameters')
            base_parameters = None

        return base_parameters

    def train(self):
        """
            ## Train Single Model ##

            Model Name:
            'lr':           Logistic Regression
            'rf':           Random Forest
            'et':           Extra Trees
            'gb':           GradientBoosting
            'xgb':          XGBoost
            'xgb_sk':       XGBoost using scikit-learn module
            'lgb':          LightGBM
            'lgb_sk':       LightGBM using scikit-learn module
            'cb':           CatBoost
        """
        TM = TrainingMode()

        """
            Global Seed
        """
        train_seed = random.randint(0, 1000)
        cv_seed = random.randint(0, 1000)
        # train_seed = 666
        # cv_seed = 216  # 425 48 461 157

        """
            Training Arguments
        """
        train_args = {'use_global_valid': False,
                      'use_custom_obj': False,
                      'show_importance': False,
                      'save_final_pred': True,
                      'save_final_pred_train': False,
                      'save_cv_pred': True,
                      'save_cv_pred_train': False,
                      'save_csv_log': True,
                      'loss_fuc': self.rmse,
                      'append_info': 'Yuanan Bike'}

        """
            Cross Validation Arguments
        """
        cv_args = {'n_cv': 10}

        """
            Base Parameters
        """
        base_parameters = self.get_base_params('dnn')

        """
            Train Single Model
        """
        TM.train_single_model('dnn', train_seed, cv_seed,
                              # num_boost_round=1000,
                              base_parameters=base_parameters, train_args=train_args, cv_args=cv_args)

        print('======================================================')
        print('Global Train Seed: {}'.format(train_seed))
        print('Global Cross Validation Seed: {}'.format(cv_seed))


if __name__ == "__main__":

    start_time = time.time()

    # Check if directories exit or not
    utils.check_dir(parameters.path_list)

    print('======================================================')
    print('Start Training...')

    T = Training()
    T.train()

    print('------------------------------------------------------')
    print('All Tasks Done!')
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')
