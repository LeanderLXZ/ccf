import random
import time

from models import utils, parameters
from models.training_mode import TrainingMode


class Training:

    def __init__(self):
        pass

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

        else:
            print('------------------------------------------------------')
            print('[W] Training without Base Parameters')
            base_parameters = None

        return base_parameters

    @staticmethod
    def get_cv_args(model_name=None):

        from models.cross_validation import CrossValidation

        if model_name == 'custom_cv':
            cv_args = {'valid_rate': 0.1,
                       'n_cv': 10,
                       'cv_generator': CrossValidation.sk_k_fold}

        else:
            cv_args = {'n_splits': 10,
                       'n_cv': 10}
            print('------------------------------------------------------')
            print('[W] Training with Base cv_args:\n', cv_args)

        return cv_args

    def train(self):
        """
            Model Name:
            'lr':           Logistic Regression
            'rf':           Random Forest
            'ab':           AdaBoost
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
        train_seed = random.randint(0, 500)
        cv_seed = random.randint(0, 500)
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
                      'save_cv_pred': False,
                      'save_cv_pred_train': False,
                      'save_csv_log': True,
                      'loss_fuc': None,
                      'append_info': 'forward_window_postscale_mdp-11_sub'}

        """
            Cross Validation Arguments
        """
        # cv_args = {'n_splits': 10,
        #            'n_cv': 10}

        cv_args = self.get_cv_args('xgb')

        """
            Base Parameters
        """
        # base_parameters = self.get_base_params('xgb')
        base_parameters = None

        """
            Train Single Model
        """
        # TM.train_single_model('xgb', train_seed, cv_seed, num_boost_round=88,
        #                       reduced_feature_list=reduced_feature_list, base_parameters=base_parameters,
        #                       train_args=train_args, use_multi_group=False)

        """
            Auto Train with Logs of Boost Round
        """
        pg_list = [
                   # [
                   #  ['n_cv', n_cv_list],
                   #  ['valid_rate', valid_rate_list]
                   #  ]
                   [
                    ['max_depth', [11]],
                    ['min_child_weight', [9]],
                    ['subsample', (0.88, 0.90, 0.92)],
                    ['colsample_bytree', (0.86, 0.88, 0.9)],
                    ['colsample_bylevel', (0.7, 0.75, 0.8)]
                    ]
                   # [['learning_rate', [0.003]]]
                   ]
        train_seed_list = [999]
        cv_seed_list = [95]
        TM.auto_train_boost_round('xgb', num_boost_round=100, n_epoch=1, full_grid_search=True,
                                  train_seed_list=train_seed_list, cv_seed_list=cv_seed_list,
                                  base_parameters=base_parameters, parameter_grid_list=pg_list, save_final_pred=False,
                                  train_args=train_args, cv_args=cv_args)

        """Train Different Rounds"""
        # num_boost_round_list = [83, 85, 87]
        # self.train_diff_round('xgb', TM, num_boost_round_list=num_boost_round_list, n_epoch=1, full_grid_search=True,
        #                       train_seed_list=train_seed_list, cv_seed_list=cv_seed_list,
        #                       base_parameters=base_parameters, parameter_grid_list=pg_list, save_final_pred=True,
        #                       train_args=train_args, cv_args=cv_args)

        """
            Auto Grid Search Parameters
        """
        # pg_list = [
        #            [
        #             ['max_depth', (8, 9, 10)],
        #             ['min_child_weight', (2, 4, 6, 8)],
        #             ['subsample', (0.81, 0.84, 0.87, 0.9)],
        #             ['colsample_bytree', (0.8, 0.85, 0.9)],
        #             ['colsample_bylevel', (0.7, 0.75, 0.8)],
        #             ]
        #            ]
        # train_seed_list = [999]
        # cv_seed_list = [95]
        # # train_seed_list = None
        # # cv_seed_list = None
        # TM.auto_grid_search('xgb', num_boost_round=95, n_epoch=1, full_grid_search=True,
        #                     train_seed_list=train_seed_list, cv_seed_list=cv_seed_list,
        #                     parameter_grid_list=pg_list, base_parameters=base_parameters, save_final_pred=False,
        #                     train_args=train_args, cv_args=cv_args)

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
