import sys
import re
import time
import copy
import numpy as np
from models import utils
from models.cross_validation import CrossValidation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class ModelBase(object):
    """
        Base Model Class of Models in scikit-learn Module
    """
    def __init__(self, x_tr, y_tr, x_te, id_te, x_va=None, y_va=None, category_index=None):

        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_global_valid = x_va
        self.y_global_valid = y_va
        self.category_index = category_index

        self.importance = np.array([])
        self.indices = np.array([])
        self.model_name = ''
        self.num_boost_round = 0
        self.use_global_valid = False
        self.use_custom_obj = False

    @staticmethod
    def get_reg(parameters):

        print('This Is Base Model!')
        reg = DecisionTreeRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('------------------------------------------------------')
        print('This Is Base Model!')

        self.model_name = 'base'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_train, x_valid, x_test

    def fit(self, x_train, y_train, x_valid, y_valid, parameters=None):

        # Get Regressor
        reg = self.get_reg(parameters)

        # Training Model
        reg.fit(x_train, y_train)

        return reg

    def get_pattern(self):
        return None

    def fit_with_round_log(self, boost_round_log_path, cv_count, x_train, y_train,
                           x_valid, y_valid, parameters, param_name_list, param_value_list, append_info=''):

        boost_round_log_path, _ = utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                                                 param_name_list, param_value_list, append_info)
        boost_round_log_path += 'cv_cache/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_cv_{}_log.txt'.format(cv_count)

        print('Saving Outputs to:', boost_round_log_path)
        print('------------------------------------------------------')

        open(boost_round_log_path, 'w+').close()

        with open(boost_round_log_path, 'a') as f:
            __console__ = sys.stdout
            sys.stdout = f
            reg = self.fit(x_train, y_train, x_valid, y_valid, parameters)
            sys.stdout = __console__

        with open(boost_round_log_path) as f:
            lines = f.readlines()
            idx_round_cv = []
            train_loss_round_cv = []
            valid_loss_round_cv = []
            global_valid_loss_round_cv = []
            pattern = self.get_pattern()
            for line in lines:
                if pattern.match(line) is not None:
                    idx_round_cv.append(int(pattern.match(line).group(1)))
                    train_loss_round_cv.append(float(pattern.match(line).group(2)))
                    valid_loss_round_cv.append(float(pattern.match(line).group(3)))
                    if self.use_global_valid:
                        global_valid_loss_round_cv.append(float(pattern.match(line).group(4)))

        if self.use_global_valid:
            return reg, idx_round_cv, train_loss_round_cv, valid_loss_round_cv, global_valid_loss_round_cv
        else:
            return reg, idx_round_cv, train_loss_round_cv, valid_loss_round_cv

    def save_boost_round_log(self, boost_round_log_path, idx_round, train_loss_round_mean,
                             valid_loss_round_mean, train_seed, cv_seed, csv_idx, parameters,
                             param_name_list, param_value_list, append_info='', global_valid_loss_round_mean=None):

        boost_round_log_upper_path = \
            utils.get_boost_round_log_upper_path(boost_round_log_path, self.model_name, param_name_list, append_info)
        boost_round_log_path, param_name = \
            utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                           param_name_list, param_value_list, append_info)
        utils.save_boost_round_log_to_csv(self.model_name, boost_round_log_path, boost_round_log_upper_path, csv_idx,
                                          idx_round, valid_loss_round_mean, train_loss_round_mean, train_seed,
                                          cv_seed, parameters, param_name_list, param_value_list, param_name)
        if self.use_global_valid:
            utils.save_boost_round_log_gl_to_csv(self.model_name, boost_round_log_path, boost_round_log_upper_path,
                                                 csv_idx, idx_round, valid_loss_round_mean, train_loss_round_mean,
                                                 global_valid_loss_round_mean, train_seed, cv_seed, parameters,
                                                 param_name_list, param_value_list, param_name)

        boost_round_log_path += 'final_logs/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_' + str(csv_idx) + '_t-' \
            + str(train_seed) + '_c-' + str(cv_seed) + '_log.csv'

        if self.use_global_valid:
            utils.save_final_boost_round_gl_log(boost_round_log_path, idx_round, train_loss_round_mean,
                                                valid_loss_round_mean, global_valid_loss_round_mean)
        else:
            utils.save_final_boost_round_log(boost_round_log_path, idx_round,
                                             train_loss_round_mean, valid_loss_round_mean)

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = reg.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, reg, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test...')

        pred_test = np.array(reg.predict(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test)

        return pred_test

    def get_pred_train(self, reg, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train...')

        pred_train = np.array(reg.predict(x_train))[:, 1]

        if pred_path is not None:
            utils.save_pred_train_to_csv(pred_path, pred_train, self.y_train)

        return pred_train

    def save_csv_log(self, mode, csv_log_path, param_name_list, param_value_list, csv_idx, loss_train_mean,
                     loss_valid_mean, train_seed, cv_seed, valid_rate, n_cv, parameters,
                     boost_round_log_path=None, file_name_params=None, append_info='', loss_global_valid=None):

        if mode == 'auto_grid_search':

            csv_log_path, param_name, param_info = \
                utils.get_grid_search_log_path(csv_log_path, self.model_name,
                                               param_name_list, param_value_list, append_info)
            if self.use_global_valid:
                utils.save_grid_search_log_with_glv_to_csv(csv_idx, csv_log_path + param_name + '_',
                                                           loss_train_mean, loss_valid_mean, train_seed,
                                                           loss_global_valid, cv_seed, valid_rate, n_cv,
                                                           parameters, param_name_list, param_value_list)
                csv_log_path += str(param_info) + '_'
                utils.save_grid_search_log_with_glv_to_csv(csv_idx, csv_log_path, loss_train_mean, loss_valid_mean,
                                                           train_seed, loss_global_valid, cv_seed, valid_rate, n_cv,
                                                           parameters, param_name_list, param_value_list)
            else:
                utils.save_grid_search_log_to_csv(csv_idx, csv_log_path + param_name + '_', loss_train_mean,
                                                  loss_valid_mean, train_seed, cv_seed, valid_rate,
                                                  n_cv, parameters, param_name_list, param_value_list)
                csv_log_path += str(param_info) + '_'
                utils.save_grid_search_log_to_csv(csv_idx, csv_log_path, loss_train_mean, loss_valid_mean,
                                                  train_seed, cv_seed, valid_rate, n_cv,
                                                  parameters, param_name_list, param_value_list)

        elif mode == 'auto_train_boost_round':

            boost_round_log_path, _ = utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                                                     param_name_list, param_value_list, append_info)
            boost_round_log_path += self.model_name + '_' + append_info + '_'
            if self.use_global_valid:
                utils.save_grid_search_log_with_glv_to_csv(csv_idx, boost_round_log_path,
                                                           loss_train_mean, loss_valid_mean, train_seed,
                                                           loss_global_valid, cv_seed, valid_rate, n_cv,
                                                           parameters, param_name_list, param_value_list)
            else:
                utils.save_final_loss_log_to_csv(csv_idx, boost_round_log_path, loss_train_mean, loss_valid_mean,
                                                 train_seed, cv_seed, valid_rate, n_cv, parameters)

        elif mode == 'auto_train':

            csv_log_path += self.model_name + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + '_' + append_info + '/'
            utils.check_dir([csv_log_path])
            csv_log_path += self.model_name + '_'
            if file_name_params is not None:
                for p_name in file_name_params:
                    csv_log_path += str(parameters[p_name]) + '_'
            else:
                for p_name, p_value in parameters.items():
                    csv_log_path += str(p_value) + '_'

            if self.use_global_valid:
                utils.save_log_with_glv_to_csv(csv_idx, csv_log_path, loss_train_mean, loss_valid_mean,
                                               train_seed, loss_global_valid, cv_seed, valid_rate, n_cv, parameters)
            else:
                utils.save_final_loss_log_to_csv(csv_idx, csv_log_path, loss_train_mean, loss_valid_mean,
                                                 train_seed, cv_seed, valid_rate, n_cv, parameters)

        else:

            csv_log_path += self.model_name + '_' + append_info + '_'
            if self.use_global_valid:
                utils.save_log_with_glv_to_csv(csv_idx, csv_log_path, loss_train_mean, loss_valid_mean,
                                               train_seed, loss_global_valid, cv_seed, valid_rate, n_cv, parameters)
            else:
                utils.save_final_loss_log_to_csv(csv_idx, csv_log_path, loss_train_mean, loss_valid_mean,
                                                 train_seed, cv_seed, valid_rate, n_cv, parameters)

    def save_final_pred(self, mode, save_final_pred, pred_test_mean, pred_path, parameters,
                        csv_idx, train_seed, cv_seed, boost_round_log_path=None, param_name_list=None,
                        param_value_list=None, file_name_params=None, append_info=''):

        params = '_'
        if file_name_params is not None:
            for p_name in file_name_params:
                params += utils.get_simple_param_name(p_name) + '-' + str(parameters[p_name]) + '_'
        else:
            for p_name, p_value in parameters.items():
                params += utils.get_simple_param_name(p_name) + '-' + str(p_value) + '_'

        if save_final_pred:

            if mode == 'auto_train':

                pred_path += self.model_name + '/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + append_info + '/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + params + 'results/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + str(csv_idx) + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + '_'
                utils.save_pred_to_csv(pred_path, self.id_test, pred_test_mean)

            elif mode == 'auto_train_boost_round':

                boost_round_log_path, _ = utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                                                         param_name_list, param_value_list, append_info)
                pred_path = boost_round_log_path + 'final_results/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + str(csv_idx) + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + '_'
                utils.save_pred_to_csv(pred_path, self.id_test, pred_test_mean)

            else:
                pred_path += 'final_results/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_' + append_info + '/'
                utils.check_dir([pred_path])
                pred_path += self.model_name + '_t-' + str(train_seed) + '_c-' + str(cv_seed) + params
                utils.save_pred_to_csv(pred_path, self.id_test, pred_test_mean)

    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, boost_round_log_path=None,
              train_seed=None, cv_args=None, parameters=None, show_importance=False,
              save_cv_pred=True, save_cv_pred_train=False, save_final_pred=True, save_final_pred_train=False,
              save_csv_log=True, csv_idx=None, use_global_valid=False, return_pred_test=False,
              mode=None, param_name_list=None, param_value_list=None, use_custom_obj=False,
              file_name_params=None, append_info=None, loss_fuc=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        # Global Validation
        self.use_global_valid = use_global_valid

        # Use Custom Objective Function
        self.use_custom_obj = use_custom_obj

        cv_args_copy = copy.deepcopy(cv_args)
        valid_rate = cv_args_copy['valid_rate']
        n_cv = cv_args_copy['n_cv']
        cv_seed = cv_args_copy['cv_seed']

        # Append Information
        if append_info is None:
            append_info = 'v-' + str(valid_rate) + '_c-' + str(n_cv)

        if csv_idx is None:
            csv_idx = self.model_name

        # Print Start Information and Get Model Name
        self.print_start_info()

        if use_global_valid:
            print('------------------------------------------------------')
            print('[W] Using Global Validation...')

        cv_count = 0
        pred_test_total = []
        pred_train_total = []
        loss_train_total = []
        loss_valid_total = []
        idx_round = []
        train_loss_round_total = []
        valid_loss_round_total = []
        global_valid_loss_round_total = []
        pred_global_valid_total = []
        loss_global_valid_total = []

        # Get Cross Validation Generator
        if 'cv_generator' in cv_args_copy:
            cv_generator = cv_args_copy['cv_generator']
            if cv_generator is None:
                cv_generator = CrossValidation.random_split
            cv_args_copy.pop('cv_generator')
        else:
            cv_generator = CrossValidation.random_split
        print('------------------------------------------------------')
        print('[W] Using CV Generator: {}'.format(getattr(cv_generator, '__name__')))

        # Training on Cross Validation Sets
        for x_train, y_train, x_valid, y_valid in cv_generator(x=self.x_train, y=self.y_train, **cv_args_copy):

            # CV Start Time
            cv_start_time = time.time()

            cv_count += 1

            # Fitting and Training Model
            if mode == 'auto_train_boost_round':
                if use_global_valid:
                    reg, idx_round_cv, train_loss_round_cv, valid_loss_round_cv, global_valid_loss_round_cv = \
                        self.fit_with_round_log(boost_round_log_path, cv_count, x_train, y_train, x_valid,
                                                y_valid, parameters, param_name_list, param_value_list,
                                                append_info=append_info)
                    global_valid_loss_round_total.append(global_valid_loss_round_cv)
                else:
                    reg, idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                        self.fit_with_round_log(boost_round_log_path, cv_count, x_train, y_train, x_valid,
                                                y_valid, parameters, param_name_list, param_value_list,
                                                append_info=append_info)

                idx_round = idx_round_cv
                train_loss_round_total.append(train_loss_round_cv)
                valid_loss_round_total.append(valid_loss_round_cv)
            else:
                reg = self.fit(x_train, y_train, x_valid, y_valid, parameters)

            # Feature Importance
            if show_importance:
                self.get_importance(reg)

            # Prediction
            if save_cv_pred:
                cv_pred_path = pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(cv_count)
            else:
                cv_pred_path = None
            pred_test = self.predict(reg, self.x_test, pred_path=cv_pred_path)

            # Save Train Prediction to CSV File
            if save_cv_pred_train:
                cv_pred_train_path = pred_path + 'cv_pred_train/' + self.model_name + '_cv_{}_'.format(cv_count)
            else:
                cv_pred_train_path = None
            pred_train = self.get_pred_train(reg, x_train, pred_path=cv_pred_train_path)
            pred_train_all = self.get_pred_train(reg, self.x_train, pred_path=cv_pred_train_path)

            # Predict Global Validation Set
            if use_global_valid:
                pred_global_valid = self.predict(reg, self.x_global_valid)
            else:
                pred_global_valid = np.array([])

            # Get Prediction sof Validation Set
            pred_valid = self.predict(reg, x_valid)

            # Print LogLoss
            loss_train, loss_valid = utils.print_loss(pred_train, y_train, pred_valid, y_valid, loss_fuc)

            # Print Loss and Accuracy of Global Validation Set
            if use_global_valid:
                loss_global_valid = utils.print_global_valid_loss(pred_global_valid, self.y_global_valid, loss_fuc)
                pred_global_valid_total.append(pred_global_valid)
                loss_global_valid_total.append(loss_global_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + self.model_name + '_', cv_count, parameters, valid_rate, n_cv,
                                loss_train, loss_valid, train_seed, cv_seed)

            pred_test_total.append(pred_test)
            pred_train_total.append(pred_train_all)
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)

            # CV End Time
            print('------------------------------------------------------')
            print('CV Done! Using Time: {}s'.format(time.time() - cv_start_time))

        print('======================================================')
        print('Calculating Final Result...')

        # Calculate Means of pred and losses
        pred_test_mean, pred_train_mean, loss_train_mean, loss_valid_mean = \
            utils.calculate_means(pred_test_total, pred_train_total, loss_train_total, loss_valid_total)

        # Save Logs of num_boost_round
        if mode == 'auto_train_boost_round':
            if use_global_valid:
                train_loss_round_mean, valid_loss_round_mean, global_valid_loss_round_mean = \
                    utils.calculate_boost_round_means(train_loss_round_total, valid_loss_round_total,
                                                      global_valid_loss_round_total=global_valid_loss_round_total)
                self.save_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean,
                                          valid_loss_round_mean, train_seed, cv_seed, csv_idx,
                                          parameters, param_name_list, param_value_list, append_info=append_info,
                                          global_valid_loss_round_mean=global_valid_loss_round_mean)
            else:
                train_loss_round_mean, valid_loss_round_mean = \
                    utils.calculate_boost_round_means(train_loss_round_total, valid_loss_round_total)
                self.save_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean,
                                          valid_loss_round_mean, train_seed, cv_seed, csv_idx,
                                          parameters, param_name_list, param_value_list, append_info=append_info)

        # Save 'num_boost_round'
        if self.model_name in ['xgb', 'lgb']:
            parameters['num_boost_round'] = self.num_boost_round

        # Save Final Result
        if save_final_pred:
            self.save_final_pred(mode, save_final_pred, pred_test_mean, pred_path, parameters, csv_idx,
                                 train_seed, cv_seed, boost_round_log_path, param_name_list, param_value_list,
                                 file_name_params=file_name_params, append_info=append_info)

        # Save Final pred_train
        if save_final_pred_train:
            utils.save_pred_train_to_csv(pred_path + 'final_pred_train/' + self.model_name + '_',
                                         pred_train_mean, self.y_train)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + self.model_name + '_', parameters, valid_rate, n_cv,
                                  loss_train_mean, loss_valid_mean, train_seed, cv_seed)

        # Print Global Validation Information and Save
        if use_global_valid:
            # Calculate Means of Predictions and Losses
            loss_global_valid_mean = utils.calculate_global_valid_means(loss_global_valid_total)

            # Save csv log
            if save_csv_log:
                self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx, loss_train_mean,
                                  loss_global_valid_mean, train_seed, cv_seed, valid_rate, n_cv, parameters,
                                  boost_round_log_path=boost_round_log_path, file_name_params=file_name_params,
                                  append_info=append_info, loss_global_valid=loss_global_valid_mean)

        # Save Loss Log to csv File
        if save_csv_log:
            if not use_global_valid:
                self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx, loss_train_mean,
                                  loss_valid_mean, train_seed, cv_seed, valid_rate, n_cv, parameters,
                                  boost_round_log_path=boost_round_log_path, file_name_params=file_name_params,
                                  append_info=append_info)

        # Remove 'num_boost_round' of parameters
        if 'num_boost_round' in parameters:
            parameters.pop('num_boost_round')

        # Return Final Result
        if return_pred_test:
            return pred_test_mean


class LRegression(ModelBase):
    """
        Logistic Regression
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = LogisticRegression(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Logistic Regression...')

        self.model_name = 'lr'

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')
        self.importance = np.abs(reg.coef_)[0]
        indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, indices[f], self.importance[indices[f]]))


class DecisionTree(ModelBase):
    """
        Decision Tree
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = DecisionTreeRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Decision Tree...')

        self.model_name = 'dt'


class RandomForest(ModelBase):
    """
        Random Forest
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = RandomForestRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Random Forest...')

        self.model_name = 'rf'


class ExtraTrees(ModelBase):
    """
        Extra Trees
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = ExtraTreesRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Extra Trees...')

        self.model_name = 'et'


class AdaBoost(ModelBase):
    """
        AdaBoost
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = AdaBoostRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training AdaBoost...')

        self.model_name = 'ab'


class GradientBoosting(ModelBase):
    """
        Gradient Boosting
    """
    @staticmethod
    def get_reg(parameters):

        print('Initialize Model...')
        reg = GradientBoostingRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training Gradient Boosting...')

        self.model_name = 'gb'


class XGBoost(ModelBase):
    """
        XGBoost
    """
    def __init__(self, x_tr, y_tr, x_te, id_te, x_va=None, y_va=None, num_boost_round=None):

        super(XGBoost, self).__init__(x_tr, y_tr, x_te, id_te, x_va, y_va)

        self.num_boost_round = num_boost_round

    def print_start_info(self):

        print('======================================================')
        print('Training XGBoost...')

        self.model_name = 'xgb'

    @staticmethod
    def logloss_obj(pred, d_train):

        y = d_train.get_label()

        grad = (pred - y) / ((1.0 - pred) * pred)
        hess = (pred * pred - 2.0 * pred * y + y) / ((1.0 - pred) * (1.0 - pred) * pred * pred)

        return grad, hess

    def fit(self, x_train, y_train, x_valid, y_valid, parameters=None):

        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid)

        # Booster
        if self.use_global_valid:
            d_gl_valid = xgb.DMatrix(self.x_global_valid, label=self.y_global_valid)
            eval_list = [(d_train, 'Train'), (d_valid, 'Valid'), (d_gl_valid, 'Global_Valid')]
        else:
            eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]

        if self.use_custom_obj:
            bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                            obj=self.logloss_obj, evals=eval_list)
        else:
            bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        return bst

    def prejudge_fit(self, x_train, y_train, w_train, x_valid, y_valid, w_valid, parameters=None, use_weight=True):

        if use_weight:
            d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)
        else:
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid)

        # Booster
        eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]
        bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        return bst

    def get_pattern(self):

        if self.use_global_valid:
            return re.compile(r'\[(\d*)\]\tTrain-logloss:(.*)\tValid-logloss:(.*)\tGlobal_Valid-logloss:(.*)')
        else:
            return re.compile(r'\[(\d*)\]\tTrain-logloss:(.*)\tValid-logloss:(.*)')

    def get_importance(self, model):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = model.get_fscore()
        sorted_importance = sorted(self.importance.items(), key=lambda d: d[1], reverse=True)

        feature_num = len(self.importance)

        for i in range(feature_num):
            print('{} | feature {} | {}'.format(i + 1, sorted_importance[i][0], sorted_importance[i][1]))

    def predict(self, model, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test...')

        pred_test = model.predict(xgb.DMatrix(x_test))

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test)

        return pred_test

    def get_pred_train(self, model, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train...')

        pred_train = model.predict(xgb.DMatrix(x_train))

        if pred_path is not None:
            utils.save_pred_train_to_csv(pred_path, pred_train, self.y_train)

        return pred_train


class SKLearnXGBoost(ModelBase):
    """
        XGBoost using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):

        print('Initialize Model...')
        reg = XGBRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training XGBoost(sklearn)...')

        self.model_name = 'xgb_sk'

    def fit(self, x_train, y_train, x_valid, y_valid, parameters=None):

        # Get Regressor
        reg = self.get_reg(parameters)

        # Training Model
        reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                early_stopping_rounds=100, eval_metric='logloss', verbose=True)

        return reg

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = reg.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))


class LightGBM(ModelBase):
    """
        LightGBM
    """
    def __init__(self, x_tr, y_tr, x_te, id_te, x_va=None, y_va=None, num_boost_round=None):

        super(LightGBM, self).__init__(x_tr, y_tr, x_te, id_te, x_va, y_va)

        self.num_boost_round = num_boost_round

    def print_start_info(self):

        print('======================================================')
        print('Training LightGBM...')

        self.model_name = 'lgb'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_g_train, x_g_valid, x_g_test

    def fit(self, x_train, y_train, x_valid, y_valid, parameters=None):

        # Create Dataset
        print('------------------------------------------------------')
        print('Index of categorical feature: {}'.format(self.category_index))
        print('------------------------------------------------------')

        d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=self.category_index)
        d_valid = lgb.Dataset(x_valid, label=y_valid, categorical_feature=self.category_index)

        # Booster
        if self.use_global_valid:
            d_gl_valid = lgb.Dataset(self.x_global_valid, label=self.y_global_valid,
                                     categorical_feature=self.category_index)
            bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                            valid_sets=[d_valid, d_gl_valid, d_train],
                            valid_names=['Valid', 'Global_Valid', 'Train'])
        else:
            bst = lgb.train(parameters, d_train, num_boost_round=self.num_boost_round,
                            valid_sets=[d_valid, d_train], valid_names=['Valid', 'Train'])

        return bst

    def get_pattern(self):

        if self.use_global_valid:
            return re.compile(r"\[(\d*)\]\tTrain\'s binary_logloss: (.*)\tValid\'s binary_logloss:(.*)\tGlobal_Valid\'s binary_logloss:(.*)")
        else:
            return re.compile(r"\[(\d*)\]\tTrain\'s binary_logloss: (.*)\tValid\'s binary_logloss:(.*)")

    @staticmethod
    def logloss_obj(y, pred):

        grad = (pred - y) / ((1 - pred) * pred)
        hess = (pred * pred - 2 * pred * y + y) / ((1 - pred) * (1 - pred) * pred * pred)

        return grad, hess

    def get_importance(self, bst):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = bst.feature_importance()
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

        print('\n')

    def predict(self, bst, x_test, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Test...')

        pred_test = bst.predict(x_test)

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, pred_test)

        return pred_test

    def get_pred_train(self, bst, x_train, pred_path=None):

        print('------------------------------------------------------')
        print('Predicting Train...')

        pred_train = bst.predict(x_train)

        if pred_path is not None:
            utils.save_pred_train_to_csv(pred_path, pred_train, self.y_train)

        return pred_train


class SKLearnLightGBM(ModelBase):
    """
        LightGBM using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):

        print('Initialize Model...')
        reg = LGBMRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training LightGBM(sklearn)...')

        self.model_name = 'lgb_sk'

    def fit(self, x_train, y_train, x_valid, y_valid, parameters=None):

        # Get Regressor
        reg = self.get_reg(parameters)

        print('Index of categorical feature: {}'.format(self.category_index))

        # Fitting and Training Model
        reg.fit(x_train, y_train, categorical_feature=self.category_index,
                eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_names=['train', 'eval'],
                early_stopping_rounds=100, eval_metric='logloss', verbose=True)

        return reg


class CatBoost(ModelBase):
    """
        CatBoost
    """
    @staticmethod
    def get_reg(parameters=None):

        reg = CatBoostRegressor(**parameters)

        return reg

    def print_start_info(self):

        print('======================================================')
        print('Training CatBoost...')

        self.model_name = 'cb'

    @staticmethod
    def select_category_variable(x_train, x_g_train, x_valid, x_g_valid, x_test, x_g_test):

        return x_g_train, x_g_valid, x_g_test

    def fit(self, x_train, y_train, x_valid, y_valid, parameters=None):

        # Get Regressor
        reg = self.get_reg(parameters)

        print('------------------------------------------------------')
        print('Index of categorical feature: {}'.format(self.category_index))
        print('------------------------------------------------------')

        # Fitting and Training Model
        reg.fit(X=x_train, y=y_train, cat_features=self.category_index,
                baseline=None, use_best_model=None, eval_set=(x_valid, y_valid), verbose=True, plot=False)

        return reg

    def get_pattern(self):

        return re.compile(r'(\d*):\tlearn (.*)\ttest (.*)\tbestTest')

    def get_importance(self, reg):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = reg.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))
