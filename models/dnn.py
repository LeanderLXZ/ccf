import time
import os
import re
import sys
import copy
import numpy as np
import math
import tensorflow as tf
from os.path import isdir
from models.regressors import ModelBase
from models import utils
from models.cross_validation import CrossValidation


class DeepNeuralNetworks(ModelBase):
    """
        Deep Neural Networks
    """
    def __init__(self, x_tr, y_tr, x_te, id_te, x_va=None, y_va=None, parameters=None):

        super(DeepNeuralNetworks, self).__init__(x_tr, y_tr, x_te, id_te, x_va, y_va)

        # Hyperparameters
        self.parameters = parameters
        self.version = parameters['version']
        self.epochs = parameters['epochs']
        self.unit_number = parameters['unit_number']
        self.learning_rate = parameters['learning_rate']
        self.keep_probability = parameters['keep_probability']
        self.batch_size = parameters['batch_size']
        self.train_seed = parameters['seed']
        self.display_step = parameters['display_step']
        self.save_path = parameters['save_path']
        self.log_path = parameters['log_path']

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Deep Neural Networks...')
        print('------------------------------------------------------')

        self.model_name = 'dnn'

    def get_pattern(self):

        # [0] | CV: 10 | Epoch: 4/4 | Batch: 7300 | Time: 352.85s | Train_Loss: 0.71237314 | Valid_Loss: 0.72578128
        return re.compile(r'\[(\d*)\].*Train_Loss: (.*) \| Valid_Loss: (.*)')

    # Input Tensors
    def input_tensor(self):

        feature_num = self.x_train.shape[1]

        inputs_ = tf.placeholder(tf.float32, [None, feature_num], name='inputs')
        labels_ = tf.placeholder(tf.float32, None, name='labels')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
        is_training_ = tf.placeholder(tf.bool, name='is_training')

        return inputs_, labels_, learning_rate_, keep_prob_, is_training_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, is_training):

        if is_training is True:
            print('Using Batch Normalization')

        with tf.name_scope(layer_name):

            # x_shape = x_tensor.get_shape().as_list()
            # weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(x_shape[1]))
            weights_initializer = tf.truncated_normal_initializer()
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            #  weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)
            #  weights_reg = tf.contrib.layers.l1_regularizer(1e-3)
            #  normalizer_fn = tf.contrib.layers.batch_norm
            #  normalizer_params = {'is_training': is_training}

            fc = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                   num_outputs=num_outputs,
                                                   activation_fn=tf.nn.sigmoid,
                                                   weights_initializer=weights_initializer,
                                                   #  weights_regularizer=weights_reg,
                                                   #  normalizer_fn=normalizer_fn,
                                                   #  normalizer_params=normalizer_params,
                                                   biases_initializer=tf.zeros_initializer(dtype=tf.float32)
                                                   )

            tf.summary.histogram('fc_layer', fc)

            # fc = tf.nn.dropout(fc, keep_prob)

        return fc

    # Output Layer
    def output_layer(self, x_tensor, layer_name, num_outputs):

        with tf.name_scope(layer_name):

            # x_shape = x_tensor.get_shape().as_list()
            # weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(x_shape[1]))
            weights_initializer = tf.truncated_normal_initializer()
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            #  weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)

            out = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                    num_outputs=num_outputs,
                                                    activation_fn=None,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=tf.zeros_initializer(dtype=tf.float32)
                                                    )

        return out

    # Model
    def model(self, x, n_unit, keep_prob, is_training):

        #  fc1 = fc_layer(x, 'fc1', n_unit[0], keep_prob)
        #  fc2 = fc_layer(fc1, 'fc2', n_unit[1], keep_prob)
        #  fc3 = fc_layer(fc2, 'fc3', n_unit[2], keep_prob)
        #  fc4 = fc_layer(fc3, 'fc4', n_unit[3], keep_prob)
        #  fc5 = fc_layer(fc4, 'fc5', n_unit[4], keep_prob)
        #  logit_ = self.output_layer(fc5, 'output', 1)

        fc = [x]

        for i in range(len(n_unit)):
            fc.append(self.fc_layer(fc[i], 'fc{}'.format(i + 1), n_unit[i], keep_prob, is_training))

        logit_ = self.output_layer(fc[len(n_unit)], 'output', 1)

        return logit_

    # LogLoss
    @staticmethod
    def rmse_loss(logit, y):

        return tf.sqrt(tf.reduce_mean(tf.square(logit - y)))

    # Get Batches
    @staticmethod
    def get_batches(x, y, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num + 1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x, batch_y = x[ii: ii + batch_num], y[ii: ii + batch_num]
            else:
                batch_x, batch_y = x[ii:], y[ii:]

            yield batch_x, batch_y

    # Get Batches for Prediction
    @staticmethod
    def get_batches_for_predict(x, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num + 1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x = x[ii: ii + batch_num]
            else:
                batch_x = x[ii:]

            yield batch_x

    # Get Probabilities
    def get_pred(self, sess, logits, x, batch_num, inputs, keep_prob, is_training):

        pred = np.array([])

        for x_batch in self.get_batches_for_predict(x, batch_num):
            pred_batch = sess.run(logits, {inputs: x_batch, keep_prob: 1.0, is_training: False})
            pred_batch = pred_batch.flatten()
            pred = np.concatenate((pred, pred_batch))

        return pred

    # Trainer
    def trainer(self, sess, cv_counter, x_train, y_train, x_valid, y_valid,
                optimizer, merged, cost_, inputs, labels, lr, keep_prob, is_training, start_time):

        train_log_path = self.log_path + self.version + '/cv_{}/train'.format(cv_counter)
        valid_log_path = self.log_path + self.version + '/cv_{}/valid'.format(cv_counter)

        if not isdir(train_log_path):
            os.makedirs(train_log_path)
        if not isdir(valid_log_path):
            os.makedirs(valid_log_path)

        train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        sess.run(tf.global_variables_initializer())

        batch_counter = 0
        idx = 0

        for epoch_i in range(self.epochs):

            for batch_i, (batch_x, batch_y) in enumerate(self.get_batches(x_train, y_train, self.batch_size)):

                batch_counter += 1

                _, cost = sess.run([optimizer, cost_],
                                   {inputs: batch_x,
                                    labels: batch_y,
                                    lr: self.learning_rate,
                                    keep_prob: self.keep_probability,
                                    is_training: True})

                if str(cost) == 'nan':
                    raise ValueError('NaN BUG!!! Try Another Seed!!!')

                if batch_counter % self.display_step == 0 and batch_i > 0:

                    idx += 1

                    cost_valid_all = []
                    cost_train_all = []

                    for valid_batch_x, valid_batch_y in self.get_batches(x_valid, y_valid, self.batch_size):

                        summary_valid_i, cost_valid_i = sess.run([merged, cost_],
                                                                 {inputs: valid_batch_x,
                                                                  labels: valid_batch_y,
                                                                  keep_prob: 1.0,
                                                                  is_training: False})

                        valid_writer.add_summary(summary_valid_i, batch_counter)

                        cost_valid_all.append(cost_valid_i)

                    cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                    for train_batch_x, train_batch_y in self.get_batches(x_train, y_train, self.batch_size):
                        summary_train_i, cost_train_i = sess.run([merged, cost_],
                                                                 {inputs: train_batch_x,
                                                                  labels: train_batch_y,
                                                                  keep_prob: 1.0,
                                                                  is_training: False})

                        train_writer.add_summary(summary_train_i, batch_counter)

                        cost_train_all.append(cost_train_i)

                    cost_train = sum(cost_train_all) / len(cost_train_all)

                    total_time = time.time() - start_time

                    print('[{}] |'.format(idx),
                          'CV: {} |'.format(cv_counter),
                          'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                          'Batch: {} |'.format(batch_counter),
                          'Time: {:3.2f}s |'.format(total_time),
                          'Train_Loss: {:.8f} |'.format(cost_train),
                          'Valid_Loss: {:.8f}'.format(cost_valid))

    def train_with_round_log(self, boost_round_log_path, sess, cv_counter, x_train, y_train, x_valid,
                             y_valid, optimizer, merged, cost_, inputs, labels, lr, keep_prob, is_training,
                             start_time, param_name_list, param_value_list, append_info=''):

        boost_round_log_path, _ = utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                                                 param_name_list, param_value_list, append_info)
        boost_round_log_path += 'cv_cache/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_cv_{}_log.txt'.format(cv_counter)

        print('Saving Outputs to:', boost_round_log_path)
        print('------------------------------------------------------')

        open(boost_round_log_path, 'w+').close()

        with open(boost_round_log_path, 'a') as f:
            __console__ = sys.stdout
            sys.stdout = f
            self.trainer(sess, cv_counter, x_train, y_train, x_valid, y_valid,
                         optimizer, merged, cost_, inputs, labels, lr, keep_prob, is_training, start_time)
            sys.stdout = __console__

        with open(boost_round_log_path) as f:
            lines = f.readlines()
            idx_round_cv = []
            train_loss_round_cv = []
            valid_loss_round_cv = []
            pattern = self.get_pattern()
            for line in lines:
                if pattern.match(line) is not None:
                    idx_round_cv.append(int(pattern.match(line).group(1)))
                    train_loss_round_cv.append(float(pattern.match(line).group(2)))
                    valid_loss_round_cv.append(float(pattern.match(line).group(3)))

        return idx_round_cv, train_loss_round_cv, valid_loss_round_cv

    # Training
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

        cv_args_copy = copy.deepcopy(cv_args)
        n_cv = cv_args_copy['n_cv']
        cv_seed = cv_args_copy['cv_seed']
        valid_rate = 1 / float(n_cv)

        # Append Information
        if append_info is None:
            append_info = 'v-' + str(valid_rate) + '_c-' + str(n_cv)

        if csv_idx is None:
            csv_idx = self.model_name

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, lr, keep_prob, is_training = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_training)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                cost_ = tf.sqrt(tf.reduce_mean(tf.square(logits - labels)))
                # cost_ = tf.reduce_mean(tf.square(logits - labels))

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        # Training
        self.print_start_info()

        if use_global_valid:
            print('------------------------------------------------------')
            print('[W] Using Global Validation...')

        with tf.Session(graph=train_graph) as sess:

            # Merge all the summaries
            merged = tf.summary.merge_all()

            start_time = time.time()
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

                # Training
                if mode == 'auto_train_boost_round':
                    idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                        self.train_with_round_log(boost_round_log_path, sess, cv_count, x_train, y_train,
                                                  x_valid, y_valid, optimizer, merged, cost_,
                                                  inputs, labels, lr, keep_prob, is_training, start_time,
                                                  param_name_list, param_value_list, append_info=append_info)
                    idx_round = idx_round_cv
                    train_loss_round_total.append(train_loss_round_cv)
                    valid_loss_round_total.append(valid_loss_round_cv)
                else:
                    self.trainer(sess, cv_count, x_train, y_train, x_valid, y_valid, optimizer,
                                 merged, cost_, inputs, labels, lr, keep_prob, is_training, start_time)

                # Save Model
                # print('Saving model...')
                # saver = tf.train.Saver()
                # saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')

                # Prediction
                print('------------------------------------------------------')
                print('Predicting Probabilities...')
                pred_train = self.get_pred(sess, logits, x_train, self.batch_size, inputs, keep_prob, is_training)
                pred_train_all = self.get_pred(sess, logits, self.x_train, self.batch_size, inputs, keep_prob,
                                               is_training)
                pred_valid = self.get_pred(sess, logits, x_valid, self.batch_size, inputs, keep_prob, is_training)
                pred_test = self.get_pred(sess, logits, self.x_test, self.batch_size, inputs, keep_prob, is_training)

                # Predict Global Validation Set
                if use_global_valid:
                    pred_global_valid = self.get_pred(sess, logits, self.x_global_valid,
                                                      self.batch_size, inputs, keep_prob, is_training)
                else:
                    pred_global_valid = np.array([])

                # Print Losses of CV
                loss_train, loss_valid = utils.print_loss(pred_train, y_train, pred_valid, y_valid, loss_fuc)

                # Print Loss and Accuracy of Global Validation Set
                if use_global_valid:
                    loss_global_valid = utils.print_global_valid_loss(pred_global_valid,
                                                                      self.y_global_valid, self.rmse_loss)
                    pred_global_valid_total.append(pred_global_valid)
                    loss_global_valid_total.append(loss_global_valid)

                # Save Losses to File
                utils.save_loss_log(loss_log_path + self.model_name + '_', cv_count, parameters, valid_rate, n_cv,
                                    loss_train, loss_valid, train_seed, cv_seed)

                pred_test_total.append(pred_test)
                pred_train_total.append(pred_train_all)
                loss_train_total.append(loss_train)
                loss_valid_total.append(loss_valid)

                if save_cv_pred:
                    utils.save_pred_to_csv(pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(cv_count),
                                           self.id_test, pred_test)

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
