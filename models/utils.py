import pickle
import pandas as pd
import numpy as np
import os
import time
import random
import csv
from os.path import isdir

prejudged_data_path = './results/prejudge/'


# Save Data
def save_data_to_pkl(data, data_path):

    print('Saving ' + data_path + '...')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


# Save predictions to csv file
def save_pred_to_csv(file_path, index, pred):

    print('Saving Predictions To CSV File...')

    df = pd.DataFrame({'id': index, 'number': pred})

    df.to_csv(file_path + 'result.csv', sep=',', index=False)


# Save probabilities of train set to csv file
def save_pred_train_to_csv(file_path, pred, label):

    print('Saving Probabilities of Train Set To CSV File...')

    df = pd.DataFrame({'pred_train': pred, 'label': label})

    df.to_csv(file_path + 'pred_train.csv', sep=',', index=True)


# Save Grid Search Logs
def save_grid_search_log(log_path, params, params_grid, best_score, best_parameters, total_time):

    with open(log_path + 'grid_search_log.txt', 'a') as f:

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
        f.write('Total Time: {:.3f}s\n'.format(total_time))
        f.write('Best Score: {:.8f}\n'.format(best_score))
        f.write('Parameters:\n')
        f.write('\t' + str(params) + '\n\n')
        f.write('Parameters Grid:\n')
        f.write('\t' + str(params_grid) + '\n\n')
        f.write('Best Parameters Set:\n')
        for param_name in sorted(params_grid.keys()):
            f.write('\t' + str(param_name) + ': {}\n'.format(str(best_parameters[param_name])))


# Save Final Losses
def save_loss_log(log_path, count, parameters, valid_rate, n_cv, loss_train,
                  loss_valid, train_seed=None, cv_seed=None):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('===================== CV: {}/{} =====================\n'.format(count, n_cv))
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Valid Rate: {}\n'.format(valid_rate))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tCV Train Loss: {:.8f}\n'.format(loss_train))
        f.write('\tCV Validation Loss: {:.8f}\n'.format(loss_valid))


def save_final_loss_log(log_path, parameters, valid_rate, n_cv, loss_train_mean,
                        loss_valid_mean, train_seed=None, cv_seed=None):

    with open(log_path + 'loss_log.txt', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Final Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('==================== Final Losses ===================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Valid Rate: {}\n'.format(valid_rate))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tTotal Train Loss: {:.8f}\n'.format(loss_train_mean))
        f.write('\tTotal Validation Loss: {:.8f}\n'.format(loss_valid_mean))
        f.write('=====================================================\n')

    with open(log_path + 'final_loss_log.txt', 'a') as f:

        print('Saving Final Losses...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))

        f.write('=====================================================\n')
        f.write('Time: {}\n'.format(local_time))
        f.write('------------------------------------------------------')
        f.write('Train Seed: {}\n'.format(train_seed))
        f.write('CV Seed: {}\n'.format(cv_seed))
        f.write('Valid Rate: {}\n'.format(valid_rate))
        f.write('Validation Spilt Number: {}\n'.format(n_cv))
        f.write('Parameters:\n')
        f.write('\t' + str(parameters) + '\n\n')
        f.write('Losses:\n')
        f.write('\tTotal Train Loss: {:.8f}\n'.format(loss_train_mean))
        f.write('\tTotal Validation Loss: {:.8f}\n'.format(loss_valid_mean))


# Save Loss Log to csv File
def save_final_loss_log_to_csv(idx, log_path, loss_train_mean, loss_valid_mean,
                               train_seed, cv_seed, valid_rate, n_cv, parameters):

    if not os.path.isfile(log_path + 'csv_log.csv'):

        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            header = ['id', 'time', 'loss_train', 'loss_valid',
                      'train_seed', 'cv_seed', 'valid_rate', 'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:

        print('------------------------------------------------------')
        print('Saving Final Losses to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
        log = [idx, local_time, loss_train_mean, loss_valid_mean,
               train_seed, cv_seed, valid_rate, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Loss Log with Global Validation to csv File
def save_log_with_glv_to_csv(idx, log_path, loss_train_mean, loss_valid_mean,
                             train_seed, loss_global_valid_mean,
                             cv_seed, valid_rate, n_cv, parameters):

    if not os.path.isfile(log_path + 'csv_log.csv'):
        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log with Global Validation...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            header = ['id', 'time', 'loss_train', 'valid_loss', 'global_valid_loss',
                      'train_seed', 'cv_seed', 'valid_rate', 'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Final Losses with Global Validation to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
        log = [idx, local_time, loss_train_mean, loss_valid_mean, loss_global_valid_mean,
               train_seed, cv_seed, valid_rate, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Grid Search Log to csv File
def save_grid_search_log_to_csv(idx, log_path, loss_train_mean, loss_valid_mean, train_seed,
                                cv_seed, valid_rate, n_cv, parameters, param_name_list, param_value_list):

    if not os.path.isfile(log_path + 'csv_log.csv'):
        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            header = ['id', 'time', 'loss_train', 'loss_valid', *param_name_list,
                      'train_seed', 'cv_seed', 'valid_rate', 'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Final Losses to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
        log = [idx, local_time, loss_train_mean, loss_valid_mean,
               *param_value_list, train_seed, cv_seed, valid_rate, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Loss Log with Global Validation to csv File
def save_grid_search_log_with_glv_to_csv(idx, log_path, loss_train_mean, loss_valid_mean,
                                         train_seed, loss_global_valid_mean,  cv_seed, valid_rate, n_cv,
                                         parameters, param_name_list, param_value_list):

    if not os.path.isfile(log_path + 'csv_log.csv'):
        print('------------------------------------------------------')
        print('Creating csv File of Final Loss Log with Global Validation...')

        with open(log_path + 'csv_log.csv', 'w') as f:
            header = ['id', 'time', 'loss_train', 'valid_loss', 'global_valid_loss',
                      *param_name_list, 'train_seed', 'cv_seed', 'valid_rate', 'n_cv', 'parameters']
            writer = csv.writer(f)
            writer.writerow(header)

    with open(log_path + 'csv_log.csv', 'a') as f:
        print('------------------------------------------------------')
        print('Saving Final Losses with Global Validation to csv File...')

        local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
        log = [idx, local_time, loss_train_mean, loss_valid_mean, loss_global_valid_mean,
               *param_value_list, train_seed, cv_seed, valid_rate, n_cv, str(parameters)]
        writer = csv.writer(f)
        writer.writerow(log)


# Save Boost Round Log to csv File
def save_boost_round_log_to_csv(model_name, boost_round_log_path, boost_round_log_upper_path, csv_idx, idx_round,
                                valid_loss_round_mean, train_loss_round_mean, train_seed, cv_seed,
                                parameters, param_name_list, param_value_list, param_name):

    valid_loss_dict = {}
    train_loss_dict = {}
    lowest_loss_dict = {}

    for i, idx in enumerate(idx_round):
        valid_loss_dict[idx] = valid_loss_round_mean[i]
        train_loss_dict[idx] = train_loss_round_mean[i]

    lowest_valid_loss_idx = list(np.argsort(valid_loss_round_mean)[:5])
    lowest_valid_loss = valid_loss_round_mean[lowest_valid_loss_idx[0]]
    lowest_train_loss = train_loss_round_mean[lowest_valid_loss_idx[0]]
    lowest_round = np.array(idx_round)[lowest_valid_loss_idx[0]]
    lowest_idx = np.array(idx_round)[lowest_valid_loss_idx]
    lowest_idx = np.sort(lowest_idx)

    for idx in lowest_idx:
        lowest_loss_dict[idx] = (valid_loss_dict[idx], train_loss_dict[idx])

    def _save_log(log_path):

        if not os.path.isfile(log_path + 'boost_round_log.csv'):

            print('------------------------------------------------------')
            print('Creating csv File of Boost Round Log...')

            with open(log_path + 'boost_round_log.csv', 'w') as f:
                header = ['idx', 'time', 'lowest_round', 'lowest_valid_loss', 'lowest_train_loss', 'round',
                          'valid_loss', 'train_loss', *param_name_list, 'train_seed', 'cv_seed', 'parameters']
                writer = csv.writer(f)
                writer.writerow(header)

        with open(log_path + 'boost_round_log.csv', 'a') as f:

            print('------------------------------------------------------')
            print('Saving Boost Round Log to csv File...')

            for ii, (round_idx, (valid_loss, train_loss)) in enumerate(lowest_loss_dict.items()):
                if ii == 0:
                    local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
                    log = [csv_idx, local_time, lowest_round, lowest_valid_loss, lowest_train_loss, round_idx,
                           valid_loss, train_loss, *param_value_list, train_seed, cv_seed, str(parameters)]
                else:
                    placeholder = [''] * len(param_value_list)
                    log = [csv_idx, '', '', '', '', round_idx, valid_loss,
                           train_loss, *placeholder, train_seed, cv_seed, '']
                writer = csv.writer(f)
                writer.writerow(log)

    _save_log(boost_round_log_path)
    boost_round_log_upper_path += model_name + param_name
    _save_log(boost_round_log_upper_path)


# Save Boost Round Log with Global Validation to csv File
def save_boost_round_log_gl_to_csv(model_name, boost_round_log_path, boost_round_log_upper_path, csv_idx, idx_round,
                                   valid_loss_round_mean, train_loss_round_mean, global_valid_loss_round_mean,
                                   train_seed, cv_seed, parameters, param_name_list, param_value_list, param_name):

    gl_valid_loss_dict = {}
    valid_loss_dict = {}
    train_loss_dict = {}
    lowest_loss_dict = {}

    for i, idx in enumerate(idx_round):
        gl_valid_loss_dict[idx] = global_valid_loss_round_mean[i]
        valid_loss_dict[idx] = valid_loss_round_mean[i]
        train_loss_dict[idx] = train_loss_round_mean[i]

    lowest_gl_valid_loss_idx = list(np.argsort(global_valid_loss_round_mean)[:5])
    lowest_gl_valid_loss = valid_loss_round_mean[lowest_gl_valid_loss_idx[0]]
    lowest_valid_loss = valid_loss_round_mean[np.argsort(valid_loss_round_mean)[0]]
    lowest_round = np.array(idx_round)[lowest_gl_valid_loss_idx[0]]
    lowest_idx = np.array(idx_round)[lowest_gl_valid_loss_idx]
    lowest_idx = np.sort(lowest_idx)

    for idx in lowest_idx:
        lowest_loss_dict[idx] = (gl_valid_loss_dict[idx], valid_loss_dict[idx], train_loss_dict[idx])

    def _save_log(log_path):

        if not os.path.isfile(log_path + 'boost_round_log.csv'):
            print('------------------------------------------------------')
            print('Creating csv File of Boost Round Log...')

            with open(log_path + 'boost_round_log.csv', 'w') as f:
                header = ['idx', 'time', 'lowest_round', 'lowest_global_valid_loss', 'lowest_cv_valid_loss',
                          'round', 'global_valid_loss', 'cv_valid_loss', 'cv_train_loss',
                          *param_name_list, 'train_seed', 'cv_seed', 'parameters']
                writer = csv.writer(f)
                writer.writerow(header)

        with open(log_path + 'boost_round_log.csv', 'a') as f:

            print('------------------------------------------------------')
            print('Saving Boost Round Log with Global Validation to csv File...')

            for ii, (round_idx, (gl_valid_loss, valid_loss, train_loss)) in enumerate(lowest_loss_dict.items()):
                if ii == 0:
                    local_time = time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(time.time()))
                    log = [csv_idx, local_time, lowest_round, lowest_gl_valid_loss, lowest_valid_loss, round_idx,
                           gl_valid_loss, valid_loss, train_loss, *param_value_list, train_seed, cv_seed, str(parameters)]
                else:
                    placeholder = [''] * len(param_value_list)
                    log = [csv_idx, '', '', '', '', round_idx, gl_valid_loss,
                           valid_loss, train_loss, *placeholder, train_seed, cv_seed, '']
                writer = csv.writer(f)
                writer.writerow(log)

    _save_log(boost_round_log_path)
    boost_round_log_upper_path += model_name + param_name
    _save_log(boost_round_log_upper_path)


def save_final_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean):

    print('------------------------------------------------------')
    print('Saving Final Boost Round Log...')

    df = pd.DataFrame({'idx': idx_round,
                       'train_loss': train_loss_round_mean,
                       'valid_loss': valid_loss_round_mean})
    cols = ['idx', 'train_loss', 'valid_loss']
    df = df.loc[:, cols]
    df.to_csv(boost_round_log_path, sep=',', index=False)


def save_final_boost_round_gl_log(boost_round_log_path, idx_round, train_loss_round_mean,
                                  valid_loss_round_mean, global_valid_loss_round_mean):

    print('------------------------------------------------------')
    print('Saving Final Boost Round Log with Global Validation...')

    df = pd.DataFrame({'idx': idx_round,
                       'cv_train_loss': train_loss_round_mean,
                       'cv_valid_loss': valid_loss_round_mean,
                       'global_valid_loss:': global_valid_loss_round_mean})
    cols = ['idx', 'cv_train_loss', 'cv_valid_loss', 'global_valid_loss:']
    df = df.loc[:, cols]
    print(df)
    df.to_csv(boost_round_log_path, sep=',', index=False)


# Save stacking outputs of layers
def save_stack_outputs(output_path, x_outputs, test_outputs, x_g_outputs, test_g_outputs):

    print('Saving Stacking Outputs of Layer...')

    save_data_to_pkl(x_outputs, output_path + 'x_outputs.p')
    save_data_to_pkl(test_outputs, output_path + 'test_outputs.p')
    save_data_to_pkl(x_g_outputs, output_path + 'x_g_outputs.p')
    save_data_to_pkl(test_g_outputs, output_path + 'test_g_outputs.p')


# Load Data
def load_pkl_to_data(data_path):

    print('Loading ' + data_path + '...')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


# Load Stacked Layer
def load_stacked_data(output_path):

    print('Loading Stacked Data...')

    x_outputs = load_pkl_to_data(output_path + 'x_outputs.p')
    test_outputs = load_pkl_to_data(output_path + 'test_outputs.p')
    x_g_outputs = load_pkl_to_data(output_path + 'x_g_outputs.p')
    test_g_outputs = load_pkl_to_data(output_path + 'test_g_outputs.p')

    return x_outputs, test_outputs, x_g_outputs, test_g_outputs


# Load Preprocessed Data
def load_preprocessed_data(data_file_path):

    x_train = pd.read_pickle(data_file_path + 'x_train.p')
    y_train = pd.read_pickle(data_file_path + 'y_train.p')
    x_test = pd.read_pickle(data_file_path + 'x_test.p')
    id_test = pd.read_pickle(data_file_path + 'id_test.p')

    return x_train, y_train, x_test, id_test


# Load Preprocessed Data
def load_global_valid_data(data_file_path):

    x_global_valid = pd.read_pickle(data_file_path + 'x_global_valid.p')
    x_g_global_valid = pd.read_pickle(data_file_path + 'x_g_global_valid.p')
    y_global_valid = pd.read_pickle(data_file_path + 'y_global_valid.p')

    return x_global_valid, x_g_global_valid, y_global_valid


# Load Preprocessed Category Data
def load_preprocessed_data_g(data_file_path):

    x_g_train = pd.read_pickle(data_file_path + 'x_g_train.p')
    x_g_test = pd.read_pickle(data_file_path + 'x_g_test.p')

    return x_g_train, x_g_test


# Load Preprocessed Code ID
def load_preprocessed_code_id(data_file_path):

    code_id_train = pd.read_pickle(data_file_path + 'code_id_train.p')
    code_id_test = pd.read_pickle(data_file_path + 'code_id_test.p')

    return code_id_train, code_id_test


# Print Information of Grid Search
def print_grid_info(model_name, parameters, parameters_grid):

    print('\nModel: ' + model_name + '\n')
    print("Parameters:")
    print(parameters)
    print('\n')
    print("Parameters' Grid:")
    print(parameters_grid)
    print('\n')


# Print CV Losses
def print_loss(pred_train, y_train, pred_valid, y_valid, loss_fuc):

    loss_train = loss_fuc(pred_train, y_train)
    loss_valid = loss_fuc(pred_valid, y_valid)
    print('------------------------------------------------------')
    print('Train Loss: {:.8f}\n'.format(loss_train),
          'Validation Loss: {:.8f}\n'.format(loss_valid))

    return loss_train, loss_valid


# Print Loss and Accuracy of Global Validation Set
def print_global_valid_loss(prob_global_valid, y_global_valid, loss_fuc):
    
    loss_global_valid = loss_fuc(prob_global_valid, y_global_valid)
    print('------------------------------------------------------')
    print('Global Valid Loss: {:.8f}\n'.format(loss_global_valid))
    
    return loss_global_valid


# Print Total Losses
def print_total_loss(loss_train_mean, loss_valid_mean):

    print('------------------------------------------------------')
    print('Total Train Loss: {:.8f}\n'.format(loss_train_mean),
          'Total Validation Loss: {:.8f}\n'.format(loss_valid_mean))


# Print Total Loss of Global Validation Set
def print_total_global_valid_loss(loss_global_valid_mean):

    print('------------------------------------------------------')
    print('Total Global Valid LogLoss: {:.8f}\n'.format(loss_global_valid_mean))


# Print Information of Cross Validation
def print_cv_info(cv_count, n_cv):
    print('======================================================')
    print('Training on the Cross Validation Set: {}/{}'.format(cv_count, n_cv))


# Check if directories exit or not
def check_dir(path_list):

    for dir_path in path_list:
        if not isdir(dir_path):
            os.makedirs(dir_path)


# Check if directories exit or not
def check_dir_model(pred_path, loss_log_path=None):

    if loss_log_path is not None:
        path_list = [pred_path,
                     pred_path + 'cv_results/',
                     pred_path + 'cv_pred_train/',
                     pred_path + 'final_results/',
                     pred_path + 'final_pred_train/',
                     loss_log_path]
    else:
        path_list = [pred_path,
                     pred_path + 'cv_results/',
                     pred_path + 'cv_pred_train/']

    check_dir(path_list)


# Calculate Means
def calculate_means(prob_test_total, pred_train_total, loss_train_total, loss_valid_total):

    prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
    pred_train_mean = np.mean(np.array(pred_train_total), axis=0)
    loss_train_mean = np.mean(loss_train_total)
    loss_valid_mean = np.mean(loss_valid_total)

    return prob_test_mean, pred_train_mean, loss_train_mean, loss_valid_mean


# Calculate Global Validation Means
def calculate_global_valid_means(loss_global_valid_total):

    return np.mean(loss_global_valid_total)


# Calculate Boost Round Means
def calculate_boost_round_means(train_loss_round_total, valid_loss_round_total,
                                weights=None, global_valid_loss_round_total=None):

    if weights is None:
        train_loss_round_mean = np.mean(np.array(train_loss_round_total), axis=0)
        valid_loss_round_mean = np.mean(np.array(valid_loss_round_total), axis=0)
    else:
        train_loss_round_mean = np.average(np.array(train_loss_round_total), axis=0, weights=weights)
        valid_loss_round_mean = np.average(np.array(valid_loss_round_total), axis=0, weights=weights)

    if global_valid_loss_round_total is not None:
        if weights is None:
            global_valid_loss_round_mean = np.mean(np.array(global_valid_loss_round_total), axis=0)
        else:
            global_valid_loss_round_mean = np.average(np.array(global_valid_loss_round_total), axis=0, weights=weights)
        return train_loss_round_mean, valid_loss_round_mean, global_valid_loss_round_mean
    else:
        return train_loss_round_mean, valid_loss_round_mean


# Generate random int list
def random_int_list(start, stop, length):

    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for _ in range(length):
        random_list.append(random.randint(start, stop))

    return random_list


# Get Path of Boost Round Train
def get_boost_round_log_path(boost_round_log_path, model_name, param_name_list, param_value_list, append_info):

    param_info = ''
    param_name = ''
    for i in range(len(param_name_list)):
        if param_name_list[i] == 'cv_weights':
            param_info += '_' + get_simple_param_name(param_name_list[i])
        else:
            param_info += '_' + get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])
        param_name += '_' + get_simple_param_name(param_name_list[i])

    boost_round_log_path += model_name + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + '_' + append_info + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + param_name + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + param_info + '/'
    check_dir([boost_round_log_path])

    return boost_round_log_path, param_name


# Get Path of Boost Round Train
def get_boost_round_log_upper_path(boost_round_log_path, model_name, param_name_list, append_info):

    param_name = ''
    for i in range(len(param_name_list)):
        param_name += '_' + get_simple_param_name(param_name_list[i])

    boost_round_log_path += model_name + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + '_' + append_info + '/'
    check_dir([boost_round_log_path])
    boost_round_log_path += model_name + param_name + '/'
    check_dir([boost_round_log_path])

    return boost_round_log_path


# Get Path of Grid Search
def get_grid_search_log_path(csv_log_path, model_name, param_name_list, param_value_list, append_info):

    param_info = ''
    param_name = ''
    for i in range(len(param_name_list)):
        if param_name_list[i] == 'cv_weights':
            param_info += '_' + get_simple_param_name(param_name_list[i])
        else:
            param_info += '_' + get_simple_param_name(param_name_list[i]) + '-' + str(param_value_list[i])
        param_name += '_' + get_simple_param_name(param_name_list[i])

    csv_log_path += model_name + '/'
    check_dir([csv_log_path])
    csv_log_path += model_name + '_' + append_info + '/'
    check_dir([csv_log_path])
    csv_log_path += model_name + param_name + '/'
    check_dir([csv_log_path])
    csv_log_path += model_name

    return csv_log_path, param_name, param_info


# Get Simple Parameter's Name
def get_simple_param_name(param_name):

    param_name_convert_dict = {'num_boost_round': 'nbr',
                               'learning_rate': 'lr',
                               'gamma': 'gma',
                               'max_depth': 'mdp',
                               'min_child_weight': 'mcw',
                               'subsample': 'sbs',
                               'colsample_bytree': 'cst',
                               'colsample_bylevel': 'csl',
                               'lambda': 'lmd',
                               'alpha': 'aph',
                               'early_stopping_rounds': 'esr',
                               'objective': 'obj',
                               'seed': 's',
                               'num_leaves': 'nl',
                               'min_data_in_leaf': 'mdl',
                               'min_sum_hessian_in_leaf': 'msh',
                               'feature_fraction': 'ff',
                               'feature_fraction_seed': 'ffs',
                               'bagging_fraction': 'bf',
                               'bagging_freq': 'bfq',
                               'bagging_seed': 'bs',
                               'lambda_l1': 'll1',
                               'lambda_l2': 'll2',
                               'min_gain_to_split': 'mgs',
                               'max_bin': 'mb',
                               'min_data_in_bin': 'mdb',
                               'iterations': 'itr',
                               'depth': 'dpt',
                               'l2_leaf_reg': 'l2r',
                               'bagging_temperature': 'btp',
                               'border': 'bdr',
                               'border_count': 'bct',
                               'od_pval': 'odp',
                               'od_wait': 'odw',
                               'od_type': 'odt',
                               'gradient_iterations': 'gitr',
                               'random_seed': 's',
                               'ctr_description': 'ctrd',
                               'ctr_border_count': 'ctrb',
                               'ctr_leaf_count_limit': 'ctrl',
                               'ignored_features': 'igf',
                               'epochs': 'epo',
                               'unit_number': 'unn',
                               'keep_probability': 'kpp',
                               'batch_size': 'bs',
                               'n_cv': 'cv',
                               'n_era': 'er',
                               'valid_rate': 'vr',
                               'window_size': 'ws',
                               'cv_weights': 'cw'}

    if param_name in param_name_convert_dict.keys():
        return param_name_convert_dict[param_name]
    else:
        return param_name
