import time
from models import utils
from sklearn.model_selection import GridSearchCV


class SKLearnGridSearch(object):

    def __init__(self):
        pass

    @staticmethod
    def grid_search(log_path, tr_x, tr_y, reg, params, params_grid, cv_generator, cv_args):
        """
             Grid Search
        """
        start_time = time.time()

        grid_search_model = GridSearchCV(estimator=reg,
                                         param_grid=params_grid,
                                         scoring='neg_log_loss',
                                         verbose=1,
                                         n_jobs=-1,
                                         cv=cv_generator(tr_x, tr_y, **cv_args))

        # Start Grid Search
        print('Grid Searching...')

        grid_search_model.fit(tr_x, tr_y)

        best_parameters = grid_search_model.best_estimator_.get_params()
        best_score = grid_search_model.best_score_

        print('Best score: %0.6f' % best_score)
        print('Best parameters set:')

        for param_name in sorted(params_grid.keys()):
            print('\t%s: %r' % (param_name, best_parameters[param_name]))

        total_time = time.time() - start_time

        utils.save_grid_search_log(log_path, params, params_grid, best_score, best_parameters, total_time)
