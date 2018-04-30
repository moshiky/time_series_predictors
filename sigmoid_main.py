
import multiprocessing
from multiprocessing import Pool as ThreadPool
import numpy as np
import time
import matplotlib.pyplot as plt
from gradient_descent_fitter import GradientDescentFitter
from logger import Logger
import utils
from graph_manager import GraphManager
from statistics_manager import StatisticsManager
# import sigmoid_functions_v2
import sigmoid_functions_v3


H_INDEX_CSV_FILE_PATH = r'datasets/author_h_index.csv'
GRAPH_OUTPUT_FOLDER_PATH = 'output'
SERIES_LENGTH = 25
TEST_SIZE = 10
IS_ONLINE = True
LAG_SIZE = 5
DATASET_SIZE = 500


def calculate_dataset_mean_scores():
    # create logger
    logger = Logger()

    # load series
    logger.log('# load dataset..')
    dataset = utils.load_dataset_for_gd_fitting(H_INDEX_CSV_FILE_PATH, SERIES_LENGTH, TEST_SIZE)
    if DATASET_SIZE is not None:
        dataset = dataset[:DATASET_SIZE]

    logger.log('loaded dataset size: {num_records}'.format(num_records=len(dataset)))

    # log hyper-parameters
    gamma_0 = 1e-3
    batch_size = 5
    update_batch_size = 2
    lag = LAG_SIZE
    initial_updates = 2000
    online_updates = 1000
    gradient_size_target = 0.001
    logger.log('hyper parameters: '
               'lr={lr}, initial_updates={initial_updates}, online_updates={online_updates}, batch size={batch_size}, '
               'lag={lag}, gradient_size_target={gradient_size_target}'.format(
                lr=gamma_0, initial_updates=initial_updates, online_updates=online_updates, batch_size=batch_size,
                lag=lag, gradient_size_target=gradient_size_target))

    # fit and predict
    metrics_storage = dict()
    start_time = time.time()
    record_id = 0
    initial_update_log = list()
    online_update_log = list()
    for record in dataset:
        if (record_id % 100) == 0:
            logger.log('record #{i}'.format(i=record_id))
        record_id += 1

        # split to train and test sets
        train_set, test_set = record

        # build fitter
        gd_fitter = \
            GradientDescentFitter(
                logger,
                sigmoid_functions_v3.SigmoidV3,
                gamma_0=gamma_0,
                should_shuffle=True,
                initial_updates=initial_updates,
                batch_size=batch_size,
                lag=lag,
                gradient_size_target=gradient_size_target
            )

        # fit on train
        initial_update_log.append(gd_fitter.fit(train_set))

        # get test predictions
        predictions = list()
        try:
            sorted_test_x_values = list(sorted(test_set.keys()))[:TEST_SIZE]
            for t in range(TEST_SIZE):
                x_t = sorted_test_x_values[t]
                predictions.append(gd_fitter.predict(x_t))

                if IS_ONLINE:
                    gd_fitter.update_train_set(x_t, test_set[x_t])
                    if ((t + 1) % update_batch_size) == 0:
                        online_update_log.append(gd_fitter.update_model(online_updates, gamma=gamma_0/(t+1)))

        except Exception as ex:
            if 'nan prediction' in str(ex):
                continue
            else:
                raise ex

        # calculate error
        try:
            error_metrics = utils.get_all_metrics(sorted(test_set.values())[:TEST_SIZE], predictions)
        except Exception as ex:
            if 'bad prediction' in str(ex):
                continue
            else:
                raise ex

        # if error_metrics['r2'] < 0.6 or error_metrics['mse'] > 50 or error_metrics['mape'] > 0.6:
        # if initial_update_log[-1] == initial_updates:
        #     logger.log(error_metrics)
        #     plt.plot(list(train_set.values()) + list(test_set.values()))
        #     train_set_size = len(train_set)
        #     plt.plot(list(range(train_set_size, train_set_size+len(predictions))), predictions)
        #     plt.show()

        # add to metric counters
        for metric_name in error_metrics.keys():
            if metric_name not in metrics_storage.keys():
                metrics_storage[metric_name] = list()
            metrics_storage[metric_name].append(error_metrics[metric_name])

    # log statistics
    logger.log('total time: {total_time} secs'.format(total_time=time.time()-start_time))
    logger.log('valid samples: {valid}'.format(valid=len(metrics_storage[list(metrics_storage.keys())[0]])))

    for x in [initial_update_log, online_update_log]:
        plt.hist(x, log=True, bins=100)
        plt.show()

    # log average error
    utils.log_metrics_dict(logger, metrics_storage)


if __name__ == '__main__':
    # fit_mean_h_index_graph()
    calculate_dataset_mean_scores()
