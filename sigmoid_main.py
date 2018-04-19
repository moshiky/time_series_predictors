
import multiprocessing
from multiprocessing import Pool as ThreadPool
import numpy as np
from gradient_descent_fitter import GradientDescentFitter
from logger import Logger
import utils
from graph_manager import GraphManager
from statistics_manager import StatisticsManager
# import sigmoid_functions_v2
import sigmoid_functions_v3
import time


H_INDEX_CSV_FILE_PATH = r'datasets/author_h_index.csv'
GRAPH_OUTPUT_FOLDER_PATH = 'output'
MEAN_H_INDEX_GRAPH_PARAMS = np.array([12.05, -0.18, 12.04], dtype=np.float64)
SERIES_LENGTH = 25
TEST_SIZE = 10
ONLINE_EPOCHS = 5000
IS_ONLINE = False
MAX_PROCESSES = 2
LAG_SIZE = 1
DATASET_SIZE = 500

lock = multiprocessing.Lock()
graph_manager = GraphManager(GRAPH_OUTPUT_FOLDER_PATH)
statistics_manager = StatisticsManager()


def get_mean_h_index_graph():
    y_for_x_train = {
        1: 1.038427771, 2: 1.228371716, 3: 1.522827041, 4: 1.848247451, 5: 2.196933883,
        6: 2.560174242, 7:2.948740461, 8: 3.357871277, 9: 3.789862903, 10: 4.245053015,
        11: 4.714898359, 12: 5.210474775, 13: 5.71287229, 14: 6.24127102, 15: 6.765482542,
        16: 7.3025596, 17: 7.851995678, 18: 8.402275951, 19: 8.945026001, 20: 9.44381036
    }

    y_for_x_test = {
        21: 9.890896198, 22: 10.24885209, 23: 10.59303554, 24: 10.93647228, 25: 11.23032671,
        26: 11.5080625, 27: 11.79206063, 28: 12.04333498, 29: 12.26126475, 30: 12.41589939
    }

    return y_for_x_train, y_for_x_test


def generate_series(w_vector=None):
    if w_vector is not None:
        # define settings
        l_param, a_param, c_param = w_vector
        x_range = [1, 31]
        y_range = [0, l_param]
        series_length = 30
        add_noise = False
        should_plot = False

        # generate and return series
        return utils.get_synthetic_sigmoid(
            l_param=l_param,
            a_param=a_param,
            c_param=c_param,
            x_range=x_range,
            y_range=y_range,
            length=series_length,
            add_noise=add_noise,
            should_plot=should_plot
        )

    else:
        return get_mean_h_index_graph()


def get_mean_error_rate(y_for_x, w_vector):
    # extract params
    l_param, a_param, c_param = w_vector

    # calculate mean error rate
    error_rates_total = 0.0
    for x_t in y_for_x.keys():
        y_t = y_for_x[x_t]
        function_value = l_param / (1 + c_param * np.exp(a_param * x_t))
        error_rates_total += (abs(y_t - function_value) / y_t)

    # return mean error rate
    return error_rates_total / len(y_for_x)


def get_gradient(x_t, y_t, w_vector):
    # extract params
    l_param, a_param, c_param = w_vector

    # calculate common values
    e_ax = np.exp(a_param * x_t)
    bottom_part = 1 + c_param * e_ax
    common_start = 2 * ((l_param / bottom_part) - y_t)

    # calculate dq/dl
    dq_dl = 1 / bottom_part

    # calculate dq/da
    dq_da = (-1) * l_param * c_param * x_t * e_ax / np.square(bottom_part)

    # calculate dq/dc
    dq_dc = (-1) * l_param * e_ax / np.square(bottom_part)

    # return gradient
    return np.array([
        common_start * dq_dl,
        common_start * dq_da,
        common_start * dq_dc
    ], dtype=np.float64)


def get_sigmoid_prediction(x_value, w_vector):
    # extract params
    l_param, a_param, c_param = w_vector

    # return sigmoid value
    return l_param / (1 + c_param * np.exp(a_param * x_value))


def get_sigmoid_predictions_for_values(x_values, w_vector):
    # calculate values
    y_for_x = dict()
    for x_t in x_values:
        y_for_x[x_t] = get_sigmoid_prediction(x_t, w_vector)
    return y_for_x


def fit(fitter, y_for_x, initial_w_vector, epochs=None, gamma_change_mode=GradientDescentFitter.GAMMA_INCREASING,
        gamma_0=1e-6):
    fitted_w_vector = \
        fitter.fit_and_predict_gd_online(
            y_for_x, len(initial_w_vector), is_stochastic=True, fit_limit_rank=1e-4, plot_progress=False,
            first_w=initial_w_vector, gamma_change_mode=gamma_change_mode, gamma_0=gamma_0,
            evaluation_plot_mode=GradientDescentFitter.PLOT_RECENT, max_epochs=epochs
        )
    return fitted_w_vector


def fit_mean_h_index_graph():

    # create logger
    logger = Logger()

    # load series
    logger.log('# load series..')
    train, test = generate_series()

    # create fitter
    gd_fitter = \
        GradientDescentFitter(
            logger,
            target_function=sigmoid_functions_v2.get_mean_error_rate,
            gradient_function=sigmoid_functions_v2.get_gradient
        )

    # fit sigmoid params using gradient descent
    logger.log('# fit sigmoid params..')
    initial_w_vector = np.array([0.49937523, 0.89339831, 0.08601164, -0.22586464, 0.43639608])
    fitted_w_vector = fit(gd_fitter, train, initial_w_vector, gamma_change_mode=GradientDescentFitter.GAMMA_DECREASING,
                          gamma_0=1e-4)

    logger.log('initial w vector: {original_vector}'.format(original_vector=initial_w_vector))
    logger.log('fitted w vector: {fitted_vector}'.format(fitted_vector=fitted_w_vector))

    # predict next values
    logger.log('# predict next values')
    # <offline>
    test_x_values = list(test.keys())
    predictions = sigmoid_functions_v2.get_sigmoid_predictions_for_values(test_x_values, fitted_w_vector)

    # <online>

    # calculate final rank (R^2, MSE, MAPE)
    logger.log('# final ranks:')
    utils.log_metrics_dict(
        logger,
        utils.get_all_metrics(
            series_a=[x[1] for x in sorted(test.items())],
            series_b=[x[1] for x in sorted(predictions.items())]
        )
    )


def fit_online_thread_func(sample_sets, record_index):
    # get loggers
    logger = Logger()

    # create fitter
    # gd_fitter = GradientDescentFitter(logger, target_function=get_mean_error_rate, gradient_function=get_gradient)
    gd_fitter = \
        GradientDescentFitter(
            logger,
            target_function=sigmoid_functions_v2.get_mean_error_rate,
            gradient_function=sigmoid_functions_v2.get_gradient
        )

    # extract data sets
    train_set, test_set = sample_sets
    combined = dict(train_set)
    for key in test_set:
        combined[key] = test_set[key]

    # log record index
    logger.log('record #{record_index}'.format(record_index=record_index))
    test_x_values = list(test_set.keys())

    # fit values
    history = dict(combined)
    predictions = dict()
    last_fitted_w_vector = sigmoid_functions_v2.INITIAL_W_VECTOR
    fitted_values_vector = None
    try:
        sorted_history_keys = list(sorted(history.keys()))

        for i in range(2, len(sorted_history_keys)):

            x_t = sorted_history_keys[i]

            # predict
            if i == len(train_set):
                fitted_values_vector = np.array(last_fitted_w_vector)

            if i >= len(train_set):
                predictions[x_t] = \
                    sigmoid_functions_v2.get_sigmoid_prediction(x_t, last_fitted_w_vector, should_round=False)

            # build recent history
            recent_history = dict(
                [(x_t, history[x_t])] * 20
                + [(sorted_history_keys[i-1], history[sorted_history_keys[i-1]])] * 2
                + [(sorted_history_keys[i-2], history[sorted_history_keys[i-2]])] * 1
            )

            # fit
            last_fitted_w_vector = \
                fit(gd_fitter, recent_history, last_fitted_w_vector, gamma_0=1e-6,
                    gamma_change_mode=GradientDescentFitter.GAMMA_INCREASING, epochs=ONLINE_EPOCHS)

    except Exception as ex:
        logger.log('ERROR: {ex}'.format(ex=ex))
        return None

    # calculate initial fitted values
    fitted_values = \
        sigmoid_functions_v2.get_sigmoid_predictions_for_values(
            list(range(1, min(test_x_values))), fitted_values_vector
        )

    # fit first params
    # history = dict(train_set)
    # last_fitted_w_vector = sigmoid_fuctions_v2.INITIAL_W_VECTOR
    # try:
    #     sorted_history_keys = list(sorted(history.keys()))
    #     for i in range(2, len(sorted_history_keys)):
    #         recent_history = dict(
    #             [(sorted_history_keys[i], history[sorted_history_keys[i]])] * 9
    #             + [(sorted_history_keys[i-1], history[sorted_history_keys[i-1]])] * 3
    #             + [(sorted_history_keys[i-2], history[sorted_history_keys[i-2]])] * 1
    #         )
    #         last_fitted_w_vector = \
    #             fit(gd_fitter, recent_history, last_fitted_w_vector, gamma_0=1e-4,
    #                 gamma_change_mode=GradientDescentFitter.GAMMA_DECREASING, epochs=ONLINE_EPOCHS)
    #
    # except Exception as ex:
    #     logger.log('ERROR: {ex}'.format(ex=ex))
    #     return None
    #
    # # calculate initial fitted values
    # fitted_values = \
    #     sigmoid_fuctions_v2.get_sigmoid_predictions_for_values(
    #         list(range(1, min(test_x_values))), last_fitted_w_vector
    #     )
    #
    # # predict test values in online mode
    # predictions = dict()
    # for i in range(len(test_x_values)):
    #     # predict next
    #     x_t = test_x_values[i]
    #     predictions[x_t] = sigmoid_functions_v2.get_sigmoid_prediction(x_t, last_fitted_w_vector)
    #
    #     # add observation to history
    #     history[x_t] = test_set[x_t]
    #
    #     # remove oldest observations
    #     while len(history) > LAG_SIZE:
    #         oldest_xt = min(history.keys())
    #         history.pop(oldest_xt)
    #
    #     # fit on history
    #     try:
    #         last_fitted_w_vector = fit(gd_fitter, history, last_fitted_w_vector, epochs=ONLINE_EPOCHS,
    #                                    gamma_0=1e-4,
    #                                    gamma_change_mode=GradientDescentFitter.GAMMA_DECREASING)
    #     except Exception as ex:
    #         logger.log('ERROR: {ex}'.format(ex=ex))
    #         return None

    # calculate final rank (R^2, MSE, MAPE)
    metrics = utils.get_all_metrics(
        series_a=[x[1] for x in sorted(test_set.items())],
        series_b=[x[1] for x in sorted(predictions.items())]
    )
    # statistics_manager.add_metrics(metrics)
    with lock:
        statistics_manager.write_to_csv(metrics)
        logger.log('fitting ranks:')
        utils.log_metrics_dict(logger, metrics)

        # plot graph
        graph_manager.plot_graph_and_prediction(
            original_series=list(train_set.values()) + list(test_set.values()),
            fitted_values=list(fitted_values.values()),
            predictions=list(predictions.values()),
            prediction_start_index=len(train_set) + 1,
            file_name='sigmoid_{record_index}'.format(record_index=record_index),
            store=True,
            show=False
        )


def fit_online(dataset):
    # get logger
    logger = Logger()

    # create thread pool
    thread_pool = ThreadPool(processes=MAX_PROCESSES)

    # fit all series in parallel
    logger.log('start parallel fitting..')
    thread_pool.starmap(
        fit_online_thread_func,
        [[dataset[i], i] for i in range(len(dataset))]
    )
    thread_pool.close()
    thread_pool.join()
    logger.log('parallel fitting completed')

    # log average performance
    logger.log('## average overall ranks:')
    # utils.log_metrics_dict(logger, statistics_manager.get_average_metrics())
    utils.log_metrics_dict(logger, statistics_manager.get_avg_from_csv())


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
    gamma_0 = 1e-4
    epochs = 50
    logger.log('hyper parameters: lr={lr}, epochs={epochs}'.format(lr=gamma_0, epochs=epochs))

    # fit and predict
    metrics_storage = dict()
    start_time = time.time()
    record_id = 0
    for record in dataset:
        if (record_id % 100) == 0:
            logger.log('record #{i}'.format(i=record_id))
        record_id += 1

        # split to train and test sets
        train_set, test_set = record

        # build fitter
        gd_fitter = GradientDescentFitter(logger, sigmoid_functions_v3.SigmoidV3)

        # fit on train
        gd_fitter.fit(train_set, gamma_0=gamma_0, should_shuffle=True, epochs=epochs)

        # get test predictions
        predictions = list()
        test_values = list()
        try:
            for x_t in test_set.keys():
                predictions.append(gd_fitter.predict(x_t))
                test_values.append(test_set[x_t])
        except Exception as ex:
            if 'nan prediction' in str(ex):
                continue
            else:
                raise ex

        # calculate error
        try:
            error_metrics = utils.get_all_metrics(test_values, predictions)
        except Exception as ex:
            if 'bad prediction' in str(ex):
                continue
            else:
                raise ex

        # add to metric counters
        for metric_name in error_metrics.keys():
            if metric_name not in metrics_storage.keys():
                metrics_storage[metric_name] = list()
            metrics_storage[metric_name].append(error_metrics[metric_name])

    # log statistics
    logger.log('total time: {total_time} secs'.format(total_time=time.time()-start_time))
    logger.log('valid samples: {valid}'.format(valid=len(metrics_storage[list(metrics_storage.keys())[0]])))

    # log average error
    utils.log_metrics_dict(logger, metrics_storage)

    # if IS_ONLINE:
    #     fit_online(dataset)
    #
    # else:
    #     # create fitter
    #     gd_fitter = GradientDescentFitter(logger, target_function=get_mean_error_rate, gradient_function=get_gradient)
    #
    #     # fit sigmoid params using gradient descent
    #     logger.log('# fit all series..')
    #     figure_index = 0
    #     metric_counters = dict()
    #     for train_set, test_set in dataset:
    #         logger.log('## record #{figure_index}'.format(figure_index=figure_index))
    #         test_x_values = list(test_set.keys())
    #
    #         # # plot full graph
    #         full_series_values = list(train_set.values()) + list(test_set.values())
    #
    #         # ## <offline>
    #         # fit w vector
    #         fitted_w_vector = fit(gd_fitter, train_set, MEAN_H_INDEX_GRAPH_PARAMS)
    #         logger.log('fitted w vector: {fitted_vector}'.format(fitted_vector=fitted_w_vector))
    #
    #         # predict next values
    #         logger.log('predict next values')
    #         fitted_values = get_sigmoid_predictions_for_values(list(range(1, min(test_x_values))), fitted_w_vector)
    #         predictions = get_sigmoid_predictions_for_values(test_x_values, fitted_w_vector)
    #
    #         # calculate final rank (R^2, MSE, MAPE)
    #         logger.log('final ranks:')
    #         metrics = utils.get_all_metrics(
    #                 series_a=[x[1] for x in sorted(test_set.items())],
    #                 series_b=[x[1] for x in sorted(predictions.items())]
    #             )
    #         utils.log_metrics_dict(logger, metrics)
    #
    #         # plot graph
    #         GraphManager(GRAPH_OUTPUT_FOLDER_PATH).plot_graph_and_prediction(
    #             original_series=full_series_values,
    #             fitted_values=list(fitted_values.values()),
    #             predictions=list(predictions.values()),
    #             prediction_start_index=len(train_set)+1,
    #             file_name='sigmoid_{id}.png'.format(id=figure_index),
    #             store=True,
    #             show=False
    #         )
    #         figure_index += 1
    #
    #         # add to metric counters
    #         for metric_name in metrics.keys():
    #             if metric_name not in metric_counters.keys():
    #                 metric_counters[metric_name] = list()
    #             metric_counters[metric_name].append(metrics[metric_name])
    #
    #     # log average metrics
    #     logger.log('## average overall ranks:')
    #     utils.log_metrics_dict(logger, metric_counters)


if __name__ == '__main__':
    # fit_mean_h_index_graph()
    calculate_dataset_mean_scores()
