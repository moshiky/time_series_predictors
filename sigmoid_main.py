
import numpy as np
from gradient_descent_fitter import GradientDescentFitter
from logger import Logger
import utils


H_INDEX_CSV_FILE_PATH = r'datasets/author_h_index.csv'
MEAN_H_INDEX_GRAPH_PARAMS = np.array([12.05, -0.18, 12.04], dtype=np.float64)
SERIES_LENGTH = 25
TEST_SIZE = 15
ONLINE_EPOCHS = 100
IS_ONLINE = True


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


def fit(fitter, y_for_x, initial_w_vector, epochs=None):
    fitted_w_vector = \
        fitter.fit_and_predict_gd_online(
            y_for_x, len(initial_w_vector), is_stochastic=True, fit_limit_rank=5e-2, plot_progress=False, gamma_0=1e-6,
            first_w=initial_w_vector, gamma_change_mode=GradientDescentFitter.GAMMA_INCREASING,
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
    gd_fitter = GradientDescentFitter(logger, target_function=get_mean_error_rate, gradient_function=get_gradient)

    # fit sigmoid params using gradient descent
    logger.log('# fit sigmoid params..')
    fitted_w_vector = fit(gd_fitter, train, MEAN_H_INDEX_GRAPH_PARAMS)

    logger.log('initial w vector: {original_vector}'.format(original_vector=MEAN_H_INDEX_GRAPH_PARAMS))
    logger.log('fitted w vector: {fitted_vector}'.format(fitted_vector=fitted_w_vector))

    # predict next values
    logger.log('# predict next values')
    # <offline>
    test_x_values = list(test.keys())
    predictions = get_sigmoid_predictions_for_values(test_x_values, fitted_w_vector)

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


def calculate_dataset_mean_scores():
    # create logger
    logger = Logger()

    # load series
    logger.log('# load dataset..')
    dataset = utils.load_dataset_for_gd_fitting(H_INDEX_CSV_FILE_PATH, SERIES_LENGTH, TEST_SIZE)
    logger.log('dataset size: {num_records}'.format(num_records=len(dataset)))

    # create fitter
    gd_fitter = GradientDescentFitter(logger, target_function=get_mean_error_rate, gradient_function=get_gradient)

    # fit sigmoid params using gradient descent
    logger.log('# fit all series..')
    figure_index = 0
    metric_counters = dict()
    for train_set, test_set in dataset[:20]:
        logger.log('## record #{figure_index}'.format(figure_index=figure_index))
        test_x_values = list(test_set.keys())

        # # plot full graph
        full_series_values = list(train_set.values()) + list(test_set.values())
        # utils.plot_graph_and_prediction(
        #     list(train_set.values()) + list(test_set.values()), None, 0, '', store=False, show=True
        # )

        if not IS_ONLINE:
            # ## <offline>
            # fit w vector
            fitted_w_vector = fit(gd_fitter, train_set, MEAN_H_INDEX_GRAPH_PARAMS)
            logger.log('fitted w vector: {fitted_vector}'.format(fitted_vector=fitted_w_vector))

            # predict next values
            logger.log('predict next values')
            fitted_values = get_sigmoid_predictions_for_values(list(range(1, min(test_x_values))), fitted_w_vector)
            predictions = get_sigmoid_predictions_for_values(test_x_values, fitted_w_vector)

        else:
            # ## <online>
            # fit first params
            history = dict(train_set)
            last_fitted_w_vector = fit(gd_fitter, history, MEAN_H_INDEX_GRAPH_PARAMS)

            # calculate initial fitted values
            fitted_values = \
                get_sigmoid_predictions_for_values(list(range(1, min(test_x_values))), last_fitted_w_vector)

            # predict test values in online mode
            predictions = dict()
            for i in range(len(test_x_values)):
                # predict next
                x_t = test_x_values[i]
                predictions[x_t] = get_sigmoid_prediction(x_t, last_fitted_w_vector)

                # add observation to history
                history[x_t] = test_set[x_t]

                # fit on history
                last_fitted_w_vector = fit(gd_fitter, history, last_fitted_w_vector, epochs=ONLINE_EPOCHS)

        # calculate final rank (R^2, MSE, MAPE)
        logger.log('final ranks:')
        metrics = utils.get_all_metrics(
                series_a=[x[1] for x in sorted(test_set.items())],
                series_b=[x[1] for x in sorted(predictions.items())]
            )
        utils.log_metrics_dict(logger, metrics)

        # plot graph
        utils.plot_graph_and_prediction(
            original_series=full_series_values,
            fitted_values=list(fitted_values.values()),
            predictions=list(predictions.values()),
            prediction_start_index=len(train_set)+1,
            file_name='sigmoid_{id}.png'.format(id=figure_index),
            store=True,
            show=False
        )
        figure_index += 1

        # add to metric counters
        logger.log('## average overall ranks:')
        for metric_name in metrics.keys():
            if metric_name not in metric_counters.keys():
                metric_counters[metric_name] = list()
            metric_counters[metric_name].append(metrics[metric_name])

    # log average metrics
    utils.log_metrics_dict(logger, metric_counters)


if __name__ == '__main__':
    # fit_mean_h_index_graph()
    calculate_dataset_mean_scores()
