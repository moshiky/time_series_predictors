
import numpy as np
from gradient_descent_fitter import GradientDescentFitter
from logger import Logger
import utils


def generate_series(w_vector):
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


def get_sigmoid_predictions_for_values(x_values, w_vector):
    # extract params
    l_param, a_param, c_param = w_vector

    # calculate values
    y_for_x = dict()
    for x_t in x_values:
        y_t = l_param / (1 + c_param * np.exp(a_param * x_t))
        y_for_x[x_t] = y_t
    return y_for_x


def main():

    # create logger
    logger = Logger()

    # load series
    logger.log('# load series..')
    original_w_vector = np.array([70, -0.33, 70], dtype=np.float64)
    initial_w_vector = np.array([60, -1, 80], dtype=np.float64)
    train, test = generate_series(original_w_vector)

    # create fitter
    gd_fitter = GradientDescentFitter(logger, target_function=get_mean_error_rate, gradient_function=get_gradient)

    # fit sigmoid params using gradient descent
    logger.log('# fit sigmoid params..')
    fitted_w_vector = \
        gd_fitter.fit_and_predict_gd_online(
            train, len(original_w_vector), is_stochastic=True, fit_limit_rank=5e-2, plot_progress=True, gamma_0=1e-7,
            first_w=initial_w_vector, gamma_change_mode=GradientDescentFitter.GAMMA_INCREASING
        )

    logger.log('original w vector: {original_vector}'.format(original_vector=original_w_vector))
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


if __name__ == '__main__':
    main()
