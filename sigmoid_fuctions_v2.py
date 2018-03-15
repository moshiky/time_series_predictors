
import numpy as np


INITIAL_W_VECTOR = np.array([0.49937523, 0.89339831, 0.08601164, -0.22586464, 0.43639608])


def get_sigmoid_predictions_for_values(x_values, w_vector):
    # calculate values
    y_for_x = dict()
    for x_t in x_values:
        y_for_x[x_t] = get_sigmoid_prediction(x_t, w_vector)
    return y_for_x


def get_sigmoid_prediction(x_value, w_vector):
    # extract params
    a_param, b_param, c_param, k_param, f_param = w_vector

    # return sigmoid value
    return (
        a_param +
        b_param / (c_param + np.exp(k_param * x_value + f_param))
    )


def get_mean_error_rate(y_for_x, w_vector):
    # calculate mean error rate
    error_rates_total = 0.0
    for x_t in y_for_x.keys():
        y_t = y_for_x[x_t]
        function_value = get_sigmoid_prediction(x_t, w_vector)
        error_rates_total += (abs(y_t - function_value) / y_t)

    # return mean error rate
    return error_rates_total / len(y_for_x)


def get_gradient(x_t, y_t, w_vector):
    # extract params
    a_param, b_param, c_param, k_param, f_param = w_vector

    # calculate common values
    e_kxf = np.exp(k_param * x_t + f_param)
    BP = c_param + e_kxf
    BV = 2 * (a_param - y_t + (b_param / BP))

    # calculate dq/da
    dq_da = BV

    # calculate dq/db
    dq_db = BV / BP

    # calculate dq/dc
    dq_dc = (-1) * b_param * BV / np.square(BP)

    # calculate dq/dk
    dq_dk = (-1) * b_param * x_t * e_kxf * BV / np.square(BP)

    # calculate dq/df
    dq_df = (-1) * b_param * e_kxf * BV / np.square(BP)

    # return gradient
    return np.array([
        dq_da, dq_db, dq_dc, dq_dk, dq_df
    ], dtype=np.float64)
