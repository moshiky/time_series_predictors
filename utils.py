
from matplotlib import pyplot
import csv
from scipy import stats
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from consts import Consts


def parse_csv(file_path, should_shuffle=True, smoothing_level=0):
    # read file
    with open(file_path, 'rt') as file_handle:
        reader = csv.reader(file_handle)
        file_lines = list(reader)

    record_list = [[float(x) for x in l] for l in file_lines if len(l) > Consts.MIN_RECORD_LENGTH]
    if should_shuffle:
        random.shuffle(record_list)

    if smoothing_level > 0:
        # smooth graphs
        for record in record_list:
            # init indexes
            last_change_index = 0
            changes_found = 0
            for i in range(len(record)):
                if record[i] > record[last_change_index]:
                    changes_found += 1
                    if changes_found == smoothing_level:
                        changes_found = 0
                        val_diff = record[i] - record[last_change_index]
                        step_size = float(val_diff) / (i - last_change_index)
                        for j in range(1, i - last_change_index):
                            record[last_change_index + j] = record[last_change_index] + j * step_size
                        last_change_index = i

    return record_list
    # return [
    #     [1.5, 2, 3.5, 6, 10.25, 17.5, 29.875, 51, 87.0625, 148.625, 253.71875, 433.125, 739.390625, 1262.21875],
    #     [1.5, 3.4, 0.98, 3.982, 0.7778, 4.70062, 0.463298, 5.5944142, -0.00348382, 6.713645422, -0.675545126, 8.123929019, -1.623047053, 9.911019528],
    #     [-1.1, -2, -5.3, -11.25, -28.415, -62.5825, -153.47575, -345.842625, -832.8670375, -1903.556806, -4533.106664, -10451.8476, -24718.28246, -57302.0859]
    # ]


def split_list(samples, test_percentage=Consts.TEST_SIZE_FACTOR):
    test_samples_count = int(len(samples) * test_percentage)
    return samples[:-test_samples_count], samples[-test_samples_count:]


def pad_samples(original_samples):
    # find out what is the longest sample length
    max_length = 0
    for sample in original_samples:
        if len(sample) > max_length:
            max_length = len(sample)

    # create np array with each record in length of the max length
    zero_padded_array = np.zeros((len(original_samples), max_length))

    # store original samples in the padded array
    for i in range(len(original_samples)):
        sample_length = len(original_samples[i])
        padded_array_start_index = max_length - sample_length
        for j in range(sample_length):
            zero_padded_array[i][padded_array_start_index + j] = original_samples[i][j]

    # return padded array
    return zero_padded_array


def get_r_squared_value(series_a, series_b):
    """
    0 <= value <= 1
    higher is better
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(series_a, series_b)
    return r_value ** 2


def get_mse_value(series_a, series_b):
    """
    0 <= value
    lower is better
    """
    # Mean Squared Error (MSE)
    return mean_squared_error(series_a, series_b)


def get_mape_value(series_a, series_b):
    """
    0 <= value
    lower is better
    """
    # Mean Absolute Percentage Error (MAPE)
    errors = \
        [
            float( abs(series_a[i]-series_b[i]) ) / max(series_a[i], 1)
            for i in range(len(series_a))
        ]
    return float(sum(errors))/len(errors)


def get_all_metrics(series_a, series_b):
    return {
        'r2': get_r_squared_value(series_a, series_b),
        'mse': get_mse_value(series_a, series_b),
        'mape': get_mape_value(series_a, series_b)
    }


def calc_mid_end_rate(data_records):
    MID_INDEX = 9
    END_INDEX = 29
    MIN_LENGTH = END_INDEX + 1
    rates = list()

    for record in data_records:
        if len(record) > MIN_LENGTH:
            rates.append(record[END_INDEX]/record[MID_INDEX])

    # return rate statistics
    rates.sort()
    return {
        'avg': sum(rates) / len(rates),
        'med': rates[int(len(rates)/2)],
        'min': rates[0],
        'max': rates[-1]
    }


def plot_graph_and_prediction(original_series, predictions, prediction_start_index, file_name):
    # clear plot area
    pyplot.clf()
    pyplot.grid(which='both')

    # plot original series
    pyplot.plot(
        list(range(1, len(original_series)+1)),
        original_series,
        '-r'
    )

    # plot prediction
    # pyplot.plot(
    #     list(range(prediction_start_index, prediction_start_index + len(predictions))),
    #     predictions,
    #     '.b'
    # )

    # store plotted graph
    pyplot.savefig(r'output/{file_name}.png'.format(file_name=file_name))


def log_metrics_dict(logger, metrics):
    for metric_name in sorted(metrics.keys(), reverse=True):
        if len(metrics[metric_name]) > 0:
            avg_metric = sum(metrics[metric_name]) / len(metrics[metric_name])
            logger.log('{key} : {avg_value}'.format(key=metric_name, avg_value=avg_metric))


def get_synthetic_sigmoid_ts(L_param, a_param, length, y_t0, add_noise=False, should_plot=False):
    """
    return sigmoid series values.

                       L * y_t
    y_<t+1> =  ------------------------
               y_t + exp(a)*(L - y_t)

    :param add_noise:
    :param should_plot:
    :param L_param:
    :param a_param:
    :param length:
    :param y_t0:
    :return:
    """
    series_values = [np.float64(y_t0)]

    for i in range(length):
        y_t = series_values[-1]
        e_a = np.exp(a_param)
        new_value = (L_param * y_t) / (y_t + e_a * (L_param - y_t))

        if add_noise:
            new_value += (-1.0 + 2 * random.random())

        series_values.append(np.float64(new_value))

    # ## plot
    # clear plot area
    if should_plot:
        pyplot.clf()
        pyplot.grid(which='both')

        # plot series
        pyplot.plot(series_values, '-r')
        pyplot.show()

    return series_values


def get_synthetic_sigmoid(L_param, a_param, c_param, length, add_noise=False, should_plot=False):
    """
    return sigmoid series values.

                   L
    f(x) =  ---------------
            1 + c * exp(ax)

    x = 1, 2, ...

    :param c_param:
    :param add_noise:
    :param should_plot:
    :param L_param:
    :param a_param:
    :param length:
    :param y_t0:
    :return:
    """
    series_values = np.array([], dtype=np.float64)
    noise_amp = 0.05 * L_param

    for i in range(length):
        x_t = i + 1
        e_ax = np.exp(a_param * x_t)
        new_value = L_param / (1 + c_param * e_ax)

        if add_noise:
            new_value += (2 * random.random() * noise_amp - noise_amp)

        series_values = np.append(series_values, new_value)

    # ## plot
    # clear plot area
    if should_plot:
        pyplot.clf()
        pyplot.grid(which='both')

        # plot series
        pyplot.plot(series_values, '-r')
        pyplot.show()

    return series_values


def get_inflection_point_of_sigmoid(series):

    last_slope = series[1] - series[0]
    for current_index in range(2, len(series)):
        current_slope = series[current_index] - series[current_index - 1]
        if current_slope < last_slope:
            return current_index

        last_slope = current_slope

    return None
