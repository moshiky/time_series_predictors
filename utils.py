
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
    # todo: filter all values the same cases
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
            abs(float(series_a[i]-series_b[i]) / series_a[i])
            for i in range(len(series_a)) if series_a[i] != 0
        ]
    return float(sum(errors))/len(errors)


def pop_bad_predictions(series_a, series_b):
    series_a_filtered = list()
    series_b_filtered = list()
    series_a_bad = list()
    series_b_bad = list()
    for elem_id in range(min(len(series_a), len(series_b))):
        relative_error = abs((series_a[elem_id]-series_b[elem_id])/series_a[elem_id])
        if relative_error > Consts.BAD_PREDICTION_RATE:
            series_a_bad.append(series_a[elem_id])
            series_b_bad.append(series_b[elem_id])
        else:
            series_a_filtered.append(series_a[elem_id])
            series_b_filtered.append(series_b[elem_id])

    return series_a_filtered, series_b_filtered, series_a_bad, series_b_bad


def get_all_metrics(series_a, series_b):
    r2_value = get_r_squared_value(series_a, series_b)
    mse_value = get_mse_value(series_a, series_b)
    mape_value = get_mape_value(series_a, series_b)
    if r2_value < 0.1 or mse_value > 1000 or mape_value > 100:
        raise Exception('bad prediction')

    return {
        'r2': r2_value,
        'mse': mse_value,
        'mape': mape_value
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


def log_metrics_dict(logger, metrics):
    for metric_name in sorted(metrics.keys(), reverse=True):
        if type(metrics[metric_name]) == list:
            m_sum = sum(metrics[metric_name])
            m_len = len(metrics[metric_name])
            logger.log('{key} : {metric_value}'.format(key=metric_name, metric_value=m_sum/m_len))

            pyplot.hist(metrics[metric_name], bins=100, log=True)
            pyplot.show()

        else:
            logger.log('{key} : {metric_value}'.format(key=metric_name, metric_value=metrics[metric_name]))


def get_synthetic_sigmoid_ts(l_param, a_param, length, y_t0, add_noise=False, should_plot=False):
    """
    return sigmoid series values.

                       L * y_t
    y_<t+1> =  ------------------------
               y_t + exp(a)*(L - y_t)

    :param add_noise:
    :param should_plot:
    :param l_param:
    :param a_param:
    :param length:
    :param y_t0:
    :return:
    """
    series_values = [np.float64(y_t0)]

    for i in range(length):
        y_t = series_values[-1]
        e_a = np.exp(a_param)
        new_value = (l_param * y_t) / (y_t + e_a * (l_param - y_t))

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


def get_synthetic_sigmoid(l_param, a_param, c_param, length, add_noise=False, should_plot=False,
                          x_range=None, y_range=None, train_part_factor=0.6):
    """
    return sigmoid series values.

                   L
    f(x) =  ---------------
            1 + c * exp(ax)

    x = 1, 2, ...

    :param train_part_factor:
    :param y_range:
    :param x_range:
    :param c_param:
    :param add_noise:
    :param should_plot:
    :param l_param:
    :param a_param:
    :param length:
    :param y_t0:
    :return:
    """

    # initiate default x and y ranges
    if x_range is None:
        x_range = [0.0, 1.0]

    if y_range is None:
        y_range = [0.0, 1.0]

    series_values = np.array([], dtype=np.float64)
    noise_amp = 0.05 * l_param
    x_step_size = (x_range[1]-x_range[0]) / length
    y_range_size = y_range[1] - y_range[0]
    y_for_x_train = dict()
    y_for_x_test = dict()

    for i in range(length):
        x_t = x_range[0] + i*x_step_size
        e_ax = np.exp(a_param * x_t)
        new_value = l_param / (1 + c_param * e_ax)

        if add_noise:
            new_value += (2 * random.random() * noise_amp - noise_amp)

        normalized_y_value = ((new_value / l_param) * y_range_size) + y_range[0]
        series_values = np.append(series_values, normalized_y_value)

        # store in y_for_x_train form
        if i < train_part_factor * length:
            y_for_x_train[x_t] = normalized_y_value
        else:
            y_for_x_test[x_t] = normalized_y_value

    # ## plot
    # clear plot area
    if should_plot:
        pyplot.clf()
        pyplot.grid(which='both')

        # plot series
        pyplot.plot([x_range[0]+i*x_step_size for i in range(length)], series_values, '-r')
        pyplot.show()

    return y_for_x_train, y_for_x_test


def get_inflection_point_of_sigmoid(series):

    last_slope = series[1] - series[0]
    for current_index in range(2, len(series)):
        current_slope = series[current_index] - series[current_index - 1]
        if current_slope < last_slope:
            return current_index

        last_slope = current_slope

    return None


def load_dataset_for_gd_fitting(file_path, series_length, test_size):
    # load csv
    dataset_records = parse_csv(file_path, should_shuffle=True, smoothing_level=1)

    # split to train and test groups - each is list of y_for_x dicts
    dataset_splitted = list()
    for series in dataset_records:
        if len(series) < series_length:
            continue

        # split series
        train_set_size = series_length - test_size

        # convert series to y_for_x format
        train_y_for_x = {
            i + 1: series[i] for i in range(train_set_size)
        }
        test_y_for_x = {
            i + 1: series[i] for i in range(train_set_size, series_length)
        }

        # store splitted dicts
        dataset_splitted.append((train_y_for_x, test_y_for_x))

    # return found sets
    return dataset_splitted
