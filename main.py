
import sys
import os
import time
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from utils import parse_csv, split_list
from online_auto_regression_handler import OnlineAutoRegressionHandler
from offline_auto_regression_handler import OfflineAutoRegressionHandler
from moving_average_handler import MovingAverageHandler
from weighted_moving_average_handler import WeightedMovingAverageHandler
from logger import Logger
from consts import Consts
import utils


def calculate_mean_error(test_samples, predictions):
    # Mean Squared Error (MSE)
    # mean_error = mean_squared_error(test_samples, predictions)

    # # Mean Absolute Percentage Error (MAPE)
    # errors = [float(abs(test_samples[i]-predictions[i]))/max(test_samples[i], 1) for i in range(len(test_samples))]
    # mean_error = float(sum(errors))/len(errors)

    # R^2 - Coefficient of determination
    slope, intercept, r_value, p_value, std_err = stats.linregress(test_samples, predictions)
    mean_error = r_value ** 2

    return mean_error


def calculate_average_me_with_predictor(logger_p, predictor_class, train_samples, test_samples):

    # create predictor
    predictor = predictor_class(logger_p, train_samples)

    # predict next values
    predictions = list()
    for i in range(len(test_samples)):
        predicted_value = predictor.predict_next()
        predictions.append(predicted_value)
        predictor.update_predictor([test_samples[i]])

    # calculate mean error
    mean_error = calculate_mean_error(test_samples, predictions)
    return mean_error, predictions


def draw_prediction_graph(original_data, predicted_values, predictor_name, record_index):
    # build figure path
    figure_name = '{predictor_name}/record_{record_index}.png'.format(
        predictor_name=predictor_name, record_index=str(record_index).rjust(Consts.JUSTIFICATION_LENGTH, '0')
    )
    figure_path = os.path.join(r'output', figure_name)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.mkdir(os.path.dirname(figure_path))

    # draw original series
    plt.clf()
    plt.plot(list(range(len(original_data))), original_data, color='red')

    # draw prediction
    start_index = len(original_data) - len(predicted_values)
    plt.plot(list(range(start_index, len(original_data))), predicted_values, color='blue')

    # store figure
    plt.savefig(fname=figure_path)
    plt.clf()


def run_offline_predictors(logger_p, records, predictor_class_list):
    # prompt start
    logger_p.log('-- running offline predictors')

    # init error storage
    predictor_errors = dict()

    # predict with each predictor
    for predictor_class_info in predictor_class_list:

        # create instance
        predictor_class = predictor_class_info[0]
        window_size = predictor_class_info[1]
        predictor_class_name = predictor_class.__name__

        # prompt predictor start
        logger_p.log('- running predictor: {predictor_name}'.format(predictor_name=predictor_class_name))

        record_id = 0
        valid_results = 0
        start_time = time.time()
        for record in records:

            # log record index
            if record_id % Consts.RECORD_LOG_INTERVAL == 0:
                logger_p.log('record #{record_index}'.format(record_index=record_id))

            record_id += 1

            if len(record) < Consts.NUMBER_OF_INITIAL_VALUES_FOR_ONLINE + Consts.YEARS_AHEAD:
                # logger_p.log('Warning: series too short. '
                #              'length: {series_length} class: {class_name} window size: {window_size}'.format(
                #                 series_length=len(record), class_name=predictor_class_name, window_size=window_size))
                continue

            # split to test and train sets
            train_set, test_set = \
                record[:Consts.NUMBER_OF_INITIAL_VALUES_FOR_ONLINE], \
                record[Consts.NUMBER_OF_INITIAL_VALUES_FOR_ONLINE:]

            # create predictor instance
            predictor = predictor_class(logger_p, window_size)
            predictor.learn_model_params(train_set)

            # predict values
            predicted_values = predictor.predict_using_learned_params(train_set, Consts.YEARS_AHEAD)

            # calculate error
            try:
                error_metrics = utils.get_all_metrics(test_set[:Consts.YEARS_AHEAD], predicted_values)
            except Exception as ex:
                if 'bad prediction' in str(ex):
                    continue
                else:
                    raise ex

            # store metrics
            for metric_name in error_metrics.keys():
                if metric_name not in predictor_errors.keys():
                    predictor_errors[metric_name] = list()
                predictor_errors[metric_name].append(error_metrics[metric_name])
            valid_results += 1

        logger_p.log('valid rows: {valid}'.format(valid=valid_results))
        logger_p.log('total time: {time_secs} secs'.format(time_secs=time.time()-start_time))

        # init model params with train set
        # predictor.learn_model_params(train_records)
        # avg_params = np.zeros(window_size + 1)
        # record_count = 0.0
        # for record in train_records:
        #     record_params = MovingAverageHandler.test_gd_one_param(record, window_size)
        #     avg_params = (avg_params * record_count + record_params) / (record_count+1)
        #     record_count += 1
        #
        # # predict
        # avg_arror = 0.0
        # for record in test_records:
        #     record_params = MovingAverageHandler.test_gd_one_param(record, window_size)
        #     avg_params = (avg_params * record_count + record_params) / (record_count + 1)
        #     record_count += 1

        # # train
        # model_list = list()
        # for i in range(len(train_records[:400])):
        #     # create predictor instance
        #     predictor = predictor_class(logger_p, window_size)
        #
        #     # init model params with train set
        #     predictor.learn_model_params(train_records[i])
        #
        #     # store model
        #     model_list.append(predictor)
        #
        #     # print progress
        #     if (i % 100) == 0:
        #         print('record:', i)
        #
        # # calculate average params
        # avg_params = [0.0] * window_size
        # number_of_models = len(model_list)
        # for model in model_list:
        #     model_params = model.get_model_params()
        #     for i in range(window_size):
        #         avg_params[i] += model_params[i]
        # avg_params = [float(x)/number_of_models for x in avg_params]
        #
        # # select closest model to average
        # min_error = None
        # best_model = None
        # for model in model_list:
        #     model_params = model.get_model_params()
        #     error = sum([(avg_params[i]-model_params[i])**2 for i in range(window_size)])
        #     if min_error is None or error < min_error:
        #         min_error = error
        #         best_model = model

        # # predict test set
        # record_count = 0
        # for record in test_records:
        #
        #
        #     # predict values
        #     # predicted_values = predictor.predict_using_learned_params(record[:window_size], len(record)-window_size)
        #
        #
        #     # store graph
        #     # draw_prediction_graph(record, predicted_values, predictor_class_name, record_index)
        #
        #     # store error
        #     predictor_errors[predictor_class_name].append(
        #         calculate_mean_error(record[-len(predicted_values):], predicted_values)
        #     )
        #     # predictor_errors_gd[predictor_class_name].append(
        #     #     calculate_mean_error(record[-len(predicted_values_gd):], predicted_values_gd)
        #     # )
        #
        #     # increase record index
        #     record_count += 1

    # calculate mean error for each predictor
    # predictor_mean_error = {
    #     predictor_name: float(sum(predictor_errors[predictor_name])) / len(predictor_errors[predictor_name])
    #     for predictor_name in predictor_errors.keys()
    # }
    # predictor_mean_error_gd = {
    #     predictor_name: float(sum(predictor_errors_gd[predictor_name])) / len(predictor_errors_gd[predictor_name])
    #     for predictor_name in predictor_errors_gd.keys()
    # }

    # print('normal: ', predictor_mean_error)
    # print('gd: ', predictor_mean_error_gd)

    # return mean error log
    return predictor_errors


def run_online_predictor(logger_p, records, predictor):
    # prompt start
    logger_p.log('-- running online predictor')

    # init error storage
    predictor_errors = dict()
    predictor_class_name = predictor.__class__.__name__

    # prompt predictor start
    logger_p.log('- running predictor: {predictor_name}'.format(predictor_name=predictor_class_name))

    # predict test set
    record_index = 0
    valid_results = 0
    start_time = time.time()
    for record in records:
        # increase record index
        record_index += 1

        # split to train and test samples
        train_set = record[:Consts.NUMBER_OF_INITIAL_VALUES_FOR_ONLINE]
        test_set = record[Consts.NUMBER_OF_INITIAL_VALUES_FOR_ONLINE:]

        if len(test_set) < Consts.YEARS_AHEAD:
            # logger_p.log('Warning: series too short. '
            #              'length: {series_length} class: {class_name} window size: {window_size}'.format(
            #                 series_length=len(record), class_name=predictor_class_name, window_size=window_size))
            continue

        # log record index
        if record_index % Consts.RECORD_LOG_INTERVAL == 0:
            logger_p.log('record #{record_index}'.format(record_index=record_index))

        # init predictor with test set
        predictor.init(train_set)

        # predict values
        predicted_values = list()
        predicted_next_value = predictor.predict_next()
        for sample_index in range(Consts.YEARS_AHEAD):
            series_next_value = test_set[sample_index]
            predicted_values.append(predicted_next_value)
            predictor.update_predictor(series_next_value)
            predicted_next_value = predictor.predict_next()

        # store graph
        # draw_prediction_graph(record, predicted_values, predictor_class_name, record_index)

        # calculate error
        try:
            error_metrics = utils.get_all_metrics(test_set[:Consts.YEARS_AHEAD], predicted_values)
        except Exception as ex:
            if 'bad prediction' in str(ex):
                continue
            else:
                raise ex

        # store metrics
        for metric_name in error_metrics.keys():
            if metric_name not in predictor_errors.keys():
                predictor_errors[metric_name] = list()
            predictor_errors[metric_name].append(error_metrics[metric_name])
        valid_results += 1

    logger_p.log('valid rows: {valid}'.format(valid=valid_results))
    logger_p.log('total time: {time_secs} secs'.format(time_secs=time.time() - start_time))

    # return mean error log
    return predictor_errors


def main(file_path, logger_p):

    # read input file - returns list of lists
    data_records = parse_csv(file_path, smoothing_level=0, should_shuffle=False)
    logger_p.log('{num_records} records loaded'.format(num_records=len(data_records)))

    # # run offline predictors
    # offline_predictor_errors = \
    #     run_offline_predictors(
    #         logger_p,
    #         data_records,
    #         [
    #             (OfflineAutoRegressionHandler, 1),
    #             # (MovingAverageHandler, 1),
    #         ]
    #     )

    # run offline predictors
    predictor = OnlineAutoRegressionHandler(logger_p, p=1, lag_size=13)
    online_predictor_errors = \
        run_online_predictor(logger_p, data_records, predictor=predictor)

    # log results
    # utils.log_metrics_dict(logger_p, offline_predictor_errors)
    utils.log_metrics_dict(logger_p, online_predictor_errors)

    # logger_p.log(online_predictor_errors)


if __name__ == '__main__':
    logger = Logger()
    main(r'datasets/author_h_index.csv', logger)
