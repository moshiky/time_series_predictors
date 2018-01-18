
import sys
import os
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from utils import parse_csv, split_list
from online_auto_regression_handler import OnlineAutoRegressionHandler
from offline_auto_regression_handler import OfflineAutoRegressionHandler
from moving_average_handler import MovingAverageHandler
from weighted_moving_average_handler import WeightedMovingAverageHandler
from logger import Logger
from consts import Consts


def calculate_mean_error(test_samples, predictions):
    # Mean Squared Error (MSE)
    mean_error = mean_squared_error(test_samples, predictions)

    # # Mean Absolute Percentage Error (MAPE)
    # errors = [float(abs(test_samples[i]-predictions[i]))/max(test_samples[i], 1) for i in range(len(test_samples))]
    # mean_error = float(sum(errors))/len(errors)

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
    # split to train and test
    train_records, test_records = split_list(records)

    # init error storage
    predictor_errors = dict()

    # predict with each predictor
    for predictor_class_info in predictor_class_list:
        # create instance
        predictor_class = predictor_class_info[0]
        window_size = predictor_class_info[1]
        predictor_class_name = predictor_class.__name__
        predictor = predictor_class(logger_p, window_size)

        # init error list
        predictor_errors[predictor_class_name] = list()

        # init model params with train set
        predictor.learn_model_params(train_records)

        # predict test set
        record_index = 0
        for record in test_records:
            if len(record) < window_size:
                logger_p.log('Warning: series too short. '
                             'length: {series_length} class: {class_name} window size: {window_size}'.format(
                                series_length=len(record), class_name=predictor_class_name, window_size=window_size))
                continue

            # predict values
            predicted_values = predictor.predict_using_learned_params(record[:window_size], len(record)-window_size)

            # store graph
            draw_prediction_graph(record, predicted_values, predictor_class_name, record_index)

            # store error
            predictor_errors[predictor_class_name].append(
                calculate_mean_error(record[window_size:], predicted_values)
            )

            # increase record index
            record_index += 1

    # calculate mean error for each predictor
    predictor_mean_error = {
        predictor_name: float(sum(predictor_errors[predictor_name])) / len(predictor_errors[predictor_name])
        for predictor_name in predictor_errors.keys()
    }

    # return mean error log
    return predictor_mean_error


def run_online_predictors(logger_p, records, predictor_class_list):
    # init error storage
    predictor_errors = dict()

    # predict with each predictor
    for predictor_class_info in predictor_class_list:
        # create instance
        predictor_class = predictor_class_info[0]
        window_size = predictor_class_info[1]
        predictor_class_name = predictor_class.__name__

        # init error list
        predictor_errors[predictor_class_name] = list()

        # predict test set
        record_index = 0
        for record in records:
            # split to train and test samples
            train_samples = record[:Consts.NUMBER_OF_INITIAL_VALUES_FOR_ONLINE]
            test_samples = record[Consts.NUMBER_OF_INITIAL_VALUES_FOR_ONLINE:]

            if len(train_samples) < window_size:
                logger_p.log('Warning: series too short. '
                             'length: {series_length} class: {class_name} window size: {window_size}'.format(
                                series_length=len(record), class_name=predictor_class_name, window_size=window_size))
                continue

            # create predictor
            predictor = predictor_class(logger_p, train_samples, window_size)

            # predict values
            predicted_values = list()
            next_value = predictor.predict_next()
            for series_value in test_samples:
                predicted_values.append(next_value)
                predictor.update_predictor(series_value)
                next_value = predictor.predict_next()

            # store graph
            draw_prediction_graph(record, predicted_values, predictor_class_name, record_index)

            # store error
            predictor_errors[predictor_class_name].append(
                calculate_mean_error(test_samples, predicted_values)
            )

            # increase record index
            record_index += 1

    # calculate mean error for each predictor
    predictor_mean_error = {
        predictor_name: float(sum(predictor_errors[predictor_name])) / len(predictor_errors[predictor_name])
        for predictor_name in predictor_errors.keys()
    }

    # return mean error log
    return predictor_mean_error


def main(file_path, logger_p):

    # read input file - returns list of lists
    data_records = parse_csv(file_path)
    logger_p.log('{num_records} records loaded'.format(num_records=len(data_records)))

    # run offline predictors
    offline_predictor_errors = \
        run_offline_predictors(
            logger_p,
            data_records,
            [
                (OfflineAutoRegressionHandler, 1),
            ]
        )
    logger_p.log(offline_predictor_errors)

    # run offline predictors
    online_predictor_errors = \
        run_online_predictors(
            logger_p,
            data_records,
            [
                (OnlineAutoRegressionHandler, 1),
            ]
        )
    logger_p.log(online_predictor_errors)

    # # initiate mean error lists
    # mean_error_log = dict()
    # for predictor_class in predictor_classes:
    #     mean_error_log[predictor_class.__name__] = list()
    #
    # record_index = 0
    # for record in data_records:
    #     if record_index % Consts.LOG_INTERVAL == 0:
    #         logger_p.log('-- record #{record_index} --'.format(record_index=record_index))
    #
    #     # split to train and test sets
    #     train_samples, test_samples = split_list(record)
    #     predictions_of_each_predictor = dict()
    #
    #     # calculate mean error for each info record
    #     for predictor_class in predictor_classes:
    #         predictor_class_name = predictor_class.__name__
    #
    #         try:
    #             record_mean_error, predictions = \
    #                 calculate_average_me_with_predictor(logger_p, predictor_class, train_samples, test_samples)
    #         except Exception as ex:
    #             logger_p.error('problematic record: ' + str(record))
    #             logger_p.error('predictor: {predictor_name}'.format(predictor_name=predictor_class_name))
    #             logger_p.error('error: {ex}'.format(ex=ex))
    #             continue
    #
    #         mean_error_log[predictor_class_name].append(record_mean_error)
    #
    #         # collect prediction data
    #         predictions_of_each_predictor[predictor_class_name] = predictions
    #
    #         if record_index % Consts.LOG_INTERVAL == 0:
    #             logger_p.log('{predictor_class_name} >> current avg. error = {avg_error}'.format(
    #                     predictor_class_name=predictor_class_name,
    #                     avg_error=float(sum(mean_error_log[predictor_class_name])) /
    #                               len(mean_error_log[predictor_class_name])
    #                 )
    #             )
    #
    #     # draw predictions vs. original data
    #     draw_prediction_graph(predictions_of_each_predictor, record_index, train_samples, test_samples)
    #
    #     record_index += 1
    #
    # # print final stats
    # logger_p.log('== final stats ==')
    # for predictor_class in predictor_classes:
    #     predictor_class_name = predictor_class.__name__
    #     logger_p.log('{predictor_class_name} >> final avg. error = {avg_error}'.format(
    #             predictor_class_name=predictor_class_name,
    #             avg_error=float(sum(mean_error_log[predictor_class_name])) /
    #                         max(len(mean_error_log[predictor_class_name]), 1)
    #         )
    #     )


if __name__ == '__main__':
    logger = Logger()
    main(r'datasets/ar2.csv', logger)
