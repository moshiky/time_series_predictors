
import sys
import os
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from utils import parse_csv, split_samples
from online_auto_regression_handler import OnlineAutoRegressionHandler
from offline_auto_regression_handler import OfflineAutoRegressionHandler
from moving_average_handler import MovingAverageHandler
from weighted_moving_average_handler import WeightedMovingAverageHandler
from logger import Logger
from consts import Consts


def calculate_mean_error(test_samples, predictions):
    # Mean Squared Error (MSE)
    # mean_error = mean_squared_error(test_samples, predictions)

    # Mean Absolute Percentage Error (MAPE)
    errors = [float(abs(test_samples[i]-predictions[i]))/max(test_samples[i], 1) for i in range(len(test_samples))]
    mean_error = float(sum(errors))/len(errors)

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


def draw_prediction_graph(predictions_of_each_predictor, record_index, train_samples, test_samples):
    # figure name
    figure_name = 'record_{record_index}.png'.format(record_index=record_index)
    figure_path = os.path.join(r'output', figure_name)

    # predictor colors
    graph_colors = {
        'OfflineAutoRegressionHandler': 'blue',
        # 'OnlineAutoRegressionHandler': 'orange',
        'MovingAverageHandler': 'green',
        'WeightedMovingAverageHandler': 'pink'
    }

    # draw original series
    original_data = train_samples + test_samples
    plt.clf()
    plt.plot(list(range(len(original_data))), original_data, color='red')

    for predictor_name in predictions_of_each_predictor.keys():
        plt.plot(
            list(range(len(train_samples), len(original_data))),
            predictions_of_each_predictor[predictor_name],
            color=graph_colors[predictor_name]
        )

    plt.savefig(fname=figure_path)
    plt.clf()


def main(file_path, logger_p):

    # read input file - returns list of lists
    data_records = parse_csv(file_path)
    logger_p.log('{num_records} records loaded'.format(num_records=len(data_records)))

    # set predictors list
    predictor_classes = [
        OfflineAutoRegressionHandler,
        # OnlineAutoRegressionHandler,
        MovingAverageHandler,
        WeightedMovingAverageHandler,
    ]

    # initiate mean error lists
    mean_error_log = dict()
    for predictor_class in predictor_classes:
        mean_error_log[predictor_class.__name__] = list()

    record_index = 0
    for record in data_records:
        record_index += 1
        if record_index % Consts.MEAN_ERROR_LOG_INTERVAL == 0:
            logger_p.log('-- record #{record_index} --'.format(record_index=record_index))

        # split to train and test sets
        train_samples, test_samples = split_samples(record)
        predictions_of_each_predictor = dict()

        # calculate mean error for each info record
        for predictor_class in predictor_classes:
            predictor_class_name = predictor_class.__name__
            try:
                record_mean_error, predictions = \
                    calculate_average_me_with_predictor(logger_p, predictor_class, train_samples, test_samples)
            except Exception as ex:
                logger_p.error('problematic record: ' + str(record))
                logger_p.error('predictor: {predictor_name}'.format(predictor_name=predictor_class_name))
                logger_p.error('error: {ex}'.format(ex=ex))
                continue

            mean_error_log[predictor_class_name].append(record_mean_error)

            # collect prediction data
            predictions_of_each_predictor[predictor_class_name] = predictions

            if record_index % Consts.MEAN_ERROR_LOG_INTERVAL == 0:
                logger_p.log('{predictor_class_name} >> current avg. error = {avg_error}'.format(
                        predictor_class_name=predictor_class_name,
                        avg_error=float(sum(mean_error_log[predictor_class_name])) /
                                  len(mean_error_log[predictor_class_name])
                    )
                )

        # draw predictions vs. original data
        draw_prediction_graph(predictions_of_each_predictor, record_index, train_samples, test_samples)

    # print final stats
    logger_p.log('== final stats ==')
    for predictor_class in predictor_classes:
        predictor_class_name = predictor_class.__name__
        logger_p.log('{predictor_class_name} >> final avg. error = {avg_error}'.format(
                predictor_class_name=predictor_class_name,
                avg_error=float(sum(mean_error_log[predictor_class_name])) /
                            max(len(mean_error_log[predictor_class_name]), 1)
            )
        )


if __name__ == '__main__':
    logger = Logger()
    # main(r'C:\git\time_series_predictors\datasets\cts.csv', logger)
    main(r'C:\git\time_series_predictors\datasets\posts_per_course_week__for_prediction.csv', logger)
