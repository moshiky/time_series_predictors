
import sys
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
    errors = [float(abs(test_samples[i]-predictions[i]))/test_samples[i] for i in range(len(test_samples))]
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
    return mean_error


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

        # calculate mean error for each info record
        for predictor_class in predictor_classes:
            try:
                record_mean_error = calculate_average_me_with_predictor(logger_p, predictor_class, train_samples, test_samples)
            except Exception as ex:
                logger_p.error('problematic record: ' + str(record))
                continue

            predictor_class_name = predictor_class.__name__
            mean_error_log[predictor_class_name].append(record_mean_error)

            if record_index % Consts.MEAN_ERROR_LOG_INTERVAL == 0:
                logger_p.log('{predictor_class_name} >> current avg. error = {avg_error}'.format(
                        predictor_class_name=predictor_class_name,
                        avg_error=float(sum(mean_error_log[predictor_class_name])) /
                                  len(mean_error_log[predictor_class_name])
                    )
                )

    # print final stats
    logger_p.log('== final stats ==')
    for predictor_class in predictor_classes:
        predictor_class_name = predictor_class.__name__
        logger_p.log('{predictor_class_name} >> final avg. error = {avg_error}'.format(
                predictor_class_name=predictor_class_name,
                avg_error=float(sum(mean_error_log[predictor_class_name])) /
                          len(mean_error_log[predictor_class_name])
            )
        )


if __name__ == '__main__':
    logger = Logger()
    main(r'C:\git\predictors\datasets\cts.csv', logger)
