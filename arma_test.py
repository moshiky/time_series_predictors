
import numpy as np
import utils
from arma_model import ARMAModel
from sigmoid_curve import SigmoidCurve
from logger import Logger
import time


DATASET_FILE_PATH = r'datasets/author_h_index.csv'
LAG_SIZE = 13
INITIAL_HISTORY_SIZE = 15
NUMBER_OF_PREDICTIONS_AHEAD = 10
LOGGING_INTERVAL = 1000
SHOULD_PLOT = False
IS_ONLINE = False
START_PARAMS = np.array([3.81666014, 0.9825708])


def predict_using_online_mode(
        logger, ar_order, ma_order, with_c=True, initial_history_size=5, number_of_predictions_ahead=10, lag_size=0):
    # read series data
    # read input file - returns list of lists
    data_records = utils.parse_csv(DATASET_FILE_PATH, smoothing_level=1, should_shuffle=False)
    logger.log('records loaded: {num_records}'.format(num_records=len(data_records)))

    # define min error storage
    model_error_metrics = dict()
    valid_samples_counter = 0

    # make predictions for each record
    logger.log('** ARMA settings: p={p}, q={q}'.format(p=ar_order, q=ma_order))
    start_time = time.time()
    params = list()
    for record_index in range(len(data_records)):
        if (record_index % LOGGING_INTERVAL) == 0:
            logger.log('-- record #{record_index}'.format(record_index=record_index))

        # test sample size and split to train and test sets
        current_sample = data_records[record_index]
        if len(current_sample) < initial_history_size + number_of_predictions_ahead:
            # logger.log('Not enough info in record. record size={record_size}'.format(record_size=len(current_sample)))
            continue

        train_set, test_set = \
            current_sample[:initial_history_size], \
            current_sample[initial_history_size:initial_history_size+number_of_predictions_ahead]

        # run ARMA model
        arma_model = ARMAModel(logger, p=ar_order, q=ma_order, with_c=with_c, lag_size=lag_size)

        predictions = '<not initialized>'
        try:
            arma_model.learn_model_params(train_set, start_params=START_PARAMS)

            if not IS_ONLINE:
                predictions = arma_model.predict_using_learned_params(train_set, number_of_predictions_ahead)

            else:
                predictions = list()
                for i in range(number_of_predictions_ahead):
                    # predict next value
                    predicted_value = arma_model.predict_using_learned_params(train_set, 1)

                    # store prediction
                    predictions.append(predicted_value[0])

                    # update model with test value
                    arma_model.update_model([test_set[i]])

            error_metrics = utils.get_all_metrics(test_set, predictions)

        except Exception as ex:
            if 'Not enough info' not in str(ex):
                logger.log(ex)
                # logger.log('series: {ser}'.format(ser=current_sample))
                # logger.log('predictions: {preds}'.format(preds=predictions))
            continue

        for metric_name in error_metrics.keys():
            if metric_name not in model_error_metrics.keys():
                model_error_metrics[metric_name] = list()
            model_error_metrics[metric_name].append(error_metrics[metric_name])
        valid_samples_counter += 1

    logger.log('total valid predictions: {valid_predictions}'.format(valid_predictions=valid_samples_counter))
    logger.log('total time: {total_secs} secs'.format(total_secs=time.time()-start_time))

    return model_error_metrics


def run_ar(logger):
    return run_arma(logger, (1, 0))


def run_ma(logger):
    return run_arma(logger, (0, 1))


def run_arma(logger, order=None):
    if order is None:
        order = (1, 1)

    return predict_using_online_mode(
        logger,
        ar_order=order[0],
        ma_order=order[1],
        with_c=True,
        initial_history_size=INITIAL_HISTORY_SIZE,
        number_of_predictions_ahead=NUMBER_OF_PREDICTIONS_AHEAD,
        lag_size=LAG_SIZE
    )


def main(logger):
    ar_metrics = run_ar(logger)
    # ma_values = run_ma(logger)
    # arma_values = run_arma(logger)

    # print values
    logger.log('-- avg. performance:')
    utils.log_metrics_dict(logger, ar_metrics)
    # utils.log_metrics_dict(logger, ma_values)

    # logger.log('AR avg. performance: {ar_values}'.format(ar_values=ar_values))
    # logger.log('MA avg. performance: {ma_values}'.format(ma_values=ma_values))
    # logger.log('ARMA avg. performance: {arma_values}'.format(arma_values=arma_values))


def calc_rate(logger):
    logger.log('load records..')
    data_records = utils.parse_csv(DATASET_FILE_PATH, smoothing_level=1, should_shuffle=False)
    logger.log('calculate rates..')
    logger.log(utils.calc_mid_end_rate(data_records))
    logger.log('done.')


def sigmoid_test(logger, is_online, should_plot, lag_size):
    logger.log('load records..')
    data_records = utils.parse_csv(DATASET_FILE_PATH, smoothing_level=1, should_shuffle=False)

    logger.log('get predictions')
    model_error_metrics = dict()
    for record_id in range(len(data_records)):
        record = data_records[record_id]

        if len(record) < INITIAL_HISTORY_SIZE + NUMBER_OF_PREDICTIONS_AHEAD:
            continue

        else:

            if (record_id % LOGGING_INTERVAL) == 0:
                logger.log('* record #{record_id}'.format(record_id=record_id))

            # split to train and test sets
            train_set = record[:INITIAL_HISTORY_SIZE]
            test_set = record[INITIAL_HISTORY_SIZE:INITIAL_HISTORY_SIZE+NUMBER_OF_PREDICTIONS_AHEAD]

            # fit model and calculate predictions
            sigmoid_predictions = list()
            if is_online:
                mid_max_rate = SigmoidCurve.MID_MAX_RATE
                for i in range(NUMBER_OF_PREDICTIONS_AHEAD):
                    tmp_history = train_set + test_set[:i]
                    next_prediction = \
                        SigmoidCurve.fit_and_predict_recursive(
                            tmp_history[-lag_size:], 1, mid_max_rate=mid_max_rate
                        )[0]
                    sigmoid_predictions.append(next_prediction)
                    mid_max_rate *= 0.85
                    mid_max_rate = max(1.7, mid_max_rate)

            else:
                sigmoid_predictions = \
                    SigmoidCurve.fit_and_predict_recursive(
                        train_set[-lag_size:], NUMBER_OF_PREDICTIONS_AHEAD
                    )

            # plot predictions
            if should_plot:
                utils.plot_graph_and_prediction(
                    train_set + test_set,
                    sigmoid_predictions,
                    INITIAL_HISTORY_SIZE+1,
                    'sigmoid__{record_id}'.format(record_id=record_id)
                )

            error_metrics = utils.get_all_metrics(test_set, sigmoid_predictions)
            for metric_name in error_metrics.keys():
                if metric_name not in model_error_metrics.keys():
                    model_error_metrics[metric_name] = list()
                model_error_metrics[metric_name].append(error_metrics[metric_name])

    # log metrics
    logger.log('-- avg. performance:')
    utils.log_metrics_dict(logger, model_error_metrics)


def test_gradient_descent(logger):
    add_noise = False
    should_plot = True
    series_length = 100
    epochs = 3

    # define test params- [L, a, c, x_range, y_range]
    params = [
        [80.7, -10, 70, [0, 1], [0, 80.7]],
        # [10, -0.1, 40],
        # [20, -0.02, 90],
        # [5, -0.003, 10]
    ]

    # ## direct version
    for series_index in range(len(params)):
        logger.log('## series #{ser_index}'.format(ser_index=series_index))

        logger.log('# create series..')
        series = \
            utils.get_synthetic_sigmoid(
                l_param=params[series_index][0],
                a_param=params[series_index][1],
                c_param=params[series_index][2],
                x_range=params[series_index][3],
                y_range=params[series_index][4],
                length=series_length, add_noise=add_noise, should_plot=should_plot
            )

        # find inflection point of the sigmoid
        inflection_point = utils.get_inflection_point_of_sigmoid(series)
        logger.log('inflection point: {inf_point}'.format(inf_point=inflection_point))

        # calculate sigmoid params using gradient descent
        logger.log('# fit sigmoid params..')
        l, a, c = \
            SigmoidCurve.fit_and_predict_gd_online(
                logger, series, params[series_index][3], series_length*0.5, inflection_point,
                is_stochastic=True, epochs=epochs
            )

        logger.log('original params: l: {l_param}, a: {a_param}, c: {c_param}'.format(
            l_param=params[series_index][0], a_param=params[series_index][1], c_param=params[series_index][2])
        )
        logger.log('learned params: l: {l_param}, a: {a_param}, c: {c_param}'.format(l_param=l, a_param=a, c_param=c))

        # predict next values
        # <offline>
        y_predictions_for_x = SigmoidCurve.get_sigmoid_predictions_for_values(l, a, c, )

        # <online>


if __name__ == '__main__':
    main_logger = Logger()
    main(main_logger)
    # calc_rate(main_logger)
    # sigmoid_test(main_logger, is_online=IS_ONLINE, should_plot=SHOULD_PLOT, lag_size=LAG_SIZE)
    # test_gradient_descent(main_logger)
