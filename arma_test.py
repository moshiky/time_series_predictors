
import utils
from arma_model import ARMAModel
from sigmoid_curve import SigmoidCurve
from logger import Logger


DATASET_FILE_PATH = r'datasets/author_h_index.csv'
LAG_SIZE = 6
INITIAL_HISTORY_SIZE = 10
NUMBER_OF_PREDICTIONS_AHEAD = 15
LOGGING_INTERVAL = 100
SHOULD_PLOT = False
IS_ONLINE = True


def predict_using_online_mode(
        logger, ar_order, ma_order, with_c=True, initial_history_size=5, number_of_predictions_ahead=10,
        lag_size=0, update_model=True, use_sample_data=True):
    # read series data
    # read input file - returns list of lists
    data_records = utils.parse_csv(DATASET_FILE_PATH, smoothing_level=1, should_shuffle=False)
    logger.log('records loaded: {num_records}'.format(num_records=len(data_records)))

    # define min error storage
    model_error_metrics = dict()

    # make predictions for each record
    logger.log('** ARMA settings: p={p}, q={q}'.format(p=ar_order, q=ma_order))
    for record_index in range(len(data_records)):
        if (record_index % LOGGING_INTERVAL) == 0:
            logger.log('-- record #{record_index}'.format(record_index=record_index))

        current_sample = data_records[record_index]

        # run ARMA model
        arma_model = ARMAModel(logger, p=ar_order, q=ma_order, lag_size=lag_size)

        try:
            error_metrics = \
                arma_model.predict(
                    current_sample,
                    initial_history_size,
                    number_of_predictions_ahead,
                    with_c,
                    update_model=update_model,
                    use_sample_data=use_sample_data,
                    should_plot=SHOULD_PLOT,
                    record_id=record_index
                )

        except Exception as ex:
            logger.log(ex, should_print=False)
            continue

        for metric_name in error_metrics.keys():
            if metric_name not in model_error_metrics.keys():
                model_error_metrics[metric_name] = list()
            model_error_metrics[metric_name].append(error_metrics[metric_name])

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
        with_c=False,
        initial_history_size=INITIAL_HISTORY_SIZE,
        number_of_predictions_ahead=NUMBER_OF_PREDICTIONS_AHEAD,
        lag_size=LAG_SIZE,
        update_model=IS_ONLINE,
        use_sample_data=IS_ONLINE
    )


def main(logger):
    ar_metrics = run_ar(logger)
    # ma_values = run_ma(logger)
    # arma_values = run_arma(logger)

    # print values
    logger.log('-- avg. performance:')
    utils.log_metrics_dict(logger, ar_metrics)

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
                        SigmoidCurve.fit_and_predict(
                            tmp_history[-lag_size:], 1, mid_max_rate=mid_max_rate
                        )[0]
                    sigmoid_predictions.append(next_prediction)
                    mid_max_rate *= 0.85
                    mid_max_rate = max(1.7, mid_max_rate)

            else:
                sigmoid_predictions = \
                    SigmoidCurve.fit_and_predict(
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


if __name__ == '__main__':
    main_logger = Logger()
    main(main_logger)
    # calc_rate(main_logger)
    # sigmoid_test(main_logger, is_online=IS_ONLINE, should_plot=SHOULD_PLOT, lag_size=LAG_SIZE)

