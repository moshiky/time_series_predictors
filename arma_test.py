
from utils import parse_csv
from arima_model import ARIMAModel


DATASET_FILE_PATH = r'datasets/author_h_index.csv'


def predict_using_online_mode(
        ar_order, ma_order, with_c=True, initial_history_size=5, number_of_predictions_ahead=10,
        lag_size=0, multi_step=True):
    # read series data
    # read input file - returns list of lists
    data_records = parse_csv(DATASET_FILE_PATH, smoothing_level=1)

    # define min error storage
    model_errors = list()

    # make predictions for each record
    for record_index in range(len(data_records)):
        # print('=== record #{record_index} ==='.format(record_index=record_index))
        current_sample = data_records[record_index]

        # run ARMA model
        arma_model = ARIMAModel(p=ar_order, q=ma_order, lag_size=lag_size)
        # print('>> ARMA(p={p}, q={q})'.format(p=ar_order, q=ma_order))
        try:
            prediction_error = \
                arma_model.predict(
                    current_sample,
                    initial_history_size,
                    number_of_predictions_ahead,
                    record_index,
                    should_plot=True,
                    add_c=with_c,
                    multi_step=multi_step
                )
            # print('Test MSE: %.3f' % prediction_error)
            print('+ record #{record_index}: ARMA(p={p}, q={q}) error= %.3f'
                  .format(record_index=record_index, p=ar_order, q=ma_order) % prediction_error
                  )
            model_errors.append(prediction_error)

        except Exception as ex:
            print(ex)
            continue

    # print min error info
    print('# min error: ', min(model_errors))
    print('# avg error: ', sum(model_errors)/len(model_errors))


if __name__ == '__main__':
    # online mode params:
    #   lag size: if zero- use  all available each time
    #   number of predictions- at least one
    #   multi step- feed real value each time or predict all at once using forecast value
    predict_using_online_mode(
        ar_order=1,
        ma_order=0,
        with_c=True,
        initial_history_size=10,
        number_of_predictions_ahead=15,
        lag_size=10,
        multi_step=False
    )
