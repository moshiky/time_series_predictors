
from utils import parse_csv
from arma_model import ARMAModel


DATASET_FILE_PATH = r'datasets/author_h_index.csv'


def predict_using_online_mode(
        ar_order, ma_order, with_c=True, initial_history_size=5, number_of_predictions_ahead=10,
        lag_size=0, update_model=True, use_sample_data=True):
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
        arma_model = ARMAModel(p=ar_order, q=ma_order, lag_size=lag_size)
        # print('>> ARMA(p={p}, q={q})'.format(p=ar_order, q=ma_order))
        try:
            prediction_error = \
                arma_model.predict(
                    current_sample,
                    initial_history_size,
                    number_of_predictions_ahead,
                    with_c,
                    update_model=update_model,
                    use_sample_data=use_sample_data,
                    should_plot=True,
                    record_id=record_index
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
    #   update model- update model after each prediction
    #   use sample data- use sample data to update history
    predict_using_online_mode(
        ar_order=1,
        ma_order=0,
        with_c=False,
        initial_history_size=10,
        number_of_predictions_ahead=20,
        lag_size=10,
        update_model=False,
        use_sample_data=True
    )
