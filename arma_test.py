from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from utils import parse_csv


def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


class ARIMA_model:

    def __init__(self, p=0, d=0, q=0):
        # set AR component order
        self.__p = p

        # set MA component order
        self.__q = q

        # set I component order
        self.__d = d

    def predict(self, X, record_id, should_plot=False):
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(self.__p, self.__d, self.__q))
            model_fit = model.fit(disp=0, trend='nc')
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
        if should_plot:
            # plot
            pyplot.clf()
            pyplot.plot(test)
            pyplot.plot(predictions, color='red')
            pyplot.savefig(
                r'output/arima__id_{record_id}_p_{p}_d_{d}_q_{q}.png'
                .format(record_id=record_id, p=self.__p, d=self.__d, q=self.__q)
            )

        return error


def main():
    # read series data
    # read input file - returns list of lists
    file_path = r'datasets/author_h_index.csv'
    data_records = parse_csv(file_path)

    # define min error storage
    record_min_error_params = dict()

    # initiate problematic records storage
    problematic_records = list()

    # make predictions for each record
    for i in range(len(data_records)):
        print('=== record #{record_index} ==='.format(record_index=i))
        X = data_records[i]

        # define max value to search
        max_value = 3

        # initiate performance storage
        min_error = None
        min_error_params = None

        # compare p-q values combinations
        for p_val in range(0, max_value):
            # compare q values
            for q_val in range(0, max_value):
                # verify at least one parameter is greater than zero
                if p_val + q_val == 0:
                    continue

                # run ARMA model
                arma_model = ARIMA_model(p=p_val, q=q_val)
                print('>> ARMA(p={p}, q={q})'.format(p=p_val, q=q_val))
                try:
                    prediction_error = arma_model.predict(X, i)
                except Exception as ex:
                    print(ex)
                    continue

                params_string = '{p}_{q}'.format(p=p_val, q=q_val)
                if min_error is None or prediction_error < min_error:
                    min_error = prediction_error
                    min_error_params = params_string

        # store min error info
        record_min_error_params['record_{i}'.format(i=i)] = [min_error_params, min_error]
        print('record best: ', min_error_params, min_error)

        if min_error is not None:
            # plot using min error params
            p_val, q_val = min_error_params.split('_')
            arma_model = ARIMA_model(p=int(p_val), q=int(q_val))
            arma_model.predict(X, i, should_plot=True)
        else:
            # add record id to problematic records
            problematic_records.append(i)

    # print all errors
    print(record_min_error_params)

    # print min error info
    all_errors = [x[1] for x in record_min_error_params.values()]
    print('min error: ', min(all_errors))
    print('avg error: ', sum(all_errors)/len(all_errors))
    print('problematic records: ', problematic_records)


if __name__ == '__main__':
    main()
