

from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


class ARIMAModel:

    def __init__(self, p=0, d=0, q=0, lag_size=0):
        # set AR component order
        self.__p = p

        # set MA component order
        self.__q = q

        # set I component order
        self.__d = d

        # store lag size settings
        self.__lag_size = lag_size

    def predict(
            self, x_samples, initial_history_size, number_of_predictions_ahead, record_id, should_plot=False,
            add_c=False):

        if len(x_samples) < self.__lag_size + number_of_predictions_ahead:
            raise Exception('Not enough info in record')

        # split to train and test sets
        train, test = x_samples[:initial_history_size], x_samples[initial_history_size:]
        history = [x for x in train]
        predictions = list()

        # # predict all at once
        # model = ARIMA(history, order=(self.__p, self.__d, self.__q))
        # model_fit = model.fit(disp=0, trend='nc')
        # predictions = model_fit.predict(len(train)+1, len(X)-1)

        # predict one by one
        trend = 'c' if add_c else 'nc'
        for prediction_index in range(number_of_predictions_ahead):
            # fit model
            model = ARIMA(history, order=(self.__p, self.__d, self.__q))
            model_fit = model.fit(disp=0, trend=trend)

            # predict next value
            output = model_fit.forecast()
            yhat = output[0]

            # store prediction
            predictions.append(yhat)

            # update observation history
            last_observation = test[prediction_index]
            history.append(last_observation)
            history = history[-self.__lag_size:]

            # print('predicted=%f, expected=%f' % (yhat, obs))

        # plot prediction graph if needed
        if should_plot:
            # clear plot area
            pyplot.clf()
            # plot original series
            pyplot.plot(list(range(len(x_samples))), x_samples, color='red')
            # plot prediction
            pyplot.plot(
                list(range(initial_history_size, initial_history_size+len(predictions))),
                predictions,
                color='blue'
            )
            # store plotted graph
            pyplot.savefig(
                r'output/arima__id_{record_id}_p_{p}_d_{d}_q_{q}.png'
                .format(record_id=record_id, p=self.__p, d=self.__d, q=self.__q)
            )

        # calculate and return mean error
        error = mean_squared_error(test[:len(predictions)], predictions)
        print('Test MSE: %.3f' % error)
        return error