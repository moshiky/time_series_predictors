

from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error


class ARMAModel:

    def __init__(self, p=0, q=0, lag_size=0):
        # set AR component order
        self.__p = p

        # set MA component order
        self.__q = q

        # store lag size settings
        self.__lag_size = lag_size

    def __get_fitted_model(self, history, add_c):
        # create model instance
        model = ARMA(history, order=(self.__p, self.__q))

        # fit model
        trend = 'c' if add_c else 'nc'
        model_fit = model.fit(disp=0, trend=trend)
        print(model_fit.params)

        return model_fit

    def predict(
            self, sample, initial_history_size, number_of_predictions_ahead, add_c,
            update_model=True, use_sample_data=True, should_plot=False, record_id=None):
        """
         V update_model=True, use_sample_data=True    ==> update model each iteration and update history with real data
         V update_model=True, use_sample_data=False   ==> update model each iteration and update history with predictions
         update_model=False, use_sample_data=True   ==> don't update model but predict next values using real data
         V update_model=False, use_sample_data=False  ==> don't update model and predict next values using predictions
        """

        if len(sample) < initial_history_size + number_of_predictions_ahead:
            raise Exception('Not enough info in record. record size={record_size}'.format(record_size=len(sample)))

        # split to train and test sets
        train, test = sample[:initial_history_size], sample[initial_history_size:]
        history = [x for x in train]
        history = history[-self.__lag_size:]
        predictions = list()

        # fit model
        model_fit = self.__get_fitted_model(history, add_c)

        # check for multi-step direct forecast settings
        if not update_model and not use_sample_data:
            predictions = model_fit.forecast(steps=number_of_predictions_ahead)[0]

        else:
            # predict values using fitted model one by one
            for prediction_index in range(number_of_predictions_ahead):
                # predict next value
                output = \
                    model_fit.predict(
                        start=initial_history_size + prediction_index,
                        end=initial_history_size + prediction_index,
                        exog=history,
                        dynamic=False
                    )
                yhat = output[0]

                # store prediction
                predictions.append(yhat)

                # update observation history
                if use_sample_data:
                    last_observation = test[prediction_index]
                    history.append(last_observation)
                else:
                    history.append(yhat)

                # update model if needed
                if update_model:
                    model_fit = self.__get_fitted_model(history[-self.__lag_size:], add_c)

        # plot prediction graph if needed
        if should_plot:
            # clear plot area
            pyplot.clf()
            pyplot.grid(which='both')
            # plot original series
            pyplot.plot(
                list(range(len(sample))),
                sample,
                '.r'
            )
            # plot prediction
            pyplot.plot(
                list(range(initial_history_size, initial_history_size+len(predictions))),
                predictions,
                '.b'
            )
            # store plotted graph
            pyplot.savefig(
                r'output/arma__id_{record_id}_p_{p}_q_{q}.png'
                .format(record_id=record_id, p=self.__p, q=self.__q)
            )

        # calculate and return mean error
        error = mean_squared_error(test[:len(predictions)], predictions)
        return error
