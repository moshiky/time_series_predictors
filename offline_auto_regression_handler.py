
from statsmodels.tsa.ar_model import AR


class OfflineAutoRegressionHandler:

    def __init__(self, logger, train_values):
        self.__logger = logger

        self.__model = AR(train_values)
        self.__model_fit = self.__model.fit()
        self.__window = self.__model_fit.k_ar
        self.__model_params = self.__model_fit.params

        self.__history = list(train_values[-self.__window:])

    def predict_next(self):
        lag = self.__history[-self.__window:]
        y_hat = self.__model_params[0]

        for d in range(self.__window):
            try:
                y_hat += self.__model_params[d + 1] * lag[self.__window - d - 1]
            except Exception as ex:
                self.__logger.error('faild with exp: ' + str(ex))
                self.__logger.error(
                    'd= ' + str(d)
                    + ' lag= ' + str(lag)
                    + ' params= ' + str(self.__model_params)
                    + ' window= ' + str(self.__window)
                )
                raise ex

        return y_hat

    def update_predictor(self, new_values):
        self.__history += new_values
