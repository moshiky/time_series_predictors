
from statsmodels.tsa.ar_model import AR
import numpy as np
from numpy.linalg import lstsq


class OfflineAutoRegressionHandler:

    def __init__(self, logger, train_values, window_size):
        self.__logger = logger
        self.__train_values = train_values
        self.__window_size = window_size
        self.__params = None

        # self.__model = AR(train_values)
        # self.__model_fit = self.__model.fit()
        # self.__window = self.__model_fit.k_ar
        # self.__model_params = self.__model_fit.params

        # self.__history = list(train_values[-self.__window:])

    def learn_model_params(self, train_set):
        # build x list
        x_values = list()
        for row in train_set:
            x_values += [row[i:i+self.__window_size] for i in range(len(row[:-self.__window_size]))]

        # build y list
        y_values = list()
        for row in train_set:
            y_values += row[self.__window_size:]

        # find params
        self.__params = lstsq(x_values, y_values)[0]

    def predict_using_learned_params(self, initial_values):
        pass

    # def predict_next(self):
    #     lag = self.__history[-self.__window:]
    #     y_hat = self.__model_params[0]
    #
    #     for d in range(self.__window):
    #         try:
    #             y_hat += self.__model_params[d + 1] * lag[self.__window - d - 1]
    #         except Exception as ex:
    #             self.__logger.error('faild with exp: ' + str(ex))
    #             self.__logger.error(
    #                 'd= ' + str(d)
    #                 + ' lag= ' + str(lag)
    #                 + ' params= ' + str(self.__model_params)
    #                 + ' window= ' + str(self.__window)
    #             )
    #             raise ex
    #
    #     return y_hat
    #
    # def update_predictor(self, new_values):
    #     self.__history += new_values
