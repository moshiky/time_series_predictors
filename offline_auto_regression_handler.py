
from statsmodels.tsa.ar_model import AR
import numpy as np
from numpy.linalg import lstsq


class OfflineAutoRegressionHandler:

    def __init__(self, logger, window_size):
        self.__logger = logger
        self.__window_size = window_size
        self.__params = None

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

    def predict_using_learned_params(self, initial_values, prediction_length):
        if len(initial_values) < self.__window_size:
            raise Exception('must provide initial values as at least the size of the window size. '
                            'provided: {values_length} window size: {window_size}'.format(
                                values_length=len(initial_values), window_size=self.__window_size))

        # init history
        history = initial_values[-self.__window_size:]

        # predict series values
        for i in range(prediction_length):
            next_value = sum([self.__params[j] * history[j-self.__window_size] for j in range(len(self.__params))])
            history.append(next_value)

        # return predicted values
        return history[-prediction_length:]
