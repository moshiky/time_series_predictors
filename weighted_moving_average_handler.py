
import numpy as np
from numpy.linalg import lstsq


class WeightedMovingAverageHandler:

    def __init__(self, logger, train_values, window_size=1, is_online=True):
        self.__logger = logger
        self.__window_size = window_size
        self.__is_online = is_online

        self.__lag_size = 0  # self.__window_size * 2
        self.__history = list(train_values[-self.__lag_size:])
        self.__params = None
        self.__update_params()

    def __update_params(self):
        if len(self.__history) < (2 * self.__window_size):
            raise Exception('need at least two times the window size history records')

        # split to variables lists
        all_y = self.__history[self.__window_size:]
        all_xs = list()
        for i in range(self.__window_size):
            all_xs.append(self.__history[i:len(self.__history)-self.__window_size+i])

        # iterate all sub-series and solve params
        x_matrix = np.vstack(all_xs).T
        self.__params = lstsq(x_matrix, all_y)[0]

    def predict_next(self):
        return sum([self.__params[i]*self.__history[-(self.__window_size-i)] for i in range(len(self.__params))])

    def update_predictor(self, new_values):
        self.__history += new_values
        self.__history = self.__history[-self.__lag_size:]

        if self.__is_online:
            self.__update_params()
