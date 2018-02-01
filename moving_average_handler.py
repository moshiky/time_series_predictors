
import statsmodels.tsa.api as smt
import numpy as np
from random import shuffle
from scipy.optimize import curve_fit


class MovingAverageHandler:

    def __init__(self, logger, window_size=2, model_params=None):
        self.__logger = logger
        self.__window_size = window_size

        self.__model = None
        self.__model_fit = None
        self.__params = model_params

    def learn_model_params(self, train_values):
        self.__model = smt.ARMA(train_values, order=(0, self.__window_size))
        self.__model_fit = self.__model.fit(maxlag=self.__window_size, method='mle', trend='c', disp=-1)
        self.__params = self.__model_fit.params

    def get_model_params(self):
        return list(self.__params)

    def predict_using_learned_params(self, initial_values, prediction_length):
        if len(initial_values) < self.__window_size:
            raise Exception('must provide initial values as at least the size of the window size. '
                            'provided: {values_length} window size: {window_size}'.format(
                                values_length=len(initial_values), window_size=self.__window_size))

        return self.__model_fit.predict(0, prediction_length-1, initial_values)

    @staticmethod
    def test_gd(train_set, model_order):
        # build vars list
        examples = list()
        for row in train_set:
            examples += [row[i:i + model_order + 1] for i in range(len(row[:-(model_order + 1)]))]

        # start to solve
        num_of_parameters = model_order + 1
        params = [0.0] * num_of_parameters

        # make swipe
        step_size = 0.01
        step_size_factor = 0.3
        num_epochs = 10
        for ep in range(num_epochs):
            shuffle(examples)
            last_prediction = 0
            for i in range(len(examples)):
                sample = examples[i]
                dot_sum = sum([params[k] * sample[k] for k in range(num_of_parameters - 1)]) + params[-1] - sample[-1]

                for j in range(num_of_parameters - 1):
                    new_value = 2 * sample[j] * dot_sum
                    params[j] -= step_size * new_value
                params[-1] -= step_size * 2 * dot_sum

            step_size *= step_size_factor

        # print params
        print('gd params: ', params)

        # return params
        return params

    @staticmethod
    def test_gd_one_param(train_set, model_order=1):
        # build x list
        sample = train_set[0]
        x_values = sample[:-model_order]
        y_values = sample[model_order:]

        # convert to np arrays
        x_values = np.array(x_values)
        y_values = np.array(y_values)

        # fit curve
        print('start fitting..')
        p_opt, p_cov = curve_fit(ma_func, x_values, y_values)
        print('found values:')
        return p_opt

    @staticmethod
    def get_prediction_using_params(params, lag):
        pass


def ma_func(x, t, c):
    num_of_samples = len(x)
    predictions = np.zeros(num_of_samples, dtype=np.float32)

    for j in range(1, num_of_samples):
        predictions[j] = \
            sum([((-1)**(i+1)) * (t**i) * x[j-i] for i in range(1, j+1)]) \
            + c * sum([((-1)**(k+1)) * (t**(j-k)) for k in range(1, j+1)])

    return predictions
