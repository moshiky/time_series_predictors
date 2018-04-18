
from numpy.linalg import lstsq
from random import shuffle


class OfflineAutoRegressionHandler:

    def __init__(self, logger, window_size):
        self.__logger = logger
        self.__window_size = window_size
        self.__params = None

    def learn_model_params(self, train_set, should_print_params=False):
        # build x list
        x_values = list()
        for row in train_set:
            # x_values += [row[i:i+self.__window_size] + [1.0] for i in range(len(row[:-self.__window_size]))]
            x_values += [row[i:i + self.__window_size] for i in range(len(row[:-self.__window_size]))]

        # build y list
        y_values = list()
        for row in train_set:
            y_values += row[self.__window_size:]

        # find params
        self.__params = lstsq(x_values, y_values)[0]

        # print params
        if should_print_params:
            self.__logger.log('model params: {params}'.format(params=self.__params))

    def predict_using_learned_params(self, initial_values, prediction_length):
        if len(initial_values) < self.__window_size:
            raise Exception('must provide initial values at least as the size of the window size. '
                            'provided: {values_length} window size: {window_size}'.format(
                                values_length=len(initial_values), window_size=self.__window_size))

        # init history
        history = initial_values[-self.__window_size:]

        # predict series values
        for i in range(prediction_length):
            next_value = \
                sum(
                    [self.__params[j] * history[j-self.__window_size] for j in range(self.__window_size)],
                    # self.__params[-1]
                )
            history.append(next_value)

        # return predicted values
        return history[-prediction_length:]

    def get_params(self):
        return self.__params

    @staticmethod
    def predict_using_params(params, initial_values, prediction_length):
        # init history
        num_params = len(params) - 1
        history = initial_values[-num_params:]

        # predict series values
        for i in range(prediction_length):
            next_value = sum([params[j] * history[j - num_params] for j in range(num_params)], params[-1])
            history.append(next_value)

        # return predicted values
        return history[-prediction_length:]

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
            for i in range(len(examples)):
                sample = examples[i]
                dot_sum = sum([params[k]*sample[k] for k in range(num_of_parameters-1)]) + params[-1] - sample[-1]

                for j in range(num_of_parameters-1):
                    new_value = 2 * sample[j] * dot_sum
                    params[j] -= step_size * new_value
                params[-1] -= step_size * 2 * dot_sum

            step_size *= step_size_factor
        # print params
        print('gd params: ', params)

        # return params
        return params
