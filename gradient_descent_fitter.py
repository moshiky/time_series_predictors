
import math
import matplotlib.pyplot as plt
import random
import numpy as np


class GradientDescentFitter:

    LOGGING_INTERVAL = 10000    # updates
    EVALUATION_INTERVAL = 1000  # epochs
    MIN_UPDATES = 10

    # gamma modes consts
    GAMMA_STATIC = 0
    GAMMA_INCREASING = 1
    GAMMA_DECREASING = 2

    # evaluation plot mode
    PLOT_ALL = 0
    PLOT_RECENT = 1
    RECENT_AMOUNT = 15
    STOP_LEARNING_INTERVAL = 1e-5

    def __init__(self, logger, model_class, gamma_0, should_shuffle, initial_updates, batch_size, lag, gradient_size_target):
        self.__logger = logger
        self.__model_class = model_class
        self.__target_function = model_class.get_mean_error_rate
        self.__gradient_function = model_class.get_gradient
        self.__prediction_function = model_class.get_prediction
        self.__w_vector = None

        self.__gamma_0 = gamma_0
        self.__should_shuffle = should_shuffle
        self.__updates = initial_updates
        self.__batch_size = batch_size
        self.__lag = lag
        self.__gradient_size_target = gradient_size_target

        # init empty members
        self.__train_set = None

    @staticmethod
    def __is_learning_stopped(evaluations):
        if len(evaluations) < GradientDescentFitter.RECENT_AMOUNT:
            return False

        values_to_consider = evaluations[-GradientDescentFitter.RECENT_AMOUNT:]
        recent_avg = sum(values_to_consider) / len(values_to_consider)
        return abs(recent_avg - evaluations[-1]) < GradientDescentFitter.STOP_LEARNING_INTERVAL

    def fit(self, train_set=None, updates=None, gamma=None):
        if train_set is not None:
            self.__train_set = dict(train_set)

        # extract x values
        x_values = list(self.__train_set.keys())

        if self.__w_vector is None:
            self.__w_vector = self.__model_class.get_random_w(train_set[1])

        # apply gradient improvements
        gradient_sum = np.zeros(self.__w_vector.shape)
        last_gradient_avg_sum_size = 999.0
        in_batch_index = 0
        updates_so_far = 0
        max_updates = self.__updates if updates is None else updates
        sample_index = -1
        gamma_0 = self.__gamma_0 if gamma is None else gamma
        gradient_log = list()

        while updates_so_far < max_updates and last_gradient_avg_sum_size > self.__gradient_size_target:

            # # select x_t
            # if self.__should_shuffle:
            #     sample_index = np.random.randint(low=0, high=len(x_values))
            # else:
            #     sample_index = (sample_index+1) % len(x_values)
            # x_t = x_values[sample_index]
            x_t = x_values[0]

            # get y_t
            y_t = self.__train_set[x_t]

            # calculate gradient
            gradient = self.__gradient_function(x_t, y_t, self.__w_vector)

            # store gradient
            gradient_sum += gradient
            in_batch_index += 1

            # # cut gamma
            # if ((updates_so_far + 1) % 300) == 0:
            #     gamma_0 *= 2

            # apply update at batch end
            if in_batch_index == self.__batch_size:

                # apply average gradient change
                self.__w_vector -= gamma_0 * (gradient_sum / self.__batch_size)
                self.__w_vector = self.__model_class.project(self.__w_vector)
                print(gradient_sum / self.__batch_size)

                # store gradient sum size
                last_gradient_avg_sum_size = np.sqrt(np.sum(np.square(gradient_sum / self.__batch_size)))
                gradient_log.append(last_gradient_avg_sum_size)

                diff = 5
                if len(gradient_log) > GradientDescentFitter.RECENT_AMOUNT:
                    if np.array(gradient_log[-GradientDescentFitter.RECENT_AMOUNT:-diff]).mean() \
                        - np.array(gradient_log[-(GradientDescentFitter.RECENT_AMOUNT-diff):]).mean() \
                            < GradientDescentFitter.STOP_LEARNING_INTERVAL:
                        break

                # init batch process
                gradient_sum = np.zeros(self.__w_vector.shape)
                in_batch_index = 0
                updates_so_far += 1

        # self.__logger.log('stopped at update: {updates}'.format(updates=updates_so_far))
        # if updates_so_far > 100:
        #     plt.plot(gradient_log)
        #     plt.show()

        return updates_so_far

    def get_w(self):
        return self.__w_vector

    def predict(self, x_t):
        predicted_value = self.__prediction_function(x_t, self.__w_vector)
        if np.isnan(predicted_value):
            raise Exception('nan prediction')
        return predicted_value

    def update_train_set(self, new_x, new_y):
        self.__train_set = dict()
        self.__train_set[new_x] = new_y

    def update_model(self, online_updates, gamma=None):
        return self.fit(updates=online_updates, gamma=gamma)
