
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
        self.__target_function = model_class.get_mean_error_rate
        self.__gradient_function = model_class.get_gradient
        self.__prediction_function = model_class.get_prediction
        self.__w_vector = model_class.get_initial_w()

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
        # x_values = sorted(x_values)[-self.__lag:]

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

    # def fit_and_predict_gd_online(self, y_for_x, w_size, is_stochastic=False, max_epochs=None, fit_limit_rank=None,
    #                               first_w=None, gamma_0=0.0001, plot_progress=True, gamma_change_mode=GAMMA_STATIC,
    #                               evaluation_plot_mode=PLOT_ALL):
    #     """
    #     fits using stochastic gradient descent method
    #     :param evaluation_plot_mode:
    #     :param gamma_change_mode:
    #     :param plot_progress:
    #     :param gamma_0:
    #     :param y_for_x:
    #     :param w_size:
    #     :param is_stochastic:
    #     :param max_epochs:
    #     :param fit_limit_rank:
    #     :param first_w:
    #     :return:
    #     """
    #     if max_epochs is None and fit_limit_rank is None:
    #         raise Exception('must define epochs or fit_limit_rank, or both')
    #
    #     # GD formula: w<t+1> = w<t> - gamma_t * grad(f)(w<t>)
    #
    #     # initiate w vector
    #     if first_w is not None:
    #         w_vector = np.array(first_w)
    #     else:
    #         w_vector = np.random.rand(w_size)
    #
    #     # set data type
    #     w_vector = np.array(w_vector, dtype=np.float64)
    #     gamma_0 = np.float64(gamma_0)
    #
    #     # log initial params
    #     # self.__logger.log('initial params: w={w_vector}, gamma_0={gamma_0}'.format(w_vector=w_vector, gamma_0=gamma_0))
    #
    #     # initiate progress logging
    #     evaluations = list()
    #     last_evaluation = self.__target_function(y_for_x, w_vector)
    #     evaluations.append(last_evaluation)
    #     # self.__logger.log('initial evaluation: {evaluation}'.format(evaluation=last_evaluation))
    #
    #     if plot_progress:
    #         eval_fig = pyplot.figure()
    #         eval_ax = eval_fig.add_subplot(1, 1, 1)
    #         eval_ax.plot(evaluations)
    #         pyplot.show(block=False)
    #
    #     # initiate fitting counters
    #     t_counter = 1
    #     epoch_id = 0
    #
    #     # extract x values
    #     x_values = list(y_for_x.keys())
    #
    #     # for i in range(epochs):
    #     while ((fit_limit_rank is None) or (fit_limit_rank is not None and last_evaluation > fit_limit_rank)) \
    #             and ((max_epochs is None) or (max_epochs is not None and epoch_id < max_epochs))\
    #             and not GradientDescentFitter.__is_learning_stopped(evaluations):
    #
    #         # if (epoch_id % GradientDescentFitter.LOGGING_INTERVAL) == 0:
    #         #     self.__logger.log('%% epoch #{ep_id}'.format(ep_id=epoch_id))
    #         epoch_id += 1
    #
    #         if is_stochastic:
    #             # shuffle samples
    #             random.shuffle(x_values)
    #
    #         # run for series values
    #         for x_t in x_values:
    #             # get y_t
    #             y_t = y_for_x[x_t]
    #
    #             # calculate gradient
    #             gradient_values = self.__gradient_function(x_t, y_t, w_vector)
    #
    #             # calculate gamma_t
    #             if gamma_change_mode == GradientDescentFitter.GAMMA_STATIC:
    #                 gamma_t = gamma_0
    #             elif gamma_change_mode == GradientDescentFitter.GAMMA_INCREASING:
    #                 gamma_t = gamma_0 * np.log(t_counter)
    #             elif gamma_change_mode == GradientDescentFitter.GAMMA_DECREASING:
    #                 gamma_t = gamma_0 / np.sqrt(t_counter)
    #             else:
    #                 raise Exception('gamma mode not supported')
    #
    #             # update w vector using gradient values
    #             if any(map(math.isnan, gradient_values)):
    #                 self.__logger.log('last w: {last_w}'.format(last_w=w_vector))
    #                 self.__logger.log('last gamma: {gamma_t}'.format(gamma_t=gamma_t))
    #                 self.__logger.log('gradient values: {gradient_values}'.format(gradient_values=gradient_values))
    #                 raise Exception('nan param values reached')
    #
    #             w_vector -= gamma_t * gradient_values
    #
    #             # increase t
    #             t_counter += 1
    #
    #             # log current state
    #             if (t_counter % GradientDescentFitter.LOGGING_INTERVAL) == 0:
    #                 # self.__logger.log('[t={t_counter}] >> status:\n'
    #                 #                   'gamma_t: {gamma_t}\n'
    #                 #                   'w: {w_vector}'.format(t_counter=t_counter, gamma_t=gamma_t, w_vector=w_vector))
    #
    #                 if plot_progress:
    #                     eval_ax.clear()
    #
    #                     if evaluation_plot_mode == GradientDescentFitter.PLOT_ALL:
    #                         eval_ax.plot(evaluations)
    #                     else:   # evaluation_plot_mode == GradientDescentFitter.PLOT_RECENT:
    #                         evaluation_amount = len(evaluations)
    #                         first_index = 1 + max(0, evaluation_amount-GradientDescentFitter.RECENT_AMOUNT)
    #                         indexes = \
    #                             list(range(first_index, evaluation_amount+1))
    #                         eval_ax.plot(indexes, evaluations[-GradientDescentFitter.RECENT_AMOUNT:])
    #
    #                     pyplot.pause(0.001)
    #
    #         # calculate target function value
    #         if (epoch_id % GradientDescentFitter.EVALUATION_INTERVAL) == 0:
    #             new_evaluation = self.__target_function(y_for_x, w_vector)
    #             # self.__logger.log('evaluation: {evaluation}'.format(evaluation=new_evaluation))
    #             last_evaluation = new_evaluation
    #             if math.isnan(last_evaluation):
    #                 raise Exception('nan evaluation')
    #
    #             evaluations.append(last_evaluation)
    #
    #     # log learn statistics
    #     # self.__logger.log('evaluation: {evaluation}'.format(evaluation=self.__target_function(y_for_x, w_vector)))
    #     # self.__logger.log('epochs: {epoch_id}'.format(epoch_id=epoch_id))
    #     # self.__logger.log('updates: {update_id}'.format(update_id=t_counter))
    #
    #     # end figure
    #     pyplot.close('all')
    #
    #     # return learned params
    #     return w_vector
