
import math
from matplotlib import pyplot
import random
import numpy as np


class GradientDescentFitter:

    LOGGING_INTERVAL = 10000    # updates
    EVALUATION_INTERVAL = 1000  # epochs

    # gamma modes consts
    GAMMA_STATIC = 0
    GAMMA_INCREASING = 1
    GAMMA_DECREASING = 2

    # evaluation plot mode
    PLOT_ALL = 0
    PLOT_RECENT = 1
    RECENT_AMOUNT = 10
    STOP_LEARNING_INTERVAL = 1e-4

    def __init__(self, logger, model_class):
        self.__logger = logger
        self.__target_function = model_class.get_mean_error_rate
        self.__gradient_function = model_class.get_gradient
        self.__prediction_function = model_class.get_prediction
        self.__w_vector = model_class.get_initial_w()

    @staticmethod
    def __is_learning_stopped(evaluations):
        if len(evaluations) < GradientDescentFitter.RECENT_AMOUNT:
            return False

        values_to_consider = evaluations[-GradientDescentFitter.RECENT_AMOUNT:]
        recent_avg = sum(values_to_consider) / len(values_to_consider)
        return abs(recent_avg - evaluations[-1]) < GradientDescentFitter.STOP_LEARNING_INTERVAL

    def fit(self, train_set, gamma_0, should_shuffle, epochs):
        # log initial results
        # last_evaluation = self.__target_function(train_set, self.__w_vector)
        # self.__logger.log('initial evaluation: {score}'.format(score=last_evaluation))

        # extract x values
        x_values = list(train_set.keys())

        # apply gradient improvements
        for i in range(epochs):
            # self.__logger.log('epoch #{i}'.format(i=i))

            if should_shuffle:
                # shuffle samples
                random.shuffle(x_values)

            # run for series values
            for x_t in x_values:
                # todo: add batch support
                # get y_t
                y_t = train_set[x_t]

                # calculate gradient
                gradient_values = self.__gradient_function(x_t, y_t, self.__w_vector)

                # apply gradient change
                self.__w_vector -= gamma_0 * gradient_values

            # log initial results
            # last_evaluation = self.__target_function(train_set, self.__w_vector)
            # self.__logger.log('current evaluation: {score}'.format(score=last_evaluation))

    def predict(self, x_t):
        predicted_value = self.__prediction_function(x_t, self.__w_vector)
        if np.isnan(predicted_value):
            raise Exception('nan prediction')
        return predicted_value

    def fit_and_predict_gd_online(self, y_for_x, w_size, is_stochastic=False, max_epochs=None, fit_limit_rank=None,
                                  first_w=None, gamma_0=0.0001, plot_progress=True, gamma_change_mode=GAMMA_STATIC,
                                  evaluation_plot_mode=PLOT_ALL):
        """
        fits using stochastic gradient descent method
        :param evaluation_plot_mode:
        :param gamma_change_mode:
        :param plot_progress:
        :param gamma_0:
        :param y_for_x:
        :param w_size:
        :param is_stochastic:
        :param max_epochs:
        :param fit_limit_rank:
        :param first_w:
        :return:
        """
        if max_epochs is None and fit_limit_rank is None:
            raise Exception('must define epochs or fit_limit_rank, or both')

        # GD formula: w<t+1> = w<t> - gamma_t * grad(f)(w<t>)

        # initiate w vector
        if first_w is not None:
            w_vector = np.array(first_w)
        else:
            w_vector = np.random.rand(w_size)

        # set data type
        w_vector = np.array(w_vector, dtype=np.float64)
        gamma_0 = np.float64(gamma_0)

        # log initial params
        # self.__logger.log('initial params: w={w_vector}, gamma_0={gamma_0}'.format(w_vector=w_vector, gamma_0=gamma_0))

        # initiate progress logging
        evaluations = list()
        last_evaluation = self.__target_function(y_for_x, w_vector)
        evaluations.append(last_evaluation)
        # self.__logger.log('initial evaluation: {evaluation}'.format(evaluation=last_evaluation))

        if plot_progress:
            eval_fig = pyplot.figure()
            eval_ax = eval_fig.add_subplot(1, 1, 1)
            eval_ax.plot(evaluations)
            pyplot.show(block=False)

        # initiate fitting counters
        t_counter = 1
        epoch_id = 0

        # extract x values
        x_values = list(y_for_x.keys())

        # for i in range(epochs):
        while ((fit_limit_rank is None) or (fit_limit_rank is not None and last_evaluation > fit_limit_rank)) \
                and ((max_epochs is None) or (max_epochs is not None and epoch_id < max_epochs))\
                and not GradientDescentFitter.__is_learning_stopped(evaluations):

            # if (epoch_id % GradientDescentFitter.LOGGING_INTERVAL) == 0:
            #     self.__logger.log('%% epoch #{ep_id}'.format(ep_id=epoch_id))
            epoch_id += 1

            if is_stochastic:
                # shuffle samples
                random.shuffle(x_values)

            # run for series values
            for x_t in x_values:
                # get y_t
                y_t = y_for_x[x_t]

                # calculate gradient
                gradient_values = self.__gradient_function(x_t, y_t, w_vector)

                # calculate gamma_t
                if gamma_change_mode == GradientDescentFitter.GAMMA_STATIC:
                    gamma_t = gamma_0
                elif gamma_change_mode == GradientDescentFitter.GAMMA_INCREASING:
                    gamma_t = gamma_0 * np.log(t_counter)
                elif gamma_change_mode == GradientDescentFitter.GAMMA_DECREASING:
                    gamma_t = gamma_0 / np.sqrt(t_counter)
                else:
                    raise Exception('gamma mode not supported')

                # update w vector using gradient values
                if any(map(math.isnan, gradient_values)):
                    self.__logger.log('last w: {last_w}'.format(last_w=w_vector))
                    self.__logger.log('last gamma: {gamma_t}'.format(gamma_t=gamma_t))
                    self.__logger.log('gradient values: {gradient_values}'.format(gradient_values=gradient_values))
                    raise Exception('nan param values reached')

                w_vector -= gamma_t * gradient_values

                # increase t
                t_counter += 1

                # log current state
                if (t_counter % GradientDescentFitter.LOGGING_INTERVAL) == 0:
                    # self.__logger.log('[t={t_counter}] >> status:\n'
                    #                   'gamma_t: {gamma_t}\n'
                    #                   'w: {w_vector}'.format(t_counter=t_counter, gamma_t=gamma_t, w_vector=w_vector))

                    if plot_progress:
                        eval_ax.clear()

                        if evaluation_plot_mode == GradientDescentFitter.PLOT_ALL:
                            eval_ax.plot(evaluations)
                        else:   # evaluation_plot_mode == GradientDescentFitter.PLOT_RECENT:
                            evaluation_amount = len(evaluations)
                            first_index = 1 + max(0, evaluation_amount-GradientDescentFitter.RECENT_AMOUNT)
                            indexes = \
                                list(range(first_index, evaluation_amount+1))
                            eval_ax.plot(indexes, evaluations[-GradientDescentFitter.RECENT_AMOUNT:])

                        pyplot.pause(0.001)

            # calculate target function value
            if (epoch_id % GradientDescentFitter.EVALUATION_INTERVAL) == 0:
                new_evaluation = self.__target_function(y_for_x, w_vector)
                # self.__logger.log('evaluation: {evaluation}'.format(evaluation=new_evaluation))
                last_evaluation = new_evaluation
                if math.isnan(last_evaluation):
                    raise Exception('nan evaluation')

                evaluations.append(last_evaluation)

        # log learn statistics
        # self.__logger.log('evaluation: {evaluation}'.format(evaluation=self.__target_function(y_for_x, w_vector)))
        # self.__logger.log('epochs: {epoch_id}'.format(epoch_id=epoch_id))
        # self.__logger.log('updates: {update_id}'.format(update_id=t_counter))

        # end figure
        pyplot.close('all')

        # return learned params
        return w_vector
