
import time
import math
from matplotlib import pyplot
import random
import numpy as np


class SigmoidCurve:

    MID_MAX_RATE = 3.1

    @staticmethod
    def fit_and_predict(series, prediction_length, mid_max_rate=MID_MAX_RATE):
        """
        fits logistic function params. function form:

                  L
        y = --------------
            1 + c * exp(a*x)

        :param series:
        :param prediction_length:
        :param mid_max_rate:
        :return:
        """
        # translate params according to "data linearization" method
        X_values = [[i+1, 1.0] for i in range(len(series))]
        l_param = series[-1] * mid_max_rate
        Y_values = [np.log((l_param/y_value) - 1) for y_value in series]

        # fit linear model params: Y = A*X + B
        linear_params = np.linalg.lstsq(X_values, Y_values)[0]
        # print('fitted params: {params}'.format(params=linear_params))

        # convert linear params to logistic params: a=A, c=exp(B)
        a_param = linear_params[0]
        c_param = np.exp(linear_params[1])

        # predict next values
        predictions = list()
        for i in range(prediction_length):
            current_x = len(series) + i
            predictions.append(
                l_param / (1 + c_param * np.exp(a_param * current_x))
            )

        # return predictions
        return predictions

    @staticmethod
    def fit_and_predict_recursive(series, prediction_length, mid_max_rate=MID_MAX_RATE, order=1):
        """
        fits logistic function params. function form:

                  L
        y = --------------
            1 + c * exp(a*x)

        but using recursive approach-

        N = A * a

        while-
            a is the parameter we wish to fit
            N = 1 / n, n is 'skip' size
            A = [ln(y<t>) - ln(y<t+n>) + ln(L-y<t+n>) - ln(L-y<t>)] ^ -1

        :param series:
        :param prediction_length:
        :param mid_max_rate:
        :return:
        """
        # calculate L value
        l_param = series[-1] * mid_max_rate

        # initiate X and Y value arrays
        X_values = list()
        Y_values = list()

        # translate params according to "data linearization" approach
        for n in range(1, len(series)):
            # calculate N value
            N_value = n ** -1

            for i in range(len(series)-n):
                # calculate A value
                A_value = \
                    np.log(series[i]) \
                    - np.log(series[i+n]) \
                    + np.log(l_param - series[i+n]) \
                    - np.log(l_param - series[i])

                # append to X and Y value arrays
                X_values += [[A_value]]
                Y_values.append(N_value)

        # fit linear model params: Y = a*X  [need to find 'a']
        linear_params = np.linalg.lstsq(X_values, Y_values)[0]
        # print('fitted params: {params}'.format(params=linear_params))

        # convert linear params to logistic params: a=A, c=exp(B)
        a_param = linear_params[0]

        # predict next values - using rolling forecast

        if order == 1:
            predictions = [series[-1]]
            for i in range(prediction_length):
                last_y = predictions[-1]
                predictions.append(
                    l_param / (1 + np.exp(a_param) * ((l_param / last_y) - 1))
                )

            # return predictions
            return predictions[1:]

        elif order == 2:
            predictions = series[-2:]
            for i in range(prediction_length):
                y_t = predictions[-2]
                y_t1 = predictions[-1]

                ytea = 1 + np.exp(a_param) * ((l_param/y_t) - 1)

                predictions.append(
                    l_param / (1 + np.exp(a_param) * (((ytea*2*l_param)/(ytea*y_t1+l_param)) - 1))
                )

            # return predictions
            return predictions[2:]

        else:
            raise Exception('unsupported order. must be 1 or 2')

    @staticmethod
    def __get_gradient(y_t, y_t1, l_param, a_param):
        """
        :param l_param:
        :param a_param:
        :return: gradient of F using given params
        """

        # define params
        y_t = np.float64(y_t)
        y_t1 = np.float64(y_t1)
        l_param = np.float64(l_param)
        a_param = np.float64(a_param)

        e_a = np.exp(a_param)
        G = y_t + e_a * (l_param - y_t)
        F = (l_param * y_t) / G - y_t1

        # calculate L'
        div_F_L = y_t/G - (l_param*y_t*e_a)/np.square(G)
        div_L = 2 * F * div_F_L

        # calculate a'
        K = e_a * (l_param - y_t)
        div_F_a = ((-1) * l_param * y_t * K) / np.square((y_t + K))
        div_a = 2 * F * div_F_a

        return np.array([div_L, div_a], dtype=np.float64)

    @staticmethod
    def __get_gradient_online(x_t, y_t, l_param, a_param, c_param):
        # calculate common values
        e_ax = np.exp(a_param * x_t)
        bottom_part = 1 + c_param * e_ax
        common_start = 2 * ((l_param / bottom_part) - y_t)

        # calculate dq/dL
        dq_dL = 1 / bottom_part

        # calculate dq/da
        dq_da = (-1) * l_param * c_param * x_t * e_ax / np.square(bottom_part)

        # calculate dq/dc
        dq_dc = (-1) * l_param * e_ax / np.square(bottom_part)

        # return gradient
        return np.array([
            common_start * dq_dL,
            common_start * dq_da,
            common_start * dq_dc
        ], dtype=np.float64)

    @staticmethod
    def __get_function_avg_value(samples, l_param, a_param):
        value_sum = 0.0
        for _, y_t, y_t1 in samples:
            value_sum += np.square((l_param * y_t) / (y_t + np.exp(a_param) * (l_param - y_t)) - y_t1)

        return value_sum / len(samples)

    @staticmethod
    def __get_function_avg_value_online(x_values, series_by_x, l_param, a_param, c_param):
        value_sum = 0.0
        for x_t in x_values:
            value_sum += np.square(l_param / (1 + c_param * np.exp(a_param * x_t)) - series_by_x[x_t])

        return value_sum / len(x_values)

    @staticmethod
    def __get_mean_error_rate(x_values, series_by_x, l_param, a_param, c_param):
        error_rates_total = 0.0
        for x_t in x_values:
            function_value = l_param / (1 + c_param * np.exp(a_param * x_t))
            y_t = series_by_x[x_t]
            error_rates_total += (abs(y_t - function_value) / y_t)

        return error_rates_total / len(x_values)

    @staticmethod
    def fit_and_predict_gd(
            logger, series, prediction_length, inflection_point, is_stochastic=False, epoches=1):
        """
        fits using stochastic gradient descent method
        :param inflection_point:
        :param is_stochastic:
        :param series:
        :param prediction_length:
        :return:
        """
        # GD formula: w<t+1> = w<t> - a * grad(f)(w<t>)
        # W[0] <- L     W[1] <- a
        w_vector = np.array([1.0, -1.0], dtype=np.float64)
        logger.log('L={l_param}, a={a_param}'.format(l_param=w_vector[0], a_param=w_vector[1]))
        alpha = np.float64(0.0001)

        samples = [
            [i, series[i], series[i+1]] for i in range(len(series)-1)
        ]

        last_eval = SigmoidCurve.__get_function_avg_value(samples, l_param=w_vector[0], a_param=w_vector[1])
        # while abs(last_eval) > 0.01:
        for epoch_id in range(epoches):

            if is_stochastic:
                # shuffle samples
                random.shuffle(samples)

            # update as many times as requested
            for sample_index, y_t, y_t1 in samples:
                logger.log('[#{i}] y_t= {y_t}, y_t1= {y_t1}'
                           .format(i=sample_index, y_t=series[sample_index-1], y_t1=series[sample_index]))

                gradient_values = \
                    SigmoidCurve.__get_gradient(
                        y_t=y_t,
                        y_t1=y_t1,
                        l_param=w_vector[0],
                        a_param=w_vector[1]
                    )

                w_vector -= alpha * gradient_values
                logger.log('L={l_param}, a={a_param}'.format(l_param=w_vector[0], a_param=w_vector[1]))

            logger.log('L={l_param}, a={a_param}'.format(l_param=w_vector[0], a_param=w_vector[1]))

            last_eval = SigmoidCurve.__get_function_avg_value(samples, l_param=w_vector[0], a_param=w_vector[1])
            logger.log('## func. eval.: {avg_eval}'.format(avg_eval=last_eval))

        l_param = w_vector[0]
        a_param = w_vector[1]

        return l_param, a_param

    @staticmethod
    def __update_eval_plot(eval_plot, values):
        # clear old values
        eval_plot.clf()

        # plot series
        pyplot.plot(values)

    @staticmethod
    def fit_and_predict_gd_online(
            logger, series, x_range, prediction_length, inflection_point, is_stochastic=False, epochs=1):
        """
        fits using stochastic gradient descent method
        :param x_range:
        :param epochs:
        :param logger:
        :param inflection_point:
        :param is_stochastic:
        :param series:
        :param prediction_length:
        :return:
        """

        LOGGING_INTERVAL = 10000
        EVALUATION_INTERVAL = 10000
        PERFORMANCE_MODE = False

        series_length = len(series)
        x_step_size = (x_range[1]-x_range[0]) / series_length
        x_values = [x_range[0]+i*x_step_size for i in range(series_length)]

        series_by_x = {
            x_values[i]: series[i] for i in range(len(series))
        }

        # GD formula: w<t+1> = w<t> - gamma * grad(f)(w<t>)
        # W[0] <- L     W[1] <- a      W[2] <- c
        w_vector = np.array([random.random(), -random.random(), random.random()], dtype=np.float64)
        gamma_0 = np.float64(0.00001)

        # log initial params
        logger.log('L={l_param}, a={a_param}, c={c_param}'
                   .format(l_param=w_vector[0], a_param=w_vector[1], c_param=w_vector[2]))

        evaluations = list()
        last_evaluation = \
            SigmoidCurve.__get_mean_error_rate(
                x_values, series_by_x, l_param=w_vector[0], a_param=w_vector[1], c_param=w_vector[2]
            )
        logger.log('eval: {evaluation}'.format(evaluation=last_evaluation))
        evaluations.append(last_evaluation)

        eval_fig = pyplot.figure()
        eval_ax = eval_fig.add_subplot(1, 1, 1)
        eval_ax.plot(evaluations)
        pyplot.show(block=False)

        update_number = 1
        epoch_id = 0

        # params_last_change_direction = None
        # last_dir_change_update_number = 0

        # for i in range(epochs):
        while last_evaluation > 1e-2:

            epoch_id += 1
            if not PERFORMANCE_MODE and (epoch_id % LOGGING_INTERVAL) == 0:
                logger.log('%%%%%%%%% epoch #{ep_id}'.format(ep_id=epoch_id))

            if is_stochastic:
                # shuffle samples
                random.shuffle(x_values)

            for x_t in x_values:
                y_t = series_by_x[x_t]
                # logger.log('x_t={x_t} y_t={y_t}'.format(x_t=x_t, y_t=y_t))

                gradient_values = \
                    SigmoidCurve.__get_gradient_online(
                        x_t=x_t,
                        y_t=y_t,
                        l_param=w_vector[0],
                        a_param=w_vector[1],
                        c_param=w_vector[2]
                    )

                # logger.log('gradient: {gradient}'.format(gradient=gradient_values))
                # params_change_direction = 1 if gradient_values[1] > 0 else -1
                # if params_last_change_direction is not None and params_change_direction != params_last_change_direction:
                #     params_last_change_direction = params_change_direction
                #     last_dir_change_update_number = update_number-1
                #     print('change')
                #
                # gamma = gamma_0 * np.square(np.log(update_number - last_dir_change_update_number))

                # gamma = gamma_0 / update_number
                gamma = gamma_0 * np.square(np.log(update_number))
                # print(w_vector)
                w_vector -= gamma * gradient_values
                update_number += 1

                # w_vector[0] = 1.0

                if (update_number % LOGGING_INTERVAL) == 0:
                    logger.log('updates: {update_number}'.format(update_number=update_number))
                    logger.log('gamma: {gamma}'.format(gamma=gamma))
                    logger.log('L={l_param}, a={a_param}, c={c_param}'
                               .format(l_param=w_vector[0], a_param=w_vector[1], c_param=w_vector[2]))

                    eval_ax.clear()
                    eval_ax.plot(evaluations)
                    pyplot.pause(0.001)

                if (update_number % EVALUATION_INTERVAL) == 0:
                    new_evaluation = \
                        SigmoidCurve.__get_mean_error_rate(
                            x_values, series_by_x, l_param=w_vector[0], a_param=w_vector[1], c_param=w_vector[2]
                        )
                    logger.log('eval: {evaluation}'.format(evaluation=new_evaluation))

                    last_evaluation = new_evaluation
                    evaluations.append(last_evaluation)

        # log learn statistics
        logger.log('epochs: {epoch_id}'.format(epoch_id=epoch_id))
        logger.log('updates: {update_id}'.format(update_id=update_number))

        l_param = w_vector[0]
        a_param = w_vector[1]
        c_param = w_vector[2]

        return l_param, a_param, c_param
