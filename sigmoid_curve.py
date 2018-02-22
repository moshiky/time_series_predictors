
import math
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
        L_param = series[-1] * mid_max_rate
        Y_values = [np.log((L_param/y_value) - 1) for y_value in series]

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
                L_param / (1 + c_param * np.exp(a_param * current_x))
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
        L_param = series[-1] * mid_max_rate

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
                    + np.log(L_param - series[i+n]) \
                    - np.log(L_param - series[i])

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
                    L_param / (1 + np.exp(a_param) * ((L_param / last_y) - 1))
                )

            # return predictions
            return predictions[1:]

        elif order == 2:
            predictions = series[-2:]
            for i in range(prediction_length):
                y_t = predictions[-2]
                y_t1 = predictions[-1]

                ytea = 1 + np.exp(a_param) * ((L_param/y_t) - 1)

                predictions.append(
                    L_param / (1 + np.exp(a_param) * (((ytea*2*L_param)/(ytea*y_t1+L_param)) - 1))
                )

            # return predictions
            return predictions[2:]

        else:
            raise Exception('unsupported order. must be 1 or 2')

    @staticmethod
    def __get_gradient(y_t, y_t1, L_param, a_param):
        """
        :param L_param:
        :param a_param:
        :return: gradient of F using given params
        """

        # define params
        y_t = np.float64(y_t)
        y_t1 = np.float64(y_t1)
        L_param = np.float64(L_param)
        a_param = np.float64(a_param)

        e_a = np.exp(a_param)
        G = y_t + e_a * (L_param - y_t)
        F = (L_param * y_t) / G - y_t1

        # calculate L'
        div_F_L = y_t/G - (L_param*y_t*e_a)/np.square(G)
        div_L = 2 * F * div_F_L

        # calculate a'
        K = e_a * (L_param - y_t)
        div_F_a = ((-1) * L_param * y_t * K) / np.square((y_t + K))
        div_a = 2 * F * div_F_a

        return np.array([div_L, div_a], dtype=np.float64)

    @staticmethod
    def __get_gradient_online(x_t, y_t, L_param, a_param, c_param):
        # calculate common values
        e_ax = np.exp(a_param * x_t)
        bottom_part = 1 + c_param * e_ax
        common_start = 2 * ((L_param / bottom_part) - y_t)

        # calculate dq/dL
        dq_dL = 1 / bottom_part

        # calculate dq/da
        dq_da = (-L_param) * a_param * e_ax / np.square(bottom_part)

        # calculate dq/dc
        dq_dc = (-L_param) * e_ax / np.square(bottom_part)

        # return gradient
        return np.array([
            common_start * dq_dL,
            common_start * dq_da,
            common_start * dq_dc
        ], dtype=np.float64)


    @staticmethod
    def __get_gradient_auto(y_t, y_t1, L_param, a_param):

        e_a = np.exp(a_param)

        # calculate L'
        div_L = (2 * np.square(y_t) * (e_a - 1) * ((y_t1 * e_a - y_t) * L_param - y_t * y_t1 * e_a + y_t * y_t1)) \
                / np.power((e_a * L_param - y_t * e_a + y_t), 3)

        # calculate a'
        div_a = ((-2)*y_t*L_param*(L_param-y_t)*e_a*(((y_t*L_param)/((L_param-y_t)*e_a+y_t)) - y_t1)) \
                / np.square((L_param-y_t)*e_a + y_t)

        return np.array([div_L, div_a], dtype=np.float64)

    @staticmethod
    def __get_function_avg_value(samples, L_param, a_param):
        value_sum = 0.0
        for _, y_t, y_t1 in samples:
            value_sum += np.square((L_param * y_t) / (y_t + np.exp(a_param) * (L_param - y_t)) - y_t1)

        return value_sum / len(samples)


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
        logger.log('L={L_param}, a={a_param}'.format(L_param=w_vector[0], a_param=w_vector[1]))
        alpha = np.float64(0.05)

        samples = [
            [i, series[i], series[i+1]] for i in range(len(series)-1)
        ]

        last_eval = SigmoidCurve.__get_function_avg_value(samples, L_param=w_vector[0], a_param=w_vector[1])
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
                        L_param=w_vector[0],
                        a_param=w_vector[1]
                    )

                # if sample_index < inflection_point:
                #     w_vector -= alpha * gradient_values
                # else:
                #     w_vector += alpha * gradient_values

                w_vector -= alpha * gradient_values
                logger.log('L={L_param}, a={a_param}'.format(L_param=w_vector[0], a_param=w_vector[1]))

            # else:
            #     for i in range(1, len(series)):
            #         # logger.log('[#{i}] y_t= {y_t}, y_t1= {y_t1}'.format(i=i, y_t=series[i-1], y_t1=series[i]))
            #
            #         if w_vector[0] < series[i-1]:
            #             new_value = np.float64(series[i-1] * 2.0)
            #             logger.log('L: {old} -> {new}'.format(old=w_vector[0], new=new_value))
            #             w_vector[0] = new_value
            #
            #         gradient_values = \
            #             SigmoidCurve.__get_gradient(
            #                 y_t=series[i-1],
            #                 y_t1=series[i],
            #                 L_param=w_vector[0],
            #                 a_param=w_vector[1]
            #             )
            #         if i < inflection_point:
            #             w_vector -= alpha * gradient_values
            #         else:
            #             w_vector += alpha * gradient_values

            logger.log('L={L_param}, a={a_param}'.format(L_param=w_vector[0], a_param=w_vector[1]))

            last_eval = SigmoidCurve.__get_function_avg_value(samples, L_param=w_vector[0], a_param=w_vector[1])
            logger.log('## func. eval.: {avg_eval}'.format(avg_eval=last_eval))

        L_param = w_vector[0]
        a_param = w_vector[1]

        return L_param, a_param

    @staticmethod
    def fit_and_predict_gd_online(
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
        # W[0] <- L     W[1] <- a      W[2] <- c
        w_vector = np.array([random.random(), -random.random(), random.random()], dtype=np.float64)
        logger.log(w_vector)
        alpha = np.float64(0.1)

        x_values = list(range(1, len(series)+1))

        for i in range(epoches):

            if is_stochastic:
                # shuffle samples
                random.shuffle(x_values)

            for sample_index in x_values:
                logger.log('[#{i}] y_t= {y_t}'
                           .format(i=sample_index, y_t=series[sample_index - 1]))

                gradient_values = \
                    SigmoidCurve.__get_gradient_online(
                        x_t=sample_index,
                        y_t=series[sample_index-1],
                        L_param=w_vector[0],
                        a_param=w_vector[1],
                        c_param=w_vector[2]
                    )

                w_vector -= alpha * gradient_values

                logger.log('L={L_param}, a={a_param}, c={c_param}'
                           .format(L_param=w_vector[0], a_param=w_vector[1], c_param=w_vector[2]))

        L_param = w_vector[0]
        a_param = w_vector[1]
        c_param = w_vector[2]

        return L_param, a_param, c_param
