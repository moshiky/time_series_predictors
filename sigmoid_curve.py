
import math
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


