
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
            xxx = 1

        # return predictions
        return predictions
