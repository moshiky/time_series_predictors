
import numpy as np


class SigmoidV3:
    """
    minimize MAPE based loss function

    fitted function:
                          B
            y = A + --------------
                    C + e^(K*x + F)

    loss function:
                      y - y_p
             L_sq =   -------     Loss = L_sq ^ 2
                         y

    """

    INITIAL_W_VECTOR = \
        np.array([
            0.49937523,
            0.89339831,
            0.08601164,
            -0.22586464,
            0.43639608
        ])

    @staticmethod
    def get_initial_w():
        return np.array(SigmoidV3.INITIAL_W_VECTOR)

    @staticmethod
    def get_prediction(x_value, w_vector, should_round=False):
        # extract params
        a_param, b_param, c_param, k_param, f_param = w_vector

        # return sigmoid value
        raw_value = a_param + b_param / (c_param + np.exp(k_param * x_value + f_param))

        # return value
        return round(raw_value) if should_round else raw_value

    @staticmethod
    def get_mean_error_rate(dataset, w_vector):
        # calculate mean error rate
        error_rates_total = 0.0
        for x_t in dataset.keys():
            y_t = dataset[x_t]
            function_value = SigmoidV3.get_prediction(x_t, w_vector)
            error_rates_total += abs((y_t - function_value) / y_t)

        # return mean error rate
        return error_rates_total / len(dataset)

    @staticmethod
    def get_gradient(x_t, y_t, w_vector):
        # extract params
        a_param, b_param, c_param, k_param, f_param = w_vector

        # calculate common values
        e_kxf = np.exp(k_param * x_t + f_param)
        div_bottom = c_param + e_kxf
        base_value = 2 * (y_t - a_param - (b_param / div_bottom))
        y2 = np.square(y_t)

        # calculate dq/da
        dq_da = -base_value / y2

        # calculate dq/db
        dq_db = dq_da / div_bottom

        # calculate dq/dc
        dq_dc = -b_param * dq_db / div_bottom

        # calculate dq/dk
        dq_dk = dq_dc * x_t * e_kxf

        # calculate dq/df
        dq_df = dq_dc * e_kxf

        # return gradient
        return np.array([
            dq_da, dq_db, dq_dc, dq_dk, dq_df
        ], dtype=np.float64)
