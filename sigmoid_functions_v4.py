
import numpy as np


class SigmoidV4:
    """
    minimize MSE based loss function, with projection

    fitted function:
                          B
            y = A + --------------
                    C + e^(D*x + F)

    params ranges:
    -   B, C > 0 (E R+)
    -   D < 0 (E R-)
    -   A is set using first sample

    loss function:

             Loss = (y - y_p) ^ 2

    """

    INITIAL_W_VECTOR = \
        np.array([
            0.49937523,
            0.89339831,
            0.08601164,
            -0.22586464,
            0.43639608
        ])

    MAX_RAND_VALUE = 1e1
    EPSILON = 1e-10

    @staticmethod
    def get_initial_w():
        return np.array(SigmoidV4.INITIAL_W_VECTOR)

    @staticmethod
    def get_random_w(first_sample):
        params = np.random.random(5) * SigmoidV4.MAX_RAND_VALUE

        # fix b and c
        params[1:3] += SigmoidV4.EPSILON

        # fix d
        params[3] *= -1

        # # calculate a
        # params[0] = first_sample - (params[1] / (params[2] + np.exp(params[3] + params[4])))

        return params

    @staticmethod
    def get_prediction(x_value, w_vector, should_round=False):
        # extract params
        a_param, b_param, c_param, d_param, f_param = w_vector

        # return sigmoid value
        raw_value = a_param + b_param / (c_param + np.exp(d_param * x_value + f_param))

        # return value
        return round(raw_value) if should_round else raw_value

    @staticmethod
    def get_mean_error_rate(dataset, w_vector):
        # calculate mean error rate
        error_rates_total = 0.0
        for x_t in dataset.keys():
            y_t = dataset[x_t]
            function_value = SigmoidV4.get_prediction(x_t, w_vector)
            error_rates_total += abs((y_t - function_value) / y_t)

        # return mean error rate
        return error_rates_total / len(dataset)

    @staticmethod
    def project(w_vector):
        # extract params
        a_param, b_param, c_param, d_param, f_param = w_vector

        # project b and c to defined range
        b_param = max(SigmoidV4.EPSILON, b_param)
        c_param = max(SigmoidV4.EPSILON, c_param)

        return np.array([a_param, b_param, c_param, d_param, f_param], dtype=float)

    @staticmethod
    def get_gradient(x_t, y_t, w_vector):
        # extract params
        a_param, b_param, c_param, d_param, f_param = w_vector

        # calculate common values
        e_kxf = np.exp(d_param * x_t + f_param)
        div_bottom = c_param + e_kxf
        base_value = 2 * (y_t - a_param - (b_param / div_bottom))
        y2 = np.square(y_t)

        # calculate dq/db
        dq_da = -base_value / y2
        dq_db = dq_da / div_bottom

        # calculate dq/dc
        dq_dc = -b_param * dq_db / div_bottom

        # calculate dq/dd
        dq_dd = dq_dc * x_t * e_kxf

        # calculate dq/df
        dq_df = dq_dc * e_kxf

        # return gradient
        return np.array([
            dq_da, dq_db, dq_dc, dq_dd, dq_df
        ], dtype=np.float64)
