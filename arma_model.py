

from statsmodels.tsa.arima_model import ARMA
import numpy as np
from sklearn.metrics import mean_squared_error
import utils


class ARMAModel:

    def __init__(self, logger, p=0, q=0, with_c=False, lag_size=0):
        self.__logger = logger

        # set AR component order
        self.__p = p

        # set MA component order
        self.__q = q

        # set constant setting
        self.__with_c = with_c

        # store lag size settings
        self.__lag_size = lag_size

        # init empty members
        self.__train_set = None
        self.__model = None

    def learn_model_params(self, train_set=None, print_settings=False, start_params=None):
        # store train set
        if train_set is not None:
            self.__train_set = train_set

        # validate data
        if len(set(self.__train_set)) == 1:
            raise Exception("Can't fit model since all history values are the same")

        # create model instance
        model = ARMA(self.__train_set, order=(self.__p, self.__q))

        # fit model
        trend = 'c' if self.__with_c else 'nc'

        solvers = [
            'bfgs', 'powell', 'nm', 'lbfgs', 'newton', 'cg', 'ncg'
        ]
        methods = [
            'css-mle', 'mle', 'css'
        ]
        model_fit = None
        for trans_params_mode in [True, False]:
            for solver in solvers:
                for method in methods:
                    try:
                        model_fit = \
                            model.fit(
                                disp=0,
                                trend=trend,
                                method=method,
                                solver=solver,
                                transparams=trans_params_mode,
                                start_params=start_params
                            )
                        if print_settings:
                            self.__logger.log(
                                'settings: method={method}, solver={solver}, transparams={transparams}'
                                .format(method=method, solver=solver, transparams=trans_params_mode),
                                should_print=False
                            )
                        break

                    except Exception as ex:
                        continue

                if model_fit is not None:
                    break

            if model_fit is not None:
                break
        # self.__logger.log('fitted params: {params}'.format(params=model_fit.params))

        if model_fit is None:
            raise Exception('No settings worked')

        self.__model = model_fit

    def predict_using_learned_params(self, initial_values, number_of_predictions_ahead):
        return self.__model.forecast(steps=number_of_predictions_ahead)[0]

    def update_model(self, new_observations):
        self.__train_set += new_observations
        self.learn_model_params(start_params=self.__model.params)

    def get_params(self):
        return self.__model.params
