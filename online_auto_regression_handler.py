
from offline_auto_regression_handler import OfflineAutoRegressionHandler


class OnlineAutoRegressionHandler:

    def __init__(self, logger, p, lag_size):
        self.__logger = logger

        # p is model order
        self.__p = p
        self.__lag_size = lag_size

        # init empty members
        self.__history = None
        self.__model = None

    def init(self, train_set=None):
        # store train set
        if train_set is not None:
            self.__history = list(train_set)

        # create offline model with history
        self.__model = OfflineAutoRegressionHandler(self.__logger, self.__p)
        self.__model.learn_model_params(self.__history, lag=self.__lag_size, should_print_params=False)

    def predict_next(self):
        return self.__model.predict_using_learned_params(self.__history, 1)[0]

    def update_predictor(self, new_value):
        self.__history.append(new_value)
        # todo: check option to only update existing model params
        self.init()
