
from offline_auto_regression_handler import OfflineAutoRegressionHandler


class OnlineAutoRegressionHandler:

    def __init__(self, logger, train_values, window_size):
        self.__logger = logger
        self.__window_size = window_size
        self.__history = list(train_values)
        self.__initiate_model()
        # todo: add lag size- how much samples back we should learn on

    def __initiate_model(self):
        # create offline model with history
        self.__model = OfflineAutoRegressionHandler(self.__logger, self.__window_size)
        self.__model.learn_model_params([self.__history], should_print_params=False)
        self.__logger.log(self.__model.get_params())

    def predict_next(self):
        return self.__model.predict_using_learned_params(self.__history, 1)[0]

    def update_predictor(self, new_value):
        self.__history.append(new_value)
        # self.__initiate_model()
