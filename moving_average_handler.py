
class MovingAverageHandler:

    def __init__(self, logger, train_values, window_size=1):
        self.__logger = logger
        self.__history = list(train_values)
        self.__window_size = window_size

    def predict_next(self):
        return sum(self.__history[-self.__window_size:]) / float(self.__window_size)

    def update_predictor(self, new_values):
        self.__history += new_values
