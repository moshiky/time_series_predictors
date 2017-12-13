
from statsmodels.tsa.ar_model import AR


class OnlineAutoRegressionHandler:

    def __init__(self, logger, train_values):
        self.__logger = logger
        self.__history = list(train_values)
        self.__initiate_model()

    def __initiate_model(self):
        self.__model = AR(self.__history)
        self.__model_fit = self.__model.fit()

    def predict_next(self):
        return self.__model_fit.predict(start=len(self.__history), end=len(self.__history), dynamic=False)[0]

    def update_predictor(self, new_values):
        self.__history = self.__history[len(new_values):]
        self.__history += new_values
        self.__initiate_model()
