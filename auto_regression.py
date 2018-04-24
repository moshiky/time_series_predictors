
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error


FILE_PATH = r'datasets/citations.csv'
PREDICTIONS = 30


def offline_test():
    """
    not learns online!
    :return:
    """
    # load dataset from file
    series = Series.from_csv(FILE_PATH, header=0)

    # split dataset to train and test batches
    all_samples = series.values
    train, test = all_samples[:-PREDICTIONS], all_samples[-PREDICTIONS:]

    # train AutoRegression
    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    # print('window=' + str(window))
    model_params = model_fit.params

    # walk forward over time steps in test
    history = list(train[-window:])
    # ? history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
        lag = history[-window:]
        y_hat = model_params[0]
        for d in range(window):
            y_hat += model_params[d + 1] * lag[window - d - 1]
        obs = test[t]
        predictions.append(y_hat)
        history.append(obs)
        print('predicted=%f, expected=%f' % (y_hat, obs))
    #
    error = mean_squared_error(test, predictions)
    print('Test MSE = %.3f' % error)

    # return test and predictions
    return test, predictions


def load_csv(csv_file_path):
    # load dataset from file
    series = Series.from_csv(csv_file_path, header=0)
    return series.values


def split_samples(samples):
    return samples[:-PREDICTIONS], samples[-PREDICTIONS:]


def main(csv_file_path):
    # load csv
    all_samples = load_csv(csv_file_path)

    # split to test and train
    train, test = split_samples(all_samples)

    # set history=train (duplicate train)
    history = list(train)

    # for i < number_of_predictions
    prediction_list = list()
    for prediction_index in range(PREDICTIONS):
        # train model on history
        model = AR(history)
        model_fit = model.fit()

        # predict next value and concatenate to prediction list
        predictions = model_fit.predict_using_learned_params(start=len(history), end=len(history), dynamic=False)
        prediction_list.append(predictions[0])

        # concatenate test[i] to history
        history.append(test[prediction_index])
        print('predicted={pred_value}, expected={real_value}'.format(
            pred_value=prediction_list[-1], real_value=test[prediction_index])
        )

        # keep history to same length
        history = history[1:]

    # calculate MSE with test and prediction lists
    error = mean_squared_error(test, prediction_list)
    print('Test MSE = {mse_value}'.format(mse_value=error))

    # return test and predictions
    return test, prediction_list


if __name__ == '__main__':
    # test offline method
    print('-- offline')
    test_values, offline_predictions = offline_test()

    # test online method
    print('-- online')
    _, online_predictions = main(FILE_PATH)

    # plot both
    pyplot.plot(test_values, color='red', marker='d')
    pyplot.plot(offline_predictions, color='blue', marker='s')
    pyplot.plot(online_predictions, color='green', marker='o')
    pyplot.show()
