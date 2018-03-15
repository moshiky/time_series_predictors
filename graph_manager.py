
import time
import os
from matplotlib import pyplot
import multiprocessing
from singleton import Singleton


class GraphManager(object, metaclass=Singleton):

    def __init__(self, output_folder_path):
        self.__output_folder_path = output_folder_path
        # self.__pyplot_lock = multiprocessing.Lock()
        # self.__plot_index = multiprocessing.Value('i', 0)

    def plot_graph_and_prediction(self, original_series, predictions, prediction_start_index, file_name='graph',
                                  fitted_values=None, store=True, show=False):

        # with self.__pyplot_lock:
        # clear plot area
        pyplot.clf()
        pyplot.grid(which='both')

        # plot original series
        pyplot.plot(
            list(range(1, len(original_series) + 1)),
            original_series,
            '-r'
        )

        # plot prediction
        if fitted_values is not None and predictions is not None:
            pyplot.plot(
                list(range(1, len(fitted_values) + 1)), fitted_values, '.b'
            )
            pyplot.plot(
                list(range(prediction_start_index, prediction_start_index + len(predictions))),
                predictions,
                '.g'
            )

        elif predictions is not None:
            pyplot.plot(
                list(range(prediction_start_index, prediction_start_index + len(predictions))),
                predictions,
                '.b'
            )

        # show plotted graph
        if show:
            pyplot.show()

        # store plotted graph
        if store:
            # plot_index = self.__plot_index.value
            # self.__plot_index.value += 1
            # plot_index = time.time()

            # pyplot.savefig(os.path.join(self.__output_folder_path, '{plot_index}_{file_name}.png'
            #                             .format(plot_index=plot_index, file_name=file_name)))
            pyplot.savefig(os.path.join(self.__output_folder_path, '{file_name}.png'
                                        .format(file_name=file_name)))
