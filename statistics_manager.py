
import time
import os
from singleton import Singleton
import multiprocessing


class StatisticsManager(object, metaclass=Singleton):

    def __init__(self):
        self.__metrics_storage = dict()
        # self.__metrics_storage_lock = multiprocessing.Lock()

    def add_metrics(self, metrics_dict):
        # with self.__metrics_storage_lock:
        for metric_name in metrics_dict.keys():
            if metric_name not in self.__metrics_storage.keys():
                self.__metrics_storage[metric_name] = list()
            self.__metrics_storage[metric_name].append(metrics_dict[metric_name])

    def get_average_metrics(self):
        # with self.__metrics_storage_lock:
        return {
            metric_name: sum(self.__metrics_storage[metric_name]) / len(self.__metrics_storage[metric_name])
            for metric_name in self.__metrics_storage
        }

    @staticmethod
    def write_to_csv(metrics_dict):
        metrics_file_path = r'output\metrics.csv'

        line = [
            str(metrics_dict[key_name]) for key_name in sorted(metrics_dict.keys())
        ]

        if not os.path.exists(metrics_file_path):
            with open(metrics_file_path, 'wt') as metrics_file:
                metrics_file.write('timestamp,' + ','.join(list(sorted(metrics_dict.keys()))) + '\n')

        with open(metrics_file_path, 'at') as metrics_file:
            metrics_file.write(str(time.time()) + ',' + ','.join(line) + '\n')

    @staticmethod
    def get_avg_from_csv():
        # read file
        with open(r'output\metrics.csv', 'rt') as metrics_file:
            file_content = metrics_file.read()

        # split to lines
        file_lines = file_content.split('\n')

        # read headers and create storage
        metrics = dict()
        headers = list()
        for column_name in file_lines[0].split(',')[1:]:
            metrics[column_name] = list()
            headers.append(column_name)

        # extract column values
        for line in file_lines[1:]:
            line_values = line.split(',')[1:]
            for column_index in range(len(line_values)):
                metric_value = float(line_values[column_index])
                if metric_value > 0:
                    metrics[headers[column_index]].append(float(line_values[column_index]))

        # return average
        return {
            key_name: sum(metrics[key_name]) / len(metrics[key_name]) for key_name in sorted(headers)
        }
