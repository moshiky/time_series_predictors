
import csv
from random import shuffle
from consts import Consts


def parse_csv(file_path, should_shuffle=True):
    # read file
    with open(file_path, 'rt') as file_handle:
        reader = csv.reader(file_handle)
        file_lines = list(reader)

    record_list = [[int(x) for x in l] for l in file_lines if len(l) > Consts.MIN_RECORD_LENGTH]
    if should_shuffle:
        shuffle(record_list)
    return record_list


def split_samples(samples, test_percentage=Consts.TEST_SIZE_FACTOR):
    test_samples_count = int(len(samples) * test_percentage)
    return samples[:-test_samples_count], samples[-test_samples_count:]
