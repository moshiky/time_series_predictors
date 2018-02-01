
import csv
import numpy as np
from random import shuffle
from consts import Consts


def parse_csv(file_path, should_shuffle=True):
    # read file
    with open(file_path, 'rt') as file_handle:
        reader = csv.reader(file_handle)
        file_lines = list(reader)

    record_list = [[float(x) for x in l] for l in file_lines if len(l) > Consts.MIN_RECORD_LENGTH]
    if should_shuffle:
        shuffle(record_list)
    return record_list


def split_list(samples, test_percentage=Consts.TEST_SIZE_FACTOR):
    test_samples_count = int(len(samples) * test_percentage)
    return samples[:-test_samples_count], samples[-test_samples_count:]


def pad_samples(original_samples):
    # find out what is the longest sample length
    max_length = 0
    for sample in original_samples:
        if len(sample) > max_length:
            max_length = len(sample)

    # create np array with each record in length of the max length
    zero_padded_array = np.zeros((len(original_samples), max_length))

    # store original samples in the padded array
    for i in range(len(original_samples)):
        sample_length = len(original_samples[i])
        padded_array_start_index = max_length - sample_length
        for j in range(sample_length):
            zero_padded_array[i][padded_array_start_index + j] = original_samples[i][j]

    # return padded array
    return zero_padded_array
