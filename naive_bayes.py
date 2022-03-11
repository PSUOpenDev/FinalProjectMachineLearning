# ==================================================================================================================== #
# Contributors: Tri Le + Phuoc Nguyen
# Description: Naive Bayes
# Filename: naive_bayes.py
# ==================================================================================================================== #
# Dependencies
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import math


# GLOBAL VARIABLES
MEAN = 0
STANDARD_DEVIATION = 1
GOOD = 1
NOT_GOOD = 0
GOOD_STATISTIC = 2
NOT_GOOD_STATISTIC = 3
SQRT_2PI = math.sqrt(2.0 * math.pi)

# THRESHOLD
dx_1 = 0.0001
dx_2 = 1e-308


def good_probability(target_set):
    num_of_good = np.count_nonzero(target_set)
    num_of_row = target_set.shape[0]
    return num_of_good / num_of_row


def feature_statistic(data_set, column):
    result = {}

    result[MEAN] = np.mean(data_set[:, column], dtype=np.float64)
    standard = np.std(data_set[:, column], dtype=np.float64)
    result[STANDARD_DEVIATION] = (standard == 0).astype(int) * dx_1 + standard
    return result


def probability_density(x, mean, std):
    exponent = math.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent


def log_normal_dist(data_input, mean, sdd):
    prob_density = probability_density(data_input, mean, sdd)
    return np.log(dx_2 if prob_density == 0 else prob_density)


def training(data_set, target_set):
    result = {}

    good_prob = good_probability(target_set)

    result[GOOD] = np.log(good_prob)
    result[NOT_GOOD] = np.log(1 - good_prob)

    good_index_set = np.where(target_set == 1)
    not_good_index_set = np.where(target_set == 0)

    good_set = np.array([data_set[i, :] for i in good_index_set[0]])
    not_good_set = np.array([data_set[i, :] for i in not_good_index_set[0]])

    result[GOOD_STATISTIC] = [feature_statistic(good_set, col) for col in range(0, good_set.shape[1])]
    result[NOT_GOOD_STATISTIC] = [feature_statistic(not_good_set, col) for col in range(0, not_good_set.shape[1])]

    return result


def predictor(data_input, training_result):
    good_result = training_result[GOOD]
    not_good = training_result[NOT_GOOD]

    feature_prediction = np.full((1, data_input.shape[1]), 0).astype(int)

    for i in range(0, data_input.shape[1]):
        good_norm = log_normal_dist(
            data_input[0, i],
            training_result[GOOD_STATISTIC][i][MEAN],
            training_result[GOOD_STATISTIC][i][STANDARD_DEVIATION]
        )

        not_good_norm = log_normal_dist(
            data_input[0, i],
            training_result[NOT_GOOD_STATISTIC][i][MEAN],
            training_result[NOT_GOOD_STATISTIC][i][STANDARD_DEVIATION]
        )

        good_result += good_norm
        not_good += not_good_norm

        feature_prediction[0, i] = 1 if good_norm + training_result[GOOD] > not_good_norm + training_result[NOT_GOOD] \
            else 0

    return 1 if good_result > not_good else 0, feature_prediction


def readfile(filename):
    try:
        # Read raw dataset
        raw_data = np.loadtxt(filename, delimiter=",")
        return raw_data

    # File not found
    except FileNotFoundError:
        print("Cannot read files. Please check your dataset!")
        return None


if __name__ == "__main__":
    # PATH FILES
    DATA_PATH = "./processed_data.csv"

    # Read data
    data = readfile(DATA_PATH)
