# ==================================================================================================================== #
# Contributors: Tri Le + Phuoc Nguyen
# Description: Naive Bayes
# Filename: naive_bayes.py
# ==================================================================================================================== #
# Dependencies
import math
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import math

MEAN = 0
STANDARD_DEVIATION = 1

GOOD = 1
NOTGOOD = 0

GOOD_STATICTIS = 2
NOTGOOD_STATICTIS = 3

SQRT_2PI = math.sqrt(2.0 * math.pi)

dx_1 = 0.0001
dx_2 = 1e-308


def good_probability(target_set):
    num_of_good = np.count_nonzero(target_set)
    num_of_row = target_set.shape[0]

    return num_of_good/num_of_row


def feature_statistic(data_set, column):
    result = {}

    result[MEAN] = np.mean(data_set[:, column], dtype=np.float64)

    standard = np.std(data_set[:, column], dtype=np.float64)

    result[STANDARD_DEVIATION] = (standard == 0).astype(int) * dx_1 + standard

    return result


def log_normal_dist(data, mean, sdd):
    prob_density = (1/(SQRT_2PI*sdd)) * np.exp(-0.5 * ((data-mean)/sdd)**2)

    return np.log(dx_2 if prob_density == 0 else prob_density)


def training(data_set, target_set):

    result = {}
    good_prob = good_probability(target_set)


    result[GOOD] = np.log(good_prob)
    result[NOTGOOD] = np.log(1 - good_prob)

    good_index_set = np.where(target_set == 1)
    notgood_index_set = np.where(target_set == 0)

    good_set = np.array([data_set[i, :] for i in good_index_set[0]])
    notgood_set = np.array([data_set[i, :] for i in notgood_index_set[0]])

    result[GOOD_STATICTIS] = [feature_statistic(good_set, col)
                              for col in range(0, good_set.shape[1])]

    result[NOTGOOD_STATICTIS] = [feature_statistic(notgood_set, col)
                                  for col in range(0, notgood_set.shape[1])]

    return result


def predictor(data,  training_result):

    good_result = training_result[GOOD]
    notgood = training_result[NOTGOOD]

    feature_prediction = np.full((1, data.shape[1]), 0).astype(int)

    for i in range(0, data.shape[1]):
        good_norm = log_normal_dist(data[0, i],  training_result[GOOD_STATICTIS]
                                    [i][MEAN],  training_result[GOOD_STATICTIS][i][STANDARD_DEVIATION])

        notgood_norm = log_normal_dist(data[0, i],  training_result[NOTGOOD_STATICTIS]
                                      [i][MEAN],  training_result[NOTGOOD_STATICTIS][i][STANDARD_DEVIATION])

        good_result += good_norm
        notgood += notgood_norm

        feature_prediction[0, i] = 1 if good_norm + training_result[GOOD] > notgood_norm + \
            training_result[NOTGOOD] else 0

    return 1 if good_result > notgood else 0, feature_prediction


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

    data = readfile(DATA_PATH)
    print(data.shape)
