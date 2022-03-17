# ==================================================================================================================== #
# Contributors: Tri Le + Phuoc Nguyen
# Description: Naive Bayes
# Filename: naive_bayes.py
# ==================================================================================================================== #


# Dependencies
import numpy as np
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


def read_file(filename):
    try:
        # Read raw dataset
        raw_data = np.loadtxt(filename, delimiter=",")
        return raw_data

    # File not found
    except FileNotFoundError:
        print("Cannot read files. Please check your dataset!")
        return None


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
    def train(data_set, target_set):
        result = {}
        good_prob = good_probability(target_set)

        print("======== Training Set ========")
        print("good probability = ", round(good_prob, 3))
        print("not good probability = ", round(1 - good_prob, 3))
        print("==============================")

        result[GOOD] = np.log(good_prob)
        result[NOT_GOOD] = np.log(1 - good_prob)

        good_index_set = np.where(target_set == 1)
        not_good_index_set = np.where(target_set == 0)

        good_set = np.array([data_set[i, :] for i in good_index_set[0]])
        not_good_set = np.array([data_set[i, :] for i in not_good_index_set[0]])

        result[GOOD_STATISTIC] = [feature_statistic(
            good_set, col) for col in range(0, good_set.shape[1])]
        
        result[NOT_GOOD_STATISTIC] = [feature_statistic(
            not_good_set, col) for col in range(0, not_good_set.shape[1])]

        return result 

    #Step 1: Divide training set into two parts with the ratio to A=7: B=3 
    #Step 2: Run training for A  => result_of_training
    #Step 3  Run testing for B and remove all features that lower than the average
    #Step 4: Return the result with ignoring columns for the test

    pass



def predictor(data_input, training_result,ignoring_column):
    #Don't use column in ignoring column for testing 

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

        feature_prediction[0, i] = 1 \
            if good_norm + training_result[GOOD] > not_good_norm + training_result[NOT_GOOD] \
            else 0

    return 1 if good_result > not_good else 0, feature_prediction


def create_data_set(draw_data):
    num_of_col = draw_data.shape[1]

    # get even rows
    train_set = draw_data[1::2, 0:num_of_col - 1]

    # get even row of the last column
    train_target_set = draw_data[1::2, num_of_col - 1:num_of_col]

    # get even rows
    test_set = draw_data[::2, 0:num_of_col - 1]

    # get odd row of last column
    test_target_set = draw_data[::2, num_of_col - 1:num_of_col]

    return train_set, train_target_set.astype(int), test_set, test_target_set.astype(int)


def train_and_test(train_set, train_target_set, test_set, test_target_set):

    training_result = training(train_set, train_target_set)

    print("== The Prediction for Testing set======")
    confusion_matrix = np.full((2, 2, test_set.shape[1] + 1), 0).astype(int)

    for i in range(0, test_set.shape[0]):

        predict, feature_prediction = predictor(
            test_set[i:i + 1, :], training_result)

        confusion_matrix[predict, test_target_set[i], test_set.shape[1]] += 1

        for j in range(0, test_set.shape[1]):
            
            confusion_matrix[feature_prediction[0, j],
                             test_target_set[i], j] += 1

    accuracy = None

    for i in range(0, test_set.shape[1] + 1):
        cfm = confusion_matrix[:, :, i]
        diagonal = np.diagonal(cfm)
        sum_correct_case = np.sum(diagonal)
        sum_all = np.sum(cfm)

        accuracy = 0 if sum_all == 0 else (sum_correct_case / sum_all * 100)

        if i < test_set.shape[1]:
            print(
                "- The accuracy of the feature {0}'s prediction = {1}".format(i + 1, round(accuracy, 3)))

    print("=======================================")
    print("The Confusion Matrix of Testing set:")
    print(confusion_matrix[:, :, test_set.shape[1]])
    print("## Final accuracy  = ", round(accuracy, 3))


if __name__ == "__main__":
    # PATH FILES
    DATA_PATH = "./processed_data.csv"

    # Read pre-processed file
    data = read_file(DATA_PATH)

    # Perform Naive Bayes algorithm
    if data is not None:
        training_set, training_target_set, testing_set, testing_target_set = create_data_set(
            data)
        train_and_test(training_set, training_target_set,
                       testing_set, testing_target_set)
