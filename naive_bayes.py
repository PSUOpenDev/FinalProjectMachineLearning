# ==================================================================================================================== #
# Contributors: Tri Le + Phuoc Nguyen
# Description: Naive Bayes
# Filename: naive_bayes.py
# ==================================================================================================================== #


# Dependencies
import itertools
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy

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


def readfile(filename):
    try:
        # Read raw dataset
        raw_data = np.loadtxt(filename, delimiter=",")
        # Raw training and testing data
        train_data = raw_data[:, :-1]
        # Target class
        target_class = raw_data[:, -1]
        return train_data, target_class

    # File not found
    except FileNotFoundError:
        print("Cannot read files. Please check your dataset!")
        return None


def calculate_accuracy(conf_matrix):
    diagonal = np.diagonal(conf_matrix)
    sum_correct_case = np.sum(diagonal)
    sum_all = np.sum(conf_matrix)

    accuracy = 0 if sum_all == 0 else (sum_correct_case / sum_all * 100)
    return accuracy


def get_dataset(filename):
    raw, target = readfile(filename)
    list_training, list_testing, target_training, target_testing = train_test_split(
        raw, target, test_size=0.5)
    return list_training, target_training, list_testing, target_testing


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
    def train(data_training, target_set_training):
        result = {}
        good_prob = good_probability(target_set_training)

        print("======== Training Set ========")
        print("Good probability = ", round(good_prob, 3))
        print("Not good probability = ", round(1 - good_prob, 3))
        print("==============================")

        result[GOOD] = np.log(good_prob)
        result[NOT_GOOD] = np.log(1 - good_prob)

        good_index_set = np.where(target_set_training == 1)
        not_good_index_set = np.where(target_set_training == 0)

        good_set = np.array([data_training[i, :] for i in good_index_set[0]])
        not_good_set = np.array([data_training[i, :]
                                for i in not_good_index_set[0]])

        result[GOOD_STATISTIC] = [feature_statistic(
            good_set, col) for col in range(0, good_set.shape[1])]
        result[NOT_GOOD_STATISTIC] = [feature_statistic(
            not_good_set, col) for col in range(0, not_good_set.shape[1])]

        return result

    def test(test_set, test_target_set, training_result):
        ignore_col = dict()
        prob_dict = list()

        confusion_matrix = np.full(
            (2, 2, test_set.shape[0] + 1), 0).astype(int)

        for i in range(0, test_set.shape[0]):
            predict, feature_prediction = predictor(
                test_set[i, :], training_result, None)

            # confusion matrix for final
            confusion_matrix[predict, int(
                test_target_set[i]), test_set.shape[1]] += 1

            # confusion matrix for every column
            for j in range(0, test_set.shape[1]):
                confusion_matrix[feature_prediction[j],
                                 int(test_target_set[i]), j] += 1

        accuracy = None

        for i in range(0, test_set.shape[1] + 1):
            cfm = confusion_matrix[:, :, i]
            diagonal = np.diagonal(cfm)
            sum_correct_case = np.sum(diagonal)
            sum_all = np.sum(cfm)

            accuracy = 0 if sum_all == 0 else (
                sum_correct_case / sum_all * 100)

            if i < test_set.shape[1]:
                prob_dict.append((i, accuracy))

        # Calculate the average probability
        prob_dict = np.array(prob_dict)
        avg_prob = np.array([p[1] for p in prob_dict]).mean()

        for p in prob_dict:
            if p[1] < avg_prob:
                ignore_col[int(p[0])] = True
        return ignore_col

    # Step 1: Divide training set into two parts with the ratio to A=7: B=3
    train_data, test_data, train_data_target, test_data_target = train_test_split(
        data_set,
        target_set,
        test_size=0.3,
        random_state=42
    )

    # Step 2: Run training for A  => result_of_training
    result_of_training = train(train_data, train_data_target)

    # Step 3  Run testing for B and remove all features that lower than the average
    ignore_column = test(test_data, test_data_target, result_of_training)

    # Step 4: Return the result with ignoring columns for the test
    return result_of_training, ignore_column


def predictor(data_input, training_result, ignoring_column, is_feature_predict=True):

    good_result = training_result[GOOD]
    not_good = training_result[NOT_GOOD]

    if is_feature_predict:
        feature_prediction = np.zeros_like(data_input, dtype=int)
    else:
        feature_prediction = None

    for i in range(0, data_input.shape[0]):

        # Don't use column in ignoring column for testing
        if ignoring_column is None or ((ignoring_column is not None) and (i not in ignoring_column)):
            good_norm = log_normal_dist(
                data_input[i],
                training_result[GOOD_STATISTIC][i][MEAN],
                training_result[GOOD_STATISTIC][i][STANDARD_DEVIATION]
            )

            not_good_norm = log_normal_dist(
                data_input[i],
                training_result[NOT_GOOD_STATISTIC][i][MEAN],
                training_result[NOT_GOOD_STATISTIC][i][STANDARD_DEVIATION]
            )

            good_result += good_norm
            not_good += not_good_norm

            if is_feature_predict:
                feature_prediction[i] = 1 if good_norm + \
                    training_result[GOOD] > not_good_norm + training_result[NOT_GOOD] else 0

    if is_feature_predict:
        return 1 if good_result > not_good else 0, feature_prediction

    return 1 if good_result > not_good else 0


def create_data_set(draw_data):
    num_of_col = draw_data.shape[1]

    # get odd rows
    train_set = draw_data[::2, 0:num_of_col - 1]

    # get odd row of the last column
    train_target_set = draw_data[::2, num_of_col - 1:num_of_col]

    # get even rows
    test_set = draw_data[1::2, 0:num_of_col - 1]

    # get even row of last column
    test_target_set = draw_data[1::2, num_of_col - 1:num_of_col]

    return train_set, train_target_set.astype(int), test_set, test_target_set.astype(int)


def train_and_test(train_set, train_target_set, test_set, test_target_set):
    training_result, ignore_col = training(train_set, train_target_set)

    confusion_matrix_training = np.full(
        (2, 2), 0).astype(int)

    for i in range(0, train_set.shape[0]):
        predict = predictor(
            train_set[i, :], training_result, ignore_col, is_feature_predict=False)
        confusion_matrix_training[predict, int(train_target_set[i])] += 1

    accuracy = calculate_accuracy(confusion_matrix_training)

    print("=======================================")
    print("The Confusion Matrix of Training set:")
    accuracy_train = round(accuracy, 3)

    print("##The accuracy of Training set = ", accuracy_train)

    confusion_matrix_testing = np.full(
        (2, 2), 0).astype(int)

    for i in range(0, test_set.shape[0]):

        predict = predictor(
            test_set[i, :], training_result, ignore_col, is_feature_predict=False)

        confusion_matrix_testing[predict, int(
            test_target_set[i])] += 1

    accuracy = calculate_accuracy(confusion_matrix_testing)


    print("=======================================")
    print("The Confusion Matrix of Testing set:")
    print(confusion_matrix_testing)

    accuracy_test = round(accuracy, 3)
    print("##The accuracy of Testing set = ", accuracy_test)

    return accuracy_train, accuracy_test, confusion_matrix_training, confusion_matrix_testing


def plotting_confusion_matrix(title, cm):
    plt.clf()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.OrRd)
    plt.title(title)
    plt.colorbar()
    threshold = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    plt.show()


def plotting_accuracy(acc1, acc2):
    plt.clf()
    plt.title('Compare accuracy of training and accuracy of testing')
    plt.xlabel('Accuracy')
    plt.ylabel('Accuracy Value')
    x_axis = ['For Training', 'For Testing']
    y_axis = [acc1, acc2]
    plt.bar(x_axis, y_axis)
    plt.show()


def main(filename, num_of_run):
    best_train_accuracy = 0
    best_test_accuracy = 0
    best_cm_train = []
    best_cm_test = []
    for i in range(num_of_run):
        print("-------------------------------------------------------------------------------------")
        print("Running: " + str(i))
        print("-------------------------------------------------------------------------------------")

        training_list, training_target, testing_list, testing_target = get_dataset(
            filename)
        accuracy_train, accuracy_test, cm_train, cm_test = train_and_test(
            training_list,
            training_target,
            testing_list,
            testing_target
        )

        if accuracy_test > best_test_accuracy:
            best_train_accuracy = accuracy_train
            best_test_accuracy = accuracy_test
            best_cm_train = copy.deepcopy(cm_train)
            best_cm_test = copy.deepcopy(cm_test)

    print("--------------------------------------------------------------------------------------------")
    print("Best testing accuracy: ", best_test_accuracy)
    plotting_confusion_matrix("Confusion matrix for training", best_cm_train)
    plotting_confusion_matrix("Confusion matrix for testing", best_cm_test)
    plotting_accuracy(best_train_accuracy, best_test_accuracy)


if __name__ == "__main__":
    # PATH FILES
    DATA_PATH = "./processing_dataset.csv"
    main(DATA_PATH, 10)
