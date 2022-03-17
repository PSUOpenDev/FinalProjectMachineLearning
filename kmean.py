import numpy as np
import pandas as pd
import copy
import math
from sklearn.metrics import accuracy_score


k_def = 2
r_def = 10
epsilon = 0.0025


# Read data from file
def read_data(filename):
    return pd.read_table(filename, delimiter=';', engine='python')


# Pre-process data
def preprocess(dataset):
    keys = [[],
            ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar",
             "self-employed", "retired", "technician", "services"],
            ["married", "divorced", "single", "unknown"],
            ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course",
             "university.degree", "unknown"],
            ["no", "yes", "unknown"],
            ["no", "yes", "unknown"],
            ["no", "yes", "unknown"],
            ["telephone", "cellular"],
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            ["mon", "tue", "wed", "thu", "fri"],
            [], [], [], [],
            ["failure", "nonexistent", "success"],
            [], [], [], [], [],
            ["no", "yes"]]
    # replace strings with values
    columns = dataset.columns.values
    for i in range(dataset.shape[1]):
        for j in range(len(keys[i])):
            dataset[columns[i]] = dataset[columns[i]].replace([keys[i][j]], j)
    # save to file
    # dataset.to_csv("./new_processed_data.csv", index=False)
    # normalized data to [0, 1]
    for col in dataset.columns:
        dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())


# Compute distance square
def distance_square(a, b):
    return np.sum((a - b) ** 2)


# Check equal arrays
def is_not_change(a, b):
    if len(a) == 0 or len(b) == 0:
        return False
    sub = np.abs(np.array(a - b))
    if np.max(sub) <= epsilon:
        return True
    else:
        return False


# Train with K-Means
def k_means_train(dataset, k, r):
    print("Starting K-Means with", k, "clusters")
    # init result lists
    min_errors = math.inf
    best_centroids, best_clusters, best_predicted = [], [], []
    headers = dataset.columns.values
    for ir in range(r):
        # init k centroids randomly
        centroids_list = dataset.sample(k).values
        # clustering data
        iteration = 1
        print("Random initialization:", ir + 1)
        print("Iteration:", iteration, end=' ')
        while True:
            # init clusters
            clusters_list = [pd.DataFrame(columns=headers) for _ in range(k)]
            predicted = np.zeros(shape=dataset.shape[0])
            # update clusters
            for ith in range(len(dataset.values)):
                min_distance_square = math.inf
                min_centroid_pos = -1
                for pos in range(k):
                    distance_sq = distance_square(dataset.values[ith], centroids_list[pos])
                    if distance_sq <= min_distance_square:
                        min_distance_square = distance_sq
                        min_centroid_pos = pos
                clusters_list[min_centroid_pos].loc[len(clusters_list[min_centroid_pos])] = dataset.values[ith]
                predicted[ith] = min_centroid_pos
            # update centroids, save old centroids
            old_centroids = copy.deepcopy(centroids_list)
            for i in range(k):
                for j in range(len(headers)):
                    centroids_list[i][j] = clusters_list[i][headers[j]].sum() / len(clusters_list[i])
            # check stopping condition
            if is_not_change(old_centroids, centroids_list):
                print("done...")
                break
            else:
                iteration += 1
                print("done...")
                print("Iteration:", iteration, end=' ')
        # calculate objective func and save values
        sum_errors = 0
        for i in range(k):
            for point in clusters_list[i].values:
                sum_errors += distance_square(point, centroids_list[i])
        if sum_errors <= min_errors:
            min_errors = sum_errors
            best_centroids = copy.deepcopy(centroids_list)
            best_clusters = [clusters_list[i].copy() for i in range(k)]
            best_predicted = copy.deepcopy(predicted)
    return min_errors, best_centroids, best_clusters, best_predicted


# Test the result after training with K-Means
def k_means_test(centroids_list, testset, k):
    predicted = np.zeros(shape=testset.shape[0])
    for ith in range(len(testset.values)):
        min_distance_square = math.inf
        min_centroid_pos = -1
        for pos in range(k):
            distance_sq = distance_square(testset.values[ith], centroids_list[pos])
            if distance_sq <= min_distance_square:
                min_distance_square = distance_sq
                min_centroid_pos = pos
        predicted[ith] = min_centroid_pos
    return predicted


# check accuracy
def accuracy(actual, predicted, predict_case=None):
    # case 0: cluster 0 = 'no', cluster 1 = 'yes'
    accuracies = [accuracy_score(actual, predicted)]
    # case 1: cluster 1 = 'no', cluster 0 = 'yes'
    new_predict = np.array(np.ones(shape=len(predicted)) - predicted)
    accuracies.append(accuracy_score(actual, new_predict.tolist()))
    if predict_case is None:
        # since we dont know which cluster is yes and
        # which is no, we consider two cases below:
        # so we return the highest accuracy of the two
        return np.max(np.array(accuracies)), np.argmax(np.array(accuracies))
    else:
        return accuracies[predict_case]


if __name__ == "__main__":
    # read and preprocess data
    raw_data = read_data("./bank-additional-full.csv")
    preprocess(raw_data)
    # split data for testing and training
    test_set = raw_data.sample(raw_data.shape[0] % 10000)
    test_data = test_set.drop(columns='y')
    train_set = raw_data.drop(test_set.index)
    train_data = train_set.drop(columns='y')
    # training
    errors, centroids, clusters, train_predict = k_means_train(train_data, k_def, r_def)
    train_accuracy, case = accuracy(train_set['y'].tolist(), train_predict)
    print("Train Accuracy:", train_accuracy)
    # testing
    test_predict = k_means_test(centroids, test_data, k_def)
    test_accuracy = accuracy(test_set['y'].tolist(), test_predict, case)
    print("Test Accuracy:", test_accuracy)
