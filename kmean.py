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


# K-Means implementation
def k_means(dataset, k, r):
    # init result lists
    min_errors = math.inf
    best_centroids, best_clusters, best_predicted = [], [], []
    headers = dataset.columns.values
    for _ in range(r):
        # init k centroids randomly
        centroids = dataset.sample(k).values
        # clustering data
        clusters = []
        iteration = 1
        print(_, iteration)
        while True:
            # init clusters
            clusters.clear()
            for _ in range(k):
                clusters.append(pd.DataFrame(columns=headers))
            predicted = np.zeros(shape=dataset.shape[0])
            # update clusters
            for ith in range(len(dataset.values)):
                min_distance_square = math.inf
                min_centroid_pos = -1
                for pos in range(k):
                    distance_sq = distance_square(dataset.values[ith], centroids[pos])
                    if distance_sq <= min_distance_square:
                        min_distance_square = distance_sq
                        min_centroid_pos = pos
                clusters[min_centroid_pos].loc[len(clusters[min_centroid_pos])] = dataset.values[ith]
                predicted[ith] = min_centroid_pos
            # update centroids, save old centroids
            old_centroids = copy.deepcopy(centroids)
            for ik in range(k):
                for j in range(len(headers)):
                    centroids[ik][j] = clusters[ik][headers[j]].sum() / len(clusters[ik])
            # check stopping condition
            if is_not_change(old_centroids, centroids):
                break
            else:
                iteration += 1
                print(_, iteration)
        # calculate objective func and save values
        sum_errors = 0
        for i in range(k):
            for x in clusters[i].values:
                sum_errors += distance_square(x, centroids[i])
        if sum_errors <= min_errors:
            min_errors = sum_errors
            best_centroids = copy.deepcopy(centroids)
            best_clusters = [clusters[_].copy() for _ in range(k)]
            best_predicted = copy.deepcopy(predicted)
    return min_errors, best_centroids, best_clusters, best_predicted


# check accuracy
def accuracy(actual, predict):
    # case 0: cluster 0 = 'no', cluster 1 = 'yes'
    accuracies = [accuracy_score(actual, predict)]
    # case 1: cluster 1 = 'no', cluster 0 = 'yes'
    new_predict = np.array(np.ones(shape=len(predict)) - predict)
    accuracies.append(accuracy_score(actual, new_predict.tolist()))
    return np.max(np.array(accuracies))


if __name__ == "__main__":
    raw_data = read_data("./bank-additional-full.csv")
    preprocess(raw_data)
    data = raw_data.drop(columns='y')
    results = k_means(data, k_def, r_def)
    print("Accuracy:", accuracy(raw_data['y'].tolist(), results[3]))
