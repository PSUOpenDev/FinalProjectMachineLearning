import numpy as np
import pandas as pd
import copy
import math


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
            [], [], [], [], [], [],
            ["no", "yes"]]
    columns = dataset.columns.values
    for i in range(dataset.shape[1]):
        for j in range(len(keys[i])):
            dataset[columns[i]] = dataset[columns[i]].replace([keys[i][j]], j)
    # dataset.to_csv("./new_processed_data.csv", index=False)


# Compute distance square
def distance_square(a, b):
    return np.sum((a - b) ** 2)


# Check equal arrays
def is_not_change(a, b):
    if len(a) == 0 or len(b) == 0:
        return False
    sub = np.array(math.fabs(a - b))
    if np.max(sub) <= epsilon:
        return True
    else:
        return False


def k_means(dataset, k, r):
    # init result lists
    min_errors = math.inf
    best_centroids = []
    best_clusters = []
    for _ in range(r):
        # init k centroids randomly
        centroids = dataset.sample(k).values
        # clustering data
        iteration = 1
        while True:
            # init clusters
            clusters = [pd.DataFrame(columns=dataset.columns.values) for _ in range(k)]
            # update clusters
            for datapoint in dataset.values:
                min_distance_square = math.inf
                min_centroid_pos = -1
                for pos in range(k):
                    distance_sq = distance_square(datapoint, centroids[pos])
                    if distance_sq <= min_distance_square:
                        min_distance_square = distance_sq
                        min_centroid_pos = pos
                clusters[min_centroid_pos].loc[len(clusters[min_centroid_pos])] = datapoint
            # update centroids, save old centroids
            old_centroids = copy.deepcopy(centroids)
            for ik in range(k):
                pass
            # check stopping condition
            if is_not_change(old_centroids, centroids):
                break
            else:
                iteration += 1
        # calculate objective func and save values
        sum_errors = 0
        for i in range(k):
            for x in clusters[i]:
                sum_errors += distance_square(x.values, centroids[i])
        if sum_errors <= min_errors:
            min_errors = sum_errors
            best_centroids = copy.deepcopy(centroids)
            best_clusters = copy.deepcopy(clusters)
    return min_errors, best_centroids, best_clusters


if __name__ == "__main__":
    raw_data = read_data("./bank-additional-full.csv")
    preprocess(raw_data)
    data = raw_data.drop(columns='y')
    k_means(data, k_def, r_def)
