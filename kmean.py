import numpy as np
import os

import matplotlib.pyplot as plt

""" x = np.linspace(0,20,100)
plt.plot(x, np.sin(x))
plt.show()
 """
def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2
    
def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]

def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids():
    centroids = []
    centroids.append([3,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,3,999,0,1.4,94.465,-41.8,4.864,5228.1,0])
    centroids.append([5,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,999,0,1.4,93.444,-36.1,4.97,5228.1,1])
    
    return np.array(centroids)

def plot_results(dataset,  centroids, k):
  plt.figure(figsize=(80,6))
  plt.scatter(dataset[:, 0], dataset[:, 1],s=100,c='red',label='Cluster1')
  plt.scatter(dataset[:, 1], dataset[:, 1],s=100,c='blue',label='Cluster2')
  
  plt.scatter(centroids[:, 0], centroids[:, 1], color='black', s=100)

if __name__ == "__main__":
    filename = "./processed_data.csv"

    data_points = np.genfromtxt(filename, delimiter=",")
    centroids = create_centroids()
    total_iteration = 5
    
    [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
    #print_label_data([cluster_label, new_centroids])
    print(len(cluster_label))
    #print_label_data([cluster_label, new_centroids])
    plot_results(data_points, new_centroids,k=2)
    print_label_data([cluster_label, new_centroids])
    print()