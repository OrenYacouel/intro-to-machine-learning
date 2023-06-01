import numpy as np
from kmeans import kmeans
from singlelinkage import singlelinkage
from more_itertools import locate

data = np.load('mnist_all.npz')


def init_1000_random_samples():
    # initiate X as an unlabeled random sample of size 1000 generated from all the digits in the MNIST data file mnist_all.mat 
    X = np.zeros((1000, 784))
    y = np.zeros((1000, 1))
    for i in range(10):
        X[i * 100: (i + 1) * 100] = data['train' + str(i)][np.random.randint(0, 5000, 100)]
        y[i * 100: (i + 1) * 100] = i
    return X, y

def init_300_random_samples():
    # get 30 random samples from each digit and put it in X, and save the labels in y
    X = np.zeros((300, 784))
    y = np.zeros((300, 1))
    for i in range(10):
        X[i * 30: (i + 1) * 30] = data['train' + str(i)][np.random.randint(0, 5000, 30)]
        y[i * 30: (i + 1) * 30] = i
    return X, y


def find_indices(list_to_check, item_to_find):
    indices = locate(list_to_check, lambda x: x == item_to_find)
    return list(indices)

def get_cluster_info_and_error(X, y, c, k):
    centroids, cluster_sizes = np.unique(c, return_counts = True)
    m = X.shape[0]
    missclassified_samples = 0

    if(len(centroids) != k):
        print("Error: the number of clusters is not " + str(k))
        return

    for i in range(k):
        centroid = centroids[i]
        cluster_size = cluster_sizes[i]
        indices_of_cluster = find_indices(c, centroid)
        labels_in_cluster = y[indices_of_cluster]
        labels_in_cluster = labels_in_cluster.flatten()
        labels_in_cluster = labels_in_cluster.astype(int)
        #find the most frequent element in labels_in_cluster and its frequency
        counts = np.bincount(labels_in_cluster)
        most_frequent_label = np.argmax(counts)
        most_frequent_label_frequency = counts[most_frequent_label]
        print("Cluster " + str(i) + " has " + str(cluster_size) + " samples, and the most frequent label is " + str(most_frequent_label) + " with frequency " + str(most_frequent_label_frequency))

        # calculate the classification error
        missclassified_samples += cluster_size - most_frequent_label_frequency
    error = missclassified_samples / m
    print("The classification error is " + str(error))


def q1c(k, t):
    print("k = " + str(k))

    X,y = init_1000_random_samples()

    c = kmeans(X, k = k, t = t)
    get_cluster_info_and_error(X, y, c, k)

def q1d(k):
    print("k = " + str(k))

    X,y = init_300_random_samples()

    c = singlelinkage(X, k = k)
    get_cluster_info_and_error(X, y, c, k)


q1c(10, 10)
q1d(10)

# question 1e
q1c(6, 10)
q1d(6)

