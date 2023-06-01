import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist


# def singlelinkage(X, k):
#     """
#     :param X: numpy array of size (m, d) containing the test samples
#     :param k: the number of clusters
#     :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
#     """
#     # initiate the centroids
#     m, d = X.shape
#     C = np.zeros((m, 1))
#     # run single-linkage
#     for i in range(m):
#         C[i] = i
#     for i in range(m - k):
#         print("i = ", i)
#         # find the closest pair of clusters
#         min_distance = np.inf
#         neighbors_index = (0, 0)
#         for j in range(m):
#             for n in range(m):
#                 if C[j] != C[n]:
#                     distance = np.linalg.norm(X[j] - X[n])
#                     if distance < min_distance:
#                         min_distance = distance
#                         neighbors_index = (j, n)
#         # merge the closest pair of clusters
#         for j in range(m):
#             if C[j] == C[neighbors_index[1]]:
#                 C[j] = C[neighbors_index[0]]
#     centroids, cluster_sizes = np.unique(C, return_counts = True)
#     return C

def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    m, d = X.shape
    C = np.zeros(m)
    # initialize clusters
    for i in range(m):
        C[i] = i
    distances = cdist(X, X, 'euclidean')
    while len(set(C)) > k:
        min_distance = np.inf
        merge_clusters = (-1, -1)
        for i in range(m):
            for j in range(m):
                if C[i] != C[j] and distances[i][j] < min_distance:
                    min_distance = distances[i][j]
                    merge_clusters = (C[i], C[j])
        # merge clusters with smallest distance
        for i in range(m):
            if C[i] == merge_clusters[1]:
                C[i] = merge_clusters[0]
    return C



def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    # X = np.concatenate((data['train0'], data['train1']))

    # get 30 random samples from each digit and put it in X
    X = np.zeros((300, 784))
    for i in range(10):
        X[i * 30: (i + 1) * 30] = data['train' + str(i)][np.random.randint(0, 5000, 30)]

    m, d = X.shape

    # run single-linkage
    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
