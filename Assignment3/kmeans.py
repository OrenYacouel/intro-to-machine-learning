import numpy as np


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    # initiate the centroids
    m, d = X.shape
    centroids = np.zeros((k, d))
    C = np.zeros((m,1))
    # find the minimum and maximum values from all the sample
    min_value = np.min(X)
    max_value = np.max(X)
    for i in range(k):
        for j in range(d):
            centroids[i][j] = np.random.randint(min_value , max_value)
    # run k-means
    for i in range(t):
        # assign each sample to the closest centroid
        for j in range(m):
            C[j] = np.argmin(np.linalg.norm(X[j] - centroids, axis=1))
        # update the centroids
        for j in range(k):
            # get the size of the cluster
            ci_size = np.count_nonzero(C == j)
            sum_of_x_in_ci = np.zeros(d)
            for n in range(m):
                if C[n] == j:
                    sum_of_x_in_ci += X[n]
            centroids[j] = (1 / (ci_size + 0.1)) * sum_of_x_in_ci
                
    return C


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)
    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
