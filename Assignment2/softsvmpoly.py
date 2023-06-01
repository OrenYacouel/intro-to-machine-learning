import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m, d = trainX.shape
    gram = gram_matrix(trainX, k)
    H = np.block([[gram, np.zeros((m, m))],
                  [np.zeros((m, m)), np.zeros((m, m))]])
    H *= 2*l
    # H += (1e-4) * np.eye(2*m)
    H += np.eye(2*m)

    A = np.block([[np.zeros((m, m)), np.eye(m)],
                  [np.diag(trainy) @ gram.T, np.eye(m)]])

    u = matrix(np.concatenate((np.zeros(m), np.ones(m))))
    u *= (1/m) 
    v = matrix(np.concatenate((np.zeros(m), np.ones(m)))) 


    A = sparse(matrix(A))
    H = sparse(matrix(H))


    sol = solvers.qp(H, u, -A, -v)
    sol = np.asarray(sol["x"])
    return sol[:m]



def gram_matrix(X: np.array, k: int):
    """
    :param X: numpy array of size (m, d) containing the training sample
    :param k: the degree of the polynomial kernel
    :return: numpy array of size (m, m) containing the gram matrix
    """
    m, d = X.shape
    G = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            G[i, j] = poly_kernel(X[i], X[j], k)
    return G

def poly_kernel(x1: np.array, x2: np.array, k: int):
    """
    :param x: numpy array of size (d, 1) containing a sample
    :param y: numpy array of size (d, 1) containing a sample
    :param k: the degree of the polynomial kernel
    :return: the value of the polynomial kernel between x and y
    """
    return (float(1) + x1 @ x2) ** k

def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
