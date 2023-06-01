from softsvm import softsvm
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from softsvmpoly import softsvmpoly, poly_kernel
import matplotlib.patches as mpatches
import matplotlib
import math


data = np.load('ex2q4_data.npz')
x_train = data['Xtrain']
y_train = data['Ytrain']
x_test = data['Xtest']
y_test = data['Ytest']


# Question 4a

def plot_train_set():
    negative_train = x_train[y_train.flatten() == -1]
    positive_train = x_train[y_train.flatten() == 1]

    plt.scatter(negative_train[:, 0], negative_train[:, 1], color="green", label='-1', s=7)
    plt.scatter(positive_train[:, 0], positive_train[:, 1], color="purple", label='1', s=7)
    plt.legend()
    plt.title(r'Train Set Points in ${R}^2$')
    plt.show()

plot_train_set()


# Question 4b

def poly_error(alpha, xTrain, xTest, yTest, k):
    y_predict = []
    for x in xTest:
        sum = 0
        for i in range(len(alpha)):
            if alpha[i] == 0:
                continue
            sum += alpha[i] * poly_kernel(xTrain[i], x, k)
        y_predict.append(np.sign(sum)[0])
    return np.mean(np.asarray(y_predict) != yTest.flatten())



def helper_func_poly(pair):
    alpha = softsvmpoly(pair[0], pair[1], x_train, y_train.flatten())
    err = poly_error(alpha, x_train, x_test, y_test, pair[1])
    return err, pair


k_array = [ 2, 5, 8]
lambda_array = [1, 10, 100]
def fold_cross_validation_poly(x_train,y_train,k_array,lamda_array):
    split_x_train = np.asarray(np.split(x_train, 5))
    split_y_train = np.asarray(np.split(y_train, 5))
    comb_err = []
    combination_array = [[1,2],[1,5],[1,8],[10,2],[10,5],[10,8],[100,2],[100,5],[100,8]]
    
    for l in lambda_array:
        for k in k_array:
            err = []
            for i in range(5):
                x_train_i = np.concatenate(np.delete(split_x_train, i, axis=0))
                y_train_i = np.concatenate(np.delete(split_y_train, i, axis=0))
                x_test_i = split_x_train[i]
                y_test_i = split_y_train[i]
                alpha = softsvmpoly(l, k, x_train_i, y_train_i)
                err.append(poly_error(alpha, x_train_i, x_test_i, y_test_i,k))
            comb_err.append(np.mean(err))
    min_err_index = np.argmin(np.asarray(comb_err))
    return comb_err,helper_func_poly(combination_array[min_err_index])


def svm_error(h, xTest, yTest):
    y_predict = np.sign(xTest @ h).flatten()
    return np.mean(yTest != y_predict)

def helper_func_linear(l):
    h = softsvm(l, x_train, y_train)
    err = svm_error(h, x_test, y_test)
    return err, l

def fold_cross_validation_linear(x_train,y_train,lamda_array):
    split_x_train = np.asarray(np.split(x_train, 5))
    split_y_train = np.asarray(np.split(y_train, 5))
    mean_errors = []   
    for l in lambda_array:
        err = []
        for i in range(5):
            x_train_i = np.concatenate(np.delete(split_x_train, i, axis=0))
            y_train_i = np.concatenate(np.delete(split_y_train, i, axis=0))
            x_test_i = split_x_train[i]
            y_test_i = split_y_train[i]
            h = softsvm(l, x_train_i, y_train_i)
            err.append(svm_error(h, x_test_i, y_test_i))
        mean_errors.append(np.mean(err))
    min_err_index = np.argmin(np.asarray(mean_errors))
    return mean_errors,helper_func_linear(lambda_array[min_err_index])

def Question4b():
    mean_errors,(err, optPair) = fold_cross_validation_poly(x_train,y_train,k_array,lambda_array)
    print("SoftSVMPoly results are:")
    print("The mean errors are: ", mean_errors)
    print("The optimal combination (lambda, k) is ", optPair)
    print("The optimal error is ", err)
    print()

    mean_errors,(err, opt_lambda) = fold_cross_validation_linear(x_train,y_train,lambda_array)
    print("SoftSVM results are:")
    print("The mean errors are: ", mean_errors)
    print("The optimal lambda is ", opt_lambda)
    print("The optimal error is ", err)


    print()

Question4b()


# question 4d

def predict_svm_poly(alpha: np.array, x_train: np.array, k: float, x_test: np.array):
    predictions = np.array([np.sign(sum(alpha[i] * poly_kernel(x, test, k) for i, x in enumerate(x_train))) for test in x_test])
    return predictions

def svm_poly(l: float, k: float, grid_step):
    alpha = softsvmpoly(l, k, x_train, y_train)
    (x0_min, x0_max) = (x_train[:,0].min(), x_train[:, 0].max())
    (x1_min, x1_max) = (x_train[:,1].min(), x_train[:, 1].max())   
    x = np.arange(x0_min, x0_max, grid_step)
    y = np.arange(x1_min, x1_max, grid_step)
    x_test = np.array([[[x_val, y_val] for y_val in y ] for x_val in x])
    x_test = np.flip(x_test, 0)
    x_test = x_test.reshape(x.shape[0] * y.shape[0], x_test.shape[2])
    y_preds = predict_svm_poly(alpha, x_train, k=k, x_test=x_test)
    y_preds = y_preds.reshape(x.shape[0], y.shape[0])
    y_preds +=1
    y_preds /=2

    # Plotting the result
    plt.figure()
    colors=['BLUE','RED']
    plt.imshow(y_preds, cmap = matplotlib.colors.ListedColormap(colors), extent=[x0_min, x0_max, x1_min, x1_max])
    plt.title(f'Grid map for: lambda = {l} , k = {k}')

step = 0.25
svm_poly(100, 3, step)
svm_poly(100, 5, step)
svm_poly(100, 8, step)
plt.show()


def getIdk(k,d):
    idk = []
    for i in range(k+1):
        j=0
        while i+j <= k:
            if(j <= i):
                difference = k-i-j
                idk.append([difference,i,j])
                if(j != i):
                    idk.append([difference,j,i])
            j+=1
    return idk


def getB(k,t):
    co = math.factorial(k)
    for ti in t:
        co /= math.factorial(ti)
    return co

def map(x,k,d):
    idk = getIdk(k,d)
    fi = []
    for i in range(len(idk)):
        t = idk[i]
        sqrtB = math.sqrt(getB(k,t))
        curr = sqrtB
        for j in range(len(x)):
            curr *= x[j]**t[j+1]
        fi.append(curr)
    return fi


def question4f():
    d=2
    pair=(1,5)
    alpha = softsvmpoly(pair[0], pair[1], x_train, y_train)
    w = np.zeros(21)
    for i in range(len(alpha)):
        w += alpha[i] * np.array(map(x_train[i],pair[1],d))
    return w

question4f()


def q4f4():
    alpha = softsvmpoly(1, 5, x_train, y_train)
    idk = getIdk(5,2)
    t = np.array(idk)
    w = question4f()
    for i in range(len(x_train)):
        if alpha[i] != 0:
            x = poly_kernel(t, x_train[i], 5)
            y = x * alpha[i]
            w += y
            

    x_train_labels = []
    one_labels = []
    zero_labels = []
    x_test_labels = []

    for x in x_train:
        pred = np.sign(poly_kernel(t, x) @ w)
        x_train_labels.append(pred)

    for x in x_test:
        pred = np.sign(poly_kernel(t, x) @ w)
        x_test_labels.append(pred)
    
    for i in range(len(x_train)):
        if x_train_labels[i] == 1:
            one_labels.append(x_train[i])
        else:
            zero_labels.append(x_train[i])


    for i in range(len(x_test_labels)):
        if x_test_labels[i] == 1:
            one_labels.append(x_test[i])
        else:
            zero_labels.append(x_test[i])


    x_zero_labels = []
    y_zero_labels = []
    x_one_labels = []
    y_one_labels = []

    for x in zero_labels:
        x_zero_labels.append(x[0])
        y_zero_labels.append(x[1])
    plt.plot(x_zero_labels, y_zero_labels, 's', label='Points labeled 0', color='purple')

    for x in one_labels:
        x_one_labels.append(x[0])
        y_one_labels.append(x[1])
    plt.plot(x_one_labels, y_one_labels, 's', label='Points labeled 1', color='green')

    plt.legend()
    plt.show()

q4f4()