import numpy as np
from sklearn.linear_model import LinearRegression
import math
from itertools import combinations

def print_LR(X, w, y, add_bias = True, n = 5, verbose = False):
    W = np.diag(w)
    if verbose:
        print(W)

    Z1 = np.dot(np.dot(X.T,W), X)
    Z2 = np.linalg.inv(Z1)
    Z3 = np.dot(Z2, np.dot(X.T,W))


    beta = np.dot(Z3, y)
    if verbose:
        print("Solution of regression:", beta)
        print(beta[-2])
    if add_bias:
        weight_sum = np.sum(beta[:-1])
    else:
        weight_sum = np.sum(beta)
    if verbose:
        print("Sum of weights (w.o. bias) :", weight_sum)

    if add_bias and verbose:
        print("Bias :", beta[-1])
    
    return beta[:-1]

def generate_X(n, interactions = [], add_bias = True):
    '''
    n : number of elements
    interactions : [(0,1), (2,3), ..., (4,5)]
    '''
    X = []
    m = n + len(interactions) + int(add_bias)

    for i in range(n+1):
        for index_array in list(combinations(range(n), i)):
            x = np.zeros(m)

            if len(index_array) > 0:
                x[np.array(list(index_array))] = 1

            for j, inter in enumerate(interactions):
                if set(inter).issubset(set(index_array)):
                    x[j + n] = 1
            if add_bias:
                x[-1] = 1
            X.append(x)
    return np.array(X)

def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

def kernel_shap_weight(n, inf = 100000):
    w = [inf]

    for s in range(1, n):
        for _ in range(nCr(n,s)):
            #w.append(1)
            w.append(float(1)/ nCr(n,s)/ (n-s) / s )
    w.append(inf)


    return w

def uniform_weight(n, inf = 100000):
    w = [inf]
    for _ in range(n-2):
        w.append(1)
    w.append(inf)
    return w

def getshap(n, y):
   X = generate_X(n , [], True)
   w = kernel_shap_weight(n) 
   shap = print_LR(X, w, y, False, n)
   #print(shap)
   return shap


def getshap_sample(n, X, y):
   w = uniform_weight(n) 
   shap = print_LR(X, w, y, False, n)
   #print(shap)
   return shap