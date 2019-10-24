import numpy as np

def nonlin(x, deriv=False):
    # Limitation of exponential
    x[x > 709] = 709
    x[x < -709] = -709
    if (deriv == True):
        return (x * (1 - x))
    return 1 / (1 + np.exp(-x))

def labelling(label, dim):
    leng = len(label)
    array = np.zeros((leng,dim))
    # array -= 1
    for l in range(leng):
        for i in range(dim):
            if(label[l] == i):
                array[l][i] = 1
    return array

def replaceone(x):
    array = np.zeros((len(x), len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(x[i][j] > 0):
                array[i][j] = 1
    return array

def removeOverflow(x):
    array = x
    for i in range(len(x)):
        for j in range(len(x[0])):
            if(x[i][j] > 709):
                array[i][j] = 709
            elif(x[i][j] < -708):
                array[i][j] = -708
    return array
