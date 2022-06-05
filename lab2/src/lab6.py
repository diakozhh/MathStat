import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt

def func(x):
    return 2 + 2 * x

def noiseFunc(x):
    y = []
    for i in x:
        y.append(func(i) + stats.norm.rvs(0, 1))
    return y

def LMM(parameters, x, y):
    alpha_0, alpha_1 = parameters
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - alpha_0 - alpha_1 * x[i])
    return sum 

def getMNKParams(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1

def getMNMParams(x, y):
    beta_0, beta_1 = getMNKParams(x, y)
    result = opt.minimize(LMM, [beta_0, beta_1], args=(x, y), method='SLSQP')
    coefs = result.x
    alpha_0, alpha_1 = coefs[0], coefs[1]
    return alpha_0, alpha_1

def MNK(x, y):
    beta_0, beta_1 = getMNKParams(x, y)
    print('beta_0 = ' + str(beta_0), 'beta_1 = ' + str(beta_1))
    y_new = [beta_0 + beta_1 * x_ for x_ in x]
    return y_new

def MNM(x, y):
    alpha_0, alpha_1 = getMNMParams(x, y)
    print('alpha_0= ' + str(alpha_0), 'alpha_1 = ' + str(alpha_1))
    y_new = [alpha_0 + alpha_1 * x_ for x_ in x]
    return y_new

def getDist(y_model, y_regr):
    arr = [(y_model[i] - y_regr[i])**2 for i in range(len(y_model))]
    dist_y = sum(arr)
    return dist_y

def plotLiRegression(text, x, y):
    y_mnk = MNK(x, y)
    y_mnm = MNM(x, y)
    y_dist_mnk = getDist(y, y_mnk)
    y_dist_mnm = getDist(y, y_mnm)
    print('mnk distance', y_dist_mnk)
    print('mnm distance', y_dist_mnm)
    plt.scatter(x, y, label='Выборка', color='black', marker = ".", linewidths = 0.7)
    plt.plot(x, func(x),  label='Модель', color='lightcoral')
    plt.plot(x, y_mnk, label="МНК", color='steelblue')
    plt.plot(x, y_mnm, label="МНМ", color='lightgreen')
    plt.xlim([-1.8, 2])
    plt.grid()
    plt.legend()
    plt.show()

def buildSolve():
    x = np.arange(-1.8, 2, 0.2)
    y = noiseFunc(x)
    plotLiRegression('NoPerturbations', x, y)

    x = np.arange(-1.8, 2, 0.2)
    y = noiseFunc(x)
    y[0] += 10
    y[-1] -= 10
    plotLiRegression('Perturbations', x, y)
    return

buildSolve()