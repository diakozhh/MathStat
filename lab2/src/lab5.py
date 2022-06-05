import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

def multivariateNormal(size, ro):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, ro], [ro, 1.0]], size=size)

def mixMultivariateNormal(size, ro):
    return 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + 0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)

def quadrantCoeff(x, y):
    size = len(x)
    medianX = np.median(x)
    medianY = np.median(y)
    n = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(size):
        if x[i] >= medianX and y[i] >= medianY:
            n[1] += 1
        elif x[i] < medianX and y[i] >= medianY:
            n[2] += 1
        elif x[i] < medianX and y[i] < medianY:
            n[3] += 1
        elif x[i] >= medianX and y[i] < medianY:
            n[4] += 1
    return (n[1] + n[3] - n[2] - n[4]) / size

def coeffCount(get_sample, size, ro, repeats):
    pearson, quadrant, spirman = [], [], []
    for i in range(repeats):
        sample = get_sample(size, ro)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spirman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrantCoeff(x, y))
    return pearson, spirman, quadrant

def createTable(pearson, spirman, quadrant, size, ro, repeats):
    if ro != -1:
        rows = [["rho = " + str(ro), 'r', 'r_{S}', 'r_{Q}']]
    else:
        rows = [["size = " + str(size), 'r', 'r_{S}', 'r_{Q}']]
    p = np.median(pearson)
    s = np.median(spirman)
    q = np.median(quadrant)
    rows.append(['E(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = np.median([pearson[k] ** 2 for k in range(repeats)])
    s = np.median([spirman[k] ** 2 for k in range(repeats)])
    q = np.median([quadrant[k] ** 2 for k in range(repeats)])
    rows.append(['E(z^2)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = statistics.variance(pearson)
    s = statistics.variance(spirman)
    q = statistics.variance(quadrant)
    rows.append(['D(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    return tabulate(rows, [], tablefmt="latex")

def createEllipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    radX = np.sqrt(1 + pearson)
    radY = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=radX * 2, height=radY * 2, facecolor='none', **kwargs)

    scaleX = np.sqrt(cov[0, 0]) * n_std
    meanX = np.mean(x)

    scaleY = np.sqrt(cov[1, 1]) * n_std
    meanY = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scaleX, scaleY).translate(meanX, meanY)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def printEllipse(size, ros):
    fig, ax = plt.subplots(1, 3)
    strSize = "n = " + str(size)
    titles = [strSize + r', $ \rho = 0$', strSize + r', $\rho = 0.5 $', strSize + r', $ \rho = 0.9$']
    for i in range(len(ros)):
        num, ro = i, ros[i]
        sample = multivariateNormal(size, ro)
        x, y = sample[:, 0], sample[:, 1]
        createEllipse(x, y, ax[num], edgecolor='navy')
        ax[num].grid()
        ax[num].scatter(x, y, s=5)
        ax[num].set_title(titles[num])
    plt.show()

def buildSolve():
    sizes = [20, 60, 100]
    ros = [0, 0.5, 0.9]
    REPETITIONS = 1000

    for size in sizes:
        for ro in ros:
            pearson, spirman, quadrant = coeffCount(multivariateNormal, size, ro, REPETITIONS)
            print('\n' + str(size) + '\n' + str(createTable(pearson, spirman, quadrant, size, ro, REPETITIONS)))

        pearson, spearman, quadrant = coeffCount(mixMultivariateNormal, size, 0, REPETITIONS)
        print('\n' + str(size) + '\n' + str(createTable(pearson, spirman, quadrant, size, -1, REPETITIONS)))
        printEllipse(size, ros)
    return

buildSolve()