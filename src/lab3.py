from scipy.stats import norm, poisson, cauchy, uniform
import numpy as np
import math as m
import seaborn as sns
import matplotlib.pyplot as plt

sizes = [20, 100]
NORMAL, CAUCHY, POISSON, UNIFORM = "Normal distribution", "Cauchy distribution", "Poisson distribution", "Uniform distribution"
NUMBER_OF_REPETITIONS = 1000
STR_1, STR_2 = 'Доля выбросов выборки из 20 элементов: ', 'Доля выбросов выборки из 100 элементов: '
EXPANSION = '.png'

def moustache(distribution):
    q_1, q_3 = np.quantile(distribution, [0.25, 0.75])
    return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)

def count_out(distribution):
    x1, x2 = moustache(distribution)
    filtered = [x for x in distribution if x > x2 or x < x1]
    return len(filtered)

def DrawBoxplot(tips, name):
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.boxplot(data=tips, orient='h', width=0.15)
    sns.despine(offset=10)
    x = np.array([20, 100])
    plt.xlabel("x")
    plt.yticks(range(2), ['20', '100'])
    plt.ylabel("n")
    plt.title(name)
    plt.show()
    return

def printAnswer(result):
    print(STR_1 + str(result[0]))
    print(STR_2 + str(result[1]))

def normal_distribution():
    tips, result, count = [], [], 0
    for size in sizes:
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = norm.rvs(size=size)
            distribution.sort()
            count += count_out(distribution)
        result.append(count / (size * NUMBER_OF_REPETITIONS))
        distribution = norm.rvs(size=size)
        distribution.sort()
        tips.append(distribution)
    DrawBoxplot(tips, NORMAL)
    printAnswer(result)
    return

normal_distribution()

def cauchy_distribution():
    tips, result, count = [], [], 0
    for size in sizes:
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = cauchy.rvs(size=size)
            distribution.sort()
            count += count_out(distribution)
        result.append(count / (size * NUMBER_OF_REPETITIONS))
        distribution = cauchy.rvs(size=size)
        distribution.sort()
        tips.append(distribution)
    DrawBoxplot(tips, CAUCHY)
    printAnswer(result)
    return

cauchy_distribution()

def poisson_distribution():
    tips, result, count = [], [], 0
    for size in sizes:
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = poisson.rvs(10, size=size)
            distribution.sort()
            count += count_out(distribution)
        result.append(count / (size * NUMBER_OF_REPETITIONS))
        distribution = poisson.rvs(10, size=size)
        distribution.sort()
        tips.append(distribution)
    DrawBoxplot(tips, POISSON)
    printAnswer(result)
    return

poisson_distribution()

def uniform_distribution():
    tips, result, count = [], [], 0
    for size in sizes:
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
            distribution.sort()
            count += count_out(distribution)
        result.append(count / (size * NUMBER_OF_REPETITIONS))
        distribution = uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
        distribution.sort()
        tips.append(distribution)
    DrawBoxplot(tips, UNIFORM)
    printAnswer(result)
    return

uniform_distribution()