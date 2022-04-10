import numpy as np
from scipy.stats import norm, poisson, cauchy, uniform
import math

NORMAL, CAUCHY, POISSON, UNIFORM = "Normal distribution", "Cauchy distribution", "Poisson distribution", "Uniform distribution"

sizes = [10, 100, 1000]

NUMBER_OF_REPETITIONS = 1000

#выборочное среднее
def mean(selection):
    return np.mean(selection)
#выборочная медиана
def median(selection):
    return np.median(selection)
#полусумма экстремальных выборочных элементов
def zR(selection, size):
    zR = (selection[0] + selection[size - 1]) / 2
    return zR
#выборочная квартиль
def zP(selection, np):
    if np.is_integer():
        return selection[int(np)]
    else:
        return selection[int(np) + 1]
#полусумма квартилей
def zQ(selection, size):
    z_1 = zP(selection, size / 4)
    z_2 = zP(selection, 3 * size / 4)
    return (z_1 + z_2) / 2
#усеченное среднее
def zTr(selection, size):
    r = int(size / 4)
    sum = 0
    for i in range(r + 1, size - r + 1):
        sum += selection[i]
    return (1 / (size - 2 * r)) * sum
#дисперсия
def dispersion(selection):
    return np.std(selection) ** 2

def normal_distribution():
    for size in sizes:
        mean_list, medianList, zRList, zQList, zTrList = [], [], [], [], []
        all_list = [mean_list, medianList, zRList, zQList, zTrList]
        E, D, E_minus_D, E_plus_D = [], [], [], []
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = norm.rvs(size = size)
            distribution.sort()
            mean_list.append(mean(distribution))
            medianList.append(median(distribution))
            zRList.append(zR(distribution, size))
            zQList.append(zQ(distribution, size))
            zTrList.append(zTr(distribution, size))
        for lis in all_list:
            E.append(round(mean(lis), 6))
            D.append(round(dispersion(lis), 6))
            E_minus_D.append(round(mean(lis), 6) - round(math.sqrt(dispersion(lis)), 6))
            E_plus_D.append(round(mean(lis), 6) + round(math.sqrt(dispersion(lis)), 6))
        print(NORMAL, size)
        print("E\n", E)
        print("D\n", D)
        print("E_minus_D\n", E_minus_D)
        print("E_plus_D\n", E_plus_D)
    print("\n")
    return
normal_distribution()

def cauchy_distribution():
    for size in sizes:
        mean_list, medianList, zRList, zQList, zTrList = [], [], [], [], []
        all_list = [mean_list, medianList, zRList, zQList, zTrList]
        E, D, E_minus_D, E_plus_D = [], [], [], []
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = cauchy.rvs(size=size)
            distribution.sort()
            mean_list.append(mean(distribution))
            medianList.append(median(distribution))
            zRList.append(zR(distribution, size))
            zQList.append(zQ(distribution, size))
            zTrList.append(zTr(distribution, size))
        for lis in all_list:
            E.append(round(mean(lis), 6))
            D.append(round(dispersion(lis), 6))
            E_minus_D.append(round(mean(lis), 6) - round(math.sqrt(dispersion(lis)), 6))
            E_plus_D.append(round(mean(lis), 6) + round(math.sqrt(dispersion(lis)), 6))
        print(CAUCHY, size)
        print("E\n", E)
        print("D\n", D)
        print("E_minus_D\n", E_minus_D)
        print("E_plus_D\n", E_plus_D)
    print("\n")
    return
cauchy_distribution()

def poisson_distribution():
    for size in sizes:
        mean_list, medianList, zRList, zQList, zTrList = [], [], [], [], []
        all_list = [mean_list, medianList, zRList, zQList, zTrList]
        E, D, E_minus_D, E_plus_D = [], [], [], []
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = poisson.rvs(10, size=size)
            distribution.sort()
            mean_list.append(mean(distribution))
            medianList.append(median(distribution))
            zRList.append(zR(distribution, size))
            zQList.append(zQ(distribution, size))
            zTrList.append(zTr(distribution, size))
        for lis in all_list:
            E.append(round(mean(lis), 6))
            D.append(round(dispersion(lis), 6))
            E_minus_D.append(round(mean(lis), 6) - round(math.sqrt(dispersion(lis)), 6))
            E_plus_D.append(round(mean(lis), 6) + round(math.sqrt(dispersion(lis)), 6))
        print(POISSON, size)
        print("E\n", E)
        print("D\n", D)
        print("E_minus_D\n", E_minus_D)
        print("E_plus_D\n", E_plus_D)
    print("\n")
    return
poisson_distribution()

def uniform_distribution():
    for size in sizes:
        mean_list, medianList, zRList, zQList, zTrList = [], [], [], [], []
        all_list = [mean_list, medianList, zRList, zQList, zTrList]
        E, D, E_minus_D, E_plus_D = [], [], [], []
        for i in range(NUMBER_OF_REPETITIONS):
            distribution = uniform.rvs(size=size, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
            distribution.sort()
            mean_list.append(mean(distribution))
            medianList.append(median(distribution))
            zRList.append(zR(distribution, size))
            zQList.append(zQ(distribution, size))
            zTrList.append(zTr(distribution, size))
        for lis in all_list:
            E.append(round(mean(lis), 6))
            D.append(round(dispersion(lis), 6))
            E_minus_D.append(round(mean(lis), 6) - round(math.sqrt(dispersion(lis)), 6))
            E_plus_D.append(round(mean(lis), 6) + round(math.sqrt(dispersion(lis)), 6))
        print(UNIFORM, size)
        print("E\n", E)
        print("D\n", D)
        print("E_minus_D\n", E_minus_D)
        print("E_plus_D\n", E_plus_D)
    print("\n")
    return
uniform_distribution()