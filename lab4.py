from scipy.stats import norm, laplace, poisson, cauchy, uniform
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import math as m
import seaborn as sns
import matplotlib.pyplot as plt

sizes = [20, 60, 100]
koeff = [0.5, 1, 2]
left_boarder, right_boarder = -4, 4
poisson_left_boarder, poisson_right_boarder = 4, 16
number_of_samples = 4
NORMAL, CAUCHY, POISSON, UNIFORM = "Normal distribution", "Cauchy distribution", "Poisson distribution", "Uniform distribution"


def get_rvs(size):
    rvs_list = [norm.rvs(size=size),
                cauchy.rvs(size=size),
                poisson.rvs(10, size=size), uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))]
    return rvs_list


def get_pdf(x):
    pdf_list = [norm.pdf(x),
                cauchy.pdf(x),
                poisson(10).pmf(x),
                uniform.pdf(x, loc=-m.sqrt(3), scale=2 * m.sqrt(3))]
    return pdf_list


def get_name():
    return [NORMAL, CAUCHY, POISSON, UNIFORM]


def get_cdf(x):
    cdf_list = [norm.cdf(x),
                cauchy.cdf(x),
                poisson.cdf(x, mu=10),
                uniform.cdf(x, loc=-m.sqrt(3), scale=2 * m.sqrt(3))]
    return cdf_list


def DrawGraphics():
    sns.set_style('whitegrid')
    for num in range(number_of_samples):
        figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
        for size in range(len(sizes)):
            rvs_list = get_rvs(sizes[size])
            name_list = get_name()
            if num != 2:
                x = np.linspace(left_boarder, right_boarder, 10000)
            else:
                x = np.linspace(poisson_left_boarder, poisson_right_boarder, 10000)
            y = get_cdf(x)
            sample = rvs_list[num]
            sample.sort()
            ecdf = ECDF(sample)
            axs[size].plot(x, y[num], color='blue', label='cdf')
            axs[size].plot(x, ecdf(x), color='red', label='ecdf')
            axs[size].legend(loc='lower right')
            axs[size].set(xlabel='x', ylabel='$F(x)$')
            axs[size].set_title(name_list[num] + ' n = ' + str(sizes[size]))
        figures.savefig('E:\Work\MathStat/results/EFD/' +name_list[num] + str(sizes[size]) + ".png")
    return


DrawGraphics()


def DrawKDE():
    sns.set_style('whitegrid')
    for num in range(number_of_samples):
        for size in range(len(sizes)):
            figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
            rvs_list = get_rvs(sizes[size])
            name_list = get_name()
            if num != 2:
                x = np.linspace(left_boarder, right_boarder, 10000)
                start, stop = left_boarder, right_boarder
            else:
                x = np.linspace(poisson_left_boarder, poisson_right_boarder,
                                -poisson_left_boarder + poisson_right_boarder + 1)
                start, stop = poisson_left_boarder, poisson_right_boarder
            for kf in range(len(koeff)):
                y = get_pdf(x)
                sample = rvs_list[num]
                axs[kf].plot(x, y[num], color='blue', label='pdf')
                sns.kdeplot(data=sample, bw_method='scott', bw_adjust=koeff[kf], ax=axs[kf],
                            fill=False, common_norm=False, palette="crest", linewidth=1.5, label='kde')
                axs[kf].legend(loc='upper right')
                axs[kf].set(xlabel='x', ylabel='$f(x)$')
                axs[kf].set_xlim([start, stop])
                axs[kf].set_title(' h = ' + str(koeff[kf]))
            figures.suptitle(name_list[num] + ' KDE n = ' + str(sizes[size]))
            plt.show()
            figures.savefig('E:\Work\MathStat/results/kde/' + name_list[num] + 'KDE' + str(sizes[size]) + ".png")
    return

DrawKDE()
