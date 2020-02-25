# Central Limit Theorem
# Using Distribution of Rayleigh: https://en.wikipedia.org/wiki/Rayleigh_distribution

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, norm

RAYLEIGH_LOW_LIMIT = 0
RAYLEIGH_HIGH_LIMIT = 6
# RAYLEIGH_HIGH_LIMIT = 10
# RAYLEIGH_LOC = 1
# RAYLEIGH_SCALE = 1

# RAYLEIGH_LOC = 0.7
# RAYLEIGH_SCALE = 2
SELECTION_SIZE = 1000

# Part 1: Make selection of size 1000, build histogramm and theoretical density
# Some introduction to rayleigh function https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html#scipy.stats.rayleigh


def rayleighRandomFunction(size):
    return rayleigh.rvs(
        size=size)
    # return rayleigh.rvs(
    #     size=size, loc=RAYLEIGH_LOC, scale=RAYLEIGH_SCALE)


selection = np.linspace(
    RAYLEIGH_LOW_LIMIT, RAYLEIGH_HIGH_LIMIT, num=SELECTION_SIZE)

# y = rayleigh.pdf(selection, RAYLEIGH_LOC, RAYLEIGH_SCALE)

y = rayleigh.pdf(selection)
plt.plot(selection, y, "r-",  label="Orig. density")

# Remark: normed is deprecated, density should be used instead
plt.hist(rayleighRandomFunction(SELECTION_SIZE), density=True, histtype="stepfilled",
         color="blue", bins='auto', label="Histogram")

plt.legend()
plt.xlabel("x")
plt.ylabel("F(x)")
plt.title("Rayleigh distribution (n = 1000)")
plt.show()

# Part 2: Function definitions to calculate mean and variance


def sampleMeanFunction(y):
    sampleMean = 0
    for singleY in y:
        sampleMean = sampleMean + singleY
    sampleMean = sampleMean/len(y)
    return sampleMean


def sampleVarianceFunction(y):
    sampleMean = sampleMeanFunction(y)

    sampleVariance = 0
    for singleY in y:
        sampleVariance = sampleVariance + ((singleY - sampleMean) ** 2)
    sampleVariance = sampleVariance/(len(y))
    return sampleVariance


# Test sample mean
print("Calculated mean: ", np.round(sampleMeanFunction(y), 5),
      ", proof mean: ", np.round(np.mean(y), 5))
# Test sample variance
print("Calculated variance: ", np.round(sampleVarianceFunction(
    y), 5), ", proof variance: ", np.round(np.var(y), 5))

# Part 2
# rayleighMean, rayleighVariance = rayleigh.stats(
#     loc=RAYLEIGH_LOC, scale=RAYLEIGH_SCALE, moments="mv")

rayleighMean, rayleighVariance = rayleigh.stats(moments="mv")
print("Rayleigh stats (mean, variance): ", np.round(
    rayleighMean, 4), np.round(rayleighVariance, 4))


def normalDistributionGraph(selection, size):
    mean = rayleighMean
    variance = rayleighVariance
    # mean = sampleMeanFunction(selection)
    # variance = math.sqrt(sampleVarianceFunction(selection)/size)
    print("Selection based mean and variance are: ", mean, " , ", variance)
    # Test values

    # Standard deviation (sigma) = square root of variance (in our case our sample variance)
    #sigma = math.sqrt(variance / size)
    #sigma = math.sqrt(variance)
    sigma = math.sqrt(variance / size)

    normalFunction = norm(mean, sigma)
    x = np.linspace(RAYLEIGH_LOW_LIMIT, RAYLEIGH_HIGH_LIMIT, 100)
    plt.plot(x, normalFunction.pdf(x), "r-", label="Normal appr.")


# 1000 random selections of 4 different sizes
SELECTION_SIZES = (5, 10, 50, 750)
SELECTION_AMOUNT = 1000

sampleMeans = []

for selectionSize in SELECTION_SIZES:
    for i in range(1, SELECTION_AMOUNT):
        selection = rayleighRandomFunction(selectionSize)
        sampleMeans.append(sampleMeanFunction(selection))
#   For each selection size, plot histogram
    plt.hist(sampleMeans, density=True, histtype="stepfilled",
             color="blue", bins='auto', label="Histogram")

    normalDistributionGraph(selection, selectionSize)

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Rayleigh samples distribution for n = "+str(selectionSize))
    plt.show()
