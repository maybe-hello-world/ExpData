"""
Yeah, it's forbidden to use built-in function or packages
"""
import math

def mean(arr):
    """
    Returns average value of array

    :param arr: array of values
    :return: average value
    """
    sm = 0
    for i in arr:
        sm += i
    return sm / len(arr)

def square_mean(arr):
    """
    Returns squared mean (not mean^2, but sum of i^2)
    :param arr: array of values
    :return: squared mean value
    """
    sm = 0
    for i in arr:
        sm += i**2
    return sm / len(arr)

def stand_deviation(arr):
    """
    Returns standard deviation of array

    :param arr: array of values
    :return: standard deviation of all values in array
    """
    return math.sqrt(square_mean(arr))

def variance(arr):
    """
    Calculates dispersion of values in array

    :param arr: array of values
    :return: dispersion value for the array
    """
    return moment(arr, 2)

def sqrt_variance(arr):
    """
    Calculates sqrt of variance of values in array
    :param arr: array of values
    :return: square root of variance
    """
    return math.sqrt(variance(arr))

def moment(arr, ordinal):
    """
    Calculates ordinal moment of array

    :param arr: array of values
    :param ordinal: ordinal of moment
    :return: ordinal moment value
    """
    av = mean(arr)
    d = 0
    for i in arr:
        d += (i - av) ** ordinal
    return d / len(arr)

def skewness(arr):
    """
    Calculates skewness (assymetry coeff) for values

    :param arr: array of function's values
    :return: skewness value
    """
    return moment(arr, 3) / (sqrt_variance(arr) ** 3)

def kurtosis(arr):
    """
    Calculates kurtosis (exscess coeff) for fuinction's values

    :param arr: array of values
    :return: kurtosis value
    """
    return moment(arr, 4) / (sqrt_variance(arr) ** 4)

def density(arr, M):
    """
    Calculate probability density for array of values

    :param arr: array of function's values
    :param M: number of intervals
    :return: array of intervals (len(arr) = M) with number of values in each interval
    """
    return NotImplementedError

def autocorrelation(arr, lag):
    """
    Calculates autocorrelation value for given lag for function f(x)

    :param arr: array of values
    :param lag: lag for autocorrelation (value from 0 to N-1)
    :return: autocorrelation value for given lag for values in array
    """
    return NotImplementedError

def crosscorrelation(arr_f, arr_g, lag):
    """
    Calculates cross-correlation value for given lag for functions f(x) and g(y)

    :param arr_f: array of values for f(x) function
    :param arr_g: array of values for g(x) function
    :param lag: lag for correlation (value from 0 to N-1)
    :return: cross-correlation value for given lag and given functions
    """
    return NotImplementedError