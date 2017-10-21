"""
Module for analysis of arrays of values (mean, variance, etc.)

This module realize functions for array analysis like mean, square_mean, RMS etc.
Yeah, it's forbidden to use built-in functions
"""
import math

def mean(arr) -> float:
	"""
	Returns average value of array

	:rtype: float
	:param arr: array of values
	:return: average value
	"""
	sm = 0
	for i in arr:
		sm += i
	return sm / len(arr)

def square_mean(arr) -> float:
	"""
	Returns squared mean (not mean^2, but sum of i^2)

	:rtype: float
	:param arr: array of values
	:return: squared mean value
	"""
	sm = 0
	for i in arr:
		sm += i**2
	return sm / len(arr)

def root_mean_square(arr) -> float:
	"""
	Returns standard deviation of array

	:rtype: float
	:param arr: array of values
	:return: standard deviation of all values in array
	"""
	return math.sqrt(square_mean(arr))

def variance(arr) -> float:
	"""
	Calculates dispersion of values in array

	:rtype: float
	:param arr: array of values
	:return: dispersion value for the array
	"""
	return moment(arr, 2)

def sqrt_variance(arr) -> float:
	"""
	Calculates sqrt of variance of values in array

	:rtype: float
	:param arr: array of values
	:return: square root of variance
	"""
	return math.sqrt(variance(arr))

def moment(arr, ordinal: int) -> float:
	"""
	Calculates ordinal moment of array

	:rtype: float
	:param arr: array of values
	:param ordinal: ordinal of moment
	:return: ordinal moment value
	"""
	av = mean(arr)
	d = 0
	for i in arr:
		d += (i - av) ** ordinal
	return d / len(arr)

def skewness(arr) -> float:
	"""
	Calculates skewness (assymetry coeff) for values

	:rtype: float
	:param arr: array of function's values
	:return: skewness value
	"""
	return moment(arr, 3) / (sqrt_variance(arr) ** 3)

def kurtosis(arr) -> float:
	"""
	Calculates kurtosis (exscess coeff) for function's values

	:rtype: float
	:param arr: array of values
	:return: kurtosis value
	"""
	return moment(arr, 4) / (sqrt_variance(arr) ** 4)

def density(arr, M: int) -> list:
	"""
	Calculate probability density for array of values

	:rtype: float
	:param arr: array of function's values
	:param M: number of intervals
	:return: array of intervals (len(arr) = M) with number of values in each interval
	"""

	lMax = max(arr)
	lMin = min(arr)
	divisor = (lMax - lMin) / M
	ans = [0] * M

	# Increment element in list that corresponds to given value in range [lMin, lMax]
	for i in arr:
		try:
			ans[int(math.floor((i - lMin) / divisor))] += 1
		except IndexError:
			# Last range includes max values
			ans[-1] += 1

	# Divide each element by arr length in order to get probability
	return [i / len(arr) for i in ans]

def autocorrelation(arr, lag: int) -> float:
	"""
	Calculates auto-correlation value for given lag for function f(x)

	:rtype: float
	:param arr: array of values
	:param lag: lag for auto-correlation (value from 0 to N-1)
	:return: auto-correlation value for given lag for values in array
	"""
	return crosscorrelation(arr, arr, lag)

def crosscorrelation(arr_f, arr_g, lag: int) -> float:
	"""
	Calculates cross-correlation value for given lag for functions f(x) and g(y)

	:rtype: float
	:param arr_f: array of values for f(x) function
	:param arr_g: array of values for g(x) function
	:param lag: lag for correlation (value from 0 to N-1)
	:return: cross-correlation value for given lag and given functions
	"""
	if len(arr_f) != len(arr_g):
		raise ValueError("Lengths of function arrays are different")

	avg_f = mean(arr_f)
	avg_g = mean(arr_g)
	lSum = 0
	for i in range(0, len(arr_f) - lag - 1):
		lSum += (arr_f[i] - avg_f) * (arr_g[i + lag] - avg_g)

	return lSum / (len(arr_f) * sqrt_variance(arr_f) * sqrt_variance(arr_g))
