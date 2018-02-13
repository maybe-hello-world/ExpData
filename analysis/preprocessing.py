"""
Some data preprocessing functions
"""
from analysis.statistics import mean, std
import math
import numpy as np
from numba import jit, float64, int64


def normalize_image(image_data, depth, S = 255):
	"""
	Normalize image with given image depth to desired color depth

	:param image_data: Image array
	:param depth: Color depth of image
	:param S: Desired color depth
	:return: Normalized to new color depth image
	"""
	maxX = 1 << depth
	temp = np.divide(image_data, maxX)
	modified_image = np.multiply(temp, S).astype(int)

	return modified_image


@jit
def normalize(arr) -> None:
	"""
	Normalize data in given array

	:param arr: array of data
	"""
	minX = min(arr)
	maxX = max(arr)
	diff = maxX - minX

	for i in range(len(arr)):
		arr[i] = (arr[i] - minX) / diff


@jit
def normalize_max(arr) -> None:
	"""
	Normalize data in given array to values from (-infinity; 1]. This algo scales data in such way that max positive values are 1
	and min negative value is inifinity but negative values also scales to max positive values.

	:param arr: array of data
	"""
	maxX = max(arr)

	if maxX <= 0:
		return arr

	for i in range(len(arr)):
		arr[i] = arr[i] / maxX


@jit
def anti_shift(arr) -> None:
	"""
	Shift data to zero mean

	:param arr: Array of data
	"""
	m = mean(arr)
	for i in range(len(arr)):
		arr[i] -= m


@jit
def anti_spike(arr, K: int = 4) -> None:
	"""
	Delete spikes that are more than mean + K sigmas

	:param arr: array of data with spikes
	:param K: sigma multiplicator (lesser K - lesser border value for spike detector - smoother output)
	"""
	avg = mean(arr)
	sigma = std(arr)
	for x in range(len(arr)):
		if math.fabs(arr[x]) > avg + K * sigma:
			if 0 < x < len(arr) - 1:
				arr[x] = (arr[x + 1] + arr[x + 2]) / 2
			elif x == len(arr) - 1:
				arr[x] = (arr[x - 2] + arr[x - 1]) / 2
			else:
				arr[x] = (arr[x-1] + arr[x + 1]) / 2


@jit
def anti_trend(arr, window_width: int = None) -> list:
	"""
	Use floating window to remove trends from data

	:param arr: array of values
	:param window_width: width of window
	"""
	trend = []
	if window_width is None :
		window_width = int(len(arr) / 100)

	counter = int(math.floor(len(arr) / window_width))
	for i in range(counter):
		mean_v = mean(arr[i * window_width : (i + 1) * window_width])
		trend.append(mean_v)
		for x in range(window_width):
			arr[i * window_width + x] -= mean_v

	# Resize last values
	if window_width * counter != len(arr):
		mean_v = mean(arr[window_width * counter:])
		trend.append(mean_v)
		for x in range(window_width * counter, len(arr)):
			arr[x] -= mean_v

	return trend


def LPF(Fcut: float, dT: float, m: int = 32) -> list:
	"""
	Low-Pass Filter realization

	:param Fcut: cutting frequency (all frequencies before this value will be cut)
	:param dT: step of discretization
	:param m: length of filter array (bigger value - more accurate but calculations are slower)
	:return: list of values of filter function
	"""

	# Constant list of Potter 310 window
	d = np.array(
		[0.35577019,
	     0.2436983,
	     0.07211497,
	     0.00630165],
	dtype=np.double)

	idxs = np.array([i for i in range(1, m + 1)])

	# Create array
	lpw = np.array([0] * (m + 1), dtype=np.double)

	# Some magic here
	arg = 2 * Fcut * dT
	lpw[0] = arg
	arg *= np.pi
	lpw[1:] = np.divide(
		np.sin(np.multiply(idxs, arg)),
		np.multiply(idxs, np.pi))

	# make trapezoid:
	lpw[-1] /= 2

	# Potter's window P310
	sumg = lpw[0]
	for i in range(1, m + 1):
		_sum = d[0]
		arg = math.pi * i / m
		for k in range(1, 4):
			_sum += 2 * d[k] * math.cos(arg * k)

		lpw[i] *= _sum
		sumg += 2 * lpw[i]

	# normalization
	lpw = np.divide(lpw, sumg)

	# mirror related to 0
	answer = lpw[::-1].tolist()
	answer.extend(lpw[1:].tolist())
	return answer


@jit(float64(float64, float64, int64))
def HPF(Fcut: float, dT: float, m: int = 32) -> list:
	lpw = np.array(LPF(Fcut=Fcut, dT=dT, m=m))

	lpw = np.multiply(lpw, -1)

	lpw[m] = 1 + lpw[m]

	return lpw.tolist()


def BPF(Fcut1: float, Fcut2: float, dT: float, m: int = 32) -> list:
	"""
	Band-pass filter

	:param Fcut1:
	:param Fcut2:
	:param dT:
	:param m:
	:return:
	"""
	lpw1 = np.array(LPF(Fcut=Fcut1, dT=dT, m=m))
	lpw2 = np.array(LPF(Fcut=Fcut2, dT=dT, m=m))

	lpw = np.subtract(lpw2, lpw1)

	return lpw.tolist()


@jit(float64(float64, float64, float64, int64))
def BSF(Fcut1: float, Fcut2: float, dT: float, m: int = 32) -> list:
	lpw1 = np.array(LPF(Fcut=Fcut1, dT=dT, m=m))
	lpw2 = np.array(LPF(Fcut=Fcut2, dT=dT, m=m))

	lpw = np.subtract(lpw1, lpw2)

	lpw[m] += 1

	return lpw.tolist()