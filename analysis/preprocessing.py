"""
Some data preprocessing functions
"""
from analysis.statistics import mean, std
import math

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

def anti_shift(arr) -> None:
	"""
	Shift data to zero mean

	:param arr: Array of data
	"""
	m = mean(arr)
	for i in range(len(arr)):
		arr[i] -= m

def anti_spike(arr) -> None:
	"""
	Delete spikes that are more than mean + 4 sigmas

	:param arr: array of data with spikes
	"""
	avg = mean(arr)
	sigma = std(arr)
	for x in range(len(arr)):
		if math.fabs(arr[x]) > avg + 4 * sigma:
			if x == 0:
				arr[x] = (arr[x + 1] + arr[x + 2]) / 2
			elif x == len(arr) - 1:
				arr[x] = (arr[x - 2] + arr[x - 1]) / 2
			else:
				arr[x] = (arr[x-1] + arr[x + 1]) / 2

def anti_trend(arr, window_width = None) -> None:
	"""
	Use floating window to remove trends from data

	:param arr: array of values
	:param window_width: width of window
	"""
	if window_width is None :
		window_width = int(len(arr) / 100)

	counter = int(math.floor(len(arr) / window_width))
	for i in range(counter):
		mean_v = mean(arr[i * window_width : (i + 1) * window_width])
		for x in range(window_width):
			arr[i * window_width + x] -= mean_v

	# Resize last values
	if window_width * counter == len(arr):
		return

	mean_v = mean(arr[window_width * counter:])
	for x in range(window_width * counter, len(arr)):
		arr[x] -= mean_v

def LPF(Fcut: float, dT: float, m: int = 64) -> list:
	"""
	Low-Pass Filter realization

	:param Fcut: cutting frequency (all frequencies before this value will be cut)
	:param dT: step of discretization
	:param m: length of filter array (bigger value - more accurate but calculations are slower)
	:return: list of values of filter function
	"""

	# Constant list of Potter 310 window
	d = [0.35577019,
	     0.2436983,
	     0.07211497,
	     0.00630165]

	# Create array
	lpw = [0] * (m + 1)

	# Some magic here
	arg = 2 * Fcut * dT
	lpw[0] = arg
	arg *= math.pi
	for i in range(1, m + 1):
		lpw[i] = math.sin(arg * i) / (math.pi * i)

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
	for i in range(m + 1):
		lpw[i] /= sumg

	# mirror related to 0
	answer = lpw[::-1]
	answer.extend(lpw[1:])
	return answer

def HPF(Fcut: float, dT: float, m: int = 64) -> list:
	lpw = LPF(Fcut=Fcut, dT=dT, m=m)

	for k in range(len(lpw)):
		lpw[k] *= -1

	lpw[m] = 1 + lpw[m]

	# for k in range(2*m + 1):
	# 	if k == m:
	# 		hpw[k] = 1 - lpw[k]
	# 	else:
	# 		hpw[k] = -lpw[k]

	return lpw

def BPF(Fcut1: float, Fcut2: float, dT: float, m: int = 64) -> list:
	"""
	Band-pass filter

	:param Fcut1:
	:param Fcut2:
	:param dT:
	:param m:
	:return:
	"""
	lpw1 = LPF(Fcut=Fcut1, dT=dT, m=m)
	lpw2 = LPF(Fcut=Fcut2, dT=dT, m=m)

	for k in range(len(lpw1)):
		lpw1[k] = lpw2[k] - lpw1[k]

	return lpw1

def BSF(Fcut1: float, Fcut2: float, dT: float, m: int = 64) -> list:
	lpw1 = LPF(Fcut=Fcut1, dT=dT, m=m)
	lpw2 = LPF(Fcut=Fcut2, dT=dT, m=m)

	for k in range(len(lpw1)):
		lpw1[k] = lpw1[k] - lpw2[k]

	lpw1[m] += 1

	return lpw1