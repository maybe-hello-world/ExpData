"""
Fourier transformation module
"""

import math


class FT_result:
	"""
	Class that contains result of Fourier transformation over some function at some range

	Attributes:
		frequencies     Values of frequencies
		Re              Values of real part (cosinus basis)
		Im              Values of imaginary part (sinus basis)
		deltaF          Multiplicator for x axis for frequencies
		borderF         Maximum frequency that can be found
	"""

	frequencies: list
	Re: list
	Im: list
	deltaF: int
	borderF: int


def fourier_transform(arr, deltaT: float) -> FT_result:
	"""
	Performs Fourier transformation over some function's values

	:param arr: Array of function's values
	:param deltaT: Step of discretization
	:return: Result of transformation (frequencies, Re values list, Im values list,
	deltaF value for graph scaling and borderF value for max frequency value on graph
	"""
	res = FT_result()

	res.frequencies = []
	res.Re = []
	res.Im = []

	# Calculate FT
	for n in range(len(arr)):
		Re, Im, C = __FT_step(n, arr)
		res.Re.append(Re)
		res.Im.append(Im)
		res.frequencies.append(C)

	res.borderF = __calculate_borderF(deltaT)
	res.deltaF = __calculate_deltaF(res.borderF, len(arr))

	return res


def __FT_step(n: int, arr) -> (float, float, float):
	"""
	Internal step of Fourier transformation

	:param n: Step number
	:param arr: array of values
	:return: Re value, Im value and C value in terms of Fourier Transformation algo
	"""
	Re = 0
	Im = 0
	N = len(arr)

	for k in range(N):
		Re += arr[k] * math.cos(2 * math.pi * n * k / N)
	Re /= N

	for k in range(N):
		Im += arr[k] * math.sin(2 * math.pi * n * k / N)
	Im /= N

	C = math.sqrt(Re ** 2 + Im ** 2)

	return Re, Im, C


def __calculate_deltaF(borderF: float, N: int) -> float:
	"""
	Calculate \delta F (how long is one step given FT results)

	:param borderF: Max frequency
	:param N: Length of array
	:return: \delta F value
	"""
	return borderF / (N / 2)


def __calculate_borderF(deltaT: float) -> float:
	"""
	Calculate max frequency to be found depend on step of discretization

	:param deltaT: Step of discretization
	:return: Border freq value
	"""
	return 1 / (2 * deltaT)

