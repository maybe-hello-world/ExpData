"""
Some data preprocessing functions
"""
from analysis.statistics import mean, std, histogramm, cumulative_sum, sqrt_variance
import math
import numpy as np
from scipy.misc import imresize
from numba import jit, float64, int64
from reader import bin2float
from analysis import FT, impulse_response
from analysis.statistics import autocorrelation


def remove_periodic(image_data: np.ndarray, col_nums: int, row_nums: int, turn: bool = False):
	# find peak
	N = row_nums // 2
	derivative = [image_data[N, i + 1] - image_data[N, i] for i in range(len(image_data[N]) - 1)]
	ac = [autocorrelation(derivative, i) for i in range(len(derivative))]
	res = FT.fourier_transform(ac, 1)
	d = res.frequencies[:len(res.frequencies) // 2]
	m = mean(d)
	sqv = sqrt_variance(d)
	peaks = [i * res.deltaF for i in range(len(d)) if d[i] > m + sqv / 4]
	leftBorder, rightBorder = min(peaks), max(peaks)

	# apply filter
	m = 32
	bsfilter = BSF(leftBorder, rightBorder, 1, m)
	new_image = np.empty_like(image_data)
	for i in range(len(image_data)):
		row = impulse_response.process(image_data[i], bsfilter)
		new_image[i] = row[m:-m - 1]

	return new_image


def restore_image(
		path: str, width: int, height: int,
		kernel_path: str, kernel_length: int,
		blurred: bool = False, a: float = 0.1) -> np.ndarray:

	# load kernel func
	h_func = bin2float(kernel_path, kernel_length)
	h_func = list(h_func)
	h_func.extend([0] * (width - kernel_length))
	h_func = np.array(h_func)

	# get spectre
	h_func_FT = FT.fourier_transform(h_func, 1)
	h_func_FT = make_complex(h_func_FT.Re, h_func_FT.Im)

	# Load image
	blur_229 = bin2float(path, width * height)
	blur_229 = np.array(blur_229).reshape((height, width))

	# Get image spectre
	blur_229_FT = np.empty_like(blur_229, dtype=np.complex)
	i = 0
	for x in blur_229[:, :]:
		line = FT.fourier_transform(x, 1)
		line = make_complex(line.Re, line.Im)
		blur_229_FT[i, :] = line
		i += 1

	if not blurred:
		for row in range(blur_229_FT.shape[0]):
			blur_229_FT[row, :] /= h_func_FT
	else:
		for row in range(blur_229_FT.shape[0]):
			div = blur_229_FT[row, :] * np.conjugate(h_func_FT)
			blur_229_FT[row, :] = div / (np.absolute(h_func_FT) ** 2 + a ** 2)

	blur_229_restored = np.empty_like(blur_229)
	for x in range(blur_229.shape[0]):
		line = blur_229_FT[x, :]
		new_line = []
		for i in line:
			new_line.append(i.real + i.imag)
		blur_229_restored[x, :] = FT.reverse_fourier_transform(np.array(new_line), 1)

	return blur_229_restored



def negative_image(image_data: np.ndarray, depth: int = 8) -> np.ndarray:
	return ((1 << depth) - 1) - image_data


def log_correction_image(image_data: np.ndarray, C: float, depth: int = 8) -> np.ndarray:
	temp = np.add(1,image_data, dtype=np.int64)
	result = np.log(temp) * C
	return normalize_image(result, 1 << depth - 1, dtype=image_data.dtype)


def gamma_correction_image(image_data: np.ndarray, C: float, gamma: float, depth: int = 8) -> np.ndarray:
	result = np.power(image_data, gamma) * C
	return normalize_image(result, 1 << depth - 1, dtype=image_data.dtype)


def resize_image(image_data: np.ndarray, k1, k2, method = "knn") -> np.ndarray:
	result = np.empty((int(image_data.shape[0] * k1), int(image_data.shape[1] * k2), 3), dtype=image_data.dtype)

	if method == "knn":
		for i in range(result.shape[0]):
			for j in range(result.shape[1]):
				result[i][j] = image_data[int(i/k1)][int(j/k2)]
	elif method == "bilinear":
		result = imresize(arr=image_data, size=result.shape, interp="bilinear", mode="RGB")
	else:
		raise ValueError("method isn't supported")

	return result


def histeq(image_data: np.ndarray, depth: int = 8) -> np.ndarray:

	h = histogramm(image_data, depth)
	cdf = cumulative_sum(h)
	l = (1 << depth) - 1
	if depth <= 8:
		sk = np.uint8(l * cdf)
	elif depth <= 16:
		sk = np.uint16(l * cdf)
	elif depth <= 32:
		sk = np.uint32(l * cdf)
	else:
		sk = np.uint128(l * cdf)

	s1, s2 = image_data.shape[0], image_data.shape[1]
	Y = np.zeros_like(image_data)
	for i in range(s1):
		for j in range(s2):
			Y[i, j] = sk[image_data[i, j]]

	return Y


def hist_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
	oldshape = source.shape

	if len(oldshape) > 3 or len(oldshape) < 2:
		raise ValueError(oldshape, "= oldshape")
	elif len(oldshape) == 2:
		source = np.expand_dims(source, axis=-1)
		oldshape = (oldshape[0], oldshape[1], 1)

	template = template.ravel()
	data = []
	for channel in range(oldshape[2]):
		source_local = source[:, :, channel].ravel()


		s_values, bin_idx, s_counts = np.unique(source_local, return_inverse=True,
		                                        return_counts=True)
		t_values, t_counts = np.unique(template, return_counts=True)

		s_quantiles = np.cumsum(s_counts).astype(np.float64)
		s_quantiles /= s_quantiles[-1]
		t_quantiles = np.cumsum(t_counts).astype(np.float64)
		t_quantiles /= t_quantiles[-1]

		interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

		channel_data = interp_t_values[bin_idx].reshape((oldshape[0], oldshape[1]))
		data.append(channel_data)

	data = np.stack(data, axis=2).astype(dtype=source.dtype)
	return data


def normalize_image(image_data: np.ndarray, S = 255, dtype = None):
	"""
	Normalize image with given image depth to desired color depth

	:param image_data: Image array
	:param dtype: Type of data
	:param S: Desired color depth
	:return: Normalized to new color depth image
	"""

	if dtype is None:
		dtype = image_data.dtype
	maxX = max(image_data.flat)
	minX = min(image_data.flat)
	temp = np.divide(image_data - minX, maxX - minX)
	modified_image = np.multiply(temp, S).astype(dtype)

	return modified_image


def append_zeros(first_arr: np.ndarray, second_arr: np.ndarray) -> (np.ndarray, np.ndarray):
	"""
	Append zeros to one of arrays to make them length equal

	:param first_arr: First array of data
	:param second_arr: Second array of data
	:return: None
	"""
	if len(first_arr) > len(second_arr):
		zeros_number = len(first_arr) - len(second_arr)
		second_arr = np.append(second_arr, ([0] * zeros_number))
	else:
		zeros_number = len(second_arr) - len(first_arr)
		first_arr = np.append(first_arr, ([0] * zeros_number))
	return first_arr, second_arr


def make_complex(real_part, imag_part) -> np.ndarray:
	"""
	Make complex array from real and imaginary parts

	:param real_part: Array of real parts
	:param imag_part: Array of imaginary parts
	:return: Array of complex numbers
	"""

	assert len(real_part) >= len(imag_part), "Check arrays sizes"
	result = np.empty(len(real_part), dtype=complex)
	result.real = real_part
	result.imag = imag_part

	return result


def divide_complex(real_A, img_A, real_B, img_B) -> np.ndarray:
	"""
	Implement complex division of al elements in array

	:param real_A: Array of real values
	:param img_A: Array of img values
	:param real_B: Array of real values
	:param img_B: Array of img values
	:return: Resulting array
	"""

	assert len(real_A) == len(real_B), "Lengths of array are different"

	arrA = make_complex(real_A, img_A)
	arrB = make_complex(real_B, img_B)

	return np.divide(arrA, arrB)


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