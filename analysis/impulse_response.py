"""
Module that implements impulse response function
"""
import numpy as np
from numba import jit, float64, int64, double

@jit(float64[:](float64[:], float64[:]))
def process(x_values, h_values) -> list:
	"""
	Generate impulse response of system with given delta function values and input function values

	:param x_values: input function values list
	:param h_values: delta function values list
	:return: list of generated impulse response values. Do not forget to exclude extra values
	"""
	N = len(x_values)
	m = len(h_values)

	x_values_view = np.array(x_values, dtype=np.double)
	h_values_view = np.array(h_values, dtype=np.double)
	answer = np.array([0] * (N + m), dtype=np.double)

	for k in range(N + m):
		answer[k] = __step(x_values_view, h_values_view, k, N, m)

	return list(answer.tolist())

@jit(float64(double[:], double[:], int64, int64, int64))
def __step(x_values: np.ndarray, h_values: np.ndarray, step_number: int, N: int, M: int) -> float:
	"""
	Single step of Impulse Response process

	:param x_values: array of main function values
	:param h_values: array of delta function values
	:param step_number: step number
	:return: step value
	"""

	Y_k = 0

	for l in range(M):
		if step_number >= l and step_number - l < N:
			Y_k += x_values[step_number - l] * h_values[l]

	return Y_k