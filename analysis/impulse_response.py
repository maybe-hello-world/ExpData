"""
Module that implements impulse response function
"""

def process(x_values, h_values) -> list:
	"""
	Generate impulse response of system with given delta function values and input function values

	:param x_values: input function values list
	:param h_values: delta function values list
	:return: list of generated impulse response values
	"""
	N = len(x_values)

	answer = []
	for k in range(N):
		answer.append(__step(x_values, h_values, k))

	return answer

def __step(x_values, h_values, step_number: int) -> float:
	"""
	Single step of Impulse Response process

	:param x_values: array of main function values
	:param h_values: array of delta function values
	:param step_number: step number
	:return: step value
	"""
	Y_k = 0
	M = len(h_values)
	N = len(x_values)

	upper_b = M if step_number > M else step_number
	lower_b = 0 if step_number - N < 0 else step_number

	for l in range(lower_b, upper_b):
		Y_k += x_values[step_number - l] * h_values[l]
	return Y_k