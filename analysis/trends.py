"""
Simple module that contains linear, exponential and complex function realizations
"""
import math

def linear(a: float, b: float):
	"""
	Create linear function a*x + b

	:param a: Multiplier of X
	:param b: Constant shift of function
	:return: Lambda function F(x)
	"""
	return lambda x: a*x + b

def exp(a: float, b: float):
	"""
	Create exponential function  b * (math.e ^ (-a * x))

	:param a: Multiplier of X
	:param b: Multiplier of math.e value
	:return: Lambda function
	"""
	return lambda x: b * (math.e ** ((- a) * x))

def complex(parts) -> list:
	"""
	Calculate values for complex function (concatenation of different functions)

	:param parts: List of lists, where:
		0 - left bound of X values
		1 - right bound of X values
		2 - function that takes only X (parameters are inside of function)
	:return: List of Y points
	"""
	ans = []
	for part in parts:
		ans += [part[2](x) for x in range(part[0], part[1])]

	return ans