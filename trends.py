import math

def linear(x, a, b):
	return a*x + b

def exp(x, a, b):
	return b * (math.e ** ((- a) * x))

def complex(parts):
	"""

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