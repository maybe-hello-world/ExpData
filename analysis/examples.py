"""
Some example functions
"""
import math

def sin(A0: float, f0: float, dt: float) -> function:
	"""
	Function that return math.sin function with given characteristics

	:param A0: amplitude of sin
	:param f0: frequency of sin
	:param dt: step of discretization for sin
	:return: lambda function of sin with given characteristics
	"""
	return lambda t: A0 * math.sin(2 * math.pi * f0 * dt * t)

def heartbeat(dT: float, f0 : float = 14, alpha : float = 45) -> function:
	"""
	Function that imitates heartbeat

	:param dT: step of discretization
	:param f0: given frequency for heart imitating function (not freq of heart rate but some math value for imitating)
	:param alpha: given speed of signal decreasing (some math value for imitating)
	:return: lambda function of heartbeat
	"""
	return lambda t: math.sin(2 * math.pi * f0 * dT * t) * (math.e ** (-alpha * t * dT))