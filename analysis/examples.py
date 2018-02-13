"""
Some example functions
"""
import math
import random



def sin(A0: float, f0: float, dt: float):
	"""
	Function that return math.sin function with given characteristics

	:param A0: amplitude of sin
	:param f0: frequency of sin
	:param dt: step of discretization for sin
	:return: lambda function of sin with given characteristics
	"""
	return lambda t: A0 * math.sin(2 * math.pi * f0 * dt * t)

def heartbeat(dT: float, f0 : float = 14, alpha : float = 45):
	"""
	Function that imitates heartbeat

	:param dT: step of discretization
	:param f0: given frequency for heart imitating function (not freq of heart rate but some math value for imitating)
	:param alpha: given speed of signal decreasing (some math value for imitating)
	:return: lambda function of heartbeat
	"""
	return lambda t: math.sin(2 * math.pi * f0 * dT * t) * (math.e ** (-alpha * t * dT))

def spike(chance: float, sigma: float) -> float:
	"""
	Return signal of given amplitude (positive or negative) with given chance or 0

	:param chance: chance of spike appearing (0 < x < 1)
	:param sigma: amplitude of spike
	:return: signal of given amplitude or 0
	"""
	return sigma * (-1 if random.uniform(0, 1) < 0.5 else 1) if random.uniform(0, 1) < chance else 0

def shift(c: float) -> float:
	"""
	Return constant shift for data

	:param c: shift
	:return: c
	"""
	return c

def ito_process(a = 1, b = 0, c = 1, d = 1):
	"""
	Particular case of Kiyosi Ito stochastic process

	:param a: Константное смещение вверх, разделенное на c
	:param b: отвечает за экспоненциальный вид графика (подъем - спад - отсуствие влияние экспоненты)
	:param c: Длина отрезка рандомизатора и множитель a
	:param d: отвечает за периодичность (must be non-zero)
	:return: function of this process
	"""
	return lambda t: a * c + t * math.exp(b*t) + d * d * math.sin(t/d) * random.uniform(0, c)