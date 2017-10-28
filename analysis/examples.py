"""
Some example functions
"""
import math

def sin(A0, f0, dt):
	return lambda t: A0 * math.sin(2 * math.pi * f0 * dt * t)

def heartbeat(f0, alpha, dT):
	return lambda t: math.sin(2 * math.pi * f0 * dT * t) * (math.e ** (-alpha * t * dT))