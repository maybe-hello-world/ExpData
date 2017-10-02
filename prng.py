from datetime import datetime
import math

class PRNG:
	"""
	Simple pseudorandom number generator
	"""
	prev = 0

	def __init__(self, seed = None):
		"""
		Initialize prng

		:param seed: seed for initialization. Multiple initializations with similar seed will return similar sequences of number
		"""
		if seed is None:
			self.prev = datetime.now().microsecond
		else:
			self.prev = math.fabs(seed)


	def next(self, left = 0, right = 1):
		"""
		Get next pseudorandom value

		:param left: left boundary of possible values
		:param right: right boundary of possible values (does not include value itself)
		:return: pseudorandom float number in [left; right)
		:raise ValueError
		"""
		if left > right: raise ValueError("Left boundary must be <= right")

		self.prev = self.prev*148878.553 % 1000000
		return left + ((self.prev % 100) / 100) * (right - left)