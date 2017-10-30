"""
Some data preprocessing functions
"""
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