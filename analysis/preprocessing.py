def normalize(arr):
	minX = min(arr)
	maxX = max(arr)
	diff = maxX - minX

	for i in range(len(arr)):
		arr[i] = (arr[i] - minX) / diff

	return arr

def normalize_max(arr):
	maxX = max(arr)

	if maxX < 0:
		return arr

	for i in range(len(arr)):
		arr[i] = arr[i] / maxX

	return arr