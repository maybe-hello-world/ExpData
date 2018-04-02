import struct
from PIL import Image
import numpy as np

def bin2float(filepath: str, length: int) -> tuple:
	"""
	Read (length) floats from raw byte file and return tuple of them

	:param filepath: path to binary file
	:param length: number of floats inside (each float is 32-bit number)
	:return: tuple with floats
	"""
	with open(filepath, 'rb') as f:
		ans = struct.unpack(str(length) + "f", f.read())

	return ans


def xcr_reader(filepath: str, col_num: int, row_num: int, offset: int = 0, reversed: bool = False) -> (int, int, int, np.ndarray):
	"""
	Read XCR image

	:param filepath: path to XCR file
	:param col_num: number of columns in img
	:param row_num: number of rows in img
	:param offset: offset from begin (for header)
	:param reversed: if bytes are reversed in image
	:return: (columns_number, rows_number, depth, image_array)
	"""
	with open(filepath, 'rb') as f:
		data = f.read()

	image_len = col_num * row_num

	depth = 16

	image_data = list(data[offset:(offset + image_len*2)])

	if reversed:
		for i in range(0, len(image_data), 2):
			image_data[i], image_data[i+1] = image_data[i+1], image_data[i]

	image_data = list(struct.unpack(str(image_len) + "H", bytes(image_data)))
	image_data = np.array(image_data).reshape((col_num, row_num))

	image_data = np.flipud(image_data)
	return col_num, row_num, depth, image_data


def jpg_reader(filepath: str, channels: int = 3) -> (int, int, int, np.ndarray):
	"""
	Read JPG files

	:param filepath: path to JPG file
	:return: (columns_number, rows_number, depth, image_array)
	"""
	assert channels == 1 or channels == 3, "Channels number must be 1 or 3"

	jpgfile = Image.open(filepath)

	col_num = jpgfile.width
	row_num = jpgfile.height

	depth = jpgfile.bits

	image_data = np.array(jpgfile.getdata(), dtype=np.ubyte)

	cur_channels = len(image_data.flat) / row_num / col_num
	if cur_channels // 1 != cur_channels:
		raise ValueError("Inconsistent data: can't divide into channels")
	cur_channels = int(cur_channels)

	# TODO: Refactor
	if channels == 3:
		if cur_channels == 1:
			image_data = image_data.reshape((row_num, col_num))
			image_data = np.stack((image_data,) * 3, -1)
		else:
			image_data = image_data.reshape((row_num, col_num, cur_channels))
	else:
		if cur_channels != 1:
			image_data = image_data.reshape((row_num, col_num, cur_channels))
			image_data2 = np.empty(shape=(image_data.shape[0], image_data.shape[1]))
			for i in range(len(image_data)):
				image_data2[i, :] = image_data[i, :, 0]
			image_data = image_data2
		else:
			image_data = image_data.reshape((row_num, col_num))

	return col_num, row_num, depth, image_data