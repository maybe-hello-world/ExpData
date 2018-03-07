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


def xcr_reader(filepath: str) -> (int, int, int, np.ndarray):
	"""
	Read XCR image

	:param filepath: path to XCR file
	:return: (columns_number, rows_number, image_array)
	"""
	with open(filepath, 'rb') as f:
		data = f.read()

	col_num = int(data[608:624].decode('utf-8').partition('\0')[0])
	row_num = int(data[624:640].decode('utf-8').partition('\0')[0])
	image_len = col_num * row_num

	depth = 16

	image_data = list(data[2048:(2048 + image_len*2)])

	for i in range(0, len(image_data), 2):
		image_data[i], image_data[i+1] = image_data[i+1], image_data[i]

	image_data = list(struct.unpack(str(image_len) + "H", bytes(image_data)))
	image_data = np.array(image_data).reshape((col_num, row_num))

	image_data = np.flipud(image_data)
	# image_data = np.fliplr(image_data)
	return col_num, row_num, depth, image_data


def jpg_reader(filepath: str) -> (int, int, int, np.ndarray):
	"""
	Read JPG files

	:param filepath: path to JPG file
	:return: (columns_number, rows_number, image_array)
	"""
	jpgfile = Image.open(filepath)

	col_num = jpgfile.width
	row_num = jpgfile.height

	depth = jpgfile.bits

	image_data = np.array(jpgfile.getdata(), dtype=np.ubyte)

	channels = len(image_data.flat) / row_num / col_num
	if channels // 1 != channels:
		raise ValueError("Inconsistent data: can't divide into channels")
	channels = int(channels)

	if channels == 1:
		image_data = image_data.reshape((row_num, col_num))
		image_data = np.stack((image_data,) * 3, -1)
	else:
		image_data = image_data.reshape((row_num, col_num, channels))

	return col_num, row_num, depth, image_data