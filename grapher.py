import matplotlib.pyplot as plt
from numpy import ndarray
from analysis.preprocessing import normalize_image

axs = {}

def set_subplot(subplot_number, y_arr, x_arr,
				xmin=None, xmax=None, ymin=None, ymax=None,
				grid=True, subplot_x_number=2, subplot_y_number=2,
				xlabel = None, ylabel = None, title = None,
				xticks = None, yticks = None,
				annotations = None, bar = False,
				figure_number=None, global_title=None, XKCD=False) -> None:
	"""
	Define subplot in graph

	:param subplot_number: Number of subplot
	:param y_arr: Array of function's values
	:param x_arr: Array of function's arguments
	:param xmin: Minimum value on X axis
	:param xmax: Maximum value on X asix
	:param ymin: Minimum value on Y axis
	:param ymax: Maximum value on Y axis
	:param grid: If grid on the subplot needed
	:param subplot_x_number: Number of subplots in row
	:param subplot_y_number: Number of subplots in column
	:param xlabel: Label of X axis
	:param ylabel: Label of Y axis
	:param title: Title of subplot
	:param xticks: List of 2 lists: 1 contains tick locations (usually list of numbers from start to end), 2 contains tick names
	:param yticks: List of 2 lists: 1 contains tick locations (usually list of numbers from start to end), 2 contains tick names
	:param annotations: List of annotations. Each element is a list of next elements:
		0 - text of annotation,
		1 - tuple (x, y) of desired point of annotation,
		2 - tuple (x, y) as annotation text coordinates
	:param bar: Bool param if bars or graph is needed
	:return: Nothing
	"""
	global axs

	if XKCD is True: plt.xkcd()
	fig = plt.figure(figure_number)
	if global_title is not None: fig.suptitle(global_title)

	axs[subplot_number - 1] = plt.subplot(subplot_x_number, subplot_y_number, subplot_number)

	if xmin is not None and xmax is not None: axs[subplot_number - 1].set_xlim([xmin, xmax])
	if ymin is not None and ymax is not None: axs[subplot_number - 1].set_ylim([ymin, ymax])
	if xlabel is not None: plt.xlabel(xlabel)
	if ylabel is not None: plt.ylabel(ylabel)
	if title is not None: plt.title(title)
	if xticks is not None: plt.xticks(xticks[0], xticks[1])
	if yticks is not None: plt.yticks(yticks[0], yticks[1])

	if bar:
		plt.bar(x_arr, y_arr)
	else:
		plt.plot(x_arr, y_arr)

	if annotations is not None:
		for point in annotations:
			plt.annotate(str(point[0]), xy=point[1], arrowprops=dict(arrowstyle='->'), xytext=point[2])
		plt.grid(grid)


def set_image(image_array: ndarray, title=None, cmap=None, normalize=True) -> None:
	fig = plt.figure()
	if normalize:
		image_array = normalize_image(image_array)
	plt.imshow(image_array, aspect="auto", interpolation="none", cmap=cmap)
	if title is not None: fig.suptitle(title)

def show() -> None:
	plt.show()