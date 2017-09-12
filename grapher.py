import matplotlib.pyplot as plt
import pylab

axs = {}

def init(figure_number=1, XKCD=False):
    if XKCD is True: plt.xkcd()
    plt.figure(figure_number)

def set_subplot(subplot_number, func, begin, end, step=1,
                xmin=None, xmax=None, ymin=None, ymax=None,
                grid=True, subplot_x_number=2, subplot_y_number=2,
                xlabel = None, ylabel = None, xticks = None, yticks = None,
                text_points = []):
    axs[subplot_number - 1] = plt.subplot(subplot_x_number, subplot_y_number, subplot_number)

    if xmin is not None and xmax is not None: axs[subplot_number - 1].set_xlim([xmin, xmax])
    if ymin is not None and ymax is not None: axs[subplot_number - 1].set_ylim([ymin, ymax])
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if xticks is not None: plt.xticks(xticks[0], xticks[1])
    if yticks is not None: plt.yticks(yticks[0], yticks[1])

    x_ar = pylab.frange(begin, end, step)
    plt.plot(x_ar, [func(x) for x in x_ar])

    for point in text_points:
        plt.text(point[0], point[1], point[2])
    plt.grid(grid)


def show():
    plt.show()