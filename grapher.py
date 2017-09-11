import matplotlib.pyplot as plt

axs = {}

def init(figure_number=1, XKCD=False):
    if XKCD is True: plt.xkcd()
    plt.figure(figure_number)

def set_subplot(subplot_number, func, begin, end, xmin=None, xmax=None, ymin=None, ymax=None, grid=True, subplot_x_number=2, subplot_y_number=2):
    axs[subplot_number - 1] = plt.subplot(subplot_x_number, subplot_y_number, subplot_number)
    if xmin is not None and xmax is not None: axs[subplot_number - 1].set_xlim([xmin, xmax])
    if ymin is not None and ymax is not None: axs[subplot_number - 1].set_ylim([ymin, ymax])
    plt.plot([func(x) for x in range(begin, end)])
    plt.grid(grid)


def show():
    plt.show()