import sys
sys.path.append(r'C:\Users\Kell\PycharmProjects\ExpData')

import random
from pylab import frange

import grapher
import analysis as a
import prng

p = prng.PRNG()

grapher.init()

length = 5000
left = 0
right = 1

fake_random = [p.next(left=left, right=right) for i in range(0, length)]
real_random = [random.random() for j in range(0, length)]

print("Functions:\t\t{0}\t{1}".format("My realization",
                                      "Native Realization"))
print("Mean:\t\t{0}\t\t{1}".format(round(a.mean(fake_random), 7),
                                   round(a.mean(real_random), 7)))
print("Square mean:\t{0}\t\t{1}".format(round(a.square_mean(fake_random), 7),
                                        round(a.square_mean(real_random), 7)))
print("Root Mean Square:\t{0}\t\t{1}".format(round(a.root_mean_square(fake_random), 7),
                                             round(a.root_mean_square(real_random), 7)))
print("Variance:\t\t{0}\t\t{1}".format(round(a.variance(fake_random), 7),
                                       round(a.variance(real_random), 7)))
print("SQRT Variance:\t{0}\t\t{1}".format(round(a.sqrt_variance(fake_random), 7),
                                          round(a.sqrt_variance(real_random), 7)))
print("Skewness:\t\t{0}\t\t{1}".format(round(a.skewness(fake_random), 7),
                                       round(a.skewness(real_random), 7)))
print("Kurtosis:\t\t{0}\t\t{1}".format(round(a.kurtosis(fake_random), 7),
                                       round(a.kurtosis(real_random), 7)))

bar_count = 10
grapher.set_subplot(subplot_number=1, x_arr=range(0, bar_count),
                    xticks=[range(0, bar_count),
                            [str(i) for i in frange(left, right, (right - left) / bar_count)]],
                    y_arr=a.density(fake_random, bar_count), bar=True, title='My Random()')
grapher.set_subplot(subplot_number=2, x_arr=range(0, bar_count),
                    xticks=[range(0, bar_count),
                            [str(i) for i in frange(0, 1, 0.1)]],
                    y_arr=a.density(real_random, bar_count), bar=True, title='Native Random()')
grapher.show()

ac_fake = [a.autocorrelation(fake_random, i) for i in range(0, len(fake_random) - 1)]
ac_real = [a.autocorrelation(real_random, i) for i in range(0, len(real_random) - 1)]
cca = [a.crosscorrelation(fake_random, real_random, i) for i in range(0, len(fake_random) - 1)]
ccb = [a.crosscorrelation(real_random, fake_random, i) for i in range(0, len(real_random) - 1)]

grapher.set_subplot(subplot_number=1, x_arr=range(0, len(fake_random) - 1), y_arr=ac_fake,
                    title='AutoCorr of my function')
grapher.set_subplot(subplot_number=2, x_arr=range(0, len(real_random) - 1), y_arr=ac_real,
                    title='AutoCorr of native function')
grapher.set_subplot(subplot_number=3, x_arr=range(0, len(fake_random) - 1), y_arr=cca,
                    title='CrossCorr of my R() to native')
grapher.set_subplot(subplot_number=4, x_arr=range(0, len(real_random) - 1), y_arr=ccb,
                    title='CrossCorr of native R() to mine')
grapher.show()

length = 10000
fake_random = [p.next() for i in range(0, length)]

array_f = []
for i in range(0, 10):
	array_f.append(fake_random[int(i * length / 10):int((i + 1) * length / 10)])


mean_arr = [a.mean(i) for i in array_f]
t = a.sqrt_variance(mean_arr)
print('Дисперсия средних по 10 замерам: {0},'
      ' относительно максимального результата функции: {1}%'.format(t, round(t * 100 / right, 2)))
stationarity = False if round(t * 100 / right) > 5 else True

variance_arr = [a.variance(i) for i in array_f]
t = a.sqrt_variance(variance_arr)
print('Дисперсия дисперсий по 10 замерам: {0},'
      ' относительно максимального результата функции: {1}%'.format(t, round(t * 100 / right, 2)))
stationarity = False if round(t * 100 / right) > 5 else stationarity

print('Процесс стационарен: ' + str(stationarity))