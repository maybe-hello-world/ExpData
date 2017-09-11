import grapher
import math

def linear(x, a, b):
    return a*x + b

def exp(x, a, b):
    return b * (math.e ** ((- a) * x))

def f1(x):
    return linear(x, 3, 7)

def f2(x):
    return linear(x, -2, 0.3)

def f3(x):
    return exp(x, -0.01, 10)

def f4(x):
    return exp(x, 0.005, 12)

grapher.init(XKCD=True)
grapher.set_subplot(subplot_number=1, func=f1, begin=0, end=1000)
grapher.set_subplot(subplot_number=2, func=f2, begin=0, end=1000)
grapher.set_subplot(subplot_number=3, func=f3, begin=0, end=1000)
grapher.set_subplot(subplot_number=4, func=f4, begin=0, end=1000)
grapher.show()