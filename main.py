import sys
sys.path.append(r'C:\Users\Kell\PycharmProjects\ExpData')

import grapher
import math
from analysis import FT

# Шаг дискретизации
deltaT = 0.002

# Количество точек замера
N = 1000

def sinFunc(A0, f0, dt):
	return lambda t: A0 * math.sin(2*math.pi*f0*dt*t)

f1 = sinFunc(20, 5, deltaT)
f2 = sinFunc(100, 57, deltaT)
f3 = sinFunc(35, 190, deltaT)
sinList = [f1(t) + f2(t) + f3(t) for t in range(N)]

grapher.init(1)
grapher.set_subplot(1, x_arr=range(N), y_arr=sinList, title="Sin function")

FT_res = FT.fourier_transform(sinList, deltaT)
x_arr = [i * FT_res.deltaF for i in range(N)]

grapher.set_subplot(2, x_arr=x_arr, y_arr=FT_res.Re,
                    xmin=0, xmax=FT_res.borderF,
                    title="Re list")

grapher.set_subplot(3, x_arr=x_arr, y_arr=FT_res.Im,
                    xmin=0, xmax=FT_res.borderF,
                    title="Im list")

grapher.set_subplot(4, x_arr=x_arr, y_arr=FT_res.frequencies,
                    xmin=0, xmax=FT_res.borderF,
                    title="Frequencies list")

grapher.show()
