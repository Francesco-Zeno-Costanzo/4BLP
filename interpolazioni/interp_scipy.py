import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

N = 10
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x)

#interpolazione con una, spline cucbica (k=3)
s3 = InterpolatedUnivariateSpline(x, y, k=3)

z = np.linspace(0, 1, 100)

plt.figure(1)
plt.title('Interpolazione spline cubuca')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(z, s3(z), 'b', label='interpolazione')
plt.plot(x, y, marker='.', linestyle='', c='k', label='dati')
plt.legend(loc='best')
plt.grid()
plt.show()