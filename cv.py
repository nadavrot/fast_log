import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import log

# Generate the training data:
x = np.arange(1, 2, 0.001)
y = np.log(np.arange(1, 2, 0.001))

# The polynomial to fit:
def func1(x, a, b, c, d): return a * x**3 + b*x**2 + c*x + d

params, _ = curve_fit(func1, x, y)
a, b, c, d = params[0], params[1], params[2], params[3]
yfit1 = func1(x, a, b, c, d)
yerr1 = yfit1 - y

print("The polynomial parameters:", params)

plt.plot(x, y, 'o', label="log")
plt.plot(x, yfit1, label="approximated log")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', fancybox=True)
plt.grid(True)
plt.show()

plt.plot(x, yerr1, label="error")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', fancybox=True)
plt.grid(True)
plt.show()
