import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import log

# Generate the training data:
start = 1
end = 2
start_poly = start - 0.1
end_poly = end + 0.1
x = np.arange(start_poly, end_poly, 0.001)
y = np.log(np.arange(start_poly, end_poly, 0.001))

# The polynomial to fit:
def func1(x, a, b, c, d):   
  return a * x**3 + b*x**2 + c*x + d

# The sollya expression:
# display = decimal; Q = fpminimax(log(x), 3, [|D...|], [0.95, 2.05]); Q;
def func2(x, a, b, c, d): 
  return -1.5917102950301498243135256416280753910541534423828 + \
    x * (2.3280353343302793156510688277194276452064514160156 + \
    x * (-0.88005784993956648332158465564134530723094940185547 + \
    x * 0.143731192420675585319500555669947061687707901000977))
  
params, _ = curve_fit(func1, x, y)
a, b, c, d = params[0], params[1], params[2], params[3]
yfit1 = func1(x, a, b, c, d)
yerr1 = yfit1 - y
yfit2 = func2(x, a, b, c, d)
yerr2 = yfit2 - y

print("The polynomial parameters:", params)

min_val = min(yfit1 + yfit2)
max_val = max(yfit1 + yfit2)
plt.plot(x, y, 'o', label="log")
plt.plot(x, yfit1, label="poly log", color="red")
plt.plot(x, yfit2, label="sollya log", color="gold")
plt.xlabel('x')
plt.ylabel('y')
plt.vlines(x = start, ymin = min_val, ymax = max_val, colors = 'purple')
plt.vlines(x = end, ymin = min_val, ymax = max_val, colors = 'purple')
plt.legend(loc='best', fancybox=True)
plt.grid(True)
plt.show()

min_err = min(yerr1 + yerr2)
max_err = max(yerr1 + yerr2)
plt.plot(x, yerr1, label="poly error", color="red")
plt.plot(x, yerr2, label="sollya error", color="gold")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best', fancybox=True)
plt.vlines(x = start, ymin = min_err, ymax = max_err, colors = 'purple')
plt.vlines(x = end, ymin = min_err, ymax = max_err, colors = 'purple')
plt.grid(True)
plt.show()
