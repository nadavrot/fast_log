import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import log

# Generate the training data:
start = 0.5
end = 1
start_poly = start - 0.1
end_poly = end + 0.1
x = np.arange(start_poly, end_poly, 0.001)
y = np.log(np.arange(start_poly, end_poly, 0.001))/np.log(2)

# The polynomial to fit:
def func1(x, a, b, c, d):
  return a * x**3 + b*x**2 + c*x + d

# The sollya expression:
# display = decimal; Q = fpminimax(log(x)/log(2), 3, [|D...|], [0.5, 1.0]); Q;
def func2(x, a, b, c, d):
  return -3.10688310292421920877359298174269497394561767578125 + x * (5.8216322434128127127905827364884316921234130859375 + x * (-3.75924393052939986858973497874103486537933349609375 + x * 1.04449239329354615080092116841115057468414306640625))

params, _ = curve_fit(func1, x, y)
a, b, c, d = params[0], params[1], params[2], params[3]
yfit1 = func1(x, a, b, c, d)
yerr1 = yfit1 - y
yfit2 = func2(x, a, b, c, d)
yerr2 = yfit2 - y

print("The polynomial parameters:", params)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
p0 = axes[0]
p1 = axes[1]
min_fit = min(min(yerr1), min(yfit2))
max_fit = min(max(yfit1), max(yfit2))
p0.plot(x, y, 'o', label="log")
p0.plot(x, yfit1, label="curve_fit", color="red")
p0.plot(x, yfit2, label="minimax", color="gold")
p0.vlines(x = start, ymin = min_fit, ymax = max_fit, colors = 'purple')
p0.vlines(x = end, ymin = min_fit, ymax = max_fit, colors = 'purple')
p0.legend(loc='best', fancybox=True)
p0.grid(True)

min_err = min(min(yerr1), min(yerr2))
max_err = min(max(yerr1), max(yerr2))
p1.plot(x, yerr1, label="curve_fit error", color="red")
p1.plot(x, yerr2, label="minimax error", color="gold")
p1.legend(loc='best', fancybox=True)
p1.vlines(x = start, ymin = min_err, ymax = max_err, colors = 'purple')
p1.vlines(x = end, ymin = min_err, ymax = max_err, colors = 'purple')
p1.grid(True)
fig.tight_layout()
plt.show()
