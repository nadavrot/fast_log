import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import exp

# Generate the training data:
start = 0
end = 1
start_poly = start - 0.1
end_poly = end + 0.1
x = np.arange(start_poly, end_poly, 0.001)
y = np.exp(np.arange(start_poly, end_poly, 0.001))

# The polynomial to fit:
def func1(x, a, b, c, d):
  return a * x**3 + b*x**2 + c*x + d

# The sollya expression:
# display = decimal; Q = fpminimax(exp(x), 3, [|D...|], [0, 1]); Q;
def func2(x, a, b, c, d):
 return 0.99967771959938100945208816483500413596630096435547 +\
    x * (1.01217403745740819331899729149881750345230102539062 +\
    x * (0.43418272190290696510572843180852942168712615966797 +\
    x * 0.27137130054267810663759519229643046855926513671875))

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
