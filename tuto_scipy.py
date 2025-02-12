import numpy as np
import matplotlib.pyplot as plt

# https://docs.scipy.org/doc/scipy/reference/index.html

# interpolation : combler trous entre points d'un capteur pas bonne frequence
""" x = np.linspace(0, 10, 10)
y = x**2
plt.figure()
plt.scatter(x, y)
plt.show()

from scipy.interpolate import interp1d

f = interp1d(x, y, kind="linear")

new_x = np.linspace(0, 10, 30)
result = f(new_x)

plt.figure()
plt.scatter(x, y)
plt.scatter(new_x, result, c="r")
plt.show() """

# optimization
x = np.linspace(0, 2, 100)
y = 1 / 3 * x**3 - 3 / 5 * x**2 + 2 + np.random.randn(x.shape[0]) / 20
plt.figure()
plt.scatter(x, y)
plt.show()

# pour utiliser curvefit on doit définir un modele


def f(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


from scipy import optimize

print(optimize.curve_fit(f, x, y))
