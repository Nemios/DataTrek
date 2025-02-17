import numpy as np
import matplotlib.pyplot as plt

# https://docs.scipy.org/doc/scipy/reference/index.html

# interpolation : combler trous entre points d'un capteur pas bonne frequence
""" 
x = np.linspace(0, 10, 10)
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
plt.show() 

from scipy import optimize

# optimization
x = np.linspace(0, 2, 100)
y = 1 / 3 * x**3 - 3 / 5 * x**2 + 2 + np.random.randn(x.shape[0]) / 20
plt.figure()
plt.scatter(x, y)
# pour utiliser curvefit on doit définir un modele


def f(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


print(optimize.curve_fit(f, x, y))

params, param_cov = optimize.curve_fit(f, x, y)
plt.plot(x, f(x, params[0], params[1], params[2], params[3]), c="g", lw=3)
plt.show()
# ATTENTION : mieux à faire sur scikit learn

# Minimisation


def f(x):
    return x**2 + 15 * np.sin(x)


x = np.linspace(-10, 10, 100)
plt.figure()
plt.plot(x, f(x))
plt.show()

print(optimize.minimize(f, x0=-8))  # on choisit un point de départ x0
# -6.7 = minimum local
# on change le point de départ

print(optimize.minimize(f, x0=-5))
# -1.38 = minimum global

x0 = -5
result = optimize.minimize(f, x0=x0).x

plt.plot(x, f(x), lw=3, zorder=-1)
plt.scatter(result, f(result), s=100, c="r", zorder=1)
plt.scatter(x0, f(x0), s=200, marker="+", c="g", zorder=1)
plt.show()


# on reprend exemple du contour plot
def f(x):
    return np.sin(x[0]) + np.cos(x[0] + x[1]) * np.cos(x[0])


x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)

x, y = np.meshgrid(x, y)
plt.figure()
plt.contour(x, y, f(np.array([x, y])), 20)

# on initialise avec 2 pts car 2D
x0 = np.zeros((2, 1))
plt.scatter(x0[0], x0[1], marker="+", c="r", s=100)

# on reshape car x0 doit etre de shape (n,) et non (n,1) pour minimize
x0 = x0.reshape(
    x0.shape[0],
)
print(x0.shape)
result = optimize.minimize(f, x0=x0).x
plt.scatter(result[0], result[1], c="g", s=100)
plt.show()

# Traitement du signal
from scipy import signal

x = np.linspace(0, 20, 100)
y = x + 4 * np.sin(x) + np.random.randn(x.shape[0])
plt.figure()
plt.plot(x, y, label="base signal")

# detrend pour enlever une tendance, linéaire par exemple
new_y = signal.detrend(y)
plt.plot(x, new_y, label="detrended signal")
plt.legend()
plt.show()
"""

# Transformation de Fourier : extraire les fréquences d'un signal périodique
# https://docs.scipy.org/doc/scipy/reference/fftpack.html

"""
 x = np.linspace(0, 30, 1000)
y = 3 * np.sin(x) + 2 * np.sin(5 * x) + np.sin(10 * x)
plt.figure()
plt.plot(x, y, label="signal initial")
plt.legend()
plt.show()

from scipy import fftpack

fourier = fftpack.fft(y)
power = np.abs(fourier)
frequences = fftpack.fftfreq(y.size)
# on prend les valeurs absolues pour filtrer les frequences negatives
plt.figure()
plt.plot(np.abs(frequences), power, label="spectre de Fourier")
plt.legend()
plt.show()

# exemple de débruitage avec Fourier
# 1 on prend le sprectre de Fourier
# 2 Boolean Indexing pour supprimer les fréquences faibles (bruit)
# 3 FFT inverse pour repasser au signal réel et clean

y = y + np.random.random(x.shape[0]) * 10
plt.figure()
plt.plot(x, y, label="signal bruité")
plt.legend()
plt.show()

fourier = fftpack.fft(y)
power = np.abs(fourier)
frequences = fftpack.fftfreq(y.size)
plt.figure()
plt.plot(np.abs(frequences), power, label="spectre de Fourier du signal bruité")
fourier[power < 400] = 0
plt.plot(np.abs(frequences), np.abs(fourier), label="spectre de fourier nettoyé", c="r")
plt.legend()
plt.show()

filtered_signal = fftpack.ifft(fourier)
plt.figure()
plt.plot(x, y, c="b", label="signal bruité", lw=0.5)
plt.plot(x, filtered_signal, c="orange", label="signal filtré", lw=3)
plt.legend()
plt.show()
 """

# Traitement d'image
# https://docs.scipy.org/doc/scipy/reference/ndimage.html

# Morphology
