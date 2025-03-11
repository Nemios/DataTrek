import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *

y = np.array([1, 5, 6])
y_pred = np.array([5, 6, 9])

print("MAE :", mean_absolute_error(y, y_pred))
print("MSE :", mean_squared_error(y, y_pred))
# en général on n'utilise pas la MSE mais la RMSE (root mean squared error)
print("RMSE :", np.sqrt(mean_squared_error(y, y_pred)))

# Quand utiliser la MAE ou la RMSE ?
# MAE => importance d'une erreur est linéaire avec son amplitude ou si dataset contient outliers
# RMSE => on accorde une grande importance aux grandes erreurs

print("median abs error :", median_absolute_error(y, y_pred))
# utile car si modele fait une seule mais tres grosse mauvaise prediction, score moyen tres impacté
# or moins le cas ici

# de maniere generale, il vaut mieux toutes les utilisées
# voire meme visualiser les erreurs avec un histogramme

# exemple
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

wine = load_wine()
X = wine.data
y = wine.target
model = LinearRegression()
model.fit(X, y)
print(model.score(X, y))
y_pred = model.predict(X)

plt.figure()
plt.scatter(X[:, 5], y, label="y")
plt.scatter(X[:, 5], y_pred, alpha=0.8, label="y_pred")
plt.legend()
plt.show()

plt.figure()
err_hist = np.abs(y - y_pred)
plt.hist(err_hist, bins=50)
plt.show()

# Coeff de détermination R^2
# implémenté par défaut dans la méthode score de sklearn
# évalue la performance du modèle par rapport au niveau de variation présent dans les données
# R^2 = 1 - (erreur/variance)
# plus il est proche de 1, mieux c'est
# ici , le modèle décrit 90% des variations

# pour utiliser toutes ses metrics dans une cross validation :
# https://scikit-learn.org/stable/modules/model_evaluation.html
print(
    cross_val_score(LinearRegression(), X, y, cv=3, scoring="neg_mean_absolute_error")
)
