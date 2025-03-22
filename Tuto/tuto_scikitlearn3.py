import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    chi2,
    SelectFromModel,
)
from sklearn.linear_model import SGDClassifier

###############################################################
# 23/30 SKLearn : FEATURE SELECTION
###############################################################
# donner trop de variables (features) à un modele de Machine Learning peut l'affaiblir
# on ne garde que les features utiles

# Sélecteur Variance : Variance Threshold
# on veut éliminer les variables qui varient très peu

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

plt.figure()
plt.plot(X)
plt.legend(iris.feature_names)
plt.show()
# on remarque que certaines features varient beaucoup alors que d'autres très peu

print(X.var(axis=0))

selector = VarianceThreshold(threshold=0.2)
print(selector.fit_transform(X))
print(selector.get_support())  # mask boolean
# renvoie quelle var a été sélectionnée par le seuil à 0.2
print(np.array(iris.feature_names)[selector.get_support()])

###############################################################
# Select KBest, Selectpercentile, ...
print(chi2(X, y))
selector = SelectKBest(chi2, k=1)
print(selector.fit_transform(X, y))
print(selector.get_support())

selector = SelectKBest(chi2, k=2)
print(selector.fit_transform(X, y))
print(selector.get_support())

###############################################################
# SelectFromModel : entrainer un estimateur sur le modele
# fonctionne pour les modeles qui ont des coefficients comme reseau de neurones, modeles linéaires, arbres de décisions
# incompatible KNN car ne développe pas de coefficients et retient en mémoire tous les échantillons fournis

selector = SelectFromModel(SGDClassifier(random_state=0), threshold="mean")

print(selector.fit_transform(X, y))
print(selector.get_support())

# matrice des coefficients
print(selector.estimator_.coef_)
# matrice 3x4 car X.shape = (150, 4) et y.shape = (150,) mais y a 3 classes donc 150x3
# la matrice des coef permet de passer de X à y donc taille 3x4 ou 4x3 si on a utilisé une transposée

###############################################################
# RFE + RFECV : estimateurs récursifs
