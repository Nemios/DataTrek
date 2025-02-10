from test import fibo, trier
import math
import random
import statistics
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

""""
liste = fibo(50)
print(liste)

#######################################
# MATH
#######################################
print(math.pi)
print(math.cos(2 * math.pi))


#######################################
# STATISTICS
#######################################
print(statistics.mean(liste))
print(statistics.variance(liste))


#######################################
# RANDOM
#######################################
print(random.choice(liste))
random.seed(0)
print(random.choice(liste))  # tjr 8 car seed
print(random.randint(5, 10))  # random int entre 5 et 10
print(random.randrange(100))  # int entre 0 et 100
print(random.sample(range(100), random.randrange(10)))
random.shuffle(liste)
print(liste)


#######################################
# OS
#######################################
print(os.getcwd())  # current work directory

#######################################
# GLOBAL
#######################################
print(glob.glob("*"))  # all files in workspace
print(glob.glob("*.txt"))  # all .txt files in workspace
filenames = glob.glob("*.txt")

dico_files = {}
for file in filenames:
    with open(file, "r") as f:
        dico_files[file] = f.read()

print(dico_files)




tableau = np.array([1, 2, 3, 4])
print(tableau.size)
print(tableau.ndim)
print(tableau.shape)

# np.zeros(shape) avec shape = (ligne, colonne)
# np.ones(shape)
# np.full(shape, value)

tab0 = np.zeros((3, 4))
print(tab0)

# np.random.randn(ligne, colonne) distribution normale centrée en 0
print(np.random.randn(3, 4))

# np.linspace(debut, fin, quantite)
# np.arrange(debut, fin, pas)
print(np.linspace(0, 10, 20))
print(np.arange(0, 10, 2))

tab1 = np.ones((3, 4))
# horizontal stack = np.concatenate((tab0, tab1), axis=1)
print(np.concatenate((tab0, tab1), axis=1))
# vertical stack = np.concatenate((tab0, tab1), axis=0)
D = np.concatenate((tab0, tab1), axis=0)
print(D)
D = D.reshape(4, 6)
print(D)

A = np.array([1, 2, 3])
print(A.shape)
# avoir comme shape (3,1) au lieu de (3,) est des fois utile
A = A.reshape(A.shape[0], 1)
print(A.shape)
# avoir l'inverse peut parfois etre utile aussi
A = A.squeeze()
print(A.shape)
# convertir un tab de dimension n en dimension 1 en mettant les elements à la suite
print(D.ravel())


# exo numpy random, reshape, concatenate


def initialisation(m, n):
    bout = np.ones(m)
    print(bout)
    bout = bout.reshape(bout.shape[0], 1)
    matrice = np.random.randn(m, n)
    matrice = np.concatenate((matrice, bout), axis=1)
    return matrice


print(initialisation(3, 2))

################################################################################################################
# 11/30 Numpy Indexing Slicing Masking
################################################################################################################
# Indexing
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
print(A[1, 1])

# Slicing
print(A[:, 0])  # affichage tous les elements axe 1 (colonne)
print(A[0, :])  # affichage tous les elements axe 0 (ligne)
print(A[0])  # fonctionne aussi pour ligne
# tableau dans A
A1 = A[0:2, 0:2]
print(A1)
print(A1.shape)
A[0:2, 0:2] = 10
print(A)
# exo
A2 = A[:, -2:]
print(A2)
B = np.zeros((4, 4))
print(B)
B[1:3, 1:3] = 1
print(B)
C = np.zeros((5, 5))
print(C)
C[::2, ::2] = 1  # avec un pas mais pas tres utile en Data Science
print(C)

# BOOLEAN INDEXING, tres utile en Data Science
A = np.random.randint(0, 10, [5, 5])
print(A)
A[(A < 5) & (A > 2)] = 10  # comme un Mask
print(A)

# exo mask image
from scipy import misc

face = misc.face(gray=True)
# plt.imshow(face, cmap=plt.cm.gray)
# plt.show()

print(type(face))
print(face.shape)

a = face.shape[0]
b = face.shape[1]

face_quart = face[a // 4 : -a // 4, b // 4 : -b // 4]
face_quart[face_quart > 150] = 255
face_quart[face_quart < 25] = 0
plt.imshow(face_quart, cmap=plt.cm.gray)
plt.show()

# compression image
face = face[::2, ::2]
plt.imshow(face, cmap=plt.cm.gray)
plt.show()

################################################################################################################
# 12/30 Numpy Statistiques et Mathématiques
################################################################################################################
# doc Numpy ndarray
# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
np.random.seed(0)
A = np.random.randint(0, 10, [2, 3])
print(A)
print(A.sum(axis=0))  # somme selon axe vertical
print(
    A.cumsum()
)  # somme sur tous les elements en parcourant les éléments de chaque ligne

print(A.prod())  # produit du tableau
print(A.cumprod())  # produit du tableau

print(A.min(axis=0))

# Argmin Position du minimum
print(A.argmin(axis=0))

# Argsort trie le tableau et renvoie les index du tableau initial
print(A[0].argsort())

# Fonctions mathématiques ndarray
# doc Numpy Fonctions mathématiques
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html
print(np.exp(A))
print(np.sinh(A))

# Fonctions statistiques ndarray
# doc Numpy Fonctions statistiques
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.statistics.html
print(A.mean())
print(A.std())  # Ecart-type
print(A.var())  # Variance
print(np.corrcoef(A))  # matrice corrélations
print(np.corrcoef(A)[0][1])  # coeff corrélation ligne 1 / ligne 2
print(np.unique(A, return_counts=True))  # nb occurence elements
B = np.random.randint(0, 10, [5, 5])
print(np.unique(B, return_counts=True))

# utilisation de Argsort avec np.unique
values, counts = np.unique(B, return_counts=True)
print(values[counts.argsort()])
# mise en forme
for i, j in zip(values[counts.argsort()], counts[counts.argsort()]):
    print(f"le nombre {i} apparait {j} fois")

# NAN corrections, manque de nombre ou nombre corrompu
D = np.random.randn(5, 5)
D[0, 2] = np.nan
D[4, 3] = np.nan
print(D)
print(D.mean())  # marche pas comme tout le reste
print(np.nanmean(D))
print(np.nanvar(D))
print(np.nanstd(D))
print(np.isnan(D))
print(np.isnan(D).sum() / A.size)  # rapport Nan / taille
D[np.isnan(D)] = 0  # suppression nan
print(D)


# Fonctions algèbre linéaire ndarray
# doc Numpy Fonctions algèbre linéaire
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html
A = np.ones((2, 3))
B = np.ones((3, 2))
print(f"A = {A}")
print(f"B = {B}")
print(A.T)  # transposée
print(A.dot(B))  # produit matriciel
C = np.random.randint(0, 10, [3, 3])
print(np.linalg.det(C))  # déterminant d'une matrice carrée
print(np.linalg.inv(C))  # inversion d'une matrice carrée

# pas tout le temps utilisable avec dataset car souvent colonne colinéaire
# on peut utiliser une pseudo inverse
print(np.linalg.pinv(C))

print(np.linalg.eig(C))  # valeurs propres, vecteurs propres
"""

# exo numpy stats et maths
np.random.seed(0)
A = np.random.randint(0, 100, [10, 5])
print(A[:, 0])
for j in range(A.shape[1]):
    A[:, j] = (A[:, j] - A[:, j].mean()) / A[:, j].std()
print(A)
