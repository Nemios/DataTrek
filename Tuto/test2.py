from test import fibo, trier
import math
import random
import statistics
import os
import glob

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
"""

import numpy as np

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
