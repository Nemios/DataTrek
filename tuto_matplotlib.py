import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

###############################################################
# 14/30 Matplotlib les Bases
###############################################################

# ne pas mélanger méthode fonction et méthode POO

# méthode Fonction :
"""
x = np.linspace(0, 2, 10)
y = x**2
print(x)
plt.figure(figsize=(12, 8))  # appelle une figure sans rien de taille en inch
plt.plot(x, y, label="quadratique", c="red", lw=5, ls="--")
plt.plot(x, x**3, label="cubique", lw=2, ls="-", c="yellow")
plt.title("figure 1")
plt.xlabel("axe x")
plt.ylabel("axe y")
plt.legend()  # affiche les labels
plt.show()
# plt.savefig("figure.png")  # sauvegarde figure

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x), c="red", label="sinus")
plt.xlabel("axe x")
plt.ylabel("axe y")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x), c="blue", label="cosinus")
plt.title("figure 2")
plt.xlabel("axe x")
plt.ylabel("axe y")
plt.legend()
plt.show()
# Exo
dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}


def graphique(dataset):
    n = len(dataset)
    plt.figure(figsize=(12, 8))
    for k, i in zip(dataset.keys(), range(1, n + 1)):
        plt.subplot(n, 1, i)
        plt.plot(dataset[k])
        plt.title(k)
    plt.show()


graphique(dataset)

###############################################################
# 15/30 Matplotlib Graphes importants
###############################################################
"""
# Scatter
iris = load_iris()

x = iris.data
y = iris.target
names = list(iris.target_names)

print(f"x contient {x.shape[0]} exemples et {x.shape[1]} variables")
print(f"il y a {np.unique(y).size} classes")

plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5, s=x[:, 2] * 100)
# on prend toutes les lignes et on choisit un index
# on colorie en fonction de la classe avec c=y
# on utilise s la size pour faire apparaitre la 3e variable : la taille des sépales
plt.xlabel("longueur sépale")
plt.ylabel("largeur sépale")
plt.show()
"""
# Graphique 3D
from mpl_toolkits.mplot3d import Axes3D

plt.figure()

ax = plt.axes(projection="3d")
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
plt.show()

# visualisation de modélisation mathématique en 3D
f = lambda x, y: np.sin(x) + np.cos(x + y)  # génération fonction anonyme lambda
X = np.linspace(0, 5, 100)  # on crée nos deux axes pour la base 3D
Y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(X, Y)  # on crée le quadrillage à partir des axes
Z = f(X, Y)
print(Z.shape)
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="plasma")
plt.show()

# Histogramme : distribution des données, normale ou non, symétrique ou non, etc
plt.figure()
plt.hist(x[:, 0], bins=20)
plt.hist(x[:, 1], bins=20)
plt.show()

# Histogramme 2D : distribution selon 2 variables
plt.figure()
plt.hist2d(x[:, 0], x[:, 1], cmap="Blues")
plt.xlabel("longueur sépale")
plt.ylabel("largeur sépale")
plt.colorbar()
plt.show()

# Contour plot : utile quand modele à 3D en vue du dessus
plt.figure()
plt.contour(X, Y, Z, 20, colors="black")
plt.show()

plt.figure()
plt.contourf(X, Y, Z, 20, cmap="RdGy")
plt.colorbar()
plt.show()

# Imshow : afficher n'importe quel tableau numpy
plt.figure()
plt.imshow(np.corrcoef(x.T), cmap="Blues")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Z)
plt.show()
"""


# exo
n = x.shape[1]
plt.figure(figsize=(12, 8))
for i in range(n):
    plt.subplot(n // 2, n // 2, i + 1)
    plt.scatter(x[:, 0], x[:, i], c=y)
    plt.xlabel("0")
    plt.ylabel(i)
plt.show()
