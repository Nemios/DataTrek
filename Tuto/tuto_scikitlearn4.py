import os

os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

###############################################################
# 24/30 SKLearn : Apprentissage non-supervisé
###############################################################
# Deep Learning : machine apprend toute seule à différencier les éléments
# Clustering : regroupement par certains points communs (classification documents, photos, tweets, ...)
# Détection d'anomalies : sécurité, fraude bancaire, défaillance usine, ...
# Réduction dimensionalité : simplification de la structure en gardant éléments principaux

###############################################################
# K-means clustering
# utilisation centroid (centre de masse)
# algo itératif en 2 étapes
# 1. affectation des points au centroid le plus proche
# 2. déplacement du centroid en centre des points affectés
# on répète jusqu'à ce que le centroid soit fixe

# erreur possible en fonctions de la position initiale des centroids
# SOLUTION : on lance plusieurs fois l'algo avec des positions initiales différentes
# et on garde le cas où les distances finales aux centroids sont les plus faibles (ie inertia / min variance)

# génération données
X, y = make_blobs(n_features=100, centers=3, cluster_std=0.8)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.show()

model = KMeans(n_clusters=3)  # n_init=10, max_iter=300, init="k-means++"
# init="k-means++" permet d'initialiser aux points les plus éloignés pour converger pluys rapidement, très performant

model.fit(X)
print(model.predict(X))

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X))
print(model.cluster_centers_)  # positions centroids
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="r")
plt.show()

print(model.inertia_)  # ou model.score mais exprimée de façon négative

###############################################################
# Elbow Method
# solution quand bcp de dimension dans dataset et/ou données mélangées => clusters pas évidents
# Elbow Method trace l'évolution du cout du modèle en fonction du nb de clusters
# on détecte "le coude" dans ce graphe qui est le bon nombre de cluster à utiliser

inertia = []
K_range = range(1, 20)
for k in K_range:
    model = KMeans(n_clusters=k).fit(X)
    inertia.append(model.inertia_)

plt.figure()
plt.plot(K_range, inertia)
plt.xlabel("nombre de clusters")
plt.ylabel("cout du modele (Inertia)")
plt.show()

# ATTENTION : Kmeans peu performant sur clusters non convexes et anisotropes
# SOLUTION : dans sklearn.cluster, Agglomerative Clustering ou DBSCAN

###############################################################
# Détection d'anomalie : détecter échantillon très différent des autres
# Algo Isolation Forest

# génération des données avec une anomalie
X, y = make_blobs(n_samples=50, centers=1, cluster_std=0.5)
X[-1, :] = np.array([2.25, 5])

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.show()

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.01)
# on dit au modele "je pense qu'il y a 1% d'anomalie dans le dataset"

model.fit(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X))
plt.show()

# application nettoyage des outliers du dataset digits
from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images
X = digits.data
y = digits.target

print(X.shape)
# (1797, 64) ie 1797 images écrites chacune sur 64 pixels

plt.figure()
plt.imshow(images[42])
plt.show()

model = IsolationForest(random_state=0, contamination=0.02)
model.fit(X)
print(model.predict(X))  # tableau deF 1 (non anomalie) et -1 (anomalie)

# on filtre avec boolean indexing
outliers = model.predict(X) == -1
print(outliers)

plt.figure()
plt.imshow(images[outliers][0])  # premiere anomalie du dataset
plt.title(y[outliers][0])
plt.show()

###############################################################
# Réduction dimensionalité : simplification de la structure en gardant éléments principaux
# ACP : préserver un maximum la variance pour obtenir la projection la plus fidele a nos données
# 1. calcul matrice covariance
# 2. détermiantion des vecteurs propres de la matrice : ce sont les Composantes Principales
# 3. Projection des données sur ces axes

from sklearn.decomposition import PCA

model = PCA(n_components=10)

model.fit_transform(X)

# comment choisir n_components ?
# 1er cas : espace 2D ou 3D alors 2 ou 3
# 2e cas : compression pour accéléer machine sur taches de classification ou regression
# => on doit choisir n_components pour conserver 95-99% de la variance des données

# visualisation des données avec dataset digits
model = PCA(n_components=2)
X_reduced = model.fit_transform(X)

plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.colorbar()
plt.show()

# réduction de dimension avec dataset digits
model = PCA(n_components=64)
X_reduced = model.fit_transform(X)

plt.figure()
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.show()

print(np.argmax(np.cumsum(model.explained_variance_ratio_) > 0.99))
# renvoie 40 : a partir de la 40e composante principale, on a atteint 99% de conservation de la variance
# d'ou :
model = PCA(n_components=40)
X_reduced = model.fit_transform(X)

# si on veut observer les images apres compression :
# on les decompresse pour les refaire passer en 64 pixels$
X_recovered = model.inverse_transform(X_reduced)

plt.figure()
plt.imshow(X_recovered[0].reshape((8, 8)))
plt.show()

# ASTUCE : on peut aussi directement demander au modele de conserver un certain % de variance
model = PCA(n_components=0.95)
X_reduced = model.fit_transform(X)
X_recovered = model.inverse_transform(X_reduced)
plt.figure()
plt.imshow(X_recovered[0].reshape((8, 8)))
plt.show()
print(model.n_components_)

# ATTENTION :
# il faut standardiser les données avant d'utiliser PCA (StandardScaler)
# PCA est normalement conçue pour var continues
# PCA pas efficace sur datasets non linéaires => Alternatives : Manifold Learning (IsoMap, T-SNE)
