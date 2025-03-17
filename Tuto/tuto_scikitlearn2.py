import numpy as np
import matplotlib.pyplot as plt

###############################################################
# 22/30 SKLearn : PRE-PROCESSING + PIPELINE
###############################################################

# Preprocessing ou pre traitement des données
# 1. Encodage : conversion qualitatives en quantitatives
# [chien, chat, chien, oiseau] = [0, 1, 0, 2]
# 2. Normalisation : mise sur mm echelle des quantitatives
# [2, 10, 4, 6] = [0, 1, 0.25, 0.5]
# 3. Imputation : remplacemennt certaines val par val statistiques
# [2, 1, 3, 'nan'] = [2, 1, 3, 2]
# 4. Selection : utilise test (ex khi2) pour garder uniquement les var les plus utiles au modele
# 5. Extraction : generation de nouvelles variables à partir d'infos cachées dans dataset

# SKLearn a donc plusieurs modules pour ces tâches
# sklearn.preprocessing pour 1. Encodage et 2. Normalisation
# sklearn.impute pour 3. Imputation
# sklearn.feature_selection pour 4. Selection
# sklearn.feature_extraction pour 5. Extraction

###############################################################
# sklearn.preprocessing pour 1. Encodage
###############################################################
# Encodage ordinal : associe chaque catégorie ou classe d'une var à une valeur décimale unique
# ex : [chien, chat, chien, oiseau] = [0, 1, 0, 2]
# pour cela il existe 2 Transformers :
# LabelEncoder() pour y (label/target)
# OrdinalEncoder()
# méthode inverse_transform permet de décoder les données

from sklearn.preprocessing import LabelEncoder

y = np.array(["chat", "chien", "chat", "oiseau"])

encoder = LabelEncoder()
encoder.fit(y)

print(encoder.classes_)

print(encoder.transform(y))

# pour aller plus vite :
print(encoder.fit_transform(y))

# pour décoder :
print(encoder.inverse_transform(np.array([0, 0, 2, 2, 1])))

# si plusieurs colonnes, on utilise OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder

X = np.array(
    [["chat", "poils"], ["chien", "poils"], ["chat", "poils"], ["oiseau", "plumes"]]
)

encoder = OrdinalEncoder()
print(encoder.fit_transform(X))

# ATTENTION : pas top d'un pov arithmétique car revient à dire chat < chien < oiseau ce qui n'a pas de sens
# et peu rendre le modele moins efficace

# SOLUTION : Encodage One-Hot, chaque catégorie est représentée de manière binaire dans une colonne qui lui est propre
# ex : [chat, chat, chien, oiseau, chien] = [[1,1,0,0,0] (chat), [0,0,1,0,1] (chien), [0,0,0,1,0] (oiseau)]]
# pour cela il existe 3 Transformers :
# LabelBinarizer()
# MultiLabelBinarizer()
# OneHotEncoder()
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, OneHotEncoder

encoder = LabelBinarizer()
print(encoder.fit_transform(y))
# INCONVENIENT : peut renvoyer des tableaux tres larges si bcp de variables
# SOLUTION : sparse matrix : la matrice obtenue est en grande partie remplie de zéro donc on peut supprimer les zeros du stockage
# ex :
encoder = LabelBinarizer(sparse_output=True)
print(encoder.fit_transform(y))
# cette compression est faite par defaut pour OneHotEncoder
encoder = OneHotEncoder()
print(encoder.fit_transform(X))

###############################################################
# sklearn.preprocessing pour 2. Normalisation
###############################################################
# Normalisation MinMax, transformation de chaque var pour que val soit compris entre 0 et 1
# formule : Xscaled = (X-Xmin)/(Xmax-Xmin)
from sklearn.preprocessing import MinMaxScaler

X = np.array([[70], [80], [120]])
scaler = MinMaxScaler()
print(scaler.fit_transform(X))

# ex avec iris
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

X_minmax = MinMaxScaler().fit_transform(X)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], label="Original")
plt.scatter(X_minmax[:, 2], X_minmax[:, 3], label="X_minmax")
plt.title("dataset Iris Normalisation MinMax")
plt.legend()
plt.show()

# Standardisation : transformation où toutes les var ont moyenne=0 et écart-type=1
# formule : Xscaled = (X-µ)/sigma
from sklearn.preprocessing import StandardScaler

X = np.array([[70], [80], [120]])
scaler = StandardScaler()
print(scaler.fit_transform(X))

# ex avec iris
X = iris.data
X_stdscl = StandardScaler().fit_transform(X)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], label="Original")
plt.scatter(X_minmax[:, 2], X_minmax[:, 3], label="X_minmax")
plt.scatter(X_stdscl[:, 2], X_stdscl[:, 3], label="X_stdscl")
plt.title("dataset Iris Normalisation MinMax + Standardisation")
plt.legend()
plt.show()

# ATTENTION : ces deux techniques sont sensibles aux outliers
# SOLUTION : si outliers => RobustScaler
# Robust Scaler : soustraction non plus avec la moyenne mais avec la médiane qui est moins sensible aux outliers
# formule : Xscaled = (X-mediane)/IQR avec IQR l'intervalle entre le 1er et le 3e quartile
X = iris.data
# on ajoute des outliers
outliers = np.full((10, 4), 100) + np.random.randn(10, 4)
X = np.vstack((X, outliers))
X_minmax = MinMaxScaler().fit_transform(X)
X_stdscl = StandardScaler().fit_transform(X)

from sklearn.preprocessing import RobustScaler

X_robustscl = RobustScaler().fit_transform(X)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], label="Original")
plt.scatter(X_minmax[:, 2], X_minmax[:, 3], label="X_minmax")
plt.scatter(X_stdscl[:, 2], X_stdscl[:, 3], label="X_stdscl")
plt.scatter(X_robustscl[:, 2], X_robustscl[:, 3], label="X_robustscl")
plt.title(
    "dataset Iris outliers Normalisation MinMax + Standardisation + Robust_Scaler"
)
plt.legend()
plt.show()

###############################################################
# sklearn.preprocessing PolynomialFeatures
###############################################################
# Feature Engineering = créer des var polynomiales à partir de var existantes
# => modeles de ML plus riches et developpés
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# exemple : regression lineaire ax + b devient en polynomiale ax^2 + bx + c
X = np.linspace(0, 4, 100).reshape((100, 1))
y = X**2 + 5 * np.cos(X) + np.random.randn(100, 1)

plt.figure()
plt.scatter(X, y)

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

plt.plot(X, y_pred, c="r", lw=3)
plt.show()

X_poly = PolynomialFeatures(3).fit_transform(X)
model = LinearRegression().fit(X_poly, y)
y_pred = model.predict(X_poly)
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_pred, c="r", lw=3)
plt.show()

# ATTENTION à bien normaliser données après avoir utilisé PolynomialFeatures

###############################################################
# sklearn.preprocessing Discrétisation : découper var continue en plusieurs parties
###############################################################
# Binarizer(threshold=n) permet de mettre toutes les var en dessous de n en 0 et les autres en 1
# KBinsDiscretizer(n_bins=n) idem mais en plus que 2 catégories
