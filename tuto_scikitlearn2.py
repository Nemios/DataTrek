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
