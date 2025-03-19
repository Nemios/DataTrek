import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    Binarizer,
    LabelBinarizer,
    MultiLabelBinarizer,
    RobustScaler,
    PolynomialFeatures,
)
from sklearn.linear_model import (
    LinearRegression,
    SGDClassifier,
)  # SGD = Stochastic gradient descent
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline


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

""" y = np.array(["chat", "chien", "chat", "oiseau"])

encoder = LabelEncoder()
encoder.fit(y)

print(encoder.classes_)

print(encoder.transform(y))

# pour aller plus vite :
print(encoder.fit_transform(y))

# pour décoder :
print(encoder.inverse_transform(np.array([0, 0, 2, 2, 1])))

# si plusieurs colonnes, on utilise OrdinalEncoder

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

X = np.array([[70], [80], [120]])
scaler = MinMaxScaler()
print(scaler.fit_transform(X))

# ex avec iris

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

###############################################################
# sklearn pipeline
###############################################################
# Transformer Estimator

# Xtrain => Transformer (.fit_transform(Xtrain)) => Estimator (.fit(Xtraintransformed,ytrain))
# Xtest => Transformer (.transform(Xtest)) => Estimator (.predict(Xtesttransformed))

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Transformer
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)

# Estimator
model = SGDClassifier(random_state=0)
model.fit(X_train_transformed, y_train)

# Test
X_test_transformed = scaler.transform(X_test)
model.predict(X_test_transformed)

# Pipeline = Transformer + Estimator ie un Composite Estimator
# ie un estimateur composé de plusieurs éléments
# quand on utilise la méthode .fit du Composite Estimator, on utilise en une seule ligne
# les méthodes .fit de chaque élément
# idem pour les autres méthodes

# même code mais sous forme de Pipeline


model = make_pipeline(StandardScaler(), SGDClassifier())
model.fit(X_train, y_train)
model.predict(X_test)
# avantage Pipeline : permet de faire de la cross validation sur tous les éléments en mm temps

# Pipeline et GridSearchCV pour trouver les meilleurs parametres de la pipeline
# grid = GridSearchCV(pipeline, params, cv)
# params = {
# <Composant>__<parametre>:[]
# } dictionnaire des parametres de la pipeline
# grid.fit(X_train, y_train)
# grid.best_estimator


model = make_pipeline(
    PolynomialFeatures(), StandardScaler(), SGDClassifier(random_state=0)
)

print(model)

# dictionnaire de parametres
params = {
    "polynomialfeatures__degree": [2, 3, 4],
    "sgdclassifier__penalty": ["l1", "l2"],
}

grid = GridSearchCV(
    model, param_grid=params, cv=4
)  # cv = nb splits pour crossvalidation

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.score(X_test, y_test))

# Comparaison avec model sans preprocessing

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SGDClassifier(random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

###############################################################
# sklearn Pipeline Avancée : meilleur model pour dataset hétérogène
###############################################################

titanic = sns.load_dataset("titanic")
print(titanic.head())
y = titanic.loc[:, "survived"]
X = titanic.drop("survived", axis=1)

# on a un dataset hétérogène (var continues, discretes, ...) donc on ne peut pas
# utiliser l'expression : model = make_pipeline(StandardScaler(),...)
# car StandardScaler ne peut pas traiter tous les types de var (discretes, ...)
# donc on va trier les colonnes :
from sklearn.compose import make_column_transformer

# transformer sur colonnes
transformer = make_column_transformer((StandardScaler(), ["age", "fare"]))
transformer.fit_transform(X)  # ne transformer que les colonnes données précédemment

# utilisation avec pipeline
model = make_pipeline(transformer, SGDClassifier())


# de maniere generale, on regroupe les var en deux groupes : catégorielle ou numérique
numerical_features = ["pclass", "age", "fare"]
categorical_features = ["sex", "deck", "alone"]
from sklearn.impute import SimpleImputer  # enleve valeurs manquantes

numerical_pipeline = make_pipeline(SimpleImputer(), StandardScaler())
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), OneHotEncoder()
)
# SimpleImputer(strategy="most_frequent") remplace les missing values par les most frequent values
preprocessor = make_column_transformer(
    (numerical_pipeline, numerical_features),
    (categorical_pipeline, categorical_features),
)

model = make_pipeline(preprocessor, SGDClassifier())
model.fit(X, y)

# pour sélectionner les variables, on peut aussi utiliser make_column_selector
from sklearn.compose import make_column_selector

numerical_features = make_column_selector(dtype_include=np.number)
# toutes les var numeriques sont sélectionnées
categorical_features = make_column_selector(dtype_exclude=np.number)
# toutes les var non numeriques ie catégorielles sont sélectionnées
numerical_pipeline = make_pipeline(SimpleImputer(), StandardScaler())
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), OneHotEncoder()
)

preprocessor = make_column_transformer(
    (numerical_pipeline, numerical_features),
    (categorical_pipeline, categorical_features),
)

model = make_pipeline(preprocessor, SGDClassifier())
model.fit(X, y)

# Feature Union :  pipelines en parallèle
numerical_features = X.loc[:, ["age", "fare"]]
from sklearn.pipeline import make_union

pipeline = make_union(StandardScaler(), Binarizer())

pipeline.fit_transform(numerical_features)
"""

###############################################################
# sklearn Imputer : nettoyage de données
###############################################################

# SimpleImputer
from sklearn.impute import SimpleImputer

X = np.array([[10, 3], [0, 4], [5, 3], [np.nan, 3]])

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# strategy : "mean", "median", "most_frequent", "constant" avec fill_value = constant
print(imputer.fit_transform(X))

# KNNImputer
from sklearn.impute import KNNImputer

X = np.array([[1, 100], [2, 30], [3, 15], [np.nan, 20]])
imputer = KNNImputer(n_neighbors=1)
print(imputer.fit_transform(X))

# MissingImputer
from sklearn.impute import MissingIndicator

X = np.array([[1, 100], [2, 30], [3, 15], [np.nan, np.nan]])
print(MissingIndicator().fit_transform(X))

# très utile dans certains cas, exemple :
# imaginons qu'une personne du titanic est nan pour la classe car c'est un matelot
# on pourrait alors créer une var "membre équipage" pour ne pas pas perdre cette info
from sklearn.pipeline import make_union

pipeline = make_union(
    SimpleImputer(strategy="constant", fill_value=-99), MissingIndicator()
)
print(pipeline.fit_transform(X))

# ATTENTION : plus intelligent que juste remove nan avec pandas

# exemple application Transformers
titanic = sns.load_dataset("titanic")
print(titanic.head())
X = titanic.loc[:, ["pclass", "age"]]
y = titanic.loc[:, "survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = make_pipeline(KNNImputer(), SGDClassifier())

params = {"knnimputer__n_neighbors": [1, 2, 3, 4]}

grid = GridSearchCV(model, param_grid=params, cv=5)

grid.fit(X_train, y_train)

print(grid.best_params_)
