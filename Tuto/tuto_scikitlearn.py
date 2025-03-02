import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

###############################################################
# 20/30 SKLearn : KNN, LinearRegression et SUPERVISED LEARNING
###############################################################
# Machine Learning : donner à machine capacité d'apprendre sans la programmer de façon explicite
# dans les faits : développer modèle mathématique partir de données expérimentales
# 3 moyens : apprentissage supervisé, non-supervisé, par renforcement

# Apprentissage supervisé:
# Machine reçoit données caractérisées par variables xi (features) et annotées par variable y (label/target)
# But : prédire y (label/target) en fonction des xi (features)
# Etape 1 : on donne données (dataset)
# Etape 2 : On spécifie type de modèle (linéaire, polynomial, arbre de décisions, réseau de neurones,...)
# que machine doit apprendre
# + hyperparamètres dans le modèle (nb branches arbre décision, nb neurones réseau)
# Etape 3 : Machine suit algorithme optimisation pour trouver paramètres qui donnent best perf sur dataset
# = phase d'entrainement
# But : Régression (y est continue ie quantitative) ou Classification (y discrète ie qualitative)

# https://scikit-learn.org/stable/

# SKLearn = modèles déjà programmés en tant que classes orientés objet = estimateurs
# Ecriture : model = LinearRegression(....) (objet = Constructeur(Hyperparamètres))
# Exemple descente de Gradient : model = SGDRegressor(eta0=0.3) #Learning_rate=0.3
# Exemple Random Forest : model = RandomForestClassifier(n_estimatores=100) #nb arbes =100


# Etape 1 : Sélectionner un estimateur
# https://scikit-learn.org/stable/machine_learning_map.html
# model = LinearRegression(....)

# Etape 2 : Entraîner modèle sur données X (xi features), y (label/target) (divisées en 2 tab numpy)
# ==> X doit être de dimension [n_samples, n_features] et y de dimension [n_samples, 1]
# model.fit(X, y)

# Etape 3 : Evaluer le modèle
# model.score(X, y)

# Etape 4 : Utiliser le modèle
# model.predict(X)

###############################################################
# Régression
###############################################################
""" np.random.seed(0)
m = 100  # creating 100 samples
X = np.linspace(0, 10, m).reshape(m, 1)
y = X + np.random.randn(m, 1)

plt.figure()
plt.scatter(X, y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
# LinearRegression implémente la méthode des moindres carrés qui ne nécessite pas d'hyperparamètre
model.fit(X, y)
score = model.score(X, y)  # R^2 de la méthode des moindres carrés
print(score)
predictions = model.predict(X)
# tableau Numpy qui contient autant de resultats qu'il y a de données dans tableau X
print(predictions)


plt.plot(X, predictions, c="r")
plt.show()

# Si pblm non linéaire :
y2 = X**2 + np.random.randn(m, 1)
model2 = LinearRegression()
model2.fit(X, y2)
model2.score(X, y2)
predictions2 = model2.predict(X)
plt.figure()
plt.scatter(X, y2)
plt.plot(X, predictions2, c="r")
plt.show()

# pas le bon estimateur, il faut changer par exemple avec un modèle de Support Vector Machine (SVM)
from sklearn.svm import SVR  # regression

model = SVR(C=100)
y = y2
model.fit(X, y)
model.score(X, y)
predictions = model.predict(X)
plt.figure()
plt.scatter(X, y)
plt.plot(X, predictions, c="r")
plt.show() """

###############################################################
# Classification
###############################################################
""" titanic = sns.load_dataset("titanic")
print(titanic.shape)
print(titanic.head())

titanic = titanic[["survived", "pclass", "sex", "age"]]
titanic.dropna(axis=0, inplace=True)
titanic.loc[:, "sex"] = titanic["sex"].replace(["male", "female"], [0, 1])
print(titanic.head())

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

# dissocier label/target y du tableau
y = titanic.loc[:, "survived"]
print(y.head())
X = titanic.drop("survived", axis=1)
print(X.head())

# entrainement, test, validation
model.fit(X, y)
print(model.score(X, y))
predictions = model.predict(X)
print(predictions)


# fonction pour savoir si on aurait survécu à la catastrophe du titanic
def survie(model, pclass=3, sex=0, age=21):
    x = pd.DataFrame([[pclass, sex, age]], columns=X.columns)
    print(model.predict(x))
    # proba appartenance classe 0 ou 1
    # ATTENTION : méhode pas dispo pour tous les estimateurs
    print(model.predict_proba(x))


survie(model)

# exo
n = 0
highscore = 0
for i in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X, y)
    score = model.score(X, y)
    print(score)
    if score > n:
        n = i
        highscore = score

print((n, highscore))
 """
######################################################################################
# 21/30 SKLearn :  MODEL SELECTION : Train_test_split, Cross Validation, GridSearchCV
######################################################################################
# ON NE VALIDE JAMAIS UN MODELE AVEC LES DONNEES D'ENTRAINEMENT
# on fait des Xtrain, Ytrain pour model.fit() et Xtest, Ytest pour model.score()

# exemple avec iris
""" from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# test_size = % de données pour les tests
print("Train set:", X_train.shape)
print("Test set:", X_test.shape)

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
plt.title("Train set")
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8)
plt.title("Test set")
plt.show()

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
# score de 100% car test sur données entrainement

print("Test score:", model.score(X_test, y_test))

# ATTENTION : si on optimise les hyperparametres directement en regardant le score du test set
# on ne pourra plus l'utiliser ensuite
# on découpe donc le dataset en train-validation-test sets
# comme ça on regarde le meilleur des modeles pour validation set et on ne teste que ce modele
# avec le test set

# mais comment savoir si le dataset a été découpé de la bonne façon ?

# ==>

###############################################
# Cross-Validation
###############################################
# Cross validation inutile en DeepLearning
# le modele est découpé en n splits
# on test les n cas ou le validation set est en i-eme position
# on peut utiliser la méthode KFold (tres utile pour regression) LeaveOneOut (tres couteuse en calcul), 
# ou ShuffleSplit (on mélange, le train set et le validation set avant de les séparer, on valide et on recommence)

from sklearn.model_selection import cross_val_score

result = cross_val_score(
    KNeighborsClassifier(), X_train, y_train, cv=5, scoring="accuracy"
)
# cv (cross validation) = nb de splits dans le modele si juste un entier


######################## pour aller plus loin #################################################
# on peut aussi faire
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit
cv1 = KFold(5, random_state=0) # random_state pour définir seed
cv2 = LeaveOneOut()
cv3 = ShuffleSplit(4, test_size=0.2) # 4=nb splits et test_size = 20% en validation, 80% en train

# on utilise la méthode StratifiedKFold quand les classes sont déséquilibrées
# cela consiste à séparer les stratas (classes) ex: a survécu / est mort
# ensuite on divise les stratas en un certains nombre de groupe (le meme) par exemple 4 (survécu1, survécu2,..., mort1, mort2,...)
# on réassemble en 4 splits (survécu1mort1, survécu2mort2,...)
# on a donc 4 splits avec les memes proportions de strata 1 et 2 dans chaque, on évite de se retrouver avec des splits où manque un strata
from sklearn.model_selection import StratifiedKFold
cv4 = StratifiedKFold(4)

# on utilise la méthode Group KFold quand il y a dépendances entre des variables
# ie des groupes qui représentent un facteur d'influence
# ex : on regarde le cancer pour une meme famille, si il y a bcp de cancer alors on va considerer un risque genetique
# on separe en groupe : on fait des tas de cartes en fonction des familles (coeur, trefle, pique, carreau)
# on fait des splits dans chaque famille ex :4
# on assemble 4 splits (un de chaque famille) pour constituer le validation set
# on rassemble le reste pour former le train set
# on recommence en reprenant les 4 splits (ie 16 au total) et on réassemble 4 splits (un de chaque famille)
# jusqu'à tester toutes les combinaisons
from sklearn.model_selection import GroupKFold
cv5 = GroupKFold(5).get_n_splits(X, y, groups=X[:,0]) #groups pour titanic : les gens en pclass 1 sont dans le groupe 1, les gens dans en pclass2 dans le groupe 2

result = cross_val_score(
    KNeighborsClassifier(), X, y, cv=cv1,2,3,4 ou 5, scoring="accuracy"
)
##################################### fin de la parenthèse ##################################################

print(result)
print("moyenne des resultats:", result.mean())

# maintenant on veut trouver les meilleurs hyperparametres

#########################################################
# Validation Curve - optimisation d'un hyperparametre
#########################################################
from sklearn.model_selection import validation_curve

model = KNeighborsClassifier()
k = np.arange(1, 50)

train_score, val_score = validation_curve(
    model, X_train, y_train, param_name="n_neighbors", param_range=k, cv=5
)

# "n_neighbors" = nom de l'hyperparametre qu'on veut régler

plt.figure()
plt.plot(k, val_score.mean(axis=1))
plt.show()

#########################################################
# GridSearchCV - optimisation d'un hyperparametre
#########################################################
from sklearn.model_selection import GridSearchCV

param_grid = {"n_neighbors": np.arange(1, 20), "metric": ["euclidean", "manhattan"]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_score_)  # meilleur score

print(grid.best_params_)  # meilleur parametres

model = grid.best_estimator_  # meilleur modele

print(model.score(X_test, y_test))

# Confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, model.predict(X_test)))
# la matrice montre que parmi les éléments de la classe 3, un élément a été classé dans la classe 2

#################################################################################################
# Learning curve - évolution des performances du modele en fonction quantité données fournie
#################################################################################################
# performance finit toujours par plafonner donc on arrete de collecter des donnees si cela
# n'apporte plus rien au modele
from sklearn.model_selection import learning_curve

N, train_score, val_score = learning_curve(
    model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
)
print(N)

plt.figure()
plt.plot(N, train_score.mean(axis=1), label="train")
plt.plot(N, val_score.mean(axis=1), label="validation")
plt.xlabel("train_sizes")
plt.legend()
plt.show() """

# exo Titanic avec model Selection

titanic = sns.load_dataset("titanic")

titanic = titanic[["survived", "pclass", "sex", "age"]]
titanic.dropna(axis=0, inplace=True)
titanic.loc[:, "sex"] = titanic["sex"].replace(["male", "female"], [0, 1])
print(titanic.head())

from sklearn.neighbors import KNeighborsClassifier


# dissocier label/target y du tableau
y = titanic.loc[:, "survived"]
print(y.head())
X = titanic.drop("survived", axis=1)
print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# test_size = % de données pour les tests
print("Train set:", X_train.shape)
print("Test set:", X_test.shape)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
# score de 100% car test sur données entrainement

print("Test score:", model.score(X_test, y_test))

# ATTENTION : si on optimise les hyperparametres directement en regardant le score du test set
# on ne pourra plus l'utiliser ensuite
# on découpe donc le dataset en train-validation-test sets
# comme ça on regarde le meilleur des modeles pour validation set et on ne teste que ce modele
# avec le test set

# mais comment savoir si le dataset a été découpé de la bonne façon ?

# ==>

###############################################
# Cross-Validation
###############################################
# le modele est découpé en n splits
# on test les n cas ou le validation set est en i-eme position

from sklearn.model_selection import cross_val_score

result = cross_val_score(
    KNeighborsClassifier(), X_train, y_train, cv=5, scoring="accuracy"
)
# cv = nb de splits dans le modele
print(result)
print("moyenne des resultats:", result.mean())

# maintenant on veut trouver les meilleurs hyperparametres

#########################################################
# Validation Curve - optimisation d'un hyperparametre
#########################################################
from sklearn.model_selection import validation_curve

model = KNeighborsClassifier()
k = np.arange(1, 50)

train_score, val_score = validation_curve(
    model, X_train, y_train, param_name="n_neighbors", param_range=k, cv=5
)

# "n_neighbors" = nom de l'hyperparametre qu'on veut régler

plt.figure()
plt.plot(k, val_score.mean(axis=1))
plt.show()

#########################################################
# GridSearchCV - optimisation d'un hyperparametre
#########################################################
from sklearn.model_selection import GridSearchCV

param_grid = {"n_neighbors": np.arange(1, 20), "metric": ["euclidean", "manhattan"]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_score_)  # meilleur score

print(grid.best_params_)  # meilleur parametres

model = grid.best_estimator_  # meilleur modele

print(model.score(X_test, y_test))

# Confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, model.predict(X_test)))
# la matrice montre que parmi les éléments de la classe 3, un élément a été classé dans la classe 2

#################################################################################################
# Learning curve - évolution des performances du modele en fonction quantité données fournie
#################################################################################################
# performance finit toujours par plafonner donc on arrete de collecter des donnees si cela
# n'apporte plus rien au modele
from sklearn.model_selection import learning_curve

N, train_score, val_score = learning_curve(
    model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
)
print(N)

plt.figure()
plt.plot(N, train_score.mean(axis=1), label="train")
plt.plot(N, val_score.mean(axis=1), label="validation")
plt.xlabel("train_sizes")
plt.legend()
plt.show()
