import numpy as np
import matplotlib.pyplot as plt

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
np.random.seed(0)
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
plt.show()

###############################################################
# Classification
###############################################################
