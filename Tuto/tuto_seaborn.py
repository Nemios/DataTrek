import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

###############################################################
# 19/30 Seaborn Pairplot
###############################################################
# https://seaborn.pydata.org/index.html
# Fonctions Seaborn TREEEEES UTILE
# https://seaborn.pydata.org/api.html#

iris = pd.read_csv(r"Tuto\seaborn-data-master\iris.csv")
print(iris.head())

# Exemple :
# avec Matplotlib
plt.figure()
plt.scatter(iris["sepal_length"], iris["sepal_width"])
plt.show()
# peu lisible, nécessite encore beaucoup de code pour avoir un affichage efficace (couleurs, axes,...)

# avec Seaborn
sns.pairplot(iris, hue="species")
plt.show()

# fonction type seaborn
# sns.fonction(x, y, data, hue, size, style)

titanic = sns.load_dataset("titanic")
titanic.drop(
    ["alone", "alive", "who", "adult_male", "embark_town", "class"],
    axis=1,
    inplace=True,
)
titanic.dropna(axis=0, inplace=True)
print(titanic.head())
sns.pairplot(titanic)
# beaucoup de graphe avec peu de points car beaucoup de variable de catégorie
# on checke les fonctions (API reference du lien) de Categorical plots

# Categorical plots
sns.catplot(x="pclass", y="age", data=titanic, hue="sex")
plt.show()
sns.boxplot(x="pclass", y="age", data=titanic, hue="sex")
plt.show()

# Distribution plots
# selon 1 varriable
sns.displot(titanic["fare"])
plt.show()

# selon 2 variables
sns.jointplot(data=titanic, x="age", y="fare", kind="kde")  # kde, hex, ...
plt.show()

# Heatmap (surtout corrélation)
sns.heatmap(titanic.corr())
plt.show()

# Fonctions les plus utiles
# Pairplot()
# Catplot()
# Boxplot()
# Distplot()
# Jointplot()
# Heatmap()


# Quand utiliser Seaborn ou Matplotlib ?

# Data
# Exploration Statistique
# Vision gloabale
# ==> Seaborn (très global)

# Fonctions math, matrices, etc
# Mathématique, science, ingénierie, etc
# Graphique spécialisé
# ==> Matplotlib
