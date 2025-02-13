import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(r"C:\Users\baudo\Desktop\DataTrek\titanic3.xls")
print(data.shape)
print(data.columns)
print(data.head())
data = data.drop(
    [
        "name",
        "sibsp",
        "parch",
        "ticket",
        "fare",
        "cabin",
        "embarked",
        "boat",
        "body",
        "home.dest",
    ],
    axis=1,  # axis=1 car on élimine des colonnes
)
print(data.head())
print(data.describe())
# manque des données pour age
# soit data.fillna car remplit valeurs manquantes par valeur moyenne mais corrompt les data
# soit data.dropna car enleve les entrées avec données manquantes

data = data.dropna(axis=0)  # axis = 0 car on élimine des lignes
# ou data.dropna(axis=0, inplace = True) pour ne pas réécrire data = ...
print(data.shape)
print(data.describe())

# répartition en classes
print(data["pclass"].value_counts())
plt.figure()
data["pclass"].value_counts().plot.bar()
plt.show()
plt.figure()
data["age"].hist()
plt.show()

# groupby
print(data.groupby(["sex", "pclass"]).mean())

#######################################################
# Fonctionnement DataFrames
#######################################################
# DataFrame = assemblement de plusieurs Series (colonnes)
# une série = une colonne value et une colonne index
# on peut chosir d'indexer sur les noms par exemple plutot que sur des numéros
# data = pd.read_excel(r"C:\Users\baudo\Desktop\DataTrek\titanic3.xls")
# data = data.set_index("name")
# print(data["age"])

# un DataFrame fonctionne comme un dictionnaire de Series
# Df["column"] = une Serie
# apres ça fonctionne comme un tableau numpy
print(data[data["age"] < 18].groupby(["sex", "pclass"]).mean())

# ATTENTION, indexing pas comme numpy, on a besoin de fonction df.iloc pour manipuler index
print(data.iloc[0:2, 0:2])

# idem pour les colonnes avec df.loc
print(data.loc[0:2, ["age", "sex"]])

# CONSEIL
data.iloc[0:2]  # à éviter car sous-entend qu'on veut toutes les colonnes
data.iloc[0:2, :]  # mieux car on dit explicitement qu'on veut toutes les colonnes

# exo
dico_age = {}
for i in range(1, 5):
    print(i)
    if i == 1:
        dico_age[f"Cat{i-1}"] = data[(data["age"] < 20)]
    elif i != 4:
        dico_age[f"Cat{i-1}"] = data[
            (data["age"] > i * 10) & (data["age"] < 10 * i + 1)
        ]
    else:
        dico_age[f"Cat{i-1}"] = data[data["age"] > 10 * i]

print(dico_age)
