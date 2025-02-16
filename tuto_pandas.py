import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################
# 17/30 Intro + Titanic
###############################################################
data = pd.read_excel(r"C:\Users\baudo\Desktop\DataTrek\titanic3.xls")
"""
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
"""
# data = pd.read_excel(r"C:\Users\baudo\Desktop\DataTrek\titanic3.xls")
# data = data.set_index("name")
# print(data["age"])
"""
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

print(dico_age) """

# correction


###############################################################
# 18/30 Pandas Time Series
###############################################################

bitcoin = pd.read_csv(
    r"C:\Users\baudo\Desktop\DataTrek\aapl_us_d.csv", index_col="Date", parse_dates=True
)
""" print(bitcoin.head())
print(bitcoin.index)

plt.figure()
bitcoin["Volume"].plot()
plt.show()

plt.figure()
bitcoin.loc["2015":"2019", "Close"].plot()
plt.show()

# ATTENTION Pandas plutot libre niveau format de date mais si prbl : .to_datetime("date")

# resample
plt.figure()
bitcoin.loc["2019", "Close"].plot()
bitcoin.loc["2019", "Close"].resample("ME").mean().plot(
    label="moyenne par mois", lw=3, ls=":", alpha=0.8
)
bitcoin.loc["2019", "Close"].resample("W").mean().plot(
    label="moyenne par semaine", lw=2, ls="--", alpha=0.8
)
plt.legend()
plt.show()

# si on veut plusieurs stats en mm temps sur un resample
stats = bitcoin["Close"].resample("W").agg(["mean", "std", "min", "max"])

plt.figure(figsize=(12, 8))
stats.loc["2019", "mean"].plot(label="moyenne par semaine")
plt.fill_between(
    stats.index, stats["max"], stats["min"], alpha=0.2, label="min-max par semaine"
)
plt.legend()
plt.show()

# Moving Average : moyenne sur une fenêtre de valeurs qui se décale (rolling)
# Exponenting Moving Average
plt.figure()
bitcoin.loc["2019-09", "Close"].plot()
bitcoin.loc["2019-09", "Close"].rolling(window=7, center=True).mean().plot(
    label="Moving Average", lw=3, ls=":", alpha=0.8
)
bitcoin.loc["2019-09", "Close"].ewm(alpha=0.6).mean().plot(
    label="Exponenting Moving Average", lw=2, ls="--", alpha=0.8
)
plt.legend()
plt.show() """

# Fusionner plusieurs Datasets
ethereum = pd.read_csv(
    r"C:\Users\baudo\Desktop\DataTrek\aapl_us_d.csv", index_col="Date", parse_dates=True
)
"""
btc_eth = pd.merge(bitcoin, ethereum, on="Date", how="inner", suffixes=("_btc", "_eth"))
# inner = assemblage sur index en commun, si un dataframe en a en dehors les données sont perdues
# outer = NaN sur données qui n'existent pas
plt.figure()
btc_eth.loc[:, ["Close_btc", "Close_eth"]].plot(subplots=True)
plt.show()

print(btc_eth.loc[:, ["Close_btc", "Close_eth"]].corr())

# Exo Turtle Strategy
data = bitcoin.copy()
data["Buy"] = np.zeros(len(data))
data["Sell"] = np.zeros(len(data))

data["RollingMax"] = data["Close"].shift(1).rolling(window=28).max()
data["RollingMin"] = data["Close"].shift(1).rolling(window=28).min()
data.loc[data["RollingMax"] < data["Close"]] = 1
data.loc[data["RollingMin"] > data["Close"]] = -1

start = "2019"
end = "2019"
fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)
ax[0].plot(data["Close"][start:end])
ax[0].plot(data["RollingMax"][start:end])
ax[0].plot(data["RollingMin"][start:end])
ax[0].legend(["close", "max", "min"])
ax[1].plot(data["Buy"][start:end], c="g")
ax[1].plot(data["Sell"][start:end], c="r")
ax[1].legend(["Buy", "Sell"])
plt.show()
"""
