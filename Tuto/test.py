import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"Data\Raw\student-depression-dataset.csv")

""" 
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.columns)
gender = df["Gender"]
print(gender)
print(df[["Gender", "Age"]])

# pandas aggregation
print(df.groupby(["City", "Profession"]).agg({"Age": "mean", "Depression": "mean"}))

# pandas merge dataframes
print(pd.merge(df1, df2, on="student_id", how="inner" / "outer" / "left" / "right"))

# matplotlitb
Age = df["Age"]
Depression = df["Depression"]
plt.scatter(Age, Depression)
# .scatter : nuage de points
# .bar : diagramem en barres
# .hist : histogramme ==> plt.hist(Age, bins=10,edgecolor="black")
# .plot : courbe
plt.xlabel("Age")
plt.ylabel("Depression")
plt.title("Nuage de points Depression/Age")
plt.show()

# subplots :
plt.subplot(221)
plt.scatter(Age, Depression)
plt.xlabel("Age")
plt.ylabel("Depression")
plt.title("Nuage de points Depression/Age")

plt.subplot(222)
...
plt.subplot(223)
...
plt.subplot(224)
...
plt.subplots_adjust(
    top=0.92, bottom=0.08, left=0.10, right=0.05, hspace=0.5, wspace=0.5
)
plt.show()
 """


""" 
villes = ["Paris", "Rome", "Londres", "Dublin", "Amsterdan", "Berlin"]
num = [24, 45, 1, 17, 26]

for index, ville in enumerate(villes):
    print(index, ville)

for a, b in zip(villes, num):
    print(a, b)
print(villes[:3])
print(villes[3::2])
print(villes[::-1])
villes.sort()
print(villes)
num.sort(reverse=True)
print(num)


def fibo(n):
    liste = []
    a = 0
    b = 1
    while a < n:
        liste.append(a)
        a, b = b, b + a
    print(liste)


fibo(1000)
dico = {"chien": 1, "chat": 2, "poule": 6, "cheval": 8}

# pour le Deep Learning
# parametres reseau de neurones a stocker dans dico
parametres = {
    "W1": np.random.randn(10, 100),
    "B1": np.random.randn(10, 1),
    "W2": np.random.randn(10, 10),
    "B2": np.random.randn(10, 1),
}


print(dico.keys())
print(dico.values())
print(len(dico))
dico["ecureuil"] = 9
print(dico)
print(dico.get("elephant", "pas dans le dico"))
# .fromkeys pour créer nouveau dico dont les clés sont la liste passées en argument
dico2 = dico.fromkeys(villes, "default")
print(dico2)
# .pop pour extraire une association spécifique du dico ou un élément d'une liste
# l'élément est supprimé du dico / liste et donne en sortie la valeur associée

animal = dico.pop("chat")
print(animal)
print(dico)

# boucles dico
# sur les clés
for i in dico:
    print(i)
# sur les values
for i in dico.values():
    print(i)
# pour les deux sous forme de tuples
for i, j in dico.items():
    print(i, j)

liste_test_signe = [1, -5, 65, -158, 2, -69]

classeur = {"positif": [], "negatif": []}


def trier(classeur, nb):
    if nb > 0:
        classeur["positif"].append(nb)
    elif nb < 0:
        classeur["negatif"].append(nb)
    else:
        print("c'est zero wsh")
    return classeur


for i in liste_test_signe:
    print(trier(classeur, i))

# passer au carré les éléments d'une liste
# méthode classique
import time

start = time.time()

num2 = []
for i in range(10000000):
    num2.append(i**2)

end = time.time()
print(end - start)
# List Comprehension
start = time.time()

num2_comprehension = [i**2 for i in range(10000000)]

end = time.time()
print(end - start)
# list comprehension dans une liste
liste1 = [[i + j for i in range(3)] for j in range(3)]
print(liste1)

# dico classique
dico2 = {0: "Noe", 1: "Zachari", 2: "Gaelle", 3: "Theo"}

# dico comprehension
prenoms = ["Noe", "Zachari", "Gaelle", "Theo"]
dico3 = {k: v for k, v in enumerate(prenoms)}
print(dico3)

# zip() sur dico comprehension
ages = [21, 20, 25, 22]
dico4 = {prenom: age for prenom, age in zip(prenoms, ages) if age < 24}
print(dico4)

# tuple comprehension
tuple1 = tuple((i**2 for i in range(10)))
print(tuple1)

# exo dico
dico_exo = {k: k**2 for k in range(20)}
print(dico_exo)

prenoms = ["Noe", "Zachari", "Gaelle", "Theo"]

dico_test = {k: v for k, v in enumerate(prenoms)}
print(dico_test)
 """

prenoms = ["Noe", "Zachari", "Gaelle", "Theo"]

table_verite = [False, True, True]
# all() returns True if all booleans are True or ints not 0
print(all(table_verite))
# any() returns True if at least one boolean is True or int not 0
print(any(table_verite))

# conversion function
# variables : int(),  float(), str()
# structure de données : dict(), list(), tuple(), set()
# binaire : bin(), bytes(), hex()
x = 10
print(type(x))
x = str(x)
print(type(x))

y = "15"
print(type(y))
y = int(y)
print(type(y))

liste_num = [0, 2, 3, 6, 84, -142]
tuple_num = tuple(liste_num)
print(tuple_num)

liste_num2 = list(tuple_num)
print(liste_num)

dico = {"0": 5, "1": 6}
print(type(dico.keys()))

# input()
# x = int(input("Entrez un nombre\n"))

# format()
x = 25
ville = "Paris"
message = f"Aujourd'hui, il fait {x} degrés à {ville}"
print(message)

parametres_neurones = {
    "W1": np.random.randn(2, 4),
    "b1": np.zeros((2, 1)),
    "W2": np.random.randn(2, 2),
    "b2": np.zeros((2, 1)),
}

for i in range(1, 3):
    print("couche", i)
    print(parametres_neurones[f"b{i}"])

print("W2" + str(1))

file = open("fichier.txt", "w")
file.write("bonjour")
file.close()

file = open("fichier.txt", "r")
file.read()
file.close()

# ou
with open("fichier.txt", "r") as f:
    f.read()
    f.close()

# exo open()

with open("carres.txt", "w") as f:
    for i in range(10):
        f.write(f"{i}^2 = {i**2}\n")

liste_txt = []

with open("carres.txt", "r") as f:
    for i in range(10):
        liste_txt.append(f.readline().removesuffix("\n"))

print(liste_txt)
