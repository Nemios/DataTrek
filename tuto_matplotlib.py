import matplotlib.pyplot as plt
import numpy as np

###############################################################
# 14/30 Matplotlib les Bases
###############################################################

# ne pas mélanger méthode fonction et méthode POO

# méthode Fonction :
"""
x = np.linspace(0, 2, 10)
y = x**2
print(x)
plt.figure(figsize=(12, 8))  # appelle une figure sans rien de taille en inch
plt.plot(x, y, label="quadratique", c="red", lw=5, ls="--")
plt.plot(x, x**3, label="cubique", lw=2, ls="-", c="yellow")
plt.title("figure 1")
plt.xlabel("axe x")
plt.ylabel("axe y")
plt.legend()  # affiche les labels
plt.show()
# plt.savefig("figure.png")  # sauvegarde figure

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x), c="red", label="sinus")
plt.xlabel("axe x")
plt.ylabel("axe y")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x), c="blue", label="cosinus")
plt.title("figure 2")
plt.xlabel("axe x")
plt.ylabel("axe y")
plt.legend()
plt.show()

# Exo
dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}
print(dataset)
print(dataset.keys())
for exp in dataset.values():
    print(exp)
    print(len(exp))


def graphique(dataset):
    plt.figure()
    i = 1
    for exp in dataset.values():
        x = np.linspace(0, len(exp), len(exp))
        plt.subplot(4, 1, i)
        plt.plot(x, exp)
        plt.title(f"experience {i}")
        i += 1
    plt.show()


graphique(dataset)
"""
