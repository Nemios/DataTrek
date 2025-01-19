import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"..\Data\Raw\student-depression-dataset.csv")


"""
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.columns)
gender = df["Gender"]
print(gender)
print(df[["Gender", "Age"]])

#pandas aggregation
print(df.groupby(["City", "Profession"]).agg({"Age": "mean", "Depression": "mean"}))

#pandas merge dataframes
print(pd.merge(df1,df2,on="student_id",how="inner"/"outer"/"left"/"right"))

#matplotlitb
Age = df["Age"]
Depression = df["Depression"]
plt.scatter(Age, Depression)
#.scatter : nuage de points
#.bar : diagramem en barres
#.hist : histogramme ==> plt.hist(Age, bins=10,edgecolor="black")
#.plot : courbe
plt.xlabel("Age")
plt.ylabel("Depression")
plt.title("Nuage de points Depression/Age")
plt.show()

#subplots :
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
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.05, hspace=0.5, wspace=0.5)
plt.show()
"""
