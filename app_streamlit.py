
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.title("DataTrek")
st.write("Analyse d'un Dataset Kaggle : Indian Student Depression")

data = pd.read_csv(r"Data/Raw/student-depression-dataset.csv")
df = data.copy()
st.write(df)

st.title("Modèles")
st.write("Courbes d'apprentissage des modèles")

st.badge("AdaBoost", color="green")
st.image("Figures/learning_curve_AdaBoost.png")
st.write("Le modèle est très intéressant car les courbes se rapprochent vers 0.87 de score donc valeur assez élevée et pas d'overfitting.")
st.markdown("---")

st.badge("DecisionTree", color="green")
st.image("Figures/learning_curve_DecisionTreeClassifier.png")
st.write("Situation d'overfitting évidente (le modèle connaît par coeur les données d'entraînement).")
st.markdown("---")

st.badge("KNN", color="green")
st.image("Figures/learning_curve_KNN.png")
st.write("Le modèle est moins performant que l'AdaBoost car les courbes ne se rapprochent pas mais pas d'overfitting pour autant.")
st.markdown("---")

st.badge("RandomForest", color="green")
st.image("Figures/learning_curve_RandomForest.png")
st.write("Situation d'overfitting évidente (le modèle connaît par coeur les données d'entraînement).")
st.markdown("---")

st.badge("SVM", color="green")
st.image("Figures/learning_curve_SVM.png")
st.write("Le modèle est intéressant car les courbes semblent se rapprocher vers des valeurs de score élevé mais nécessitant plus de samples. Pas d'overfitting.")
st.markdown("---")

st.title("Optimisation des hyperparamètres de l'AdaBoostClassifier avec GridSearchCV")
grid_best_params = {
    'adaboostclassifier__algorithm': 'SAMME.R',
    'adaboostclassifier__estimator': DecisionTreeClassifier(max_depth=3),
    'adaboostclassifier__learning_rate': 0.001,
    'adaboostclassifier__n_estimators': 200
}
params_df = pd.DataFrame(grid_best_params.items(), columns=["Hyperparamètre", "Valeur"])
st.write("### Hyperparamètres optimaux")
st.table(params_df)

adaboost_grid_classification_report = {
    "precision": [0.84, 0.81, "", 0.82, 0.82],
    "recall": [0.70, 0.90, "", 0.80, 0.82],
    "f1-score": [0.76, 0.85, 0.82, 0.81, 0.81],
    "support": [2300, 3253, 5553, 5553, 5553]
}
index = ["Classe 0", "Classe 1", "Accuracy", "Macro avg", "Weighted avg"]
report_df = pd.DataFrame(adaboost_grid_classification_report, index=index).round(2)

st.write("### Rapport de classification")
st.table(report_df)

st.write("### Interprétations : ")
st.write("**Accuracy globale** : Le modèle classe correctement 78% des cas totaux.")
st.markdown("---")
st.write("**Classe 0** (pas en dépression) : ")
st.write("**Précision** : Elle permet de connaître le nombre de prédictions positives bien effectuées. En d'autres termes, c'est le nombre de positifs bien prédits (Vrais Positifs) divisé par l'ensemble des positifs prédits (Vrais Positifs + Faux Positifs).")
st.latex(r'''
\text{Précision} = \frac{\text{Vrais Positifs}}{\text{Vrais Positifs} + \text{Faux Positifs}}
''')
st.write("Donc parmi les prédictions 'pas en dépression', 84% sont correctes.")
st.write("**Rappel** : Le recall permet de savoir le pourcentage de positifs bien prédits par notre modèle. En d'autres termes, c'est le nombre de positifs bien prédits (Vrais Positifs) divisé par l'ensemble des positifs (Vrais Positifs + Faux Négatifs).")
st.latex(r'''
\text{Recall} = \frac{\text{Vrais Positifs}}{\text{Vrais Positifs} + \text{Faux Négatifs}}
''')
st.write("Donc le modèle identifie 70% des vrais 'non dépressifs'.")
st.write("**F1-score** : Ici, le F1 Score est ce qu'on appelle la moyenne harmonique. C'est une « moyenne » adaptée aux calculs de taux/pourcentage (ici le recall et la précision).")
st.latex(r'''
\text{F1-score} = 2 \times \frac{\text{Recall} \times \text{Précision}}{\text{Recall} + \text{Précision}}
''')
st.write("Ici F1-score = 0.76, équilibre entre précision et rappel relativement bon mais possibilité d'améliorations si on réduisait le nombre de faux négatifs")
st.markdown("---")
st.write("**Classe 1** (en dépression) : ")
st.write("**Précision** : Parmi les prédictions 'pas en dépression', 81% sont correctes.")
st.write("**Rappel** : Le modèle identifie 90% des vrais 'non dépressifs'.")
st.write("**F1-score** : 0.85, bonne performance du modèle.")
st.markdown("---")

st.title("Optimisation des hyperparamètres de l'AdaBoostClassifier avec RandomizedSearchCV")
grid_best_params = {
    'adaboostclassifier__algorithm': 'SAMME.R',
    'adaboostclassifier__estimator': DecisionTreeClassifier(max_depth=2),
    'adaboostclassifier__learning_rate': 0.001,
    'adaboostclassifier__n_estimators': 100
}
params_df = pd.DataFrame(grid_best_params.items(), columns=["Hyperparamètre", "Valeur"])
st.write("### Hyperparamètres optimaux")
st.table(params_df)

adaboost_grid_classification_report = {
    "precision": [0.86, 0.75, "", 0.80, 0.79],
    "recall": [0.56, 0.93, "", 0.75, 0.78],
    "f1-score": [0.68, 0.83, 0.78, 0.75, 0.7],
    "support": [2300, 3253, 5553, 5553, 5553]
}
index = ["Classe 0", "Classe 1", "Accuracy", "Macro avg", "Weighted avg"]
report_df = pd.DataFrame(adaboost_grid_classification_report, index=index).round(2)

st.write("### Rapport de classification")
st.table(report_df)

st.write("### Interprétations : ")
st.write("**Accuracy globale** : Le modèle classe correctement 78% des cas totaux.")
st.markdown("---")
st.write("**Classe 0** (pas en dépression) : ")
st.write("**Précision** : Parmi les prédictions 'pas en dépression', 81% sont correctes.")
st.write("**Rappel** : Le modèle identifie 73% des vrais 'non dépressifs'.")
st.write("**F1-score** : 0.77, équilibre entre précision et rappel relativement bon mais possibilité d'améliorations si on réduisait le nombre de faux négatifs")
st.markdown("---")
st.write("**Classe 1** (en dépression) : ")
st.write("**Précision** : Parmi les prédictions 'pas en dépression', 82% sont correctes.")
st.write("**Rappel** : Le modèle identifie 88% des vrais 'non dépressifs'.")
st.write("**F1-score** : 0.85, bonne performance du modèle.")
st.markdown("---")
