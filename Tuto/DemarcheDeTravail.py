# 1. définir un objectif mesurable :
# prédire si une personne est en dépression en fonction des données
# metrics :
# - Précision : réduit faux positifs
# - Recall (sensibilité) : réduit faux négatifs => objectif Recall >= 70%
# - Score F1 >= 50%

# 2. EDA (Exploratory Data Analysis) Analyse exploratoire des données
# Analyse de la forme :
# - Identification de la target
# - Nb lignes / colonnes
# - Types de variables
# - Identification des valeurs manquantes
# Analyse du fond :
# - Visualisation de la target (Histogramme / Boxplot)
# - Compréhension des différentes variables
# - Visualisation des relations features - target (Histogramme / Boxplot)
# Identification des outliers

# 3. Preprocessing (NON-EXHAUSTIF)
# - Création du Train set / Test set
# - Elimination des NaN
# - Encodage
# - Suppression outliers néfastes au modèle
# - Feature Selection
# - Feature Engineering
# - Feature Scaling

# 4. Modelling (NON-EXHAUSTIF)
# - Définir une fonction d'évaluation
# - Entrainement de différents modèles
# - Optimisation avec GridSearchCV
# - (Optionnel) Analyse des erreurs et retour au Preprocessing / EDA
# - Learning Curve et prise de décision
