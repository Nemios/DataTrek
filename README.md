Le but du projet DataTrek était de me familiariser avec la démarche d’un Data Scientist type lors de l’analyse d’un dataset et de l’élaboration d’un modèle de Machine Learning. Je veux devenir Data Scientist et je cherchais à développer mes connaissances et compétences dans le domaine du Machine Learning. J’ai choisi de suivre des vidéos tutorielles YouTube de la chaîne “Machine Learnia” où figure une playlist d’une trentaine de vidéos tutorielles, partant des bases de modules et packages incontournables de Python (Numpy, Pandas, …) jusqu’à des modules et packages spécialisés et orientés Machine Learning comme ScikitLearn. 

Le dataset que j’ai sélectionné est “Student Depression Dataset”, un dataset disponible sur Kaggle et sur lequel une compétition de Machine Learning avait été faite. Je précise que le but du projet n’était pas d’aboutir à une analyse concluante du jeu de données, du moins ce n’est pas le “produit final”, dans le sens où le dataset est un prétexte aux mises en application des différents outils de Data Science pour l’analyse et l’élaboration d’un modèle. Le “produit final” étant l’expérience et les compétences acquises lors du projet.

Installation
Ouvrir VSCode;
Cloner le repository Github “DataTrek” ou ouvrir le dossier sur VSCode;
Installer Anaconda Navigator version 2.6.4;
Aller dans l’onglet “Environnements”;
Importer l’environnement DataTrek sur Anaconda Navigator en important en local le fichier “/CondaEnvironnement/DataTrek.yaml” présent dans le repository cloné;
Lancer VSCode depuis Anaconda Navigator (onglet “Home”) (si vous fermez VSCode il faudra également passer par Anaconda Navigator les prochaines fois);
Installer les extensions VSCode suivantes : 
Jupyther,
Jupyter Cell Tags,
Jupyter Keymap,
Jupyter Notebook Renderers,
Jupyter Slide Show,
Pylance,
Python,
Python Debugger,
(optionnel) Black Formatter et le configurer en tant que formateur par défaut dans les Settings;
Ouvrir un terminal;
Se placer dans le dossier “DataTrek”, dossier parent de “.vscode”, “CondaEnvironnement”, etc;
Exécuter la commande “conda activate DataTrek” dans le terminal;
Vous pouvez maintenant exécuter les différents fichiers Python (.py) et le Jupyter NoteBook (.ipynb). L’exécution se fait en cliquant sur le bouton en haut à droite pour les fichiers Python (.py) : . Pour le NoteBook, il faut utiliser les différentes options présentes à chaque cellule : .
ATTENTION  : Dans le NoteBook il est préférable de ne pas exécuter certaines cellules car leur temps d’exécution est TRÈS long. Ces cellules sont reconnaissables par un message d’avertissement juste au-dessus d’elles. Il s’agit globalement des cellules faisant intervenir la méthode .fit() qui entraîne un modèle sur les données.
