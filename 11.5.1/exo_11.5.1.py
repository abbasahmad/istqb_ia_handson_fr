from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns



# Générer un jeu de données synthétique
X, y = make_classification(
    n_samples=10000, 
    n_features=4, 
    n_informative=3, 
    n_redundant=0, 
    random_state=42
)

# Conversion en DataFrame pour affichage
synthetic_data = pd.DataFrame(X, columns=["Complexité", "Lignes de code", "Couverture de tests (%)", "Nombre de méthodes"])
synthetic_data['Défaut'] = y

# Afficher les données générées avec des explications détaillées
print("\n=== Explication du graphique ===")
print("Le graphique pairplot montre les relations entre toutes les variables:")
print("- Diagonale: Distribution de chaque variable (histogrammes)")
print("- Hors diagonale: Relations entre paires de variables (nuages de points)")
print("- Couleurs: Orange = Pas de défaut (0), Bleu = Défaut présent (1)")
print("- Ce graphique aide à identifier les patterns qui distinguent les classes")
print("===============================\n")

# Créer le pairplot avec des légendes améliorées
pairplot = sns.pairplot(
    synthetic_data, 
    hue="Défaut", 
    diag_kind="hist", 
    palette={0: "orange", 1: "blue"},
    plot_kws={'alpha': 0.6, 's': 20},
    diag_kws={'alpha': 0.7}
)
pairplot.fig.suptitle("Analyse des corrélations entre les métriques de code et les défauts", 
                      y=1.02, fontsize=14, fontweight='bold')

# Améliorer la légende
pairplot._legend.set_title("Présence de défaut")
for text in pairplot._legend.texts:
    if text.get_text() == "0":
        text.set_text("Pas de défaut (0)")
    elif text.get_text() == "1":
        text.set_text("Défaut présent (1)")

plt.show()


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer un modèle avec des hyperparamètres ajustés
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer le modèle
print("Exactitude : ", accuracy_score(y_test, y_pred))
print("F1 Score : ", f1_score(y_test, y_pred))
print("Précision : ", precision_score(y_test, y_pred))
print("Rappel : ", recall_score(y_test, y_pred))


