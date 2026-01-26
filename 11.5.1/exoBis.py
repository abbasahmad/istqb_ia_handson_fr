import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

# Charger les données
data = {
    "Complexité": [8, 4, 7, 3, 6, 2, 5, 1],
    "Lignes de code": [300, 150, 250, 100, 200, 80, 180, 50],
    "Couverture de tests (%)": [80, 95, 70, 98, 85, 99, 90, 100],
    "Nombre de méthodes": [10, 6, 8, 4, 7, 3, 6, 2],
    "Défaut": [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Séparer les caractéristiques (X) et la cible (y)
X = df[["Complexité", "Lignes de code", "Couverture de tests (%)", "Nombre de méthodes"]]
y = df["Défaut"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle
#model = RandomForestClassifier(random_state=42)

model = RandomForestClassifier(
    n_estimators=1000,      # Augmente le nombre d'arbres
    max_depth=5,          # Limite la profondeur des arbres
    min_samples_split=5,   # Ajuste le nombre minimum d'échantillons pour une division
    min_samples_leaf=2,    # Ajuste la taille des feuilles
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Afficher les résultats
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("Exactitude : ", accuracy_score(y_test, y_pred))


print("F1 Score : ", f1_score(y_test, y_pred))
print("Précision : ", precision_score(y_test, y_pred))
print("Rappel : ", recall_score(y_test, y_pred))

# Importance des caractéristiques
importances = model.feature_importances_
features = X.columns
print("\nImportance des caractéristiques :")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.2f}")


import matplotlib.pyplot as plt

importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title("Importance des caractéristiques")
plt.show()