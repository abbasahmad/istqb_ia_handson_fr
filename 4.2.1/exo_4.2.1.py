# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# -------------------------------------
# Étape 1 : Chargement des données nettoyées
# -------------------------------------
df = pd.read_csv('cleaned_dataset.csv')

# Vérifier les données et supprimer les colonnes inutiles si nécessaire
print(df.head())
print(df.info())

# Supposons que la colonne 'Purchase' est la cible (label) pour le modèle
X = df.drop(columns=['Purchase'])  # Données d'entrée (features)
y = df['Purchase']  # Label (variable cible)

# Encodage de la cible ('Yes' -> 1, 'No' -> 0)
y = y.map({'Yes': 1, 'No': 0})
# Encodage de la colonne 'Gender'
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0, 'Unknown': -1})
# Encodage de la colonne 'City' (assignation d'un identifiant unique à chaque ville)
X['City'] = X['City'].astype('category').cat.codes

# Vérification après encodage
print("\nDonnées après encodage manuel :")
print(X.head(16))



# -------------------------------------
# Étape 2 : Division des données
# -------------------------------------
# Division initiale entre apprentissage+(validation&Test )
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3, random_state=42)
print(len(X_train))


X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50, random_state=42)
print(f"Taille des ensembles : Apprentissage={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")


# -------------------------------------
# Étape 3 : Création et entraînement du modèle
# -------------------------------------
# Utiliser un modèle de régression logistique
model = LogisticRegression(max_iter=5000, class_weight='balanced')

# Entraîner le modèle sur l'ensemble d'apprentissage
model.fit(X_train, y_train)

# -------------------------------------
# Étape 4 : Évaluation sur les ensembles de validation et de test
# -------------------------------------
# Précision sur l'ensemble de validation
y_val_pred = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Précision sur l'ensemble de validation : {accuracy_val:.2f}")



# Précision sur l'ensemble de test
y_test_pred = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Précision sur l'ensemble de test : {accuracy_test:.2f}")



# -------------------------------------
# Étape 5 : Rapport final
# -------------------------------------
# Comparer les résultats
print("\nRapport détaillé (Validation) :")
print(classification_report(y_val, y_val_pred, zero_division=1))


print("\nRapport détaillé (Test) :")
print(classification_report(y_test, y_test_pred, zero_division=1))

print("MSE Train :", mean_squared_error(y_val, y_val_pred))
print("MSE Test :", mean_squared_error(y_test, y_test_pred))
print("R² Train :", r2_score(y_val, y_val_pred))
print("R² Test :", r2_score(y_test, y_test_pred))




# Interpréter les différences entre validation et test
print("\nInterprétation :")
print("La précision sur l'ensemble de validation est utilisée pour ajuster les hyperparamètres.")
print("La précision sur l'ensemble de test reflète les performances du modèle sur des données jamais vues.")


