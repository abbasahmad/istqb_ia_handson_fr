from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')  # Force l'utilisation d'un backend interactif

# Générer des données pour surajustement (peu de données)
np.random.seed(42)
X_small = np.random.rand(100, 1) * 100  # 100 points
y_small = 2 * X_small + np.random.randn(100, 1) * 2

# Générer des données pour sous-ajustement (faible corrélation)
X_large = np.random.rand(1000, 1) * 10  # 1000 points
# Bruit important, pas de relation claire
y_large = np.random.randn(1000, 1) * 2

# Visualisation des deux ensembles
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_small, y_small, color='blue')
plt.title("Surajustement : Forte corrélation")
plt.xlabel("Caractéristiques")
plt.ylabel("Cible")

plt.subplot(1, 2, 2)
plt.scatter(X_large, y_large, color='red')
plt.title("Sous-ajustement : Faible corrélation")
plt.xlabel("Caractéristiques")
plt.ylabel("Cible")

plt.tight_layout()
plt.show(block=True)


# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Évaluation
    print(f"\n--- {title} ---")
    print("MSE Train :", mean_squared_error(y_train, y_pred_train))
    print("MSE Test :", mean_squared_error(y_test, y_pred_test))
    print("R² Train :", r2_score(y_train, y_pred_train))
    print("R² Test :", r2_score(y_test, y_pred_test))

    # Visualisation des résultats
    plt.scatter(X, y, color='gray', label='Données réelles')
    plt.plot(X, model.predict(X), color='orange', label='Prédiction du modèle')
    plt.title(title)
    plt.legend()
    plt.show()


# Entraîner et évaluer sur les deux ensembles
train_and_evaluate(X_small, y_small, "Surajustement")
train_and_evaluate(X_large, y_large, "Sous-ajustement")
