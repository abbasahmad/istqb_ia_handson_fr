from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Génération de données synthétiques
X, y = make_classification(n_samples=10000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dictionnaire des modèles
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "MLPClassifier": MLPClassifier(max_iter=500)
}

# Fonction pour entraîner et évaluer un modèle
def train_and_evaluate(model, metric, train_size=0.7):
    # Ajuster la taille des données d’entraînement
    X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
    
    # Entraînement
    model.fit(X_train_sub, y_train_sub)
    
    # Prédiction
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Évaluation selon la métrique
    if metric == "Accuracy":
        return accuracy_score(y_test, y_pred)
    elif metric == "Precision":
        return precision_score(y_test, y_pred)
    elif metric == "Recall":
        return recall_score(y_test, y_pred)
    elif metric == "F1-Score":
        return f1_score(y_test, y_pred)
    elif metric == "ROC-AUC":
        return roc_auc_score(y_test, y_proba)

# Exemple d'appel de la fonction
accuracy = train_and_evaluate(models["RandomForest"], "Accuracy", train_size=0.8)
print(f"Accuracy: {accuracy}")



# Exemple de combinaisons générées par l'outil PICT
'''test_combinations = [
    {"Model": "LogisticRegression", "Estimators": None, "LearningRate": 0.1, "TrainSize": 0.7, "Metric": "Accuracy"},
    {"Model": "RandomForest", "Estimators": 100, "LearningRate": None, "TrainSize": 0.8, "Metric": "Precision"},
    {"Model": "MLPClassifier", "Estimators": None, "LearningRate": 0.001, "TrainSize": 0.9, "Metric": "F1-Score"},
    # Ajouter d'autres combinaisons...
]'''

import pandas as pd

# Lire les données à partir du fichier
file_path = "Combi.txt"  # Chemin vers votre fichier
data = pd.read_csv(file_path, sep="\t")  # PICT sépare les colonnes par des tabulations

# Convertir les valeurs NaN ou manquantes en None pour compatibilité Python
data = data.where(pd.notnull(data), None)

# Transformer chaque ligne en un dictionnaire formaté
test_combinations = [
    {
        "Model": row["Model"],
        "Estimators": row["Estimators"],
        "LearningRate": row["LearningRate"],
        "TrainSize": row["TrainSize"] / 100 if row["TrainSize"] else None,  # Convertir pourcentage en valeur décimale
        "Metric": row["Metric"]
    }
    for _, row in data.iterrows()
]

# Afficher la liste de combinaisons
for combo in test_combinations:
    print(combo)


    
results = []
for combo in test_combinations:
    # Configurer le modèle
    model = models[combo["Model"]]
    if combo["Model"] == "RandomForest" and combo["Estimators"]:
        model.set_params(n_estimators=combo["Estimators"])
    if combo["Model"] == "MLPClassifier" and combo["LearningRate"]:
        model.set_params(learning_rate_init=combo["LearningRate"])
    
    # Exécuter et collecter les résultats
    result = train_and_evaluate(model, combo["Metric"], combo["TrainSize"])
    results.append({"Combination": combo, "Result": result})

# Afficher les résultats
for res in results:
    print(f"Combination: {res['Combination']}, Result: {res['Result']}")