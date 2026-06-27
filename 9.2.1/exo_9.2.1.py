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


# ---------------------------------------------------------------------------
# Interface graphique : affichage de tous les résultats d'entraînement
# ---------------------------------------------------------------------------
def show_results_gui(results):
    """Ouvre une fenêtre Tkinter présentant tous les résultats des
    entraînements de manière claire, colorée et accompagnée d'une légende."""
    import tkinter as tk
    from tkinter import ttk

    # Couleur d'une ligne selon la qualité du score
    def quality(score):
        if score >= 0.90:
            return "Excellent", "#1b7837"   # vert foncé
        elif score >= 0.80:
            return "Bon", "#5aae61"          # vert
        elif score >= 0.70:
            return "Moyen", "#f1a340"        # orange
        else:
            return "Faible", "#d73027"       # rouge

    root = tk.Tk()
    root.title("Résultats des entraînements — Tests par paires (PICT)")
    root.geometry("1050x680")
    root.configure(bg="#f5f5f5")

    # ----- En-tête -----
    header = tk.Frame(root, bg="#2c3e50")
    header.pack(fill="x")
    tk.Label(
        header,
        text="🧪  Résultats des entraînements de modèles",
        font=("Helvetica", 18, "bold"),
        fg="white", bg="#2c3e50", pady=12,
    ).pack()
    tk.Label(
        header,
        text=f"{len(results)} combinaisons testées  •  Données synthétiques (10 000 échantillons, 10 caractéristiques)",
        font=("Helvetica", 10),
        fg="#bdc3c7", bg="#2c3e50",
    ).pack(pady=(0, 10))

    # ----- Statistiques récapitulatives -----
    scores = [r["Result"] for r in results]
    best = max(results, key=lambda r: r["Result"])
    worst = min(results, key=lambda r: r["Result"])
    stats = tk.Frame(root, bg="#f5f5f5")
    stats.pack(fill="x", padx=15, pady=10)

    def stat_card(parent, title, value, color):
        card = tk.Frame(parent, bg="white", bd=1, relief="solid")
        card.pack(side="left", expand=True, fill="x", padx=5)
        tk.Label(card, text=title, font=("Helvetica", 9), fg="#7f8c8d", bg="white").pack(pady=(8, 0))
        tk.Label(card, text=value, font=("Helvetica", 14, "bold"), fg=color, bg="white").pack(pady=(0, 8))

    stat_card(stats, "Score moyen", f"{sum(scores)/len(scores):.3f}", "#2c3e50")
    stat_card(stats, "Meilleur score", f"{best['Result']:.3f}  ({best['Combination']['Model']})", "#1b7837")
    stat_card(stats, "Pire score", f"{worst['Result']:.3f}  ({worst['Combination']['Model']})", "#d73027")

    # ----- Tableau des résultats -----
    table_frame = tk.Frame(root, bg="#f5f5f5")
    table_frame.pack(fill="both", expand=True, padx=15, pady=5)

    columns = ("idx", "model", "estimators", "lr", "trainsize", "metric", "score", "quality")
    headings = {
        "idx": "#",
        "model": "Modèle",
        "estimators": "Estimateurs",
        "lr": "Taux d'app.",
        "trainsize": "Taille entr.",
        "metric": "Métrique",
        "score": "Score",
        "quality": "Qualité",
    }
    widths = {"idx": 40, "model": 160, "estimators": 90, "lr": 90,
              "trainsize": 90, "metric": 100, "score": 90, "quality": 100}

    style = ttk.Style()
    style.theme_use("default")
    style.configure("Treeview", rowheight=26, font=("Helvetica", 10))
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

    tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=headings[col])
        anchor = "w" if col == "model" else "center"
        tree.column(col, width=widths[col], anchor=anchor)

    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    # Remplissage du tableau (trié du meilleur au pire score)
    sorted_results = sorted(results, key=lambda r: r["Result"], reverse=True)
    for i, res in enumerate(sorted_results, start=1):
        c = res["Combination"]
        label, color = quality(res["Result"])
        tree.insert(
            "", "end",
            values=(
                i,
                c["Model"],
                "—" if c["Estimators"] is None else int(c["Estimators"]),
                "—" if c["LearningRate"] is None else c["LearningRate"],
                f"{int(c['TrainSize'] * 100)}%",
                c["Metric"],
                f"{res['Result']:.4f}",
                label,
            ),
            tags=(label,),
        )
        tree.tag_configure(label, foreground=color)

    # ----- Légende -----
    legend = tk.LabelFrame(root, text="  Légende — comment lire ce tableau  ",
                           font=("Helvetica", 10, "bold"), bg="white", fg="#2c3e50",
                           padx=12, pady=8)
    legend.pack(fill="x", padx=15, pady=12)

    legend_cols = tk.Frame(legend, bg="white")
    legend_cols.pack(fill="x")

    # Colonne 1 : signification des colonnes
    col1 = tk.Frame(legend_cols, bg="white")
    col1.pack(side="left", anchor="n", expand=True, fill="x")
    explanations = [
        ("Modèle", "Algorithme entraîné (LogisticRegression, RandomForest, SVM, MLP)"),
        ("Estimateurs", "Nb d'arbres/unités (utilisé surtout par RandomForest)"),
        ("Taux d'app.", "Vitesse d'apprentissage (utilisé par MLPClassifier)"),
        ("Taille entr.", "Part des données utilisées pour l'entraînement"),
        ("Métrique", "Mesure d'évaluation appliquée à ce test"),
        ("Score", "Valeur de la métrique sur le jeu de test (0 à 1, plus haut = mieux)"),
    ]
    for name, desc in explanations:
        line = tk.Frame(col1, bg="white")
        line.pack(anchor="w", pady=1)
        tk.Label(line, text=f"• {name} : ", font=("Helvetica", 9, "bold"),
                 bg="white", fg="#2c3e50").pack(side="left")
        tk.Label(line, text=desc, font=("Helvetica", 9), bg="white", fg="#555").pack(side="left")

    # Colonne 2 : code couleur qualité
    col2 = tk.Frame(legend_cols, bg="white")
    col2.pack(side="left", anchor="n", padx=30)
    tk.Label(col2, text="Code couleur (qualité du score) :", font=("Helvetica", 9, "bold"),
             bg="white", fg="#2c3e50").pack(anchor="w", pady=(0, 2))
    for label, color, rng in [
        ("Excellent", "#1b7837", "score ≥ 0.90"),
        ("Bon", "#5aae61", "0.80 – 0.90"),
        ("Moyen", "#f1a340", "0.70 – 0.80"),
        ("Faible", "#d73027", "< 0.70"),
    ]:
        line = tk.Frame(col2, bg="white")
        line.pack(anchor="w", pady=1)
        tk.Label(line, text="  ■  ", font=("Helvetica", 10, "bold"),
                 bg="white", fg=color).pack(side="left")
        tk.Label(line, text=f"{label}  ({rng})", font=("Helvetica", 9),
                 bg="white", fg="#555").pack(side="left")

    tk.Label(
        legend,
        text="ℹ  Les métriques (Accuracy, Precision, Recall, F1-Score, ROC-AUC) varient de 0 à 1 : "
             "plus la valeur est proche de 1, meilleur est le modèle pour le critère choisi.",
        font=("Helvetica", 9, "italic"), bg="white", fg="#7f8c8d",
        wraplength=980, justify="left",
    ).pack(anchor="w", pady=(8, 0))

    root.mainloop()


# Ouvrir l'interface graphique avec tous les résultats
show_results_gui(results)