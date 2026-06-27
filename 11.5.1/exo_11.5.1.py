# =============================================================================
#  Exercice 11.5.1 — Modèle de prédiction des défauts dans le code source
# -----------------------------------------------------------------------------
#  Objectif (cf. instructions.txt) :
#    - Construire un modèle prédisant la présence d'un défaut (1) ou non (0)
#      à partir de mesures de code source.
#    - Utiliser AU MOINS 4 caractéristiques :
#        Complexité, Lignes de code, Couverture de tests (%), Nombre de méthodes
#    - Tester DIFFÉRENTES COMBINAISONS de caractéristiques pour observer leur
#      impact sur la performance.
#    - Visualiser les résultats (matplotlib / seaborn).
#
#  Ce fichier fusionne les deux anciennes versions :
#    - exoBis.py        -> les VRAIES données de l'énoncé (8 lignes) + importances
#    - exo_11.5.1.py    -> visualisations seaborn (pairplot) + légendes détaillées
#  ... et ajoute le test des combinaisons de caractéristiques demandé par l'énoncé.
# =============================================================================

from itertools import combinations

import matplotlib
matplotlib.use("TkAgg")  # backend cohérent avec le dashboard + fenêtres positionnables

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# -----------------------------------------------------------------------------
# 1) DONNÉES — exactement le jeu fourni dans l'énoncé (instructions.txt)
# -----------------------------------------------------------------------------
#  ⚠️  Ce jeu ne contient que 8 observations : il est volontairement minuscule.
#      Les scores ci-dessous sont donc surtout PÉDAGOGIQUES (peu de recul
#      statistique). On le signale dans les commentaires des résultats.
data = {
    "Complexité":              [8, 4, 7, 3, 6, 2, 5, 1],
    "Lignes de code":          [300, 150, 250, 100, 200, 80, 180, 50],
    "Couverture de tests (%)": [80, 95, 70, 98, 85, 99, 90, 100],
    "Nombre de méthodes":      [10, 6, 8, 4, 7, 3, 6, 2],
    "Défaut":                  [1, 0, 1, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

# Liste des 4 caractéristiques disponibles (la cible = "Défaut")
FEATURES = ["Complexité", "Lignes de code", "Couverture de tests (%)", "Nombre de méthodes"]
TARGET = "Défaut"

print("=" * 70)
print("  JEU DE DONNÉES (8 modules de code — énoncé 11.5.1)")
print("=" * 70)
print(df.to_string(index=False))
print("\nLégende des colonnes :")
print("  • Complexité              : complexité cyclomatique du module (plus haut = plus risqué)")
print("  • Lignes de code          : taille du module")
print("  • Couverture de tests (%) : part du code couverte par les tests (plus haut = mieux testé)")
print("  • Nombre de méthodes      : nombre de méthodes/fonctions du module")
print("  • Défaut (CIBLE)          : 1 = un défaut est présent, 0 = aucun défaut")
print()


# -----------------------------------------------------------------------------
#  Aide à l'affichage : chaque graphique s'ouvre SEUL, avec un titre clair dans
#  la barre de fenêtre et un message console expliquant quoi regarder.
#  → on ferme la fenêtre pour passer à la suivante : plus de chevauchement.
# -----------------------------------------------------------------------------
def afficher_graphique(fig, titre_fenetre, numero, total):
    # Titre explicite dans la barre de la fenêtre
    try:
        fig.canvas.manager.set_window_title(f"[{numero}/{total}] {titre_fenetre}")
    except Exception:
        pass
    # Position fixe et centrée (évite que les fenêtres s'empilent)
    try:
        fig.canvas.manager.window.wm_geometry("+120+80")
    except Exception:
        pass
    print(f"\n📊  Fenêtre {numero}/{total} : {titre_fenetre}")
    print("    → Fermez cette fenêtre pour afficher la suivante.\n")
    plt.show()


# -----------------------------------------------------------------------------
# 2) FONCTION UTILITAIRE — entraîner et évaluer sur un sous-ensemble de features
# -----------------------------------------------------------------------------
def entrainer_evaluer(features, afficher_rapport=False):
    """Entraîne un RandomForest sur les `features` choisies et renvoie les scores.

    Permet de TESTER DIFFÉRENTES COMBINAISONS de caractéristiques, comme
    demandé dans l'énoncé.
    """
    X = df[features]
    y = df[TARGET]

    # Avec seulement 8 lignes, on garde un petit jeu de test (2 lignes).
    # stratify=y conserve la proportion défaut / pas-défaut dans les 2 ensembles.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,      # nombre d'arbres de la forêt
        max_depth=5,           # profondeur max d'un arbre (limite le surapprentissage)
        min_samples_split=2,   # min. d'échantillons pour scinder un nœud
        min_samples_leaf=1,    # min. d'échantillons dans une feuille
        random_state=42,       # reproductibilité
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores = {
        "Exactitude": accuracy_score(y_test, y_pred),
        "Précision":  precision_score(y_test, y_pred, zero_division=0),
        "Rappel":     recall_score(y_test, y_pred, zero_division=0),
        "F1":         f1_score(y_test, y_pred, zero_division=0),
    }

    if afficher_rapport:
        print("Rapport de classification (jeu de test) :")
        print(classification_report(y_test, y_pred, zero_division=0))

    return model, scores


# -----------------------------------------------------------------------------
# 3) MODÈLE PRINCIPAL — avec les 4 caractéristiques
# -----------------------------------------------------------------------------
print("=" * 70)
print("  MODÈLE PRINCIPAL — les 4 caractéristiques")
print("=" * 70)
model, scores = entrainer_evaluer(FEATURES, afficher_rapport=True)
for nom, valeur in scores.items():
    print(f"  {nom:<11}: {valeur:.3f}")
print("\nℹ️  Métriques (toutes entre 0 et 1, plus haut = mieux) :")
print("   • Exactitude (Accuracy) : % de prédictions correctes au total")
print("   • Précision             : parmi les modules prédits 'défaut', % réellement défectueux")
print("   • Rappel (Recall)       : parmi les vrais défauts, % correctement détectés")
print("   • F1                    : moyenne harmonique Précision/Rappel (équilibre des deux)")
print("   ⚠️  8 lignes seulement → scores très sensibles, à interpréter avec prudence.\n")


# -----------------------------------------------------------------------------
# 4) TEST DES COMBINAISONS DE CARACTÉRISTIQUES (demandé par l'énoncé)
# -----------------------------------------------------------------------------
print("=" * 70)
print("  IMPACT DES COMBINAISONS DE CARACTÉRISTIQUES (F1-Score)")
print("=" * 70)
resultats_combos = []
# On teste toutes les combinaisons de 1, 2, 3 et 4 caractéristiques
for taille in range(1, len(FEATURES) + 1):
    for combo in combinations(FEATURES, taille):
        _, sc = entrainer_evaluer(list(combo))
        resultats_combos.append({
            "Caractéristiques": " + ".join(combo),
            "Nb": taille,
            "F1": sc["F1"],
            "Exactitude": sc["Exactitude"],
        })

combos_df = pd.DataFrame(resultats_combos).sort_values("F1", ascending=False).reset_index(drop=True)
print(combos_df.to_string(index=False))
print("\nLégende : chaque ligne = un modèle entraîné sur un sous-ensemble de")
print("caractéristiques. 'Nb' = nombre de caractéristiques utilisées. Le tableau")
print("est trié du meilleur au moins bon F1-Score → on voit quelles variables")
print("apportent le plus de pouvoir prédictif.\n")


# -----------------------------------------------------------------------------
# 5) VISUALISATION 1 — Importance des caractéristiques (modèle à 4 features)
# -----------------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
barres = ax.barh(importances.index, importances.values, color=sns.color_palette("viridis", len(importances)))
ax.set_title("Importance des caractéristiques\n(contribution de chaque variable aux décisions du modèle)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance relative (somme = 1.0 ; plus la barre est longue, plus la variable compte)")
ax.set_ylabel("Caractéristique")
# Annoter chaque barre avec sa valeur
for barre, val in zip(barres, importances.values):
    ax.text(val + 0.01, barre.get_y() + barre.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=10, fontweight="bold")
ax.text(0.98, -0.18,
        "Lecture : le modèle s'appuie surtout sur les variables aux barres les plus longues\n"
        "pour décider si un module contient un défaut.",
        transform=ax.transAxes, ha="right", fontsize=9, style="italic", color="#555")
plt.tight_layout()
afficher_graphique(fig, "Importance des caractéristiques", 1, 3)


# -----------------------------------------------------------------------------
# 6) VISUALISATION 2 — Comparaison des combinaisons de caractéristiques
# -----------------------------------------------------------------------------
top = combos_df.head(12).iloc[::-1]  # 12 meilleures, inversées pour l'affichage horizontal
fig, ax = plt.subplots(figsize=(10, 6))
couleurs = sns.color_palette("RdYlGn", len(top))
barres = ax.barh(top["Caractéristiques"], top["F1"], color=couleurs)
ax.set_title("Quelles combinaisons de caractéristiques prédisent le mieux les défauts ?\n(12 meilleures combinaisons, F1-Score)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("F1-Score sur le jeu de test (0 = mauvais, 1 = parfait)")
ax.set_xlim(0, 1.05)
for barre, val in zip(barres, top["F1"]):
    ax.text(val + 0.01, barre.get_y() + barre.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9, fontweight="bold")
ax.text(0.99, -0.16,
        "Lecture : vert = combinaison performante, rouge = peu performante.\n"
        "Permet de voir si certaines variables suffisent ou si les 4 sont nécessaires.",
        transform=ax.transAxes, ha="right", fontsize=9, style="italic", color="#555")
plt.tight_layout()
afficher_graphique(fig, "Comparaison des combinaisons de caractéristiques (F1-Score)", 2, 3)


# -----------------------------------------------------------------------------
# 7) VISUALISATION 3 — Pairplot : relations entre variables et défauts
# -----------------------------------------------------------------------------
print("=" * 70)
print("  GRAPHIQUE PAIRPLOT — relations entre toutes les variables")
print("=" * 70)
print("  • Diagonale     : distribution de chaque variable (histogrammes)")
print("  • Hors diagonale: nuages de points entre chaque paire de variables")
print("  • Couleurs      : Orange = Pas de défaut (0), Bleu = Défaut présent (1)")
print("  • But           : repérer visuellement les patterns qui séparent les 2 classes")
print("=" * 70 + "\n")

pairplot = sns.pairplot(
    df,
    hue=TARGET,
    diag_kind="hist",
    palette={0: "orange", 1: "blue"},
    plot_kws={"alpha": 0.7, "s": 60, "edgecolor": "white"},
    diag_kws={"alpha": 0.7},
)
pairplot.figure.suptitle(
    "Analyse des corrélations entre les métriques de code et la présence de défauts",
    y=1.02, fontsize=14, fontweight="bold",
)
# Améliorer la légende (texte explicite au lieu de 0 / 1)
pairplot._legend.set_title("Présence de défaut")
for text in pairplot._legend.texts:
    if text.get_text() == "0":
        text.set_text("Pas de défaut (0)")
    elif text.get_text() == "1":
        text.set_text("Défaut présent (1)")

afficher_graphique(pairplot.figure, "Pairplot — corrélations variables / défauts", 3, 3)

print("✅ Terminé : modèle entraîné, combinaisons testées et 3 graphiques affichés.")
print("➡️  Le tableau de bord interactif va maintenant s'ouvrir...\n")


# -----------------------------------------------------------------------------
# 8) EXTENSION — Tableau de bord interactif (cf. instructions.txt « Extension »)
# -----------------------------------------------------------------------------
#  Permet d'EXPLORER les prédictions :
#    - cocher/décocher les caractéristiques utilisées par le modèle,
#    - saisir les mesures d'un nouveau module,
#    - voir en direct la probabilité de défaut prédite.
def lancer_dashboard():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Tableau de bord — Prédiction de défauts (exercice 11.5.1)")
    root.geometry("780x640+150+60")  # taille + position fixe (pas de chevauchement)
    root.configure(bg="#f4f6f8")
    # S'assurer que la fenêtre apparaît au premier plan
    root.lift()
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))

    # --- En-tête ---
    tete = tk.Frame(root, bg="#2c3e50")
    tete.pack(fill="x")
    tk.Label(tete, text="🔍  Explorateur de prédiction de défauts", fg="white", bg="#2c3e50",
             font=("Helvetica", 16, "bold"), pady=10).pack()
    tk.Label(tete, text="Choisissez les caractéristiques, saisissez un module, obtenez la prédiction en direct",
             fg="#bdc3c7", bg="#2c3e50", font=("Helvetica", 10)).pack(pady=(0, 8))

    corps = tk.Frame(root, bg="#f4f6f8")
    corps.pack(fill="both", expand=True, padx=15, pady=10)

    # === Colonne gauche : choix des caractéristiques =========================
    gauche = tk.LabelFrame(corps, text="  1. Caractéristiques utilisées par le modèle  ",
                           font=("Helvetica", 10, "bold"), bg="white", fg="#2c3e50", padx=10, pady=10)
    gauche.pack(side="left", fill="both", expand=True, padx=(0, 8))

    vars_features = {f: tk.BooleanVar(value=True) for f in FEATURES}

    # Valeurs typiques min/max (issues des données) pour guider la saisie
    bornes = {f: (df[f].min(), df[f].max()) for f in FEATURES}
    # Valeur par défaut = moyenne, pour un module « moyen »
    entrees = {}

    # === Colonne droite : saisie d'un nouveau module =========================
    droite = tk.LabelFrame(corps, text="  2. Mesures du module à évaluer  ",
                           font=("Helvetica", 10, "bold"), bg="white", fg="#2c3e50", padx=10, pady=10)
    droite.pack(side="left", fill="both", expand=True, padx=(8, 0))
    tk.Label(droite,
             text="Saisissez les mesures du module de code à analyser.\n"
                  "Valeurs pré-remplies = moyenne du jeu de données.",
             bg="white", fg="#7f8c8d", font=("Helvetica", 8, "italic"),
             justify="left").pack(anchor="w", pady=(0, 6))

    for f in FEATURES:
        # checkbox (gauche) — fg explicite sinon texte blanc sur blanc (macOS)
        cb = tk.Checkbutton(gauche, text=f, variable=vars_features[f], bg="white",
                            fg="#2c3e50", activebackground="white", activeforeground="#2c3e50",
                            selectcolor="#dfe6ec", highlightthickness=0,
                            font=("Helvetica", 10, "bold"), anchor="w")
        cb.pack(fill="x", anchor="w", pady=2)
        lo, hi = bornes[f]
        tk.Label(gauche, text=f"    plage observée : {lo} → {hi}", bg="white",
                 fg="#7f8c8d", font=("Helvetica", 8)).pack(fill="x", anchor="w")

        # champ de saisie (droite) — libellé du nom de la caractéristique
        ligne = tk.Frame(droite, bg="white")
        ligne.pack(fill="x", pady=4)
        tk.Label(ligne, text=f, bg="white", fg="#2c3e50",
                 font=("Helvetica", 9, "bold"), width=24, anchor="w").pack(side="left")
        e = tk.Entry(ligne, width=8, font=("Helvetica", 10),
                     bg="white", fg="#2c3e50", insertbackground="#2c3e50",
                     highlightthickness=1, highlightbackground="#bdc3c7")
        e.insert(0, str(round(float(df[f].mean()), 1)))  # défaut = moyenne
        e.pack(side="left")
        tk.Label(ligne, text="(valeur mesurée)", bg="white", fg="#95a5a6",
                 font=("Helvetica", 8, "italic")).pack(side="left", padx=(6, 0))
        entrees[f] = e

    # === Zone résultat =======================================================
    resultat = tk.LabelFrame(root, text="  3. Résultat de la prédiction  ",
                             font=("Helvetica", 10, "bold"), bg="white", fg="#2c3e50", padx=12, pady=10)
    resultat.pack(fill="x", padx=15, pady=(0, 8))

    lbl_pred = tk.Label(resultat, text="—", font=("Helvetica", 15, "bold"), bg="white", fg="#2c3e50")
    lbl_pred.pack(anchor="w")
    # Barre de probabilité
    barre = ttk.Progressbar(resultat, length=400, maximum=100)
    barre.pack(anchor="w", pady=4)
    lbl_proba = tk.Label(resultat, text="", font=("Helvetica", 10), bg="white", fg="#555")
    lbl_proba.pack(anchor="w")
    lbl_metrics = tk.Label(resultat, text="", font=("Helvetica", 9), bg="white", fg="#7f8c8d", justify="left")
    lbl_metrics.pack(anchor="w", pady=(4, 0))

    def predire():
        feats = [f for f in FEATURES if vars_features[f].get()]
        if not feats:
            lbl_pred.config(text="⚠️  Sélectionnez au moins une caractéristique", fg="#d35400")
            barre["value"] = 0
            lbl_proba.config(text="")
            lbl_metrics.config(text="")
            return
        try:
            valeurs = [[float(entrees[f].get()) for f in feats]]
        except ValueError:
            lbl_pred.config(text="⚠️  Saisie invalide (entrez des nombres)", fg="#d35400")
            return

        # (Ré)entraîner sur les caractéristiques choisies, puis prédire le module saisi
        mdl, sc = entrainer_evaluer(feats)
        proba = mdl.predict_proba(pd.DataFrame(valeurs, columns=feats))[0][1]  # proba classe « défaut »
        pct = proba * 100

        if proba >= 0.5:
            lbl_pred.config(text=f"🔴  Défaut PROBABLE  ({pct:.0f} %)", fg="#c0392b")
        else:
            lbl_pred.config(text=f"🟢  Pas de défaut probable  ({pct:.0f} %)", fg="#1b7837")
        barre["value"] = pct
        lbl_proba.config(text=f"Probabilité estimée qu'un défaut soit présent : {pct:.1f} %  "
                              f"(seuil de décision = 50 %)")
        lbl_metrics.config(
            text=f"Modèle entraîné sur {len(feats)} caractéristique(s) : {', '.join(feats)}\n"
                 f"Qualité sur le jeu de test → Exactitude {sc['Exactitude']:.2f} · "
                 f"Précision {sc['Précision']:.2f} · Rappel {sc['Rappel']:.2f} · F1 {sc['F1']:.2f}"
        )

    tk.Button(root, text="▶  Prédire", command=predire, font=("Helvetica", 12, "bold"),
              bg="#2980b9", fg="white", activebackground="#1f6391", padx=20, pady=6).pack(pady=4)

    # === Légende globale =====================================================
    legende = tk.LabelFrame(root, text="  Légende — comment lire ce tableau de bord  ",
                            font=("Helvetica", 9, "bold"), bg="#fbfbfb", fg="#2c3e50", padx=12, pady=8)
    legende.pack(fill="x", padx=15, pady=(0, 12))
    texte_legende = (
        "• Étape 1 : cochez les caractéristiques que le modèle doit utiliser → il est ré-entraîné "
        "à chaque prédiction sur ces variables uniquement.\n"
        "• Étape 2 : entrez les mesures du module à évaluer (les valeurs par défaut = moyenne des données ; "
        "la plage observée est indiquée à gauche).\n"
        "• Étape 3 : la barre et le % indiquent la probabilité estimée de DÉFAUT. "
        "Rouge ≥ 50 % = défaut probable ; vert < 50 % = sain.\n"
        "• Les métriques affichées mesurent la fiabilité du modèle sur le jeu de test "
        "(rappel : seulement 8 données → indicatif)."
    )
    tk.Label(legende, text=texte_legende, bg="#fbfbfb", fg="#555", font=("Helvetica", 9),
             justify="left", wraplength=720).pack(anchor="w")

    predire()  # première prédiction au démarrage
    root.mainloop()


# Lancer le tableau de bord interactif
lancer_dashboard()
