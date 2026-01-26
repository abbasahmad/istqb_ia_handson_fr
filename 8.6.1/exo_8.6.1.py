from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Charger les données Iris
iris = load_iris()
X, y = iris.data, iris.target

feature_names = iris.feature_names
class_names = iris.target_names

print("Classes disponibles:", class_names)
print("Noms des caractéristiques:", feature_names)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Distribution des classes dans l'ensemble de test:", [list(y_test).count(i) for i in range(3)])

# Entraîner un modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Modèle entraîné avec succès!")


from lime.lime_tabular import LimeTabularExplainer

# Initialiser LIME pour des données tabulaires avec paramètres améliorés
explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',  # Explicitement spécifier le mode classification
    discretize_continuous=True,
    verbose=True  # Plus d'informations de débogage
)

# Prendre un échantillon
sample_index = 29 #29
sample = X_test[sample_index].reshape(1, -1)

# Debug: Afficher des informations sur l'échantillon
print(f"\nÉchantillon #{sample_index}:")
print(f"Valeurs des caractéristiques: {X_test[sample_index]}")
print(f"Vraie classe: {class_names[y_test[sample_index]]}")
print(f"Prédiction du modèle: {class_names[model.predict(sample)[0]]}")
print(f"Probabilités: {model.predict_proba(sample)[0]}")

# Générer une explication pour cet échantillon avec des paramètres améliorés
explanation = explainer.explain_instance(
    data_row=X_test[sample_index],
    predict_fn=model.predict_proba,
    num_features=4,
    labels=[0, 1, 2]  # Explicitement demander les 3 classes
)

# Afficher l'explication dans la console
print(f"\nExplication LIME:")
print("Explication pour toutes les classes:")
for class_idx in [0, 1, 2]:
    print(f"\nClasse {class_names[class_idx]}:")
    class_explanation = explanation.as_list(label=class_idx)
    for feature, weight in class_explanation:
        print(f"  {feature}: {weight:.4f}")

# Enregistrer les explications pour chaque classe
explanation.save_to_file('explanation_all_classes.html')
print(f"Explication sauvegardée dans explanation_all_classes.html")


# Tester avec d'autres échantillons pour vérifier
print("\n" + "="*50)
print("Test avec d'autres échantillons:")
for test_idx in [0, 5, 15, 25]:
    if test_idx < len(X_test):
        test_sample = X_test[test_idx].reshape(1, -1)
        true_class = class_names[y_test[test_idx]]
        pred_class = class_names[model.predict(test_sample)[0]]
        proba = model.predict_proba(test_sample)[0]
        print(f"Échantillon #{test_idx}: Vraie={true_class}, Prédite={pred_class}, Proba={proba.round(3)}")
