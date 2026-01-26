# Importer les bibliothèques nécessaires
import pandas as pd

# -------------------------------------
# Étape 1 : Acquisition des données
# -------------------------------------

# Charger le fichier CSV
df = pd.read_csv('sample_dataset_20_entries.csv')

# Afficher les premières lignes pour examiner les données
print("Aperçu initial des données :")
print(df.head(10))

# exit()  # Supprimer cette ligne après vérification initiale
# -------------------------------------
# Étape 2 : Prétraitement des données
# -------------------------------------

# Identifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())


# Remplacer les valeurs manquantes dans la colonne 'Age' par la moyenne
df['Age'] = df['Age'].fillna(round(df['Age'].mean(), 1))

# Remplacer les valeurs manquantes dans la colonne 'Income' par le max
df['Income'] = df['Income'].fillna(df['Income'].max())

# Remplacer les valeurs manquantes dans la colonne 'Gender' par 'Unknown'
df['Gender'] = df['Gender'].fillna('Unknown')


# Supprimer les doublons sans regarder la colonne ID
df.drop_duplicates(subset=[col for col in df.columns if col != 'ID'], inplace=True)

# Supprimer les colonnes inutiles (par exemple, 'Irrelevant_Column')
if 'Irrelevant_Column' in df.columns:
    df.drop(columns=['Irrelevant_Column'], inplace=True)

print(df.head(10))


# -------------------------------------
# Étape 3 : Ingénierie des caractéristiques
# -------------------------------------

# Créer une nouvelle colonne 'Year' à partir de la colonne 'Date'
df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

# Supprimer la colonne 'Date' si elle n'est plus nécessaire
if 'Date' in df.columns:
    df.drop(columns=['Date'], inplace=True)

# -------------------------------------
# Vérification finale et sauvegarde
# -------------------------------------

# Afficher un résumé des données nettoyées
print("\nRésumé des données après nettoyage :")
print(df.info())

# Afficher les premières lignes des données nettoyées
print("\nAperçu des données nettoyées :")
print(df.head(10))

# Sauvegarder les données nettoyées dans un nouveau fichier CSV
df.to_csv('cleaned_dataset.csv', index=False)
print("\nLes données nettoyées ont été sauvegardées dans 'cleaned_dataset.csv'.")

