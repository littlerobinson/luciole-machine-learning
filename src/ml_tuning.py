import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

def get_high_correlations(X, corr_percentage = 0.5):
    """
    Identifie les paires de colonnes avec une forte corrélation dans un DataFrame.

    Cette fonction calcule la matrice de corrélation pour un DataFrame donné et identifie
    les paires de colonnes dont la corrélation absolue dépasse un certain seuil.

    Paramètres :
    X (DataFrame) : Le DataFrame contenant les données.
    corr_percentage (float) : Le seuil de corrélation absolue (entre 0 et 1). Par défaut, 0.5.

    Retourne :
    high_corr_list (list) : Une liste de tuples contenant les paires de colonnes fortement corrélées
                            et leur corrélation en pourcentage.
    """
    corr = X.corr()

    high_corr_list = []
    cols = corr.columns

    for j in cols:
        for i, item in corr[j].items():
            if (i!=j) and abs(item) > corr_percentage:
                high_corr_list.append((i,j, f"{round(abs(item)*100, 2)} %"))
    return high_corr_list

def test_regressor_models(models, data, target_name, numeric_features = [], categorical_features = [], test_size = 0.2, iterations = 50):
    """
    Teste plusieurs modèles de régression sur un ensemble de données.

    Cette fonction prépare les données, applique des transformations de prétraitement,
    et évalue plusieurs modèles de régression en utilisant une validation croisée.

    Paramètres :
    models (dict) : Un dictionnaire de modèles de régression à tester.
    data (DataFrame) : Le DataFrame contenant les données.
    target_name (str) : Le nom de la colonne cible.
    numeric_features (list) : La liste des colonnes numériques. Par défaut, [].
    categorical_features (list) : La liste des colonnes catégorielles. Par défaut, [].
    test_size (float) : La proportion des données à utiliser pour le test. Par défaut, 0.2.
    iterations (int) : Le nombre d'itérations pour la validation croisée. Par défaut, 50.

    Retourne :
    results (DataFrame) : Un DataFrame contenant les scores R² moyens pour chaque modèle.
    """
    # Initialisation des résultats
    results = {}

    # Liste des transformateurs pour le prétraitement
    transformers = []

    # Création du pipeline pour les caractéristiques numériques
    if len(numeric_features) > 0:
        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Les valeurs manquantes seront remplacées par la médiane des colonnes
                ('scaler', StandardScaler())  # Normalisation des caractéristiques numériques
            ]
        )
        transformers.append(('num', numeric_transformer, numeric_features))

    # Création du pipeline pour les caractéristiques catégorielles
    if len(categorical_features) > 0:
        categorical_transformer = Pipeline(
            steps=[
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Encodage one-hot avec suppression de la première colonne pour éviter les corrélations entre les caractéristiques
            ]
        )
        transformers.append(('cat', categorical_transformer, categorical_features))

    # Préprocesseur de colonnes
    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Séparation des données cibles et des caractéristiques
    y = data.loc[:, target_name]
    X = data.drop(target_name, axis=1)

    # Évaluation des modèles
    for model_key in models:
        r2_train = []
        r2_test = []

        for j in range(iterations):
            # Séparation des données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)

            # Prétraitement des données d'entraînement
            X_train = preprocessor.fit_transform(X_train)

            # Prétraitement des données de test
            X_test = preprocessor.transform(X_test)

            # Entraînement du modèle
            regressor = models[model_key]
            regressor.fit(X_train, y_train)

            # Prédictions sur l'ensemble d'entraînement
            y_train_pred = regressor.predict(X_train)

            # Prédictions sur l'ensemble de test
            y_test_pred = regressor.predict(X_test)

            # Calcul des scores R²
            r2_train.append(r2_score(y_train, y_train_pred))
            r2_test.append(r2_score(y_test, y_test_pred))

        # Ajout des scores R² moyens pour l'entraînement et le test
        results[model_key] = {'r2_train': np.mean(r2_train), 'r2_test': np.mean(r2_test)}

    # Retourne les résultats sous forme de DataFrame
    return pd.DataFrame(results).transpose()
