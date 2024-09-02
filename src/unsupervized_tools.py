# Let's create a loop that will collect the Within-sum-of-square (wcss) for each value K
# Let's use .inertia_ parameter to get the within sum of square value for each value K
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def elbow(X, clusters_to_display=10, random_state=42):
  """
  Create a loop that will collect the Within-sum-of-square (wcss) for each value K
  Use .inertia_ parameter to get the within sum of square value for each value K

  Parameters:
  X: training set as list

  Returns:
  tuple (wcss, k)
  """
  wcss =  []
  k = []
  for i in range (1, clusters_to_display):
      kmeans = KMeans(n_clusters = i, random_state = random_state, n_init = 'auto')
      kmeans.fit(X)
      wcss.append(kmeans.inertia_)
      k.append(i)
      print("WCSS for K={} --> {}".format(i, wcss[-1]))
      return (wcss, k)
  

def silhouette(X, clusters_range=(2,11), random_state=42):
  """
  Compute mean silhouette score

  Parameters:
  X: training set as list

  Returns:
  tuple (sil, k)
  """

  sil = []
  k = []

  ## Careful, you need to start at i=2 as silhouette score cannot accept less than 2 labels
  for i in range (clusters_range):
      kmeans = KMeans(n_clusters= i, random_state = random_state, n_init = 'auto')
      kmeans.fit(X)
      sil.append(silhouette_score(X, kmeans.predict(X)))
      k.append(i)
      print("Silhouette score for K={} is {}".format(i, sil[-1]))
      return (sil, k)
  


def evaluate_clustering_combination(ms, ep, X, min_labels=4, max_labels=14, n_jobs=None):
    """
    Effectue un clustering DBSCAN sur un ensemble de données en utilisant une combinaison spécifique
    des paramètres `eps` et `min_samples`, et évalue la qualité du clustering en fonction du score silhouette.

    Parameters:
    -----------
    ms : int
        Le nombre minimum de points requis pour qu'une région soit considérée comme un cluster.
        Ce paramètre correspond à `min_samples` dans l'algorithme DBSCAN.

    ep : float
        La distance maximale entre deux points pour les considérer comme voisins.
        Ce paramètre correspond à `eps` dans l'algorithme DBSCAN.

    X : array-like, shape (n_samples, n_features)
        Les données d'entrée à clusteriser.

    min_labels : int, optional (default=4)
        Le nombre minimum de clusters différents (étiquettes) requis pour que la combinaison soit considérée valide.

    max_labels : int, optional (default=14)
        Le nombre maximum de clusters différents (étiquettes) permis pour que la combinaison soit considérée valide.

    n_jobs : int or None, optional (default=None)
        Le nombre de cœurs à utiliser pour le calcul. None signifie 1 (par défaut). 
        -1 signifie utiliser tous les processeurs disponibles.

    Returns:
    --------
    tuple : (int, float, float or None)
        Un tuple contenant `min_samples`, `eps`, et le score silhouette correspondant si la combinaison est valide,
        sinon `None` à la place du score silhouette.

    Description:
    ------------
    La fonction exécute les étapes suivantes :
    
    1. Crée un modèle DBSCAN en utilisant les paramètres `ms` (min_samples) et `ep` (eps) avec la distance de Manhattan.
    2. Applique le modèle DBSCAN aux données d'entrée `X`.
    3. Récupère les étiquettes de clusters résultants (`labels`) et compte le nombre de clusters uniques (`nb_labels`).
    4. Vérifie si le nombre de clusters se situe entre `min_labels` et `max_labels` inclusivement.
    5. Si la condition est remplie, calcule le score silhouette pour mesurer la cohésion des clusters.
    6. Affiche les paramètres `min_samples`, `eps`, le nombre de clusters, et le score silhouette.
    7. Retourne un tuple contenant les paramètres `ms`, `ep`, et le score silhouette. 
       Si une exception se produit, retourne le tuple avec `None` pour le score.

    Exceptions:
    -----------
    - Si une erreur survient pendant le processus de clustering, l'exception est capturée et un message d'erreur est imprimé.
      Dans ce cas, la fonction retourne `ms`, `ep`, et `None`.
    """
    try:
        db = DBSCAN(eps=ep, min_samples=ms, metric="manhattan", n_jobs=n_jobs)
        db.fit(X)
        labels = db.labels_
        nb_labels = np.unique(labels)
        if ((len(nb_labels) >= min_labels) and (len(nb_labels) <= max_labels)):
            score = silhouette_score(X, labels)
            return (ms, ep, score)
    except Exception as e:
        print(f"Error processing min_samples={ms}, eps={ep}: {e}")
        return (ms, ep, None)
    

def get_time_filtered_dbscan_centroids(input_data, eps, min_samples, day, hour):
    """
    Applique l'algorithme DBSCAN sur un sous-ensemble de données de localisation
    pour un jour et une heure spécifiques, et retourne les centroïdes des clusters identifiés.

    Parameters:
    -----------
    input_data : pandas.DataFrame
        Le DataFrame contenant les données d'entrée. Il doit inclure au moins les colonnes 'Day', 'Hour', 'Lat' et 'Lon'.

    eps : float
        Le paramètre `eps` de DBSCAN, qui définit la distance maximale entre deux points pour qu'ils soient considérés comme voisins.

    min_samples : int
        Le nombre minimum de points requis pour qu'une région soit considérée comme un cluster par DBSCAN.

    day : int
        Le jour spécifique pour lequel les données doivent être filtrées (valeur de la colonne 'Day').

    hour : int
        L'heure spécifique pour laquelle les données doivent être filtrées (valeur de la colonne 'Hour').

    Returns:
    --------
    pandas.DataFrame
        Un DataFrame contenant les centroïdes des clusters identifiés par DBSCAN. 
        Les colonnes incluent 'cluster_dbscan', 'Lat', et 'Lon'.
        - 'cluster_dbscan' : l'identifiant du cluster.
        - 'Lat' : la latitude du centroïde.
        - 'Lon' : la longitude du centroïde.

    Description:
    ------------
    La fonction exécute les étapes suivantes :

    1. **Filtrage des données** : Les données d'entrée `input_data` sont filtrées pour ne conserver que les lignes correspondant
        au jour (`Day`) et à l'heure (`Hour`) spécifiés.

    2. **Normalisation** : Les colonnes de localisation ('Lat', 'Lon') sont normalisées à l'aide de `StandardScaler` afin de standardiser les distances.

    3. **Clustering avec DBSCAN** : L'algorithme DBSCAN est appliqué sur les données normalisées pour identifier les clusters de points de localisation.
        - `eps` et `min_samples` sont les paramètres de DBSCAN.
        - La métrique utilisée pour mesurer la distance entre les points est la distance de Manhattan.

    4. **Extraction des centroïdes** : Pour chaque cluster identifié (c'est-à-dire avec un label non négatif), les coordonnées moyennes 
        (latitude et longitude) sont calculées pour obtenir le centroïde du cluster.

    5. **Retour des centroïdes** : La fonction retourne un DataFrame contenant les centroïdes des clusters identifiés.

    Notes:
    ------
    - Si aucun cluster n'est trouvé (c'est-à-dire si tous les points sont classés comme bruit par DBSCAN), le DataFrame retourné sera vide.
    - Le DataFrame résultant contient uniquement les clusters valides (labels >= 0), excluant les points de bruit.

    Examples:
    ---------
    ```python
    # Supposons que 'df' est un DataFrame contenant des colonnes 'Day', 'Hour', 'Lat', et 'Lon'.
    centroids = get_uber_centroid_points(df, eps=0.5, min_samples=5, day=3, hour=14)
    print(centroids)
    ```
    """
    data = input_data[input_data['Day'] == day]
    data = input_data[input_data['Hour'] == hour]

    X = data[['Lat', 'Lon']]

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="manhattan", n_jobs=-1)
    db.fit(X)
    
    # Calculer les centroids par cluster
    centroids = data[data['cluster_dbscan'] >= 0].groupby('cluster_dbscan')[['Lat', 'Lon']].mean().reset_index()

    # Fusionner les centroids avec le dataframe original pour ajouter les coordonnées du centroïde
    data_with_centroids = pd.merge(data, centroids, on='cluster_dbscan', how='left', suffixes=('', '_centroid'))


    return data_with_centroids