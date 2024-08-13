# Let's create a loop that will collect the Within-sum-of-square (wcss) for each value K
# Let's use .inertia_ parameter to get the within sum of square value for each value K
from sklearn.cluster import KMeans
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