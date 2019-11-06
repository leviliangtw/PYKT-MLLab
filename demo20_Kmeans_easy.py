from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [1, 8],
              [4, 2], [4, 4], [4, 0], [4, 6], [4, 7]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans)
print("kmeans center: ", kmeans.cluster_centers_)
print("kmeans label: ", kmeans.labels_)
print("lmeans inertia: ", kmeans.inertia_)