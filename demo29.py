from numpy import array
from sklearn.decomposition import PCA

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
pca = PCA(n_components=1)
pca.fit(A)
print(f"components: \n{pca.components_}")
print(f"variance: {pca.explained_variance_}")
print(f"ratio: {pca.explained_variance_ratio_}")

B = pca.transform(A)
print(B)
