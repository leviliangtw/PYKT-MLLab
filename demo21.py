import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(13579)
X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
print(type(X), X.shape)
print(X[:5, ])
print(X[50:55, ])
print(X[100:105])

Ks = [2, 3, 4, 5]  # k is kinds of hyper-parameter
for k in Ks:
    kmeans1 = KMeans(n_init=1, n_clusters=k)
    kmeans1.fit(X)
    # total distance between node and center
    print("inertia: ", kmeans1.inertia_)
    # the center that the point belongs to
    # print(kmeans1.labels_)
    # # 3 centers
    # print(kmeans1.cluster_centers_)

    colors = ['c', 'm', 'y', 'k', 'r']
    markers = ['.', '*', '^', 'x', 's']
    for i in range(k):
        dataX = X[kmeans1.labels_ == i]
        plt.scatter(dataX[:, 0], dataX[:, 1],
                    c=colors[i], marker=markers[i])
        # print(dataX.size)
    plt.show()
