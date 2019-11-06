import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])
sn = NearestNeighbors(n_neighbors=2, algorithm='auto')
sn.fit(X)
distance, indices = sn.kneighbors(X, return_distance=True)

print("distance: \n", distance)
print("indices: \n", indices)