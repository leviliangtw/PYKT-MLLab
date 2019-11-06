import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2], ])
y = np.array([1, 1, 1, 2, 2, 2])

clf1 = GaussianNB()
clf1.fit(X, y)
print("clf1 predict: ", clf1.predict([[-0.8, -0.8],
                                      [1, 1.5],
                                      [-1, 1],
                                      [1, -1]]))
clf2 = GaussianNB()
clf2.partial_fit(X, y, np.unique(y))  # Incremental fit on a batch of samples
# np.unique(y): List of all the classes that can possibly appear in the y vector.
print("clf2 predict: ", clf2.predict([[-0.8, -0.8],
                                      [1, 1.5],
                                      [-1, 1],
                                      [1, -1]]))
clf2.partial_fit([[1, -0.8]], [2])  # Incremental fit on a batch of samples
print("clf2 predict: ", clf2.predict([[-0.8, -0.8],
                                      [1, 1.5],
                                      [-1, 1],
                                      [1, -1]]))
