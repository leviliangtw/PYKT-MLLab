import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X = np.array([[-1,1], [-2,-1], [-3, -3], [1, 1], [2, 1], [3, 3]])
y = np.array([1, 1, 1, 2, 2, 2])
clf1 = SVC()
clf1.fit(X, y)
print(type(clf1))
myX = np.array([[-0.8, -0.8], [4, 4], [-1, -0.8], [0, 0], [2, 2]])
myy = clf1.predict(myX)
print("prdict: ", myy)

for t, c, m in zip([1, 2], ['red', 'green'], ['o', '^']):
    d = X[y == t]
    plt.scatter(d[:, 0], d[:, 1], c=c, marker=m)
for t, c, m in zip([1, 2], ['red', 'green'], ['.', '*']):
    myd = myX[myy == t]
    plt.scatter(myd[:, 0], myd[:, 1], c=c, marker=m)
plt.show()