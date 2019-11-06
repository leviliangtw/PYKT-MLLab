import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
# y = np.array([1,2,2,1,2,2])
# y = np.array([1,1,2,1,1,2])

X_min = -4
X_max = 4
y_min = -4
y_max = 4
h = .005
xx, yy = np.meshgrid(np.arange(X_min, X_max, h),
                     np.arange(y_min, y_max, h))

clf1 = GaussianNB()
clf1.fit(X, y)
Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.pcolormesh(xx, yy, Z)

XB, YB, XR, YR = [], [], [], []
index = 0
for index in range(len(y)):
    if y[index] == 1:
        print("B equal to: ", X[index, :])
        XB.append(X[index, 0])
        YB.append(X[index, 1])
    if y[index] == 2:
        print("R equal to: ", X[index, :])
        XR.append(X[index, 0])
        YR.append(X[index, 1])
        pass
plt.scatter(XB, YB, color='b', label='blue, type1')
plt.scatter(XR, YR, color='r', label='red, type2')
plt.legend()
plt.xlabel('variable 1')
plt.ylabel('variable 2')
plt.show()