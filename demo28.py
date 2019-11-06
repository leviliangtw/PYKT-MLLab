import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
origData = iris.data
pca = PCA(n_components=2)
data = pca.fit_transform(origData)
print(data.shape)  # breakpoint here, compare data/origData

datamax = data.max(axis=0) + 0.5  # y-axis, column
print(datamax)
# foo = data.max(axis=1)  # x-axis, row
# print(foo)
datamin = data.min(axis=0) + 0.5
n = 2000
X, y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))

svc = svm.SVC()  # default kernel: RBF
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), y.ravel()])
print(np.unique(Z))
plt.contour(X, y, Z.reshape(X.shape),
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=['r', 'g', 'b', 'k'])

for i, c in zip([0, 1, 2], ['r', 'g', 'b']):
    d = data[iris.target == i]
    plt.scatter(d[:, 0], d[:, 1], c=c, marker='.')
plt.show()
