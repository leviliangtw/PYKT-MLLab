import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca = PCA(n_components=2)
data = pca.fit_transform(iris.data)
target = iris.target

print(data.shape)
print(data[:5,:])

svc = svm.SVC() # default
# svc = svm.SVC(kernel='linear')
# svc = svm.SVC(kernel='poly')
# svc = svm.SVC(kernel='rbf') # default
# svc = svm.SVC(kernel='sigmoid') # not proper here
# svc = svm.SVC(kernel='precomputed') # not work here
# svc = svm.SVC(C=1000)

svc.fit(data, target)
datamax = data.max(axis=0)+1
datamin = data.min(axis=0)-1
print(datamax)
print(datamin)

n=2000
X, y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
print(X.shape)
print(y.shape)

eachGrid = np.c_[X.ravel(), y.ravel()] # ravel(): Return a contiguous flattened array.
print(eachGrid[:10])
print(type(eachGrid), eachGrid)

Z = svc.predict(eachGrid)
print(type(Z), Z.shape)
plt.contour(X, y, Z.reshape(X.shape))

for t, c, m in zip([0, 1, 2], ['r', 'g', 'b'], ['o', '^','*']):
    print(t, c, m)
    d = data[iris.target == t]
    plt.scatter(d[:, 0], d[:, 1], c=c, marker=m)
plt.show()