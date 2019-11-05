from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import  matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris.target_names)
# print(iris.DESCR)
# print(iris.feature_names)
# print(iris.filename)

X = iris["data"][:, 3:]
# X = iris.data[:, 3:]
y = (iris["target"] == 2).astype(np.int)
print(type(X), X.shape)
print(type(y), y.shape)

reg1 = LogisticRegression()
reg1.fit(X, y)

X_plot = np.linspace(0, 3, 1000)
print(X_plot.shape)
X_plot=X_plot.reshape(-1, 1)
print(X_plot.shape)
y_porba = reg1.predict_proba(X_plot)

plt.plot(X, y, "g^")
plt.plot(X_plot, y_porba[:, 1], 'g--', label='iris-virginica')
plt.plot(X_plot, y_porba[:, 0], 'b--', label='Not iris-virginica')
plt.xlabel("Petal width", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="upper left", fontsize=10)

# print(reg1.predict_proba([[1.2], [1.4], [2.2], [2.4],[3]]))
myX_plot = [[1.2], [1.4], [2.2], [2.4],[3]]
myy_porba = reg1.predict_proba(myX_plot)
print(myy_porba[:, 1])
plt.scatter(myX_plot, myy_porba[:, 1], c="r", marker='*')

plt.show()
