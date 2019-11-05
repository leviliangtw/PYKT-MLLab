import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print(type(iris))
print(type(iris.data), type(iris.target))
labels = ["sepal length", "sepal width", "petal length", "petal width"]
X = iris.data
y = iris.target  # species
print(X.shape)
print(y)

counter = 1
for i in range(0, 4):
    for j in range(i + 1, 4):
        xData = X[:, i]
        yData = X[:, j]
        x_min, x_max = xData.min() - 0.5, xData.max() + 0.5
        y_min, y_max = yData.min() - 0.5, yData.max() + 0.5
        plt.clf()  # Clear the current figure.
        plt.scatter(xData, yData, c=y, cmap=plt.cm.Paired, marker='.')
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks()
        plt.yticks()
        plt.show()