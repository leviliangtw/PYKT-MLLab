import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets

data1 = datasets.make_regression(6, 5, noise=5)
X = data1[0] # 6 samples, 5 features
y = data1[1] # 6 targets/values
print(X)

for i in range(5): # traverse each feature
    plt.scatter(X[:, i], y)
    plt.show()