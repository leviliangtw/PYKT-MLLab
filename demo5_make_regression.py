import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import numpy as np

# np.random.seed(5353)
data1 = datasets.make_regression(100, 1, noise=5, random_state=150)
print(type(data1))
print(type(data1[0]), type(data1[1]))
print(data1[0].shape, data1[1].shape)

plt.scatter(data1[0], data1[1], c='red', marker='.')
plt.show()

regression1 = linear_model.LinearRegression()
regression1.fit(data1[0], data1[1])
print(f'coefficient={regression1.coef_[0]}')
print(f'intercept={regression1.intercept_}')
print(f'score={regression1.score(data1[0], data1[1])}')

range1 = [-3, 3]
plt.plot(range1, regression1.coef_*range1+ regression1.intercept_, c='black')
plt.scatter(data1[0], data1[1], c='red', marker='*')
plt.show()