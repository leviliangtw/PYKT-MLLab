import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
rgrssn1 = linear_model.LinearRegression()
rgrssn1.fit(features, values)
print(rgrssn1)

plt.scatter([[0], [1], [2]], [1, 4, 5.5], c='g')
plt.scatter([[0], [3], [8]], [1, 4, 5.5], c='b')
plt.show()

print(f'coefficient={rgrssn1.coef_}')
print(f'intercept={rgrssn1.intercept_}')
print(f'x1 coef={rgrssn1.coef_[0]}, x2 coef={rgrssn1.coef_[1]}')
print('predict1=', rgrssn1.predict([[0.8, 0.8], [2, 4], [3, 5]]))