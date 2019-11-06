from sklearn import tree

X = [[0, 0], [1, 1]]
y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
print(clf)

print(clf.predict([[2, 2], [2, -2], [-2, -2], [3, 5], [5, -3]]))