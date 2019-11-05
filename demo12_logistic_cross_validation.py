import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
data = iris.data
target= iris.target

logisticReg1 = LogisticRegression()
score = model_selection.cross_val_score(logisticReg1, data, target, cv=5)
# cv: Determines the cross-validation splitting strategy.

print(score)