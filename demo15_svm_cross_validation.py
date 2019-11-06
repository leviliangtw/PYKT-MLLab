from sklearn import datasets
from sklearn import model_selection
from sklearn import svm

iris = datasets.load_iris()

svc1 = svm.SVC()
svc2 = svm.SVC(kernel='linear')
svc3 = svm.SVC(kernel='poly')
svcs=[svc1, svc2, svc3]

for svc in svcs:
    scores = model_selection.cross_val_score(svc, iris.data, iris.target, cv=5)
    print(scores)
    print("accuracy: ", scores.mean())