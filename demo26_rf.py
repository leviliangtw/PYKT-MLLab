import seaborn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(type(df), df.columns)

df['species'] = np.array([iris.target_names[i] for i in iris.target])
print(type(df), df.columns)
# seaborn.pairplot(df, hue='species')
# plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names],
                                                    iris.target,
                                                    test_size=0.5,
                                                    stratify=iris.target)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
print(rf)

from sklearn.metrics import accuracy_score

predicted = rf.predict(X_test)
print(f'OOB estimation={rf.oob_score_}')
accuracy = accuracy_score(y_test, predicted)
print(f'my mean accuracy={accuracy}')

from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(confusion_matrix(y_test, predicted),
                  columns=iris.target_names,
                  index=iris.target_names)
seaborn.heatmap(cm, annot=True)
plt.show()
