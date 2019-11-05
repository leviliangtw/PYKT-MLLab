import numpy as np
from sklearn import linear_model, datasets, model_selection

diabetes = datasets.load_diabetes()
print(type(diabetes))
print(type(diabetes.data), diabetes.data.shape)
print(type(diabetes.target), diabetes.target.shape)

dataForTest = 50
data_train = diabetes.data[dataForTest:,:]
target_train= diabetes.target[dataForTest:]
print(f'data train shape:{data_train.shape}')
print(f'target train shape:{target_train.shape}')
data_test = diabetes.data[:dataForTest,:]
target_test= diabetes.target[:dataForTest]
print(f'data test shape:{data_test.shape}')
print(f'target test shape:{target_test.shape}')


X_train, X_test, y_train, y_test = model_selection.train_test_split(diabetes.data, diabetes.target, test_size=dataForTest)


print(f'X_train shape:{X_train.shape}')
print(f'y_train shape:{y_train.shape}')
print(f'X_test shape:{X_test.shape}')
print(f'y_test shape:{y_test.shape}')

reg1 = linear_model.LinearRegression()
reg1.fit(X_train, y_train)
print('score1: ', reg1.score(X_train, y_train))
print('score2: ', reg1.score(X_test, y_test))

# target_predict = reg1.predict(data_test)
# print(type(target_predict), target_predict.shape)
# for i in range(0, dataForTest):
#     print('[I]predict:%.2f/%.2f' % (target_predict[i], target_test[i]))

for i in range(0, dataForTest):
    print("data_test: ", data_test.shape)
    data = np.array(data_test[i])
    print("data_test[i] before reshape: ", data.shape)
    dataArray = data.reshape(1, 10)
    print("data_test[i] after reshape: ", dataArray.shape)
    print(dataArray.shape)
    print('predict/real: %.2f/%.2f'%(reg1.predict(dataArray)[0], target_test[i]))