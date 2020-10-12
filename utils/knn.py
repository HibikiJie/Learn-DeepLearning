from sklearn import datasets, neighbors, preprocessing
import sklearn
import numpy
data = []
target = []
with open('testBiggerN.txt','r') as file:
    for line in file.readlines():
        line = line.strip().split()
        if len(line) > 0:
            target.append(float(line[0]))
            data.append([float(line[1][2:]), float(line[2][2:])])
data = numpy.array(data, dtype=numpy.float64)
target = numpy.array(target, dtype=numpy.int32)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=0.3)
# print(x_train)

"""数据预处理"""
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
# print(x_train)
x_test = scaler.transform(x_test)

"""
            创建模型
对于KNeighborsClassifier 来说，里面只保存了训练集。
想要基于训练集来构建模型，需要调用 knn 对象的 fit 方法，
输入参数为 X_train 和 y_train，二者都是 NumPy 
数组，前者包含训练数据，后者包含相应的训练标签 
"""
knn = neighbors.KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

"""交叉验证"""
scores = sklearn.model_selection.cross_val_score(knn,x_train,y_train,cv=7,scoring='accuracy')
print(scores)
print(scores.mean())

"""预测"""
y_prediction = knn.predict(x_test)

"""评估"""
print(sklearn.metrics.accuracy_score(y_test,y_prediction))

