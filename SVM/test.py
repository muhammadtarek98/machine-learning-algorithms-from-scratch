from sklearn import datasets
from svm import SVM
from sklearn.model_selection import train_test_split
import numpy as np
dataset = datasets.load_breast_cancer()
X, Y = dataset.data, dataset.target
# print(Y)
np.random.seed(42)
Y = np.where(Y == 0, 1, -1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
model = SVM(learning_rate=0.1, epochs=100)
model.fit(X=X_train, Y=Y_train)
pred = model.predict(X=X_test)
acc = SVM.accuracy(Y_test, pred)
print(acc)
del model
