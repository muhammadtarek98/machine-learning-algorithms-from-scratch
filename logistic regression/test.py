from sklearn import datasets
from Logistic_Regression import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = datasets.load_breast_cancer()
X, Y = dataset.data, dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
model = LogisticRegression(learning_rate=0.001, epochs=1000)
model.fit(X=X_train, Y=Y_train)
pred = model.predicit(X=X_test)
acc = LogisticRegression.accuracy(Y_test, pred)
print(acc)
del model
