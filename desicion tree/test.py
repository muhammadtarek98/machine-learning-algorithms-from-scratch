from sklearn import datasets
from desicion_tree import DesicionTreeClassifier
from sklearn.model_selection import train_test_split

dataset = datasets.load_breast_cancer()
X, Y = dataset.data, dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)
model = DesicionTreeClassifier(max_depth=10)
model.fit(X=X_train, y=Y_train)
pred = model.predict(X=X_test)
acc = DesicionTreeClassifier.accuracy(Y_test, pred)
print(acc)
del model
