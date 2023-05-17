from knn import Knn
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()


X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=1234
)

k = 5
clf = Knn(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc=Knn.accuracy(y_true=y_test,y_pred=predictions)
print(acc)