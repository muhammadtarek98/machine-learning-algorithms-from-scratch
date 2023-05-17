import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from linearregression import LinearRegression

diabetes = datasets.load_diabetes()
X=diabetes.data
Y=diabetes.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, shuffle=True, random_state=1234)
model = LinearRegression(learning_rate=1e-2, epochs=10)
model.fit(X_train=X_train, Y_train=Y_train)
pred = model.predicit(X_test=X_test)
print(LinearRegression.mse(y_true=Y_test, y_pred=pred))
del model