import numpy as np

np.random.seed(42)
class LinearRegression:
    def __init__(self, learning_rate=1e-3, epochs=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X_train, Y_train):
        n_samples, n_feature = X_train.shape
        self.weights = np.random.random(n_feature)
        self.bias = np.random.random(1)
        for _ in range(self.epochs):
            y_pred = np.dot(X_train, self.weights) + self.bias
            dw = (1 / n_samples) * np.sum(2 * np.dot(X_train.T, (y_pred - Y_train)))
            db = (1 / n_samples) * np.sum(2 * (y_pred - Y_train))
            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw

    def predcit(self, X_test):
        """
        y=w^tx+b
        :param X_test:
        :return:
        """
        return np.dot(X_test, self.weights) + self.bias
