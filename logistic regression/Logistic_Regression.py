import numpy as np

np.random.seed(42)


class LogisticRegression:

    def __init__(self, learning_rate=1e-3, epochs=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.random.random(size=(n_features,))
        self.bias = np.random.random(size=(1,))
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = LogisticRegression.sigmoid(linear_model)
            dw = (1 / n_samples) * np.sum(2 * np.dot(X.T, (y_pred - Y)))
            db = (1 / n_samples) * np.sum(2 * (y_pred - Y))
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def predcit(self, X):
        pred = LogisticRegression.sigmoid(np.dot(X, self.weights) + self.bias)
        return [1 if i > 0.5 else 0 for i in pred]
