import numpy as np

np.random.seed(42)


class SVM:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def __init__(self, learning_rate=1e-2, epochs=1, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, Y):
        Y_ = np.where(Y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.random.random(size=(n_features))
        self.b = np.random.random(size=(1))
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                forward = Y_[idx] * (np.dot(x_i, self.w) - self.b)
                if forward >= 1:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param - np.dot(x_i, Y_[idx])
                    db = Y_[idx]
            self.w -= (self.learning_rate * dw)
            self.b -= (self.learning_rate * db)

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
