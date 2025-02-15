import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.gradient_mse = []

    def closed_form_fit(self, X, y):
        # res = (X^T * X)^(-1) * X^T * y
        X = np.column_stack((np.ones(X.shape[0]), X))

        res = np.linalg.inv(X.T @ X) @ X.T @ y

        self.closed_form_weights = res[1:]
        self.closed_form_intercept = res[0]

    def gradient_descent_fit(self, X, y, lr, epochs):
        self.gradient_descent_weights = np.zeros(X.shape[1])
        self.gradient_descent_intercept = 0

        for epoch in range(epochs):
            pred = self.gradient_descent_predict(X)

            w = - X.T @ (y - pred)* 2 / len(X)
            c = - np.sum(y - pred)* 2 / len(X)

            self.gradient_descent_weights -= lr * w
            self.gradient_descent_intercept -= lr * c

            self.gradient_mse.append(self.get_mse_loss(y, pred))

    def get_mse_loss(self, pred, real):
        mse = np.mean((pred - real) ** 2)
        return mse

    def closed_form_predict(self, X):
        return X @ self.closed_form_weights + self.closed_form_intercept

    def gradient_descent_predict(self, X):
        return X @ self.gradient_descent_weights + self.gradient_descent_intercept

    def closed_form_evaluate(self, X, y):
        return self.get_mse_loss(self.closed_form_predict(X), y)

    def gradient_descent_evaluate(self, X, y):
        return self.get_mse_loss(self.gradient_descent_predict(X), y)

    def plot_learning_curve(self):
        plt.plot(range(1, len(self.gradient_mse) + 1), self.gradient_mse)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Learning Curve')
        plt.show()


# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()

    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.0001, epochs=2000000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")

    # LR.plot_learning_curve()