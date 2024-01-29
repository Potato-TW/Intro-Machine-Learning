# You are not allowed to import any additional packages/libraries.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

# class LinearRegression:
#     def __init__(self):
#         self.closed_form_weights = None
#         self.closed_form_intercept = None
#         self.gradient_descent_weights = None
#         self.gradient_descent_intercept = None

#     # This function computes the closed-form solution of linear regression.
#     def closed_form_fit(self, X, y):
#         # Compute closed-form solution.
#         # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
#         pass

#     # This function computes the gradient descent solution of linear regression.
#     def gradient_descent_fit(self, X, y, lr, epochs):
#         # Compute the solution by gradient descent.
#         # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
#         pass


#     # This function compute the MSE loss value between your prediction and ground truth.
#     def get_mse_loss(self, prediction, ground_truth):
#         # Return the value.
#         pass

#     # This function takes the input data X and predicts the y values according to your closed-form solution.
#     def closed_form_predict(self, X):
#         # Return the prediction.
#         pass

#     # This function takes the input data X and predicts the y values according to your gradient descent solution.
#     def gradient_descent_predict(self, X):
#         # Return the prediction.
#         pass

#     # This function takes the input data X and predicts the y values according to your closed-form solution,
#     # and return the MSE loss between the prediction and the input y values.
#     def closed_form_evaluate(self, X, y):
#         # This function is finished for you.
#         return self.get_mse_loss(self.closed_form_predict(X), y)

#     # This function takes the input data X and predicts the y values according to your gradient descent solution,
#     # and return the MSE loss between the prediction and the input y values.
#     def gradient_descent_evaluate(self, X, y):
#         # This function is finished for you.
#         return self.get_mse_loss(self.gradient_descent_predict(X), y)

#     # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
#     # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
#     def plot_learning_curve(self):
#         pass


class LinearRegression_:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.gradient_mse = []

    def closed_form_fit(self, X, y):
        # 计算封闭式解。
        # 将权重和截距保存到 self.closed_form_weights 和 self.closed_form_intercept

        # 封闭式解：theta = (X^T * X)^(-1) * X^T * y
        # 在 X 中添加一列全为 1 的项作为截距
        X = np.column_stack((np.ones(X.shape[0]), X))
        # 使用封闭式解计算 theta
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.closed_form_weights = theta[1:]  # 排除截距
        self.closed_form_intercept = theta[0]  # 截距

    def gradient_descent_fit(self, X, y, lr, epochs):
        # 使用梯度下降计算解。
        # 将权重和截距保存到 self.gradient_descent_weights 和 self.gradient_descent_intercept

        # 在 X 中添加一列全为 1 的项作为截距
        # X = np.column_stack((np.ones(X.shape[0]), X))

        # 初始化权重和截距

        num_features = X.shape[1]
        self.gradient_descent_weights = np.zeros(num_features)
        self.gradient_descent_intercept = 0

        

        # 执行梯度下降
        for epoch in range(epochs):
            # 计算预测值
            predictions = X @ self.gradient_descent_weights + self.gradient_descent_intercept

            # 计算梯度
            gradient_weights = -(2 / len(y)) * X.T @ (y - predictions)
            gradient_intercept = -(2 / len(y)) * np.sum(y - predictions)

            # 更新权重和截距
            self.gradient_descent_weights -= lr * gradient_weights
            self.gradient_descent_intercept -= lr * gradient_intercept

            self.gradient_mse.append(self.get_mse_loss(y, predictions))

    def get_mse_loss(self, prediction, ground_truth):
        # 计算均方误差（MSE）损失
        # print(f'pred = {prediction}\ngrou = {ground_truth}')
        mse = np.mean((prediction - ground_truth) ** 2)
        return mse

    def closed_form_predict(self, X):
        # 使用封闭式解进行预测
        return X @ self.closed_form_weights + self.closed_form_intercept

    def gradient_descent_predict(self, X):
        # 使用梯度下降解进行预测
        return X @ self.gradient_descent_weights + self.gradient_descent_intercept
        # return X @ self.gradient_descent_weights[1:] + self.gradient_descent_intercept

    def closed_form_evaluate(self, X, y):
        # 使用封闭式解计算 MSE 损失
        return self.get_mse_loss(self.closed_form_predict(X), y)

    def gradient_descent_evaluate(self, X, y):
        # 使用梯度下降解计算 MSE 损失
        return self.get_mse_loss(self.gradient_descent_predict(X), y)

    def plot_learning_curve(self):
        # 绘制学习曲线
        plt.plot(range(1, len(self.gradient_mse) + 1), self.gradient_mse)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Learning Curve')
        plt.show()


# 導入迴歸模型套件

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
    LR = LinearRegression_()

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
  
#   -------------------

    record = f'lr = {0.0001} ep = {2000000} cept not in'

    with open('fff.txt', 'a') as file:
        file.write(f'{record}\nError Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%\n\n')

    LR.plot_learning_curve()
    
# -------------------