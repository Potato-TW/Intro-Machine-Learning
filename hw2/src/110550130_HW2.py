# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None
        self.cee = []

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.zeros(X.shape[1])

        for i in range(self.iteration):
            tmp = X @ self.weights
            prob = self.sigmoid(tmp)
            
            self.cee = - (y * np.log(prob) + (1 - y) * np.log(1 - prob))
            gradient = X.T @ (prob - y) / y.size
            self.weights -= self.learning_rate * gradient

        self.intercept = self.weights[0]
        self.weights = self.weights[1:]
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        
        return  np.round(self.sigmoid(X @ np.concatenate(([self.intercept], self.weights))))

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        
        self.xt0 = None
        self.xt1 = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        self.xt0 = X[y == 0]
        self.xt1 = X[y == 1]
        
        self.m0 = np.mean(X[y == 0], axis=0)
        self.m1 = np.mean(X[y == 1], axis=0)

        self.sw = (X[y == 0] - self.m0).T @ (X[y == 0] - self.m0) + (X[y == 1] - self.m1).T @ (X[y == 1] - self.m1)
        self.sb = np.outer((self.m1 - self.m0), (self.m1 - self.m0))
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        line = np.dot(X, self.w)
        
        return np.where(line < np.mean(line), 0, 1)

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X, y):
        a = self.w[1] / self.w[0]
        b = 0
        intv =  np.linspace(-60, 200)
        yax = a * intv + b
        plt.plot(intv, yax, color='black', linewidth=0.2)
        
        ax = X[y==0].T[0]
        ay = X[y==0].T[1]
        bx = X[y==1].T[0]
        by = X[y==1].T[1]
        plt.plot(bx, by, '.', color='red', markersize=5)
        plt.plot(ax, ay, '.', color='blue', markersize=5)
       
        for i in range(ax.size):
            tmpx=1
            tmpy=a
            k=(ax[i]*1+ay[i]*a)/(a**2+1)
            tmpx=tmpx*k+0
            tmpy=tmpy*k+b
            
            # print(f'b: {b} k: {k} ax: {ax[i]} ay: {ay[i]} tx: {tmpx} ty: {tmpy}')
            plt.plot(tmpx, tmpy, '.', color='blue', markersize=5)
            plt.plot([tmpx, ax[i]], [tmpy, ay[i]], color='blue', linewidth=0.1)
        
        for i in range(bx.size):
            tmpx=1
            tmpy=a
            k=(bx[i]*1+by[i]*a)/(a**2+1)
            tmpx=tmpx*k+0
            tmpy=tmpy*k+b
            
            # print(f'b: {b} k: {k} bx: {bx[i]} by: {by[i]} tx: {tmpx} ty: {tmpy}')
            plt.plot(tmpx, tmpy, '.', color='red', markersize=5)
            plt.plot([tmpx, bx[i]], [tmpy, by[i]], color='red', linewidth=0.1)
        
        plt.xlim(-60,200)
        plt.ylim(-60,200)
        plt.title(f'Projection Line: w={a}, b={b}')
        plt.show()
     
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.00015, iteration=100000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    
    # 刪掉
    # with open('res2.txt', 'a')as file:
    #     file.write(f'm0: {FLD.m0}\nm1: {FLD.m1}\nsw: {FLD.sw}\nsb: {FLD.sb}\nw: {FLD.w}\narr: {accuracy}\n\n')
    FLD.plot_projection(X_test, y_test)
    # 刪掉
    
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"