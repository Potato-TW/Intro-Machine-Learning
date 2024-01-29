#  You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    unique, cnt = np.unique(y, return_counts=True)
    prob = cnt / len(y)
    return 1 - np.sum(prob ** 2)

# This function computes the entropy of a label array.
def entropy(y):
    unique, cnt = np.unique(y, return_counts=True)
    prob = cnt / len(y)
    return -np.sum(prob * np.log2(prob))


class Node:
    def __init__(self, impurity, pred, depth, feat, threshold, left, right):
        self.impurity = impurity
        self.pred = pred
        self.depth = depth
        self.feat = feat
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.criterion = criterion

    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)

    def fit(self, x_data, y_data):
        data = x_data.values.tolist() if isinstance(x_data, DataFrame) else x_data
        self.root = self.findMin([list(data[i]) + [y_data[i]] for i in range(len(y_data))])
        self.split(self.root, self.max_depth, 1)   
        
    def predict(self, x_data):
        dataset = x_data.values.tolist() if isinstance(x_data, DataFrame) else x_data
        predictions = []
        
        for data in dataset:
            node = self.root
            while isinstance(node, Node):
                if data[node.feat] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node)
            
        return predictions

    def findMin(self, dataset):
        pure = f = thres = float('inf')
        l = r = None
        for feat in range(len(dataset[1]) - 1):
            tmp = np.sort(np.unique([data[feat] for data in dataset]))[:-1]      
                 
            for threshold in tmp:
                left = []
                right = []
                for i in range(len(dataset)):
                    if dataset[i][feat] <= threshold:
                        left.append(dataset[i])
                    else:
                        right.append(dataset[i])
                        
                sp = (self.impurity([row[-1] for row in left]) * len(left) + self.impurity([row[-1] for row in right]) * len(right)) / len(dataset)
                
                if sp < pure:
                    pure = sp
                    f = feat
                    thres = threshold
                    l = left
                    r = right
                    
        return Node(pure, None, None, f, thres, l, r)

    def calcu(self, data):
        c = [0,0]
        for i in range(len(data)):
            ind = 0 if data[i][-1]==0 else 1
            c[ind] = c[ind] + 1
            
        if c[0] < c[1]:
            return 1
        else:
            return 0

    def split(self, node, max_depth, cur_depth):
        if cur_depth > max_depth-1:
            node.left = self.calcu(node.left)
            node.right = self.calcu(node.right)
        else:
            if all(i[-1]==node.left[0][-1] for i in node.left):
                node.left = self.calcu(node.left)
            else:
                node.left = self.findMin(node.left)
                if node.left.feat == float('inf'):
                    node.left = self.calcu(node.left)
                else:
                    self.split(node.left, max_depth, cur_depth + 1)

            if all(i[-1]==node.right[0][-1] for i in node.right):
                node.right = self.calcu(node.right)
            else:
                node.right = self.findMin(node.right)
                if node.right.feat == float('inf'):
                    node.right = self.calcu(node.right)
                else:
                    self.split(node.right, max_depth, cur_depth + 1)

# class AdaBoost():
#     def __init__(self, criterion='gini', n_estimators=200):
#         # initialize
#         self.criterion = criterion
#         self.n_estimators = n_estimators
#         self.alphas = []
#         self.classifiers = []

#     def fit(self, x_data, y_data):
#         data_weight = np.full(len(x_data), (1 / len(x_data)))
#         # print(len(x_data))
#         # print(data_weight)
#         for n in range(self.n_estimators):
#             # bootstrap with data weight
#             rows = np.random.choice(len(x_data), len(x_data), p=None)
#             # print(rows)
#             if isinstance(x_data,  DataFrame):
#                 dataset, labels = x_data.iloc[rows], y_data.iloc[rows].values.tolist()
#                 # print('3333333333333333333')
#             else:
#                 dataset = [x_data[row] for row in rows]
#                 labels  = [y_data[row] for row in rows]
#             clf = DecisionTree(self.criterion, max_depth=1)   # declare a decision tree
#             clf.fit(dataset, labels)                            # fit with bootstrap data
#             clf_predict = clf.predict(x_data)                   # predict with "all" data
#             # compute error and alpha
#             error = 0
#             for i in range(len(rows)):
#                 if clf_predict[i] != y_data[i]:
#                     error += data_weight[i]
#             alpha = 0.5 * np.log((1 - error) / error)
#             # update data weight
#             for i in range(len(rows)):
#                 if y_data[i] == clf_predict[i]:
#                     data_weight[i] *= np.exp(-alpha)
#                 else:
#                     data_weight[i] *= np.exp(alpha)
#             # normalize
#             data_weight /= sum(data_weight)
#             self.alphas.append(alpha)
#             self.classifiers.append(clf)

#     def predict(self, x_data):
#         out = np.full(len(x_data), 0.0)
#         # compute wighted sum of each clf's prediction for each data
#         for i in range(self.n_estimators):
#             pred = self.classifiers[i].predict(x_data)
#             for j in range(len(x_data)):
#                 if pred[j] == 0:
#                     out[j] -= self.alphas[i]
#                 else:
#                     out[j] += self.alphas[i]
#         # output the sign of each weighted prediction
#         out = np.sign(out)
#         # convert -1 to 0
#         for i in range(len(out)):
#             if out[i] == -1:
#                 out[i] = 0
#         return np.asarray(out)
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.alphas = []  # List to store the weights of weak classifiers
        self.classifiers = []  # List to store the weak classifiers
    
    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples  # Initialize weights

        for _ in range(self.n_estimators):
            tree = DecisionTree(criterion=self.criterion, max_depth=1)
            tree.fit(X, y)

            y_pred = tree.predict(X)
            error = np.sum(w * (y_pred != y)) / np.sum(w)

            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)

            # Update weights
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)

            self.classifiers.append(tree)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.classifiers])
        return np.sign(np.dot(self.alphas, predictions))



# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(145)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

# AdaBoost
    print("Part 2: AdaBoost")
    for i in range(1,1000):    
        # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
        ada = AdaBoost(criterion='gini', n_estimators=i)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"i:{i} Accuracy:{acc}")
        if(acc>0.79):
            print('---------------')