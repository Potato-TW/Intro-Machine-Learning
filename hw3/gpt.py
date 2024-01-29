import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    if len(y) == 0:
        return 0
    p = np.sum(y) / len(y)
    return 1 - p**2 - (1 - p)**2

# This function computes the entropy of a label array.
def entropy(y):
    if len(y) == 0:
        return 0
    p = np.sum(y) / len(y)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Node class for decision tree
class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None, result=None):
        self.feature = feature        # Index of feature to split on
        self.threshold = threshold    # Threshold value for the feature
        self.value = value            # Value if the node is a leaf
        self.left = left              # Left child (subtree)
        self.right = right            # Right child (subtree)
        self.result = result          # Result if the node is a leaf (class label)

# The decision tree classifier class.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if len(y) == 0:
            return 0
        p = np.sum(y) / len(y)
        return 1 - p**2 - (1 - p)**2

    # Recursive function to build the decision tree
    def build_tree(self, X, y, depth):
        if depth == 0 or len(y) == 0 or np.all(y == y[0]):
            return Node(value=self.impurity(y), result=y[0] if len(y) > 0 else None)

        num_features = X.shape[1]
        min_impurity = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                left_impurity = self.impurity(y[left_indices])
                right_impurity = self.impurity(y[right_indices])

                weighted_impurity = (len(y[left_indices]) * left_impurity + len(y[right_indices]) * right_impurity) / len(y)

                if weighted_impurity < min_impurity:
                    min_impurity = weighted_impurity
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is not None and best_threshold is not None:
            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold

            left_subtree = self.build_tree(X[left_indices], y[left_indices], depth - 1)
            right_subtree = self.build_tree(X[right_indices], y[right_indices], depth - 1)

            return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

        return Node(value=self.impurity(y), result=np.argmax(np.bincount(y)))

    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, self.max_depth)
    
    # Recursive function to predict using the decision tree
    def predict_recursive(self, node, x):
        if node.result is not None:
            return node.result
        if node.feature is not None and node.threshold is not None:
            if x[node.feature] <= node.threshold:
                return self.predict_recursive(node.left, x)
            else:
                return self.predict_recursive(node.right, x)
        else:
            return node.value  # Handle cases where node is a leaf without a feature and threshold

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        return np.array([self.predict_recursive(self.tree, x) for x in X])
    
# The AdaBoost classifier class.
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
# You can only modify the value of the random seed and the arguments of your Adaboost class.
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

    np.random.seed(54)

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
    for i in range(1000,1001):
        # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
        ada = AdaBoost(criterion='gini', n_estimators=i)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        # print("Accuracy:", accuracy_score(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        
        print(f"i:{i} Accuracy:{acc}")
        if(acc>0.79):
            print(f'--------------')
