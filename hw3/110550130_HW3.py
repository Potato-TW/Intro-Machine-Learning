# You are not allowed to import any additional packages/libraries.
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
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class Node:
    def __init__(self, impurity, pred, depth, feat, threshold, left, right):
        self.impurity = impurity
        self.pred = pred
        self.depth = depth
        self.feat = feat
        self.threshold = threshold
        self.left = left
        self.right = right
        
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.criterion = criterion
        self.feature_importances = []     
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        data = X.values.tolist() if isinstance(X, DataFrame) else X
        self.root = self.findMin([list(data[i]) + [y[i]] for i in range(len(y))])
        self.split(self.root, self.max_depth, 1)   
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        feat = X.values.tolist() if isinstance(X, DataFrame) else X
        pred = []
        
        for data in feat:
            node = self.root
            while isinstance(node, Node):
                if data[node.feat] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            pred.append(node)
            
        return pred
    
    def findMin(self, data):
        pure = f = thres = float('inf')
        l = r = None
        for feat in range(len(data[1]) - 1):
            tmp = np.sort(np.unique([data[feat] for data in data]))[:-1]      
                 
            for threshold in tmp:
                left = []
                right = []
                for i in range(len(data)):
                    if data[i][feat] <= threshold:
                        left.append(data[i])
                    else:
                        right.append(data[i])
                        
                sp = self.impurity([row[-1] for row in left]) * len(left) + self.impurity([row[-1] for row in right]) * len(right)
                
                if sp / len(data) < pure:
                    pure = sp / len(data)
                    f = feat
                    thres = threshold
                    l = left
                    r = right
                    
        return Node(pure, None, None, f, thres, l, r)


    def calcu(self, data):
        cnt = [0,0]
        for i in range(len(data)):
            ind = 0 if data[i][-1]==0 else 1
            cnt[ind] = cnt[ind] + 1
            
        return cnt[0] < cnt[1]

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
    
    def traval(self, node):
        if isinstance(node, Node) == 0:
            return
            # print(node.feat)
            # self.traval(node.left)
            # self.traval(node.right)

        self.feature_importances[node.feat]+=1
            
        self.traval(node.left)
        self.traval(node.right)
        
    
    def compute_feature_importance(self, node):
        # travel the tree to compute feature importance
        # if isinstance(node, dict):
        self.feature_importances[node.feat] += 1
        self.compute_feature_importance(node.left)
        self.compute_feature_importance(node.right)
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, X_train):
        dic = ['age', 'sex','cp','fbs','thalach','thal']

        self.feature_importances=[0]*len(dic)
        self.traval(self.root)

        plt.barh(dic, self.feature_importances)
        plt.tick_params(axis='x', length=0)
        plt.tick_params(axis='y', length=0)
        # ax = plt.gca()
        plt.title('Feature Importance')
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.alphas = []
        self.classifiers = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        w = np.ones(len(X))/len(X)

        for _ in range(self.n_estimators):
            rows = np.random.choice(len(X), len(X), p=w)            

            feat = [X[row] for row in rows]
            label = [y[row] for row in rows]
            
            clf = DecisionTree(self.criterion, max_depth=1)
            clf.fit(feat, label)
            y_pred = clf.predict(X)

            error = np.sum(w * (y_pred != y)) / np.sum(w)
            alpha = 0.5 * np.log((1 - error) / error)

            w *= np.exp(alpha * (y_pred != y))
            w /= np.sum(w)

            self.alphas.append(alpha)
            self.classifiers.append(clf)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        out = np.zeros(len(X))

        for i in range(self.n_estimators):
            out += self.alphas[i] * np.array([-1 if j == 0 else 1 for j in self.classifiers[i].predict(X)])

        out = np.sign(out)
        out[out == -1] = 0
        return out

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
    np.random.seed(500)

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
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=10)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
