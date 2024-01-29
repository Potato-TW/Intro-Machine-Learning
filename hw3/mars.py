# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def gini(sequence):
    c1, c2, l = 0, 0, len(sequence)
    for i in range(l):
        if sequence[i] == 1:    # count class 1
            c1 += 1
        else:                   # count class 2
            c2 += 1
    p1, p2 = c1 / l, c2 / l     # count prob. of each class
    return 1 - p1 ** 2 - p2 ** 2

def entropy(sequence):
    c1, c2, l = 0, 0, len(sequence)
    for i in range(l):
        if sequence[i] == 1:    # count class 1
            c1 += 1
        else:                   # count class 2
            c2 += 1
    p1, p2 = c1 / l, c2 / l     # count prob. of each class
    if not p1 or not p2:        # all in same class
        return 0
    else:
        return -p1 * np.log2(p1) - p2 * np.log2(p2)

class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        # initialzation
        self.max_depth = max_depth
        self.root = None
        self.criterion = None
        self.feature_importances = [0] * 20

        if criterion == 'gini':
            self.criterion = gini
        else:
            self.criterion = entropy

    def fit(self, x_data, y_data, rf=None): # this is the train function
        self.rf = rf
        # deal with different input type
        if isinstance(x_data,  DataFrame):
            dataset = x_data.values.tolist()
        else:
            dataset = x_data
        # merge x_data and y_data
        train_dataset = [list(dataset[i]) + [y_data[i]] for i in range(len(y_data))]
        # build the tree
        self.root = self.best_split(train_dataset)
        self.split_tree(self.root, self.max_depth, 1)

    def predict(self, x_data):
        # deal with different input type
        if isinstance(x_data,  DataFrame):
            dataset = x_data.values.tolist()
        else:
            dataset = x_data
        # return prediction of each data
        ret = [self.predict_data(dataset[i], self.root) for i in range(len(x_data))]
        return ret

    def predict_data(self, data, node):
        # split predict data with attribute and threshold of each node
        if data[node['attribute']] <= node['threshold']:
            if isinstance(node['left'], dict):
                return self.predict_data(data, node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_data(data, node['right'])
            else:
                return node['right']

    def test_split(self, attribute, threshold, dataset):
        # split dataset base on given attribute and threshold
        left, right = [], []
        for i in range(len(dataset)):
            if dataset[i][attribute] <= threshold:
                left.append(dataset[i])
            else:
                right.append(dataset[i])
        return left, right

    def best_split(self, dataset):
        b_purity, b_attribute, b_threshold, b_left, b_right = 10, 999, 9000, None, None
        # generate random vector of random forest
        if self.rf != None:
            cols = np.random.choice(len(dataset[0]) - 1, int(self.rf), replace=False)
        for attribute in range(len(dataset[0]) - 1):
            # skip the attribute not in random vector of random forest
            if self.rf != None and attribute not in cols:
                continue
            # find unique threshold for the attribute
            thresholds = np.array([data[attribute] for data in dataset])
            thresholds = np.unique(thresholds)
            thresholds = np.sort(thresholds)
            # pop maximum to prevent left or right = None
            if len(thresholds) == 1:
                continue
            thresholds = thresholds[:-1]
            # iterate through thresholds to find best split
            for threshold in thresholds:
                left, right = self.test_split(attribute, threshold, dataset)
                split_purity = (self.criterion([row[-1] for row in left]) * len(left) + self.criterion([row[-1] for row in right]) * len(right)) / len(dataset)
                if split_purity < b_purity:
                    b_purity, b_attribute, b_threshold, b_left, b_right = split_purity, attribute, threshold, left, right
        return {'attribute':b_attribute, 'threshold':b_threshold, 'left':b_left, 'right':b_right}

    def leaf_pred(self, dataset):
        # output the most class in dataset
        c1, c2 = 0, 0
        for i in range(len(dataset)):
            if dataset[i][-1] == 0:
                c1 += 1
            else:
                c2 += 1
        if c1 > c2:
            return 0
        else:
            return 1

    def check_same_class(self, dataset):
        # check whether all data in the dataset are in the same class
        c = dataset[0][-1]
        for i in range(len(dataset)):
            if dataset[i][-1] != c:
                return False
        return True

    def split_tree(self, node, max_depth, cur_depth):
        # recursivly split the tree
        left, right = node['left'], node['right']
        del(node['left'])
        del(node['right'])
        # left or right = None -> predict
        if not left or not right:
            node['left'] = node['right'] = self.leaf_pred(left + right)
            return 
        # reach max depth -> predict
        if max_depth != None and cur_depth >= max_depth:
            node['left'], node['right'] = self.leaf_pred(left), self.leaf_pred(right)
            return
        
        if self.check_same_class(left):             # in the same class -> predict
            node['left'] = self.leaf_pred(left)
        else:
            node['left'] = self.best_split(left)    
            if node['left']['attribute'] == 999:    # if can not split anymore -> predict
                node['left'] = self.leaf_pred(left)
            else:                                   # keep split
                self.split_tree(node['left'], max_depth, cur_depth + 1)
        
        if self.check_same_class(right):            # in the same class -> predict
            node['right'] = self.leaf_pred(right)
        else:
            node['right'] = self.best_split(right)
            if node['right']['attribute'] == 999:   # if can not split anymore -> predict
                node['right'] = self.leaf_pred(right)
            else:                                   # keep split
                self.split_tree(node['right'], max_depth, cur_depth + 1)
            
    def print_tree(self, node, depth=0):
        # recursivly print the tree
        if isinstance(node, dict):
            print('%s[Attribute %d < %.3f]' % ((depth*'\t', (node['attribute']), node['threshold'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*'\t', node)))

    def compute_feature_importance(self, node):
        # travel the tree to compute feature importance
        if isinstance(node, dict):
            self.feature_importances[node['attribute']] += 1
            self.compute_feature_importance(node['left'])
            self.compute_feature_importance(node['right'])


class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        # initialize
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.alphas = []
        self.clfs = []

    def fit(self, x_data, y_data):
        data_weight = np.full(len(x_data), (1 / len(x_data)))
        ns, nf = x_data.shape
        # data_weight = np.ones(ns)/ns
        for n in range(self.n_estimators):
            # bootstrap with data weight
            # rows = np.random.choice(len(x_data), len(x_data), p=data_weight)
            rows = np.load('rows.npy')
            # print(rows)
            if isinstance(x_data,  DataFrame):
                dataset, labels = x_data.iloc[rows], y_data.iloc[rows].values.tolist()
            else:
                dataset = [x_data[row] for row in rows]
                labels  = [y_data[row] for row in rows]
            clf = DecisionTree(self.criterion, max_depth=1)   # declare a decision tree
            clf.fit(dataset, labels)                            # fit with bootstrap data
            clf_predict = clf.predict(x_data)                   # predict with "all" data
            # compute error and alpha
            # print(dataset)
            # print(clf_predict)
            error = 0
            for i in range(len(rows)):               
                # print(f'i:{i}, pred:{clf_predict[i]}, y:{y_data[i]}, err:{error}')
                if clf_predict[i] != y_data[i]:

                    error += data_weight[i]
                    
            # print(error)
            
            alpha = 0.5 * np.log((1 - error) / error)
            # update data weight
            for i in range(len(rows)):
                if y_data[i] == clf_predict[i]:
                    data_weight[i] *= np.exp(-alpha)
                else:
                    data_weight[i] *= np.exp(alpha)
            # normalize
            data_weight /= sum(data_weight)
            print(data_weight)
            self.alphas.append(alpha)
            self.clfs.append(clf)

    def predict(self, x_data):
        out = np.full(len(x_data), 0.0)
        # compute wighted sum of each clf's prediction for each data
        for i in range(self.n_estimators):
            pred = self.clfs[i].predict(x_data)
            for j in range(len(x_data)):
                if pred[j] == 0:
                    out[j] -= self.alphas[i]
                else:
                    out[j] += self.alphas[i]
                    
        
        # output the sign of each weighted prediction
        out = np.sign(out)
        # convert -1 to 0
        for i in range(len(out)):
            if out[i] == -1:
                print('5555555555555')
                out[i] = 0
        
        # print(self.alphas)
        return np.asarray(out)


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
    for i in range(5,6):
        ada = AdaBoost(criterion='gini', n_estimators=i)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))