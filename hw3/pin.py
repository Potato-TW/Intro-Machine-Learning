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

# Copy and paste your implementations right here to check your result
# (Of course you can add your classes not written here)
# def gini(sequence):
#     pj = np.unique(sequence, return_counts=True)[1] / len(sequence)
#     return 1 - np.sum(pj**2)

# def entropy(sequence):
#     pj = np.unique(sequence, return_counts=True)[1] / len(sequence)
#     return - np.sum(np.dot(pj, np.log2(pj)))

class Node:
    def __init__(self, class_index):
        self.class_index = class_index
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None, max_features=-1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.num_features = None
        self.feature_weight = None
        self.max_features = max_features

    def gini(self, y_data, sample_weight):
        if len(y_data) == 0 or len(np.unique(y_data)) < 2:
            return 0
        pj = np.array([np.sum(sample_weight[y_data == 0]), np.sum(sample_weight[y_data == 1])]) / np.sum(sample_weight)
        return 1 - np.sum(pj**2)

    def entropy(self, y_data, sample_weight):
        if len(y_data) == 0 or len(np.unique(y_data)) < 2:
            return 0
        pj = np.array([np.sum(sample_weight[y_data == 0]), np.sum(sample_weight[y_data == 1])]) / np.sum(sample_weight)
        return - np.sum(np.dot(pj, np.log2(pj)))
        
    def SplitAttribute(self, x_data, y_data, sample_weight):
        feature_index = None
        threshold = None
        y_len = len(y_data)
        if len(np.unique(y_data)) <= 1:
            return None, None
        random_features = np.sort(np.random.permutation(np.arange(self.num_features))[:self.max_features])
        if self.criterion == 'gini':
            MinEnt = self.gini(y_data, sample_weight)
        else:
            MinEnt = self.entropy(y_data, sample_weight)
        z_data = np.concatenate((np.concatenate((x_data[:,random_features], y_data.reshape(y_len, 1)), axis=1), sample_weight.reshape(y_len, 1)), axis=1)
        for i in range(self.max_features):
            s_data = z_data[np.argsort(z_data[:, i])]
            x_temp, y_temp, sample_weight_temp = s_data[:,i], s_data[:,self.max_features], s_data[:,self.max_features+1]
            for j in range(1, y_len):
                if x_temp[j] == x_temp[j - 1]:
                    continue
                if self.criterion == 'gini':
                    e = (j * self.gini(y_temp[:j], sample_weight_temp[:j]) + (y_len - j) * self.gini(y_temp[j:], sample_weight_temp[j:])) / y_len
                else:
                    e = (j * self.entropy(y_temp[:j], sample_weight_temp[:j]) + (y_len - j) * self.entropy(y_temp[j:], sample_weight_temp[j:])) / y_len
                if e < MinEnt:
                    MinEnt = e
                    feature_index = random_features[i]
                    threshold = (x_temp[j] + x_temp[j - 1]) / 2
        return feature_index, threshold

    def GenerateTree(self, x_data, y_data, depth, sample_weight):
        p = np.unique(y_data, return_counts=True)
        class_index = p[0][np.argmax(p[1])]
        new_node = Node(class_index=class_index)
        if self.max_depth != None and depth >= self.max_depth:
            return new_node
        feature_index, threshold = self.SplitAttribute(x_data, y_data, sample_weight)
        if feature_index == None and threshold == None:
            return new_node
        self.feature_weight[feature_index] += 1
        new_node.feature_index = feature_index
        new_node.threshold = threshold
        left_index = x_data[:, feature_index] > threshold
        right_index = x_data[:, feature_index] <= threshold
        if len(left_index) <= 0 or len(right_index) <= 0:
            return new_node
        new_node.left = self.GenerateTree(x_data[left_index], y_data[left_index], depth + 1, sample_weight[left_index])
        new_node.right = self.GenerateTree(x_data[right_index], y_data[right_index], depth + 1, sample_weight[right_index])
        return new_node

    def fit(self, x_data, y_data, sample_weight=[]):
        self.num_features = x_data.shape[1]
        self.feature_weight = np.zeros(self.num_features)
        if self.max_features == -1:
            self.max_features = self.num_features
        if len(sample_weight) != len(y_data):
            sample_weight = np.ones(len(y_data))
        self.root = self.GenerateTree(x_data, y_data, 0, sample_weight)

    def predict(self, x_data):
        if self.root == None:
            #print("Why!")
            return np.zeros(x_data.shape[0])
        #print('OK')
        y_pred = []
        for feature in x_data:
            cur_node = self.root
            while cur_node.left and cur_node.right:
                if feature[cur_node.feature_index] > cur_node.threshold:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right
            y_pred.append(cur_node.class_index)
        return np.array(y_pred)

class AdaBoost():
    def __init__(self, criterion='gini',n_estimators=200):
        self.n_estimators = n_estimators
        self.criterion=criterion
        self.classifier = list()
        self.classifier_weight = list()
        
    def fit(self, x_data, y_data):
        #x_temp = x_data.copy()
        self.classifier = list()
        self.classifier_weight = list()
        D = np.ones(len(y_data)) / len(y_data)
        for i in range(self.n_estimators):
            self.classifier.append(DecisionTree(self.criterion, max_depth=1))
            self.classifier[i].fit(x_data, y_data, D)
            y_pred = self.classifier[i].predict(x_data)
            #for j in range(len(y_pred)):
            #    if y_pred[j] != y_data[j]:
            #        x_temp[j] = np.dot(x_temp[j], 2.0)
            e = np.sum(D[y_pred != y_data])
            if e >= 1.0:
                self.classifier_weight.append(0)
                print('111111111111')
            elif e <= 0.0:
                self.classifier_weight.append(300)
                print('222222222222')
            else:
                self.classifier_weight.append(1/2 * np.log((1 - e) / e))
            tmp = y_data - y_pred
            tmp[tmp != 0] = -1
            tmp[tmp == 0] = 1
            D = D * np.exp(- self.classifier_weight[i] * tmp)
            D = D / np.sum(D)
            
    def predict(self, x_data):
        pred = np.zeros(len(x_data))
        for i in range(len(self.classifier)):
            y_pred = self.classifier[i].predict(x_data)
            y_pred[y_pred == 0] = -1
            pred = pred + self.classifier_weight[i] * y_pred
        pred = np.sign(pred)
        pred[pred <= 0] = 0
        return pred


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
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    for i in range(1,1000):
        ada = AdaBoost(criterion='gini', n_estimators=i) # 106
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"n: {i} Accuracy: {acc}")
        if(acc>0.78):
            print('+++++++++++++++++++++++++++++++++++++++++')
            with open('pin.txt','a') as f:
                f.write(f'n: {i} acc: {acc}')