import numpy as np
import util

def binary(X):
    '''
    Returns binary output.
    '''
    return np.sign(X)

def one_vs_all(X):
    '''
    Returns one vs. all output.
    '''
    return np.argmax(X, axis = 1).reshape((len(X), 1))

class BinaryClassifier:
    def __init__(self, class1, class2):
        self.class1 = class1
        self.class2 = class2
        self.model = None
        self.classifier = binary

    def fit(self, X, y):
        '''
        Least squares estimation.
        '''
        inv = np.linalg.pinv(X)
        self.model = inv.dot(y)

    def predict(self, X):
        sign = self.classifier(np.dot(X, self.model))
        sign[sign == -1] = 0
        sign[sign == 1] = 1
        return sign


class OneVsAllClassifier:
    def __init__(self):
        self.model = None
        self.classifier = one_vs_all

    def fit(self, X, y):
        '''
        Least squares estimation.
        '''
        inv = np.linalg.pinv(X)
        self.model = inv.dot(y)

    def predict(self, X):
        return self.classifier(np.dot(X, self.model))

def OneVsOneClassifier(X_train, y_train, X_test, y_test):
    '''
    One vs. one classifier that returns predicted labels for both train and test data.
    '''
    # Voting system
    votes_train = np.zeros((y_train.shape[0], 10))
    votes_test = np.zeros((y_test.shape[0], 10))
    classifiers = util.generate_classifiers(10)

    for classifier in classifiers:
        lr = BinaryClassifier(classifier[0], classifier[1])
        X_train_temp, y_train_temp = util.generate_binary_dataset(X_train, y_train, classifier[0], classifier[1])
        X_test_temp, y_test_temp = util.generate_binary_dataset(X_test, y_test, classifier[0], classifier[1])
        idxs_train = util.generate_binary_index(y_train, classifier[0], classifier[1])
        idxs_test = util.generate_binary_index(y_test, classifier[0], classifier[1])
        lr.fit(X_train_temp, y_train_temp)
        y_pred_train = lr.predict(X_train_temp)
        y_pred_test = lr.predict(X_test_temp)

        for pred, idx in zip(y_pred_train, idxs_train):
            if pred == 1:
                votes_train[idx][classifier[0]] += 1
            if pred == -1:
                votes_train[idx][classifier[1]] += 1

        for pred, idx in zip(y_pred_test, idxs_test):
            if pred == 1:
                votes_test[idx][classifier[0]] += 1
            if pred == -1:
                votes_test[idx][classifier[1]] += 1

    # Predictions
    y_pred_ovo_train = np.zeros((y_train.shape))
    y_pred_ovo_test = np.zeros((y_test.shape))

    # Tiebreaking: picks random index from the indices with most votes
    for i in range(votes_train.shape[0]):
        max_idx = np.where(votes_train[i] == votes_train[i].max())[0]
        y_pred_ovo_train[i] = np.random.choice(max_idx)

    for i in range(votes_test.shape[0]):
        max_idx = np.where(votes_test[i] == votes_test[i].max())[0]
        y_pred_ovo_test[i] = np.random.choice(max_idx)

    return y_pred_ovo_train, y_pred_ovo_test