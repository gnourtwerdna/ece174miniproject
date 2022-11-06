import numpy as np


def binary(X):
    '''
    Binary classifier that calculates the sign of the input.

    '''
    return np.sign(X)

def one_vs_rest(X):
    '''
    Multiclass classifier that calculates the argmax of the input.

    '''
    return np.argmax(X, axis = 1).reshape((len(X), 1))


class LinearRegression:
    def __init__(self, method):
        self.method = method
        self.model = None

        if self.method == 'binary':
            self.classifier = binary
        
        if self.method == 'ovr':
            self.classifier = one_vs_rest

    def fit(self, X, y):
        '''
        Least squares estimation.
        '''
        inv = np.linalg.pinv(X)
        self.model = inv.dot(y)

    def predict(self, X):
        return self.classifier(np.dot(X, self.model))

    def accuracy_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)