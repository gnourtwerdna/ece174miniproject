import scipy.io as sio
import numpy as np
from scipy import optimize
from itertools import combinations

def load_data(path, train = True):
    '''
    Loads the MNIST data file.

    Parameters
    ----------
    path : string
        path to the data

    type : string
        indicates what type of data to load, 'train' or 'test'

    Returns
    -------
    tuple containing dataset in (data, label) format

    '''
    mat_contents = sio.loadmat(path)

    X_train = mat_contents['trainX']
    y_train = mat_contents['trainY']

    X_test = mat_contents['testX']
    y_test = mat_contents['testY']
    
    if train == True:
        return (X_train, y_train.T)

    else:
        return (X_test, y_test.T)

def normalize(X):
    '''
    Normalizes the data to 0-1.

    Parameters
    ----------
    X : N x d array

    Returns
    -------
    normalized input

    '''
    return X.astype('float32') / 255.0

def generate_binary_dataset(images, labels, class1, class2):
    '''
    Generates dataset with two classes

    Parameters
    ----------
    images : array
        N x d array containing images
    labels: array
        d x 1 array containging labels

    class : int
        input desired classes

    Returns
    -------
    tuple containing dataset in (data, label) format

    '''
    idx = generate_binary_index(images, labels, class1, class2)

    x = images[idx]
    y = labels[idx]
    
    y[y == class1] = 0
    y[y == class2] = 1
            
    return (x, y)

def generate_binary_index(images, labels, class1, class2):
    '''
    Generates indexes where images are labeled as class 1 and class 2.

    Parameters
    ----------
    images : array
        N x d array containing images
    labels: array
        d x 1 array containging labels

    class : int
        input desired classes

    Returns
    -------
    tuple containing dataset in (data, label) format

    '''
    return np.where((labels == class1) | (labels == class2))[0]

def onehot_encode(y, num_classes):
    '''
    Does one hot encoding on y.

    Parameters
    ----------
    y : array
        d x 1 array containging labels
    num_classes: int
        number of classes in labels

    Returns
    -------
    one hot encoded array

    '''
    y_one_hot = []
    for i in range(len(y)):
        encoding = np.zeros(num_classes, dtype=int)
        encoding[y[i]] = 1
        y_one_hot.append(np.array(encoding))

    return np.stack(y_one_hot, axis=0)

def append_bias(X):
    """
    Append bias term for dataset. Essentially adds a column of 1's.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)
    Returns
    -------
        2d numpy array with shape (N,(d+1))
    """
    bias = np.ones([X.shape[0], 1])
    return np.append(bias, X, axis = 1)

def generate_classifiers(num_classes):
    '''
    Generates N(N-1)/2 combinations of classes.
    '''
    classes = range(num_classes)
    return list(combinations(classes, 2))

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def error(y_true, y_pred):
    return 1 - np.sum(y_true == y_pred)

def error_rate(y_true, y_pred):
    return error(y_true, y_pred) / len(y_true)