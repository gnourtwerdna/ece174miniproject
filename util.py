import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from itertools import combinations

def load_data(path, train = True):
    '''
    Loads the MNIST data mat file.
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
    Returns dataset with data normalized from 0 to 1.
    '''
    return X.astype('float32') / 255.0

def generate_binary_dataset(images, labels, class1, class2):
    '''
    Returns dataset with two classes
    '''
    idx = generate_binary_index(labels, class1, class2)

    x = images[idx]
    y = labels[idx]
    
    y[y == class1] = 0
    y[y == class2] = 1
            
    return (x, y)

def generate_binary_index(labels, class1, class2):
    '''
    Returns indexes where images are labeled as class 1 and class 2.
    '''
    return np.where((labels == class1) | (labels == class2))[0]

def onehot_encode(y, num_classes):
    '''
    Returns one hot encoded labels.
    '''
    y_one_hot = []
    for i in range(len(y)):
        encoding = np.zeros(num_classes, dtype=int)
        encoding[y[i]] = 1
        y_one_hot.append(np.array(encoding))

    return np.stack(y_one_hot, axis=0)

def append_bias(X, random = False):
    """
    Returns dataset with bias appended. If random=false, will append a column of 1's. If random=true, will append a column of numbers from 0 to 1.
    """
    if random == False:
        bias = np.ones([X.shape[0], 1])
        return np.append(bias, X, axis = 1)
    if random == True:
        bias = np.random.random([X.shape[0], 1])
        return np.append(bias, X, axis = 1)

def generate_classifiers(num_classes):
    '''
    Returns N(N-1)/2 combinations of classes.
    '''
    classes = range(num_classes)
    return list(combinations(classes, 2))

def error(y_true, y_pred):
    '''
    Returns the error.
    '''
    return len(y_true) - np.sum(y_true == y_pred)

def error_rate(y_true, y_pred):
    '''
    Returns the error rate.
    '''
    return error(y_true, y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred, num_classes):
    '''
    Returns confusion matrix
    '''
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(y_true.shape[0]):
        confusion_matrix[int(y_true[i][0])][int(y_pred[i][0])] += 1
    return confusion_matrix

def plot_confusion_matrix(cm, title):
    '''
    Plots confusion matrix.
    '''
    figsize = (7 , 7)
    fig , ax = plt.subplots(figsize = figsize)
    ax.matshow(cm, cmap = plt.cm.Blues)

    for (i, j), value in np.ndenumerate(cm.astype(int)):
        ax.text(j, i, value, ha='center', va='center')

    ax.set(title = title, xlabel = 'Predicted Label', ylabel = 'True Label')

def identity(x):
    return x

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sinusoidal(x):
    return np.sin(x)

def ReLU(x):
    return np.maximum(x, 0)