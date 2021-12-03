# Some support code for the exercise that suggests to use CIFAR-10 dataset
# In case this helps you loading data once the dataset has been downloaded

import numpy as np
import matplotlib.pyplot as plt
import pickle
import platform
import matplotlib.pylab as plt

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

default_name="/ssd/datasets/cifar10/cifar-10-batches-py/data_batch_1"

def load_CIFAR_batch(filename=None):
    """ load single batch of cifar """
    if filename is None:
        filename = default_name
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols,3)
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        #X = X.reshape(10000,3072)
        #X = X.reshape((10000,)+input_shape)
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        Y = np.array(Y)
        return X, Y


if __name__ == "__main__":
    filename = default_name
    X,Y=load_CIFAR_batch(filename)
    print("X",X.shape)
    plt.imshow(X[1])
    plt.show(block=True)
