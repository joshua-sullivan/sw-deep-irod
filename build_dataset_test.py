import numpy as np
import os
import scipy.io as sio


X_train = np.load(os.path.normpath('datasets/tensor/train/X.npy'))
sio.savemat('X_train_fromNP.mat', {'X_train': X_train})
X_dev = np.load(os.path.normpath('datasets/tensor/dev/X.npy'))
sio.savemat('X_dev_fromNP.mat', {'X_dev': X_dev})
X_test = np.load(os.path.normpath('datasets/tensor/test/X.npy'))
sio.savemat('X_test_fromNP.mat', {'X_test': X_test})

Y_train = np.load(os.path.normpath('datasets/tensor/train/Y.npy'))
sio.savemat('Y_train_fromNP.mat', {'Y_train': Y_train})
Y_dev = np.load(os.path.normpath('datasets/tensor/dev/Y.npy'))
sio.savemat('Y_dev_fromNP.mat', {'Y_dev': Y_dev})
Y_test = np.load(os.path.normpath('datasets/tensor/test/Y.npy'))
sio.savemat('Y_test_fromNP.mat', {'Y_test': Y_test})

