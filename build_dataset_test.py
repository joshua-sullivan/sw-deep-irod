import numpy as np
import os
import scipy.io as sio


X_train = np.load(os.path.normpath('datasets/tensor/train/X.npy'))
print("X_train shape is: " + str(X_train.shape))
sio.savemat('X_train_fromNP.mat', {'X_train': X_train})
X_dev = np.load(os.path.normpath('datasets/tensor/dev/X.npy'))
print("X_dev shape is: " + str(X_dev.shape))
sio.savemat('X_dev_fromNP.mat', {'X_dev': X_dev})
X_test = np.load(os.path.normpath('datasets/tensor/test/X.npy'))
print("X_test shape is: " + str(X_test.shape))
sio.savemat('X_test_fromNP.mat', {'X_test': X_test})

Y_train = np.load(os.path.normpath('datasets/tensor/train/Y.npy'))
print("Y_train shape is: " + str(Y_train.shape))
sio.savemat('Y_train_fromNP.mat', {'Y_train': Y_train})
Y_dev = np.load(os.path.normpath('datasets/tensor/dev/Y.npy'))
print("Y_dev shape is: " + str(Y_dev.shape))
sio.savemat('Y_dev_fromNP.mat', {'Y_dev': Y_dev})
Y_test = np.load(os.path.normpath('datasets/tensor/test/Y.npy'))
print("Y_test shape is: " + str(Y_test.shape))
sio.savemat('Y_test_fromNP.mat', {'Y_test': Y_test})

