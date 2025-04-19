import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



mnist = sio.loadmat(r'C:\Data\mnist-original.mat')





# data = mnist['data']
# labels = mnist['label']
# indices_0 = np.where(labels == 0)[1][:1000]
# data0 = data[:, indices_0]
# xPrime = data0
# print(xPrime.shape)
# for x in range(9):
#     indeces = np.where(labels == x+1)[1][:1000]
#     datax = data[:,indeces]
#
#     xPrime = np.concatenate((xPrime, datax), axis=1)
#
#
# X = xPrime-np.mean(xPrime, axis=0)

# Xcov = np.dot(X, X.T)
#
# pcaX = PCA()
#
# pcaX.fit(Xcov)
# values = pcaX.explained_variance_
# vectors = pcaX.components_

#np.savez('data_demeaned.npz', X=X)

data = np.load('pca_vectors.npz')
vectors = data['vectors']

data = np.load('data_demeaned.npz')
X = data['X']

# weight_vectors = np.zeros((784, 10))
#
# a = np.array([700, 825, 261, 634, 879, 757, 3, 307, 623, 797, 34, 682, 55, 113, 657, 603, 91, 654,
#  114, 773, 545, 403, 633, 194, 529, 714, 474, 701, 441, 177, 314, 999, 76, 395, 910, 61,
#  840, 543, 676, 531, 13, 41, 512, 476, 420, 627, 574, 86, 906, 860, 120, 738, 38, 130,
#  308, 940, 721, 844, 358, 671, 455, 866, 921, 926, 598, 202, 581, 453, 24, 755, 904, 673,
#  272, 245, 284, 900, 857, 792, 315, 991, 65, 927, 17, 77, 650, 357, 482, 388, 609, 193,
#  326, 518, 605, 961, 303, 817, 376, 632, 365, 928])
#
# for class_idx in range(10):
#     class_indices = a + (class_idx * 1000)
#     mean_X = X[:, class_indices].mean(axis=1)
#     weight_vectors[:, class_idx] = np.dot(vectors, mean_X)
# print(weight_vectors.shape)
# np.savez('weights', weights=weight_vectors)