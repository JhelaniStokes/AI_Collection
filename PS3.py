import scipy.io as sio
import numpy as np

def F(w, X, Y, lam):
    hinge = np.max(0, 1 - Y*(w@X))
    return ((lam/2)*(np.linalg.norm(w)**2))+np.mean(hinge)



mnist = sio.loadmat(r'C:\Data\mnist-original.mat')
data = mnist['data']
labels = mnist['label']
indices_0 = np.where(labels == 0)[1][:5000]
indices_1 = np.where(labels == 1)[1][:5000]
data0 = data[:, indices_0]
data1 = data[:, indices_1]
print(data0.shape)
print(data1.shape)

Xtrain = np.hstack((data0, data1))
norms = np.linalg.norm(Xtrain, axis=0, keepdims=True)
norms[norms==0] = 1
Xtrain = Xtrain/norms

ytrain = np.hstack((-1*np.ones(data0.shape[1], dtype=int), +1*np.ones(data1.shape[1], dtype=int))).reshape(1, -1)

perm = np.random.permutation(Xtrain.shape[1])
Xtrain = Xtrain[:, perm]
ytrain = ytrain[:, perm]

indices_0 = np.where(labels == 0)[1][5000:5500]
indices_1 = np.where(labels == 1)[1][5000:5500]
data0 = data[:, indices_0]
data1 = data[:, indices_1]

Xtest = np.hstack((data0, data1))
norms = np.linalg.norm(Xtest, axis=0, keepdims=True)
norms[norms==0] = 1
Xtest = Xtest/norms

ytest = np.hstack((-1*np.ones(data0.shape[1], dtype=int), +1*np.ones(data1.shape[1], dtype=int))).reshape(1, -1)




