import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt


def F(w, X, Y, lam):
    hinge = np.maximum(0, 1 - Y * (w @ X))
    return (lam / 2) * np.linalg.norm(w) ** 2 + np.mean(hinge)

def accuracy(w, X, y):
    preds = np.sign(w@X)
    correct = preds == y.flatten()
    return np.mean(correct)



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




lam = 1
def SVM(X, y, T, lam, eta_func):
    w = np.zeros(X.shape[0])
    loss_list = []

    for t in range(1, T + 1):
        eta = eta_func(t, lam)
        index = np.random.randint(X.shape[1])
        xi = X[:, index]
        yi = y[0, index]

        if (xi * yi) @ w < 1:
            w = w - eta * (lam * w - xi * yi)
        else:
            w = w - eta * (lam * w)

        loss_list.append(F(w, X, y, lam))

    return w, loss_list

# _, loss = SVM(Xtrain, ytrain, 20000, lam, eta_func=lambda t, lam: 1 / (t*lam))
#
# _, loss = SVM(Xtrain, ytrain, 20000, lam, eta_func=lambda t, lam: 0.01)



lambdas = [1e-6, 1e-3, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 10000]

train_acc = []
test_acc = []
w_norms = []

for lam in lambdas:
    w, _ = SVM(Xtrain, ytrain, 10000, lam, eta_func=lambda t, lam=lam: 2 / (t * lam))
    train_acc.append(accuracy(w, Xtrain, ytrain))
    test_acc.append(accuracy(w, Xtest, ytest))
    w_norms.append(np.linalg.norm(w)**2)

# Plot accuracy vs log10(lambda)
log_lambdas = np.log10(lambdas)

print("lambda\t\t||w||^2")
for lam, norm in zip(lambdas, w_norms):
    print(f"{lam:<10}\t{norm:.10f}")




