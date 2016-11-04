import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def linearKernel(x1, x2):
    return np.dot(x1, x2)

def polynomialKernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussianKernel(x, y, sigma=100.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=gaussianKernel, C=0.1):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        nSamples, nFeatures = X.shape
        # Gram matrix
        K = np.zeros((nSamples, nSamples))
        for i in range(nSamples):
            for j in range(nSamples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(nSamples) * -1)
        A = cvxopt.matrix(y, (1,nSamples), 'd')
        #A = np.reshape((y.T), (1,nSamples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(nSamples) * -1))
            h = cvxopt.matrix(np.zeros(nSamples))
        else:
            tmp1 = np.diag(np.ones(nSamples) * -1)
            tmp2 = np.identity(nSamples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(nSamples)
            tmp2 = np.ones(nSamples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5   # sv -> Support Vectors
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.svY = y[sv]
        print "%d support vectors out of %d points" % (len(self.a), nSamples)

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.svY[n]
            self.b -= np.sum(self.a * self.svY * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linearKernel:
            self.w = np.zeros(nFeatures)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.svY[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            yPredict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, svY, sv in zip(self.a, self.svY, self.sv):
                    s += a * svY * self.kernel(X[i], sv)
                yPredict[i] = s
            return yPredict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
