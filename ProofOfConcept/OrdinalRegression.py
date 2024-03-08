import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

def sig1(z): # sigma
    return(1/(1+np.exp(-z)))
def sig2(z): # sigma'(z)
    phat = sig1(z)
    return(phat*(1-phat))

class y2ord(): # Convert ordinal to 1, 2, ... K
    def __init__(self):
        self.di = {}
    def fit(self,y):
        self.uy = np.sort(np.unique(y))
        self.di = dict(zip(self.uy, np.arange(len(self.uy))+1))
    def transform(self,y):
        return(np.array([self.di[z] for z in y]))

def alpha2theta(alpha,K): # theta[t] = theta[t-1] + exp(alpha[t])
    return(np.cumsum(np.append(alpha[0], np.exp(alpha[1:]))))

def theta2alpha(theta,K): # alpha[t] = log(theta[t] - theta[t-1])
    return(np.append(theta[0],np.log(theta[1:] - theta[:-1])))

def alpha_beta_wrapper(alpha_beta, X, lb=20, ub=20):
    K = len(alpha_beta) + 1
    if X is not None:
        K -= X.shape[1]
        beta = alpha_beta[K - 1:]
    else:
        beta = np.array([0])
    alpha = alpha_beta[:K - 1]
    theta = alpha2theta(alpha, K)
    theta = np.append(np.append(theta[0] - lb, theta), theta[-1] + ub)
    return(alpha, theta, beta, K)

# Likelihood function
def nll_ordinal(alpha_beta, X, idx_y, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
    score = np.dot(X,beta)
    ll = 0
    for kk, idx in enumerate(idx_y):
        ll += sum(np.log(sig1(theta[kk+1]-score[idx])-sig1(theta[kk]-score[idx])))
    nll = -1*(ll / X.shape[0])
    return(nll)

# Gradient wrapper
def gll_ordinal(alpha_beta, X, idx_y, lb=20, ub=20):
    grad_alpha = gll_alpha(alpha_beta, X, idx_y)
    grad_X = gll_beta(alpha_beta, X, idx_y)
    return(np.append(grad_alpha,grad_X))

# gradient function for beta
def gll_beta(alpha_beta, X, idx_y, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
    score = np.dot(X, beta)
    grad_X = np.zeros(X.shape[1])
    for kk, idx in enumerate(idx_y):  # kk = 0; idx=idx_y[kk]
        den = sig1(theta[kk + 1] - score[idx]) - sig1(theta[kk] - score[idx])
        num = -sig2(theta[kk + 1] - score[idx]) + sig2(theta[kk] - score[idx])
        grad_X += np.dot(X[idx].T, num / den)
    grad_X = -1 * grad_X / X.shape[0]  # negative average of gradient
    return(grad_X)

# gradient function for theta=exp(alpha)
def gll_alpha(alpha_beta, X, idx_y, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
    score = np.dot(X, beta)
    grad_alpha = np.zeros(K - 1)
    for kk in range(K-1):
        idx_p, idx_n = idx_y[kk], idx_y[kk+1]
        den_p = sig1(theta[kk + 1] - score[idx_p]) - sig1(theta[kk] - score[idx_p])
        den_n = sig1(theta[kk + 2] - score[idx_n]) - sig1(theta[kk+1] - score[idx_n])
        num_p, num_n = sig2(theta[kk + 1] - score[idx_p]), sig2(theta[kk + 1] - score[idx_n])
        grad_alpha[kk] += sum(num_p/den_p) - sum(num_n/den_n)
    grad_alpha = -1* grad_alpha / X.shape[0]  # negative average of gradient
    grad_alpha *= np.append(1, np.exp(alpha[1:])) # apply chain rule
    return(grad_alpha)

# inference probabilities
def prob_ordinal(alpha_beta, X, lb=20, ub=20):
    alpha, theta, beta, K = alpha_beta_wrapper(alpha_beta, X, lb, ub)
    score = np.dot(X, beta)
    phat = (np.atleast_2d(theta) - np.atleast_2d(score).T)
    phat = sig1(phat[:, 1:]) - sig1(phat[:, :-1])
    return(phat)

# Wrapper for training/prediction
class ordinal_reg:
    def __init__(self,standardize=True):
        self.standardize = standardize
    def fit(self,data,lbls):
        self.p = data.shape[1]
        self.Xenc = StandardScaler().fit(data)
        self.yenc = y2ord()
        self.yenc.fit(y=lbls)
        ytil = self.yenc.transform(lbls)
        idx_y = [np.where(ytil == yy)[0] for yy in list(self.yenc.di.values())]
        self.K = len(idx_y)
        theta_init = np.array([(z + 1) / self.K for z in range(self.K - 1)])
        theta_init = np.log(theta_init / (1 - theta_init))
        alpha_init = theta2alpha(theta_init, self.K)
        param_init = np.append(alpha_init, np.repeat(0, self.p))
        self.alpha_beta = minimize(fun=nll_ordinal, x0=param_init, method='L-BFGS-B', jac=gll_ordinal,
                                   args=(self.Xenc.transform(data), idx_y)).x
    def predict(self,data):
        phat = prob_ordinal(self.alpha_beta,self.Xenc.transform(data))
        return(np.argmax(phat,axis=1)+1)
    def predict_proba(self,data):
        phat = prob_ordinal(self.alpha_beta,self.Xenc.transform(data))
        return(phat)
