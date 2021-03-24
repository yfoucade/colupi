import scipy
import scipy.stats
import sklearn.cluster
import generic_collaborator

import numpy as np
import pandas as pd

from numba import njit
from copy import deepcopy
from abc import abstractmethod


class Gmm:
    """
    A class for Gaussian mixture models.

    Attributes:
        K (int): number of components.
        tol (float): convergence threshold. EM stops when variation of the
            parameters between two steps is below this threshold.
        n_iter (int): number of iterations to convergence.
        X (np.array): n*D array of data.

        n (int): number of observations.
        D (int): dimensionality of one data point.
        prop (np.array): 1*K array with the probabilities of each component.
        mu    (np.array): K*D array: the K means of the densities.
        sigma (np.array): K*D*D array: K covariance matrices of densities.
        tau   (np.array): n*K responsability matrix.
        pairwise_entropy (np.array): n*n array of pairwise cross-entropy.
        average_entropy (np.array): = (1/n) * trace(pairwise_entropy).
        uncertainty (np.array): 1 minus the highest responsability,
            for each observation.
    """

    @njit
    def __init__(self, K=3, tol=1e-2, max_iter=0):
        """
        Parameters:
            K (int): Number of clusters
            tol (float): when to stop the algorithm
        """

        self.K, self.tol, self.max_iter = K, tol, max_iter

        # number of iterations
        self.n_iter = None

        # the design matrix and its shape
        self.X, self.n, self.D = None, None, None

        # parameters of the model
        self.prop, self.mu, self.sigma, self.tau = 4*[None]

        # some information-theoretic variables
        self.pairwise_entropy, self.average_entropy, self.uncertainty = 3*[None]

    @njit
    def fit(X, resp=None):
        """
        Infers on the data X

        Args:
            X (np.ndarray): 2-dimensional array of shape n*D (number of observations * number of variables).
            resp (np.ndarray): 2-dimensional array of shape n*K: responsibility matrix.

        Returns:
            sets the following attributes:
            prop  (np.ndarray): K-dimensional vector, the proportions of the mixture model.
            mu    (np.ndarray): K*D array: the K means of the densities.
            sigma (np.ndarray): K*D*D array: K covariance matrices of densities.
            tau   (np.ndarray): n*K responsability matrix.
        """

        self.X = X.copy()

        # if X is a line vector, transform it to a column vector (we need a 2D array)
        if self.X.ndim==1:
            self.X = self.X.reshape(-1, 1)

        self.n, self.D = self.X.shape

        # initialization
        if resp is None:
            new_tau = self.warm_start(self.X, self.K)
        else:
            # one could add ```assert is_partition_matrix(resp)```
            new_tau = resp.copy()
        new_prop, new_mu, new_sigma = self.m_step(X, new_tau)

        delta, self.n_iter = 1, 0

        while True:
            self.n_iter += 1
            # print("iter number {}".format(self.n_iter))
            # save old values
            old_prop, old_mu, old_sigma, old_tau = deepcopy(new_prop), deepcopy(new_mu), deepcopy(new_sigma), deepcopy(
                new_tau)

            # compute new values: e-step followed by m-step
            new_tau = self.e_step(X, old_prop, old_mu, old_sigma)
            new_prop, new_mu, new_sigma = self.m_step(X, new_tau)

            # compute variation of the parameters
            if self.n_iter>2:
                delta = np.linalg.norm(new_mu-old_mu)+np.linalg.norm(new_sigma-old_sigma)

            if delta<=self.tol or (self.max_iter and self.n_iter==self.max_iter):
                break

        self.prop, self.mu, self.sigma, self.tau = new_prop.copy(), new_mu.copy(), new_sigma.copy(), new_tau.copy()

        return self.prop, self.mu, self.sigma, self.tau

    @njit
    @staticmethod
    def get_density(value, mean=None, cov=None):
        try:
            return scipy.stats.multivariate_normal.pdf(value, mean=mean, cov=cov, allow_singular=True)
        except:
            print("Trying to get density for value {}, mean {}, and cov {}".format(value, mean, cov))

    @njit
    @staticmethod
    def warm_start(X, K=3, epsilon=1e-3):
        """
        Initializes the parameters prop, mu, and sigma using sklearn.cluster.k_means()

        Args:
            X (np.ndarray): 2-dimensional array of shape n*D (number of observations * number of variables)
            K (int): Number of clusters.
            epsilon (float): small pertubation to cluster assignements (avoids clusters with one object).

        Return:
            resp (np.ndarray): responsibility matrix.
        """

        centroids, pred_labels, inertia, n_iter = sklearn.cluster.k_means(X, K, return_n_iter=True)

        resp = pd.get_dummies(pred_labels).values+epsilon
        resp /= 1+K*epsilon

        return resp

    @njit
    def e_step(self, X, prop, means, covs, epsilon=1e-6):
        """
        Proceeds to the e-step of the EM algorithm.

        Args:
            X     (np.ndarray): 2-dimensional array of shape n*D (number of observations * number of variables)
            prop  (np.ndarray): K-dimensional vector, the proportions of the mixture model.
            means (np.ndarray): K*D array: the K means of the densities.
            covs  (np.ndarray): K*D*D array: K covariance matrices of densities.

        Returns:
            resp  (np.ndarray): n*K responsability matrix.
        """

        n, D = X.shape
        K = means.shape[0]

        resp = np.zeros((n, K))
        for i in range(n):
            for k in range(K):
                resp[i, k] += prop[k]*self.get_density(X[i], means[k], covs[k])
        resp += epsilon
        resp /= 1+K*epsilon
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    @njit
    @staticmethod
    def m_step(X, resp):
        """
        Proceeds to the e-step of the EM algorithm.

        Args:
            X     (np.ndarray): 2-dimensional array of shape n*D (number of observations * number of variables)
            resp  (np.ndarray): n*K responsability matrix.

        Returns:
            new_prop  (np.ndarray): K-dimensional vector, the proportions of the mixture model.
            new_means (np.ndarray): K*D array: the K means of the densities.
            new_covs  (np.ndarray): K*D*D array: K covariance matrices of densities.
        """

        n, D = X.shape
        K = resp.shape[1]
        nk = resp.sum(axis=0, keepdims=False)

        # new_prop
        new_prop = resp.sum(axis=0)/n
        # print("new_prop: {}".format(new_prop))

        # new_means
        new_means = resp.T.dot(X)/resp.T.sum(axis=1, keepdims=True)
        # print("new_means: {}".format(new_means))

        # new_covs
        new_covs = np.zeros((K, D, D))
        for k in range(K):
            diff = X-new_means[k]
            new_covs[k] = np.dot(resp[:, k]*diff.T, diff)/nk[k]
        # print("new_covs: {}".format(new_covs))

        return new_prop, new_means, new_covs

    @njit
    def get_params(self):
        return self.prop, self.mu, self.sigma, self.tau