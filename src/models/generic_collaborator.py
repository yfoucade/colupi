import ot
import scipy

import numpy as np
import sklearn as sk

from numba import njit
from copy import deepcopy
from sklearn import metrics
from abc import abstractmethod


class Collaborator:
    """
    An abstract class representing a collaborator.
    Collaborators can be of different types (running with different algorithms)
    e.g. Gtms, gaussian mixture models, other mixture models, or any kind of probabilistic model.
    """

    @njit
    def __init__(self, data_Id: np.array, X: np.array, Y=None, K=3, use_criterion=None, *args, **kwargs):
        """
        Parameters:
        data_Id: np.array(int)
            N-dimensional array, Id of each individual.
        X: np.ndarray(float)
            N*D array of features.
        use_criterion: bool
            Whether to use criterion to accept collaboration or not. Default to False.

        Optional:
        Y : np.array(int)
            N-dimensional array, labels. Default to None.
        K : int
            Number of clusters. Default to 3.
        add_noise: bool
            Whether to add white noise or not. Default to False.
        noise_scale: float
            If add_noise==True, the variance of the noise in each dimension is equal to this value
            multiplied by the variance of the data. Default to 0.1.

        define:
        self.Id: int
            Id of the collaborator. Set by the setter method.
        self.K: int
            The number of clusters.
        self.N: int
            Number of lines in the dataset.
        self.D: int
            Dimensionality of the data.
        self.data_Id: np.array(int)
            N-dimensional array, Id of each individual.
        self.X: np.array
        self.R: np.array
            Partition matrix.
        self.history_R: list(np.array)
            List of all partition matrices.
        self.params: dict
            Set of parameters of the distribution (depends on what model is implemented).
            e.g. Gaussian mixture models will have:
            pi (components weights), mu, sigma.
            This attribute can be modified by:
            local_step(), save_old_values(), save_new_values.
        self.H: np.array
            N-dimensional array. Entropy of the classification of each individual.
        self.use_criterion: str
            If not None, one of 'db', 'purity', 'silhouette'
        self.criterion: float
            Current value of the criterion used to decide whether to accept collaboration.
            Computed with self.get_criterion()
        """

        self.Id = None  # set by the master algorithm using self.set_id
        self.K = K
        self.N, self.D = None, None
        self.data_Id, self.X = data_Id, self.parseX(X, **kwargs)
        if Y:
            self.Y = deepcopy(Y)
        self.R = None
        self.history_R = []
        self.params = None
        self.H = None
        self.use_criterion, self.criterion = use_criterion, None
        self.validation_indices_history = []
        self.confidence_coefficients_history = []

    @njit
    def parseX(self, X, *args, **kwargs):
        """
        parse the dataset
        """

        res = deepcopy(X)

        # we want a 2-D array
        if res.ndim==1:
            res = res.reshape(-1, 1)
        self.N, self.D = res.shape

        # If add_noise is set to True, then add noise
        if kwargs.get('add_noise', False):
            std = np.std(res, axis=0)
            noise_std = kwargs.get('noise_scale', .1)*np.diag(std)
            # noinspection PyUnresolvedReferences
            noise = scipy.random.multivariate_normal(mean=np.zeros(self.D), cov=noise_std, size=self.N)
            res += noise

        return res

    @abstractmethod
    def local_step(self):
        """
        Fit the parameters to the dataset.
        Add first partition matrix to history.
        Also initialize the validation indices, and in particular the criterion
        (self.criterion = self.validation_indices_history[-1][self.use_criterion])
        """
        pass

    @abstractmethod
    def refit(self, R):
        """
        Fit the parameters of the model starting from matrix R.

        Parameters:
            R: np.ndarray
                N*K array, responsibility matrix.

        Returns:
            A list, the first elements are the fitted parameters of the model.
            The last element is a dictionary with the validation criteria (at leats db index and silhouette).
        """
        pass

    @njit
    def log_local(self):
        """
        First log, after local step. Log the values of the various validation indices (db, purity, silhouette).

        Returns:
            log: dict
            Dictionary with the values to log: at least the validation indices (db, purity, silhouette).
        """

        db = self.compute_db()
        purity = self.compute_purity()
        silhouette = self.compute_silhouette()

        res = {
            "db": db,
            "purity": purity,
            "silhouette": silhouette
        }

        self.validation_indices_history.append(res)

        """
        or simply:
        return self.validation_indices_history[0]
        """

        """
        TODO:
        add Normalized Mutual Information.
        """

        return res


    @njit
    def collaborate(self, remote_Ids, remote_Rs):  # horizontal collab for now
        """
        Compute a new collaboration matrix, given all remote matrices.

        Parameters:
        remote_Ids: list(int)
            List of Ids of the collaborators.
        remote_Rs: list(np.array)
            List of P N*K(p)-dimensional arrays.
            Where K(p) is the number of clusters in data site number p (p=1,...,P).

        returns:
            res: np.array
                N*K array, the collaborated partition matrix.
            confidence_coefficients: np.array
                P-dimensional array. The confidence coefficient of collab with each remote site.
        """

        # number of collaborators
        P = len(remote_Rs)+1

        # vector of confidence coefficients.
        confidence_coefficients = np.zeros(P)

        # res
        res = np.zeros_like(self.R)

        # entropy of local classification
        local_H = self.compute_entropy(self.R)

        for p, (remote_Id, remote_R) in enumerate(zip(remote_Ids, remote_Rs)):
            # optimal transport
            remote_R = self.optimal_transport(remote_R)
            remote_H = self.compute_entropy(remote_R)
            # compute the local and remote coefficients (one coeff for each individual)
            l, r = (1/(P-1))*remote_H*(1-local_H), local_H*(1-remote_H)
            res += l*self.R+r*remote_R
            # update confidence vector
            confidence_coefficients[remote_Id] += r.sum()
            confidence_coefficients[self.Id] += l.sum()

        # normalize partition matrix
        res /= res.sum(axis=1, keepdims=True)

        # decide whether to accept collaboration
        update = True
        params_and_indices = self.refit(deepcopy(res))
        params, indices = params_and_indices[:-1], params_and_indices[-1]
        if self.use_criterion:
            update = True if self.compare_criterion(indices) else False

        if update:
            successful_collab = True
            self.save_new_values(res, params, indices)
        else:
            successful_collab = False
            confidence_coefficients = np.zeros(P)
            self.save_old_values()

        self.confidence_coefficients_history.append(confidence_coefficients)
        return successful_collab

    @njit
    def compare_criterion(self, new_indices):
        """
        Assess whether the criterion was improved.

        Parameters
        ----------
        new_indices: dict
            Dictionary containing the indices.

        Returns
        -------
        A bool. True if we improved the criterion. False otherwise.
        """

        # If we have no criterion, then we always accept collaboration
        if self.use_criterion is None:
            return True

        if self.use_criterion == 'db':
            return \
                True if new_indices['db'] < self.validation_indices_history[-1]['db'] \
                else False

        if self.use_criterion == 'silhouette':
            return \
                True if new_indices['silhouette'] > self.validation_indices_history[-1]['silhouette'] \
                else False

    @njit
    def save_new_values(self, R, params, indices):
        """
        save values after succesful collaboration.
        In particular, update self.criterion.
        """

        self.R = R
        self.history_R.append(deepcopy(R))
        self.params = params
        self.validation_indices_history.append(indices)
        if self.use_criterion is not None:
            self.criterion = indices[self.use_criterion]

    @njit
    def save_old_values(self, R, params, indices):
        """
        no collaboration: save the old values
        """

        self.history_R.append(self.history_R[-1])
        self.validation_indices_history.append(self.validation_indices_history[-1])

    @njit
    def optimal_transport(self, remote_R, local_R=None):
        """
        compute the optimal transport plan between the remote partition matrix remote_R and the
        local one.
        Returns the transported remote partition in the local space.
        """

        if local_R is None:
            local_R = self.R

        # compute the mass distribution (weight of each cluster)
        local_w = self.local_R.sum(axis=0)/local_R.sum()
        remote_w = remote_R.sum(axis=0)/remote_R.sum()

        # compute the cost matrix
        M = sk.metrics.pairwise_distances(local_R, remote_R)

        # compute the optimal transport plan
        gamma = ot.lp.emd(local_w, remote_w, M)

        # transport
        res = np.dot(remote_R, gamma.T)/np.dot(remote_R, gamma.T).sum(axis=1, keepdims=True)

        return res

    @njit
    def compute_entropy(self, R=None):
        """
        compute normalized entropy

        Args:
            tau: ndarray
                the probabilistic partition matrix

        Returns:
            H: ndarray
                N-dimensional vector: entropy for each data point.

        """

        K = R.shape[1]

        # compute pairwise entropies
        pairwise_entropy = -R.dot(np.log2(R, where=R>0).T)
        # normalize, maximum entropy is given by uniform distribution over K
        pairwise_entropy /= np.log2(K)

        H = pairwise_entropy.diagonal().reshape(-1, 1)

        return H

    @njit
    def log_collab(self):
        """
        Log the results of a collaboration step:
        the validation indices (db, purity, silhouette) and the confidence vector.
        """
        indices = deepcopy(self.validation_indices_history[-1])
        confidence_vector = deepcopy(self.confidence_coefficients_history[-1])

        return indices, confidence_vector

    @njit
    def get_partition_matrix(self):
        """
        Accessor. Returns the partition matrix.
        """

        return self.R

    @njit
    def set_id(self, Id):
        """
        Mutator
        """
        self.Id = Id

    @njit
    def get_id(self):
        """
        Accessor
        """
        return self.Id

    @njit
    def compute_db(self, resp=None):
        """
        compute the DB index of a dataset, given a clustering for this dataset

        Args:
            resp: array-like, (n_samples, n_clusters)
                reponsibility matrix

        Returns:
            float, the DB index
        """

        resp = resp if resp is not None else self.R

        try:
            # a hard partition is required
            y_pred = resp.argmax(axis=1)

            return metrics.davies_bouldin_score(self.X, y_pred)

        except:
            return None

    @njit
    def compute_purity(self, y_true=None, y_pred=None):
        """
        compute the purity score of a clustering

        Args:
            y_true: array-like, (n_samples,)
                labels of each observation
            y_pred: array-like, (n_samples,) or (n_samples, n_clusters)
                predicted hard clustering

        Returns: float
                purity score.
        """

        # if we do not have the labels, return None.
        if y_true is None:
            return None

        y_pred = y_pred if y_pred is not None else self.R
        if y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis=1)

        # compute contingency matrix (also called confusion matrix).
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

        return np.sum(np.amax(contingency_matrix, axis=0))/np.sum(contingency_matrix)

    @njit
    def compute_silhouette(self, y_pred=None):
        """
        Compute the silhouette index of the classification.
        Args:
            y_pred: array-like, (n_samples,) or (n_samples, n_clusters)
                predicted hard clustering or partition matrix.

        Returns: float
            silhouette index.
        """

        y_pred = y_pred if y_pred is not None else self.R
        if y_pred.ndim==2:
            y_pred = np.argmax(y_pred, axis=1)

        try:
            return metrics.silhouette_score(self.X, y_pred)
        except:
            return None
