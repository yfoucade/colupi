from copy import deepcopy

from numba import njit


class Master(object):
    """
    Master algorithm managing the collaboration between the local data sites
    """

    @njit
    def __init__(self, collabs=None, max_iter=0, *args, **kwargs):
        """
        Parameters:
        collabs: list(Collaborator)
            List of instances of type "Collaborator".
        max_iter (optional): int
            Maximum number of iterations in the collaborative phase

        optional:
        X (optional): list(pandas.DataFrame)
            List of dataframes with same size than collabs.
            The view of each collaborators, with acces to possibly different
            set of individuals (identified by the index column).
        Y (optional): list(pandas.DataFrame)
            List of dataframes, one for each collaborator. With the indices and the labels.

        notes:
        02/02 14:16 - Xs and Ys are not necessary parameters, as we can get them from each collab.
                      Consider removing them.
        """

        self.collabs, self.P = collabs, len(collabs)
        for i, collab in enumerate(self.collabs):
            collab.set_id(i)

        self.max_iter = max_iter
        # create a log that will contain information about each step of the process
        self.log = []  # validation indices (for each step, a list of dicts)
        self.collaboration_history = []  # confidence matrices

    @njit
    def launch_collab(self):
        """
        Proceed to the collaboration.
        """

        # each collaborator fits its parameters in the local step
        to_log = []
        for collab in self.collabs:
            collab.local_step()
            to_log.append(deepcopy(collab.log_local()))
        # log
        self.log.append(deepcopy(to_log))

        n_iter = 0
        while True:
            stop, to_log = True, []

            # save all collaboration matrices

            # each data site is in turn considered as the local data site
            for p, collab in enumerate(self.collabs):
                # it is provided with a tuple for each remote data site: (Id, partition_matrix)
                remote_Ids, remote_partitions = self.get_partitions_except_p(p)
                successful_collab = collab.collaborate(remote_Ids, remote_partitions)
                to_log.append(deepcopy(collab.log_collab()))
                if successful_collab:
                    stop = False

            # log the results
            self.log.append(deepcopy(to_log))

            # should we stop ?
            if stop:
                break
            # if stop==False, then a collaboration occured, add 1 to counter
            n_iter += 1

            if n_iter == self.max_iter:
                break

    @njit
    def get_partitions_except_p(self, p):
        """
        Get all partition matrices except number p.
        This is used to get every remote partition matrix when p is the local data site.

        Parameters:
            p: int
                id of the partition matrix to ignore.
        """

        res_Ids, res_partitions = [], []
        for i, collab in self.collabs:
            if i != p:
                res_Ids.append(collab.get_id())
                res_partitions.append(deepcopy(collab.get_partition_matrix()))
        return res_Ids, res_partitions
