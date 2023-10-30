import torch

class DiscreteEnergyStatistic:

    def __init__(self, n_1, n_2, clustering_function=None):
        """
        Computes discrete energy between two samples.
        Class design based on the following repo:
        https://github.com/josipd/torch-two-sample

        Parameters
        ----------
        n_1: Sample size of first sample (test sample, if learn_clusters=True)
        n_2: Sample size of second sample (test sample, if learn_clusters=True)
        coarsening_function: typically a cluster prediction function with batch support
            (optional) if None, assumes samples have already been clustered

        Returns
        -------
        Initialized class
        """

        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = -1. / (n_1 * n_1)
        self.a11 = -1. / (n_2 * n_2)
        self.a01 = 1. / (n_1 * n_2)
        self.coarsening_fn = clustering_function

    def __call__(self, sample_1, sample_2):
        
        sample_12 = torch.cat((sample_1, sample_2), 0)
        if self.coarsening_fn is not None:
            cats = self.coarsening_fn(sample_12)
        cats = cats.expand(cats.size(0), cats.size(0))
        cdist = (cats != cats.transpose(0, 1)).float()
        for i in range(cdist.size(0)): cdist[i, i] = 0
        d_1 = cdist[:self.n_1, :self.n_1].sum()
        d_2 = cdist[-self.n_2:, -self.n_2:].sum()
        d_12 = cdist[:self.n_1, -self.n_2:].sum()

        loss = 2 * self.a01 * d_12 + self.a00 * d_1 + self.a11 * d_2

        return loss