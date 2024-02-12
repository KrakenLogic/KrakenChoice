import numpy as np
from ..mcdm_base import MADMBase

class Dematel_Model(MADMBase):
    def __init__(self, data, weights, q, p):
        self.result = None
        self.weights = weights
        self.q = q
        self.p = p
        super().__init__(data)

    def solve(self):
        concordance_matrix = np.sum(
            (self.data[:, None, :] >= self.data[None, :, :]) * self.weights[None, None, :],
            axis=2
        )   
        discordance_matrix = np.sum(
            (self.data[None, :, :] - self.data[:, None, :]) * (self.data[None, :, :] > self.data[:, None, :]) * self.weights[None, None, :],
            axis=2
        )
        outranking_relation = (concordance_matrix >= self.q) & (discordance_matrix <= self.p)
        np.fill_diagonal(outranking_relation, 0)
        outranking_score = np.sum(outranking_relation, axis=1)
        ranking = np.argsort(outranking_score)[::1]
        self.result = ranking + 1  

    def report(self):
        return self.result