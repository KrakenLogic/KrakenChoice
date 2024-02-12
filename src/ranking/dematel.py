import numpy as np
from ..mcdm_base import MADMBase

class Dematel_Model(MADMBase):
    def __init__(self, data):
        self.result = None
        super().__init__(data)

    def solve(self):
        normalized_matrix = self.data / np.sum(self.data, axis=0)
        total_effect = np.sum(normalized_matrix, axis=1)
        total_cause = np.sum(normalized_matrix, axis=0)
        normalized_total_effect = total_effect / (1 + total_effect)
        normalized_total_cause = total_cause / (1 + total_cause)
        impact_matrix = np.outer(normalized_total_effect, normalized_total_cause)
        self.result = impact_matrix

    def report(self):
        return self.result