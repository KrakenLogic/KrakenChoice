import numpy as np
from ..mcdm_base import MADMBase

class Dematel_Model(MADMBase):
    def __init__(self, data, weights, criterion_type):
        self.result = None
        self.weights = weights
        self.criterion_type = criterion_type
        super().__init__(data)

    def solve(self):
        X = np.copy(self.data) 
        root = np.sqrt(np.sum(X**2, axis=0))
        X = X / root
        X = X * self.weights
        id1 = np.where(np.array(self.criterion_type) == 'max')[0]
        id2 = np.where(np.array(self.criterion_type) == 'min')[0]
        s_p = np.sum(X[:, id1], axis=1) if len(id1) > 0 else np.zeros(X.shape[0])
        s_m = np.sum(X[:, id2], axis=1) if len(id2) > 0 else np.zeros(X.shape[0])
        Y = s_p - s_m
        self.result = np.column_stack((np.arange(1, Y.shape[0]+1), Y))

    def report(self):
        return self.result
    
