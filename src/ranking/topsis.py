import numpy as np
from ..mcdm_base import MADMBase

class Topsis(MADMBase):

    def solve(self, weights):

        #Normalizing
        col_2 = self.data ** 2
        col_sum = np.sum(col_2, axis = 0)
        col_sqrt = np.sqrt(col_sum)
        self.data = self.data / col_sqrt
    
        #Weighting
        self.data = self.data * weights
        
        #Calculating Best option and the worst option
        best_option = np.max(self.data, axis = 0)
        worst_option = np.min(self.data, axis = 0)
        
        #Calculating the distances
        s_positive = np.sqrt(((self.data - best_option) ** 2).sum(axis=1))
        s_negative = np.sqrt(((self.data - worst_option) ** 2).sum(axis=1))
        
        #Calculating the results
        C = s_negative / (s_positive + s_negative) 

        self.best_option = best_option
        self.worst_option = worst_option
        self.s_positive = s_positive
        self.s_negative = s_negative
        self.C = C

    def report(self):

        print("Best option: ", self.best_option, "\n\nWorst option: ", self.worst_option, "\n\nPositive S: ", self.s_positive,
              "\n\nNegative S: ", self.s_negative, "\n\nResults: ", self.C, "\n\n","Maximum Results", max(self.C) )