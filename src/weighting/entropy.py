from ..mcdm_base import MADMBase
import numpy as np

class Entropy(MADMBase):

    def solve(self):
    
        col_sums = np.sum(self.data, axis=0)
    
        entropy = np.zeros(self.data.shape[1])
    
        #Normalizing
        self.data = self.data / col_sums
    
        #Calculating Entropy
        entropy = np.sum((self.data * np.log(self.data)) / np.log((self.data.shape[0])), axis = 0) * -1
        
        #Calculating d
        d = 1 - entropy
    
        #Calculating w
        w = d / sum(d)

        self.entropy = entropy
        self.d = d
        
    def report(self):

        print("Entropy: ", self.entropy, "\nd: ", self.d, "\nw: ", self.weights)