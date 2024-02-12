import numpy as np
from ..mcdm_base import MADMBase

class AHP_Model_Weight(MADMBase):
    def __init__(self, data):
        self.result = None
        super().__init__(data)
    
    def ahp_weighting (self):
        col_sum=np.sum(self.data,axis=0)
        normalize_matrix=self.data/col_sum
        criteria_weight=np.mean(normalize_matrix,axis=1)
        return criteria_weight
    
    def solve(self):
        self.result = self.ahp_weighting() 
    
    def report(self):
        return self.result

        

class AHP_Model_Rank(MADMBase):
    def __init__(self, data, ahp_weight=None):
        super().__init__(data)
        self.weight_class = AHP_Model_Weight(self.data)
        self.weight = ahp_weight
        self.result = None

    def ahp_ranking(self):
        self.weight_class.solve()
        result = self.weight_class.report().T
        if self.weight is not None:
            weighted_sum = np.dot(result, self.weight)
            sorted_weight = np.argsort(weighted_sum)  
            return weighted_sum, result
        else:
            self.weight = result
            return result
        
    def rate_stability(self):
        weight = self.weight
        lanada_matrix=np.dot(self.data,weight)
        lanada_matrix=np.divide(lanada_matrix,weight)
        landa_max=np.divide(np.sum(lanada_matrix),len(lanada_matrix))
        
        Incompatibility_index=(landa_max-len(lanada_matrix))/(len(lanada_matrix)-1)
        
        IIR={1:0 ,2:0 ,3:0.58 ,4:0.9 ,5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.51}
        incompatibility_rate= (Incompatibility_index)/(IIR[len(lanada_matrix)])
        
        return incompatibility_rate

    def solve(self):
        self.result = {'ranking': self.ahp_ranking(), 'rate': self.rate_stability()}
    
    def report(self):
        return self.result