
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py

from mcdm_base import MADMBase

import numpy as np


def data_gen():
    dataset = np.array([
                    [25000, 153, 15.3, 250],   #a1
                    [33000, 177, 12.3, 380],   #a2
                    [40000, 199, 11.1, 480]    #a3
                   ])
    # Grades - From 4 (worst) to 10 (best)
    grades = np.array([9, 5, 7, 6])
    
    # Min Criteria Threshold
    lower = np.array([20300, 140, 8.2, 230])
    
    # Max Criteria Threshold
    upper = np.array([40000, 220, 20, 2000])
    
    # Criterion Type: 'max' or 'min'
    criterion_type = ['max', 'max', 'min', 'max']
    d = dataset
    w = grades
    l = lower
    u = upper
    c = criterion_type

    return d, w, l, u, c



class smart(MADMBase):
    def __init__(self, data, weights, lower, upper, criterion_type):
        self.data = data
        self.weights = weights
        self.lower = lower          
        self.upper = upper
        self.criterion_type = criterion_type
        


    def solve(self):
        pass
        
    def report(self):
        normalized_data = self.normalize_data(self.data, self.lower, self.upper, self.criterion_type)
        # Step 3: Calculate total scores
        total_scores = self.calculate_smart_scores(normalized_data, self.weights)
        # Step 4: Rank the alternatives
        ranked_indices = self.rank_alternatives(total_scores)
        
        print("\nResults:")
        for i, index in enumerate(ranked_indices):
            print(f"Rank {i + 1}: Alternative {index + 1} - Total Score: {total_scores[index]:.4f}")

        return ranked_indices
    
    def normalize_data(self, data, lower, upper, criterion_type):
        D = data.copy()
        dist = upper - lower
        D = np.log2(64*((D - np.tile(lower, (D.shape[0],1)))/dist))
        
        # max
        dmax = D[:, [np.where(np.array(criterion_type) == 'max')[0]]].copy()
        D[:, [np.where(np.array(criterion_type) == 'max')[0]]] = dmax+4
        # min
        dmin = D[:, [np.where(np.array(criterion_type) == 'min')[0]]].copy()
        D[:, [np.where(np.array(criterion_type) == 'min')[0]]] = -1*dmin+10

        return D
        

    # calculation of smart score
    def calculate_smart_scores(self, normalized_data, weights):
        sq = lambda x: np.power(np.power(2, 0.5) , x)
        g = weights.copy()
        g = list(map(sq, g))
        g = g/np.sum(g)

        scores = normalized_data @ g
        return scores

    # ranking
    def rank_alternatives(self, total_scores):

        ranked_indices = np.argsort(total_scores)[::-1]
        # ranked_indices = np.argsort(total_scores)
        return ranked_indices

def main():
    # Step 1:
    data, weights, lower, upper, criterion_type = data_gen()
      
    print(data)
    print(weights)
    print(lower)
    print(upper)
    print(criterion_type)
     
      
    smart_obj = smart(data, weights, lower, upper, criterion_type)
    ranked_indices = smart_obj.report()
      
  
if __name__ == "__main__":
    main()


