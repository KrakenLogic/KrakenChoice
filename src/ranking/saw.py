
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py

from mcdm_base import MADMBase

import numpy as np


def data_gen():
  rawdata = np.random.randint(low=1, high=15, size=np.random.randint(low=1, high=10, size=(1,2))[0])
  d = np.array(rawdata, dtype=np.longdouble)
  weights = np.random.random(size = (1, d.shape[1]))
  w = (weights/np.sum(weights))[0]

  criterion_type = (np.random.randint(low=0, high=2, size = (1, d.shape[1])))[0]
  c = list(map(lambda x: "max" if x == 1 else "min", criterion_type))
  return d, w, c



class saw(MADMBase):
    def __init__(self, data, weights, criterion_type):
        self.data = data
        self.weights = weights
        self.criterion_type = criterion_type
        
    def solve(self):
        pass
        
    def report(self):
        normalized_data = self.normalize_data(self.data, self.criterion_type)
        # Step 3: Calculate total scores
        total_scores = self.calculate_saw_scores(normalized_data, self.weights)
        # Step 4: Rank the alternatives
        ranked_indices = self.rank_alternatives(total_scores)
        
        print("\nResults:")
        for i, index in enumerate(ranked_indices):
            print(f"Rank {i + 1}: Alternative {index + 1} - Total Score: {total_scores[index]:.4f}")

        return ranked_indices
    
    def normalize_data(self, data, criterion_type):
        D = data.copy()
        # max
        dmax = D[:, [np.where(np.array(criterion_type) == 'max')[0]]].copy()
        D[:, [np.where(np.array(criterion_type) == 'max')[0]]] = dmax/dmax.max(axis=0)
        # min
        dmin = D[:, [np.where(np.array(criterion_type) == 'min')[0]]].copy()
        D[:, [np.where(np.array(criterion_type) == 'min')[0]]] = dmin.min(axis=0)/dmin

        return D
        

    # calculation of saw score
    def calculate_saw_scores(self, normalized_data, weights):
        scores = normalized_data @ weights
        return scores

    # ranking
    def rank_alternatives(self, total_scores):
        ranked_indices = np.argsort(total_scores)[::-1]
        # ranked_indices = np.argsort(total_scores)
        return ranked_indices

def main():
  # Step 1:
  data, weights, criterion_type = data_gen()
  
  print(data)
  print(weights)
  print(criterion_type)
  
  
  saw_obj = saw(data, weights, criterion_type)
  ranked_indices = saw_obj.report()
  
  
if __name__ == "__main__":
    main()


