
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py

from mcdm_base import MADMBase

import numpy as np
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')

from scipy.optimize import minimize, Bounds, LinearConstraint




def data_gen():
    # Most Important Criteria (The Second Criterion is the Best)
    mic = np.array([3, 1, 4, 3, 8])

    # Least Important Criteria (The Fifth Criterion is the Worst)
    lic = np.array([4, 8, 2, 3, 1])  

    return mic, lic



class bwm(MADMBase):
    def __init__(self, mic, lic, eps_penalty = 1):
        self.mic = mic
        self.lic = lic
        self.eps_penalty = eps_penalty          
        


    def solve(self):
        pass
        
    def report(self):
        cr = []
        mx = np.max(self.mic) 
        if (mx == 1):
            cr = 1
        else:
            for i in range(0, self.mic.shape[0]):
                cr.append((self.mic[i] * self.lic[i] - mx)/(mx**2 - mx))
        cr = np.max(cr)
        threshold = [0, 0, 0, 0.1667, 0.1898, 0.2306, 0.2643, 0.2819, 0.2958, 0.3062]
        np.random.seed(42)
        variables = np.random.uniform(low = 0.001, high = 1.0, size = self.mic.shape[0])
        variables = variables / np.sum(variables)
        variables = np.append(variables, [0])
        bounds    = Bounds([0]*self.mic.shape[0] + [0], [1]*self.mic.shape[0] + [1])
        w_cons    = LinearConstraint(np.append(np.ones(self.mic.shape[0]), [0]), [1], [1])
        results   = minimize(self.target_function, variables, method = 'trust-constr', bounds = bounds, constraints = [w_cons])
        weights   = results.x[:-1]
        return weights
        
        
        
    def target_function(self, variables):
        eps     = variables[-1]
        wx      = variables[np.argmin(self.mic)]
        wy      = variables[np.argmin(self.lic)]
        cons_1  = []
        cons_2  = []
        penalty = 0
        for i in range(0, self.mic.shape[0]):
            cons_1.append(wx - self.mic[i] * variables[i])
        cons_1.extend([-item for item in cons_1])
        for i in range(0, self.lic.shape[0]):
            cons_2.append(variables[i] - self.lic[i] * wy)
        cons_2.extend([-item for item in cons_2])
        cons = cons_1 + cons_2
        for item in cons:
            if (item > eps):
                penalty = penalty + (item - eps) * 1
        penalty = penalty + eps * self.eps_penalty
        return penalty



def main():
    # Step 1:
    mic, lic = data_gen()
      
    print(mic)
    print(lic)
    bwm_obj = bwm(mic, lic)
    
    weights = bwm_obj.report()
    for i in range(0, weights.shape[0]):
        print('w(g'+str(i+1)+'): ', round(weights[i], 8))  
  
if __name__ == "__main__":
    main()


