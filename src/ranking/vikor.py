import numpy as np
from madm import MADMBase


class Vikor(MADMBase):

    def solve(self, weights, gama):

        # Calculating the best and the worst option
        best_option = np.max(self.data, axis=0)
        worst_option = np.min(self.data, axis=0)

        # Calculating S
        S = ((best_option - self.data) /
             (best_option - worst_option) * weights).sum(axis=1)
        S_positive = min(S)
        S_negative = max(S)

        # Calculating R
        R = np.max(((best_option - self.data) /
                   (best_option - worst_option) * weights), axis=1)
        R_positive = min(R)
        R_negative = max(R)

        # Calculating Q
        Q = (gama) * ((S - S_positive) / (S_negative - S_positive)) + \
            (1 - gama) * ((R - R_positive) / (R_negative - R_positive))

        rank = np.argsort(Q)
        compromise = rank[0]

        self.Q = Q
        self.rank = rank
        self.compromise = compromise

    def report(self):

        print(f"Q: {self.Q}\nrank: {self.rank}, \ncompromise: {self.compromise}")
