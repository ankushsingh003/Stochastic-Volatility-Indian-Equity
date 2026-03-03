import numpy as np
import pandas as pd
from heston_model import HestonModel

class BatesModel(HestonModel):
    def __init__(self, s0, v0, kappa, theta, sigma, rho, lamb, mu_j, sigma_j):
        super().__init__(s0, v0, kappa, theta, sigma, rho)
        self.lamb = lamb
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def simulate_paths(self, T, dt, num_paths):
        num_steps = int(T / dt)
        S = np.zeros((num_steps + 1, num_paths))
        V = np.zeros((num_steps + 1, num_paths))
        
        S[0] = self.s0
        V[0] = self.v0
        
        r = 0.05
        m = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        
        for t in range(1, num_steps + 1):
            Z1 = np.random.standard_normal(num_paths)
            Z2 = np.random.standard_normal(num_paths)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            V[t] = V[t-1] + self.kappa * (self.theta - V[t-1]) * dt + \
                   self.sigma * np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * W2
            V[t] = np.maximum(V[t], 0)
            
            N = np.random.poisson(self.lamb * dt, num_paths)
            Jsum = np.zeros(num_paths)
            for i in range(num_paths):
                if N[i] > 0:
                    jumps = np.random.normal(self.mu_j, self.sigma_j, N[i])
                    Jsum[i] = np.sum(np.exp(jumps) - 1)

            S[t] = S[t-1] * (1 + (r - self.lamb * m) * dt + \
                             np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * W1 + Jsum)
            S[t] = np.maximum(S[t], 1e-4)
            
        return S, V
