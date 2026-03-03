import numpy as np
import pandas as pd

class HestonModel:
    def __init__(self, s0, v0, kappa, theta, sigma, rho):
        """
        s0: Initial asset price
        v0: Initial variance
        kappa: Rate of mean reversion
        theta: Long-term average variance
        sigma: Volatility of volatility
        rho: Correlation between asset price and variance
        """
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def simulate_paths(self, T, dt, num_paths):
        """
        Simulate asset price and variance paths using Euler-Maruyama discretization.
        """
        num_steps = int(T / dt)
        S = np.zeros((num_steps + 1, num_paths))
        V = np.zeros((num_steps + 1, num_paths))
        
        S[0] = self.s0
        V[0] = self.v0
        
        for t in range(1, num_steps + 1):
            # Generate correlated random variables
            Z1 = np.random.standard_normal(num_paths)
            Z2 = np.random.standard_normal(num_paths)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # Update variance (using reflection to keep it positive)
            V[t] = V[t-1] + self.kappa * (self.theta - V[t-1]) * dt + \
                   self.sigma * np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * W2
            V[t] = np.maximum(V[t], 0)
            
            # Update asset price
            # Note: We assume risk-free rate r=0 for simplicity in this research step, 
            # but it can be added. 
            r = 0.05 # 5% annual risk-free rate approx
            S[t] = S[t-1] * np.exp((r - 0.5 * V[t-1]) * dt + \
                                  np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * W1)
            
        return S, V

def estimate_initial_params(vix_series):
    """
    Rough estimation of Heston parameters from VIX series.
    vix_series: India VIX historical data (as decimals, e.g., 0.15 for 15%)
    """
    # Variance is VIX^2
    var_series = vix_series**2
    theta = np.mean(var_series)
    # This is a very rough heuristic for kappa and sigma for demonstration
    kappa = 2.0 
    sigma = np.std(var_series) * 5.0
    return theta, kappa, sigma
