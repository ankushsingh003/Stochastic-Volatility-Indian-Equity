import numpy as np
import pandas as pd

from scipy.optimize import minimize

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

    def is_feller_satisfied(self):
        """
        Check if the Feller condition (2*kappa*theta > sigma^2) is satisfied.
        If satisfied, the variance process V_t is strictly positive.
        """
        return 2 * self.kappa * self.theta > self.sigma**2

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
            r = 0.05 # 5% annual risk-free rate approx
            S[t] = S[t-1] * np.exp((r - 0.5 * np.maximum(V[t-1], 0)) * dt + \
                                  np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * W1)
            
        return S, V

def calibrate_heston(market_prices, s0, v0, strikes, T, r):
    """
    Calibrate Heston parameters to market prices using SciPy.
    market_prices: List of observed option prices for given strikes
    """
    from monte_carlo_engine import MonteCarloEngine
    
    def objective_function(params):
        kappa, theta, sigma, rho = params
        # Constraints/Penalties
        if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        # Feller condition check
        if 2 * kappa * theta < sigma**2:
            # Penalize parameters that violate Feller condition to ensure variance stays positive
            return 1e10
            
        model = HestonModel(s0, v0, kappa, theta, sigma, rho)
        engine = MonteCarloEngine(model)
        
        sim_prices = []
        for K in strikes:
            p, _ = engine.price_european_option(K, T, dt=1/252, num_paths=2000, r=r)
            sim_prices.append(p)
            
        mse = np.mean((np.array(sim_prices) - np.array(market_prices))**2)
        return mse

    # Initial guess
    initial_params = [2.0, 0.04, 0.3, -0.7] # [kappa, theta, sigma, rho]
    
    print("Starting Heston Calibration (this may take a minute)...")
    result = minimize(objective_function, initial_params, method='Nelder-Mead', tol=1e-2)
    return result.x

def estimate_initial_params(vix_series):
    """
    Rough estimation of Heston parameters from VIX series.
    vix_series: India VIX historical data (as decimals, e.g., 0.15 for 15%)
    """
    var_series = vix_series**2
    theta = np.mean(var_series)
    kappa = 2.0 
    sigma = np.std(var_series) * 3.0
    return theta, kappa, sigma
