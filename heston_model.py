import numpy as np
import pandas as pd
from scipy.optimize import minimize

class HestonModel:
    def __init__(self, s0, v0, kappa, theta, sigma, rho):
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def is_feller_satisfied(self):
        return 2 * self.kappa * self.theta > self.sigma**2

    def simulate_paths(self, T, dt, num_paths):
        num_steps = int(T / dt)
        S = np.zeros((num_steps + 1, num_paths))
        V = np.zeros((num_steps + 1, num_paths))
        
        S[0] = self.s0
        V[0] = self.v0
        
        for t in range(1, num_steps + 1):
            Z1 = np.random.standard_normal(num_paths)
            Z2 = np.random.standard_normal(num_paths)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            V[t] = V[t-1] + self.kappa * (self.theta - V[t-1]) * dt + \
                   self.sigma * np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * W2
            V[t] = np.maximum(V[t], 0)
            
            r = 0.05
            S[t] = S[t-1] * np.exp((r - 0.5 * np.maximum(V[t-1], 0)) * dt + \
                                  np.sqrt(np.maximum(V[t-1], 0)) * np.sqrt(dt) * W1)
            
        return S, V

def calibrate_heston(market_prices, s0, v0, strikes, T, r):
    from monte_carlo_engine import MonteCarloEngine
    
    def objective_function(params):
        kappa, theta, sigma, rho = params
        if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        if 2 * kappa * theta < sigma**2:
            return 1e10
            
        model = HestonModel(s0, v0, kappa, theta, sigma, rho)
        engine = MonteCarloEngine(model)
        
        sim_prices = []
        for K in strikes:
            p, _ = engine.price_european_option(K, T, dt=1/252, num_paths=2000, r=r)
            sim_prices.append(p)
            
        mse = np.mean((np.array(sim_prices) - np.array(market_prices))**2)
        return mse

    initial_params = [2.0, 0.04, 0.3, -0.7]
    
    print("Starting Heston Calibration...")
    result = minimize(objective_function, initial_params, method='Nelder-Mead', tol=1e-2)
    return result.x

def estimate_initial_params(vix_series):
    var_series = vix_series**2
    theta = np.mean(var_series)
    kappa = 2.0 
    sigma = np.std(var_series) * 3.0
    return theta, kappa, sigma
