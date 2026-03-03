import numpy as np
import pandas as pd
from scipy.optimize import minimize
from heston_model.heston_model import HestonModel
from bates_model.bates_model import BatesModel
from engines.monte_carlo_engine import MonteCarloEngine

def calibrate_to_chain(market_prices, strikes, s0, v0, T, r, model_type='heston'):
    def objective(params):
        if model_type == 'heston':
            kappa, theta, sigma, rho = params
            if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1: return 1e10
            if 2 * kappa * theta < sigma**2: return 1e10
            model = HestonModel(s0, v0, kappa, theta, sigma, rho)
        else:
            kappa, theta, sigma, rho, lamb, mu_j, sigma_j = params
            if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1: return 1e10
            if lamb < 0 or sigma_j <= 0: return 1e10
            model = BatesModel(s0, v0, kappa, theta, sigma, rho, lamb, mu_j, sigma_j)
            
        engine = MonteCarloEngine(model)
        sim_prices = []
        for K in strikes:
            p, _ = engine.price_european_option(K, T, dt=1/252, num_paths=1000, r=r)
            sim_prices.append(p)
        return np.mean((np.array(sim_prices) - np.array(market_prices))**2)

    if model_type == 'heston':
        initial = [2.0, 0.04, 0.3, -0.7]
    else:
        initial = [2.0, 0.04, 0.3, -0.7, 0.1, -0.05, 0.1]
        
    res = minimize(objective, initial, method='Nelder-Mead', tol=1e-2)
    return res.x

if __name__ == "__main__":
    s0, v0, T, r = 12000, 0.04, 1/12, 0.05
    strikes = np.linspace(11000, 13000, 5)
    market_prices = [1050, 600, 250, 80, 20]
    
    heston_params = calibrate_to_chain(market_prices, strikes, s0, v0, T, r, 'heston')
    print("Heston Params:", heston_params)
    
    bates_params = calibrate_to_chain(market_prices, strikes, s0, v0, T, r, 'bates')
    print("Bates Params:", bates_params)
