import numpy as np
from heston_model import HestonModel

class MonteCarloEngine:
    def __init__(self, model):
        self.model = model

    def calculate_greeks(self, K, T, dt, num_paths, option_type='call', r=0.05, eps=None):
        if eps is None:
            eps = self.model.s0 * 0.01

        s_orig = self.model.s0
        
        p0, _ = self.price_european_option(K, T, dt, num_paths, option_type, r)
        
        self.model.s0 = s_orig + eps
        p_plus, _ = self.price_european_option(K, T, dt, num_paths, option_type, r)
        
        self.model.s0 = s_orig - eps
        p_minus, _ = self.price_european_option(K, T, dt, num_paths, option_type, r)
        
        self.model.s0 = s_orig
        
        delta = (p_plus - p_minus) / (2 * eps)
        gamma = (p_plus - 2 * p0 + p_minus) / (eps**2)
        
        return {'delta': delta, 'gamma': gamma, 'price': p0}

    def price_european_option(self, K, T, dt, num_paths, option_type='call', r=0.05):
        S, V = self.model.simulate_paths(T, dt, num_paths)
        S_T = S[-1]
        
        if option_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
            
        price = np.exp(-r * T) * np.mean(payoffs)
        std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(num_paths)
        
        return price, std_err

if __name__ == "__main__":
    h_model = HestonModel(s0=12000, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    engine = MonteCarloEngine(h_model)
    price, err = engine.price_european_option(K=12000, T=1/12, dt=1/252, num_paths=10000)
    print(f"Option Price: {price:.2f} +/- {err:.2f}")
