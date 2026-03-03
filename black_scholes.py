import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def implied_volatility(target_value, S, K, T, r, option_type='call'):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5
    
    for i in range(MAX_ITERATIONS):
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)
            
        diff = target_value - price
        if abs(diff) < PRECISION:
            return sigma
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 1e-10:
            return sigma
            
        sigma = sigma + diff / vega
        
    return sigma
