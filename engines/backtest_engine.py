import numpy as np

def run_delta_hedge_backtest(model, engine, spot_prices, strikes, T, r, dt, num_paths=10000):
    num_days = len(spot_prices)
    portfolio_value = np.zeros(num_days)
    cash = np.zeros(num_days)
    holdings = np.zeros(num_days)
    
    K = strikes
    initial_res = engine.calculate_greeks(K, T, dt, num_paths, r=r)
    initial_price = initial_res['price']
    initial_delta = initial_res['delta']
    
    holdings[0] = initial_delta
    cash[0] = initial_price - initial_delta * spot_prices[0]
    portfolio_value[0] = initial_price
    
    for t in range(1, num_days):
        time_left = T - (t * dt)
        if time_left <= 0:
            portfolio_value[t:] = holdings[t-1] * spot_prices[t:] + cash[t-1] * np.exp(r * dt)
            break
            
        res = engine.calculate_greeks(K, time_left, dt, num_paths, r=r)
        new_delta = res['delta']
        
        cash[t] = cash[t-1] * np.exp(r * dt) - (new_delta - holdings[t-1]) * spot_prices[t]
        holdings[t] = new_delta
        portfolio_value[t] = holdings[t] * spot_prices[t] + cash[t]
        
    option_payoff = np.maximum(spot_prices[-1] - K, 0)
    hedging_error = portfolio_value[-1] - option_payoff
    
    return {
        'portfolio_value': portfolio_value,
        'hedging_error': hedging_error,
        'payoff': option_payoff
    }
