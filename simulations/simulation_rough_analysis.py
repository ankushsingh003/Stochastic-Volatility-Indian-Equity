import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from rough_volatility.rough_heston_model import RoughHestonModel
from heston_model.heston_model import HestonModel
from black_scholes.black_scholes import implied_volatility

def compare_rough_heston():
    s0, v0, r = 10000, 0.04, 0.05
    kappa, theta, sigma, rho = 2.0, 0.04, 0.4, -0.7
    T = 1/52 
    dt = 1/252
    num_paths = 5000
    h_model = HestonModel(s0, v0, kappa, theta, sigma, rho)
    h_s, _ = h_model.simulate_paths(T, dt, num_paths)
    rh_model = RoughHestonModel(s0, v0, kappa, theta, sigma, rho, H=0.1)
    rh_s, _ = rh_model.simulate_paths(T, dt, num_paths)
    strikes = np.linspace(s0 * 0.9, s0 * 1.1, 10)
    h_ivs = []
    rh_ivs = []
    for K in strikes:
        hp = np.mean(np.maximum(h_s[-1] - K, 0)) * np.exp(-r * T)
        rhp = np.mean(np.maximum(rh_s[-1] - K, 0)) * np.exp(-r * T)
        h_ivs.append(implied_volatility(hp, s0, K, T, r))
        rh_ivs.append(implied_volatility(rhp, s0, K, T, r))
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, [i*100 for i in h_ivs], 'b-o', label='Standard Heston IV (H=0.5)')
    plt.plot(strikes, [i*100 for i in rh_ivs], 'r-s', label='Rough Heston IV (H=0.1)')
    plt.title(f"Volatility Smile: Standard vs Rough Heston (T = {T*365:.1f} days)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.legend()
    plt.grid(True)
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig("output/rough_comparison.png")
    print("Rough Heston comparison plot saved to output/rough_comparison.png")

if __name__ == "__main__":
    compare_rough_heston()
