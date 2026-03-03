import numpy as np
import math
from heston_model.heston_model import HestonModel

class RoughHestonModel(HestonModel):
    def __init__(self, s0, v0, kappa, theta, sigma, rho, H):
        super().__init__(s0, v0, kappa, theta, sigma, rho)
        self.H = H
        self.alpha = H + 0.5

    def simulate_paths(self, T, dt, num_paths):
        num_steps = int(T / dt)
        S = np.zeros((num_steps + 1, num_paths))
        V = np.zeros((num_steps + 1, num_paths))
        S[0] = self.s0
        V[0] = self.v0
        r = 0.05
        dW_v = np.random.standard_normal((num_steps, num_paths)) * np.sqrt(dt)
        dW_s_raw = np.random.standard_normal((num_steps, num_paths)) * np.sqrt(dt)
        dW_s = self.rho * dW_v + np.sqrt(1 - self.rho**2) * dW_s_raw
        increments = np.zeros((num_steps, num_paths))
        for t in range(num_steps):
            drift = self.kappa * (self.theta - V[t]) * dt
            diffusion = self.sigma * np.sqrt(np.maximum(V[t], 0)) * dW_v[t]
            increments[t] = drift + diffusion
            V_t_next = self.v0
            for j in range(t + 1):
                lag = (t + 1 - j)
                kernel = (lag * dt)**(self.H - 0.5) / math.gamma(self.H + 0.5)
                V_t_next += kernel * increments[j]
            V[t+1] = np.maximum(V_t_next, 0)
            S[t+1] = S[t] * np.exp((r - 0.5 * V[t]) * dt + np.sqrt(np.maximum(V[t], 0)) * dW_s[t])
        return S, V
