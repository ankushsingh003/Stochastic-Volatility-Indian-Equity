# Stochastic Volatility & Jump-Diffusion Modeling Suite

A professional-grade research and simulation engine for pricing equity derivatives using advanced stochastic volatility models. This suite implements **Heston**, **Bates (Jump-Diffusion)**, and **Rough Heston (Fractional)** models, calibrated to market data and validated through delta-hedging backtests.

## 🚀 Key Features

- **Multi-Model Support**:
  - **Heston Model**: Captures stochastic volatility and leverage effect.
  - **Bates Model**: Extends Heston with Merton-style Poisson jumps.
  - **Rough Heston**: Uses fractional Brownian motion ($H \approx 0.1$) to capture steep short-term skews.
- **Robust Greeks Engine**: High-performance finite difference estimation for **Delta** and **Gamma**.
- **Market Calibration**: Automated fitting of model parameters to market option chains (implied volatility surfaces).
- **Strategy Backtesting**: Comprehensive Delta-Hedging framework to evaluate model performance against historical market regimes (e.g., March 2020 crash).
- **Modular Architecture**: Fully decoupled packages for models, engines, and simulations.

## 📂 Directory Structure

```mermaid
graph TD
    A[Root] --> B[black_scholes]
    A --> C[heston_model]
    A --> D[bates_model]
    A --> E[rough_volatility]
    A --> F[engines]
    A --> G[simulations]
    A --> H[calibration]
    A --> I[data_loader]
    
    F --> F1[Monte Carlo Engine]
    F --> F2[Backtest Engine]
    G --> G1[Full Analysis]
    G --> G2[Bates vs Heston]
    G --> G3[Rough vs Heston]
    G --> G4[Tripartite Backtest]
```

- **`black_scholes/`**: Analytical baseline for European options.
- **`heston_model/`**: Standard stochastic volatility with Feller condition enforcement.
- **`bates_model/`**: Heston + Jump diffusion components.
- **`rough_volatility/`**: Non-Markovian simulation using hybridized convolution schemes.
- **`engines/`**: Core computational logic for path simulation and hedging.
- **`simulations/`**: End-user scripts for analysis and visualization.
- **`calibration/`**: Optimization logic for parameter estimation from market prices.

## 🛠 Usage & Execution

Initialize the environment by setting the project root in your `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "."
```

### 1. Full Market Analysis
Runs data fetching, Heston calibration, Greeks calculation, and visualizes the IV smile.
```powershell
python -m simulations.simulation_full_analysis
```

### 2. Rough Heston Comparative Study
Visualizes the impact of the Hurst index (H) on short-term volatility skew.
```powershell
python -m simulations.simulation_rough_analysis
```

### 3. Delta-Hedging Backtest
Evaluates the hedging performance of Heston, Bates, and Rough Heston models.
```powershell
python -m simulations.simulation_backtest
```

## 📊 Modeling Workflow

```mermaid
sequenceDiagram
    participant D as Data Loader
    participant C as Calibration
    participant M as Model (Heston/Bates/Rough)
    participant E as MC Engine
    participant S as Simulation
    
    D->>C: Historical Spot & VIX
    C->>M: Calibrated Parameters
    M->>E: Path Specifications
    E->>E: simulate_paths()
    E->>S: Option Prices & Greeks
    S->>S: Generate Plots & PnL
```

## 📈 Results
All visual outputs (PNG) are stored in `output/` and `backtest_results/`. The suite provides side-by-side comparisons of volatility smiles and cumulative hedging errors across different market regimes.

---
*Developed for advanced stochastic calculus research and derivative strategy validation.*
