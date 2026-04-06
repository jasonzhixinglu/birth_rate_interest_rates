# Birth Rate Collapse and Its Effect on Interest Rates — Python Replication

Replication of Lu, J. and Teulings, C. (2015). Fertility Rates and the Age Distribution. University of Cambridge Working Paper. First version: September 2015, last edited: October 2017.

## Overview

This project replicates the core quantitative results of Lu and Teulings (2015) in Python. The paper shows that the collapse in birth rates around 1970 across advanced economies can account for a significant share of the observed decline in real interest rates since the mid-1980s. The mechanism operates through life-cycle saving: the baby boomers — the last large birth cohort — accumulate savings ahead of retirement, driving down interest rates to a trough near zero around 2028 that overshoots the new long-run equilibrium. This replication omits the land extension and the bequest motive from the original paper.

## Model structure

**Demographic module** (`src/demographics.py`)
Cohort sizes evolve via a birth rate equation driven by observed TFR data. A two-parameter fertility shock fits the observed age distribution well for Germany.

**Firm module** (`src/firm.py`)
CES production function with capital share alpha and elasticity of substitution sigma. Factor prices derived from profit maximisation.

**Household module** (`src/household.py`)
Standard CRRA lifetime utility with no bequest motive. Euler equation links consumption growth to the real interest rate.

**BGP solver** (`src/bgp.py`)
Analytically pins down the pre-shock and post-shock balanced growth path interest rates by equating household asset demand to capital supply.

**OLG transition solver** (`src/olg.py`)
Iterative damped algorithm solving for the full transitional interest rate path between the two BGPs. Converges when the implied and assumed interest rate paths are sufficiently close.

## Repository structure

```
birth_rate_interest_rates/
├── src/
│   ├── __init__.py
│   ├── demographics.py     # Cohort size transition, TFR → N_t
│   ├── firm.py             # CES production, factor prices
│   ├── household.py        # CRRA Euler equation, asset accumulation
│   ├── bgp.py              # BGP interest rate solver
│   ├── olg.py              # Iterative transition algorithm
│   └── calibration.py      # Country-specific TFR data
├── notebooks/
│   ├── 01_demographics.ipynb        # Age pyramids, TFR, cohort sizes
│   ├── 02_germany_baseline.ipynb    # Main result: Germany transition
│   └── 03_cross_country.ipynb       # US, Japan, China comparison
├── figures/                # Output charts
├── docs/                   # Rendered Quarto site (GitHub Pages)
├── paper_summary.qmd       # Abridged replication document
├── requirements.txt
└── README.md
```

## Installation and usage

```bash
git clone https://github.com/jasonzhixinglu/birth_rate_interest_rates.git
cd birth_rate_interest_rates
pip install -r requirements.txt
jupyter notebook
```

Then open any notebook under `notebooks/` and run all cells. A `.devcontainer/devcontainer.json` is provided for one-click setup in GitHub Codespaces or VS Code Dev Containers.

## Key results (Germany calibration)

| Quantity | Value | Notes |
|----------|-------|-------|
| r_init | 3.5% | Pre-shock BGP, calibration target |
| r_terminal | 1.5% | Post-shock BGP |
| Trough | ~0.7% | Year ~2028 |
| Overshoot | ~84bp | Below new BGP |
| Peak-to-trough decline | ~3pp | Share of observed 4pp decline |

## Calibration

| Parameter | Value | Source |
|-----------|-------|--------|
| alpha | 0.33 | Standard capital share |
| sigma | 0.4 | Production literature |
| delta | 0.05 | Standard depreciation |
| beta | 0.946 | Calibrated to r_init = 3.5% |
| theta | 2.0 | Standard inverse EIS |
| gamma | 0.015 | German per-capita GDP growth |
| J | 80 | Lifespan |
| chi | 20 | Age entering labour market |
| psi | 60 | Retirement age |

## References

- Lu, J. and Teulings, C. (2015). Fertility Rates and the Age Distribution. University of Cambridge Working Paper. First version: September 2015, last edited: October 2017.
- Samuelson, P. A. (1958). An exact consumption-loan model of interest. *Journal of Political Economy*, 66(6), 467–482.
- Rachel, L. and Smith, T. (2015). Secular drivers of the global real interest rate. Bank of England Working Paper No. 571.
- Carvalho, C., Ferrero, A. and Nechio, F. (2016). Demographics and real interest rates. *European Economic Review*, 88, 208–226.
