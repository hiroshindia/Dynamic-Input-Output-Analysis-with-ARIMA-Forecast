# Dynamic Input-Output Analysis with ARIMA Forecast

**Modernization of an undergraduate thesis (1986–1988) from RATS to Python**

---

## Overview

This repository contains a Python implementation of input-output economic analysis originally developed as an undergraduate thesis at the **Faculty of Political Science and Economics, Waseda University** (Nakamura Laboratory, 1986–1988).

The original program was written in **RATS (Regression Analysis of Time Series)**, a specialized econometric software developed by [Estima](https://www.estima.com/). Nearly four decades later, the entire codebase has been modernized and extended in **Python**, with the assistance of [Claude](https://www.anthropic.com/claude) (Anthropic's AI assistant).

The thesis was written under the supervision of **[Professor Shinichiro Nakamura](https://researchmap.jp/read0030000?lang=en)**, currently Professor Emeritus, Faculty of Political Science and Economics, Waseda University — a leading scholar in ecological economics and input-output analysis.

---

## Research Background

The research was conducted in the aftermath of the **two oil shocks of the 1970s** (1973 and 1979), a period in which Japan's heavy dependence on imported oil and its structural impact on domestic prices were central concerns of economic policy and academic research.

The Nakamura Laboratory at Waseda University specialized in **econometrics and input-output analysis**, and this thesis pursued a particularly ambitious goal for its time: combining **static input-output analysis** with **dynamic ARIMA forecasting** of input coefficients — an early attempt at what is now called *dynamic input-output analysis*.

---

## Research Contents

### I. Spread Effect of Import Oil Price on Domestic Prices

Analyzes how a change in the price of imported oil propagates through the domestic economy via inter-industry linkages.

**Key variables:**
- `A` (28×28): Input coefficient matrix
- `Q` (28×28): Leontief inverse matrix
- `M` (28×1): Import vector
- `COEFM`: Import coefficient (oil dependency by sector)
- `MHAT`: Diagonal matrix of import coefficients
- `DPD1`: Direct price effect
- `DPD2`: Full spread effect via modified Leontief inverse

$$
\Delta P_{D2} = L^T \cdot \hat{M} \cdot A^T \cdot \Delta P_M, \quad L = [I - (I - \hat{M})A]^{-1}
$$

**Data source:** `GULF.DAT`

---

### II. Factor Analysis of Petroleum Consumption

#### II-1. Fluctuation of Petroleum Consumption

Decomposes the change in petroleum demand between two periods into four contributing factors:

$$
\Delta X_p = \underbrace{\Delta\hat{A} \cdot L_0 \cdot F_0}_{\text{Input coeff. change}} +
\underbrace{\hat{A}_0 \cdot \Delta L \cdot F_0}_{\text{Structure change}} +
\underbrace{\hat{A}_0 \cdot L_0 \cdot \Delta F}_{\text{Final demand change}} +
\underbrace{\text{Cross terms}}_{\text{CROSS}}
$$

**Data sources:** `MTO60.DAT` (base year), `MTO65.DAT` (comparison year)

---

#### II-2. Fluctuation of Petroleum Input Coefficient

Decomposes the change in petroleum input coefficient $\Delta P_{ij}$ into:

$$
\Delta P_{ij} = \underbrace{\Delta R_p \cdot E_{ij,0}}_{\text{Substitution (SUBS)}} +
\underbrace{R_{p,0} \cdot \Delta E_{ij}}_{\text{Energy saving (REFE)}} +
\underbrace{\Delta R_p \cdot \Delta E_{ij}}_{\text{Cross term (CROS)}}
$$

where $R_p$ = oil/total-energy ratio, $E_{ij}$ = energy intensity.

**Data sources:** `RIO69.N70` (1970 input-output table), `RIO69.N80` (1980 input-output table)

---

### III. Dynamic Extension: ARIMA Forecast of Input Coefficients

The most ambitious component of the original thesis — and the one that has been most substantially modernized — involves **forecasting each element $a_{ij}(t)$ of the input coefficient matrix using ARIMA models**, and then feeding the predicted coefficients back into the Leontief framework to simulate future price spread effects.

**Workflow:**

```
① Historical input coefficients aᵢⱼ(t)  [e.g. 1960, 1965, 1970, 1975, 1980]
         ↓
② Fit ARIMA(p, d, q) to each aᵢⱼ time series
         ↓
③ Forecast aᵢⱼ(t+1), aᵢⱼ(t+2), ... with 95% confidence intervals
         ↓
④ Reconstruct input coefficient matrix A(t) from forecasts
         ↓
⑤ Compute Leontief inverse  L(t) = [I - A(t)]⁻¹
         ↓
⑥ Simulate spread effect of oil price shock at each future period
```

Since `statsmodels` is unavailable in the target environment, **ARIMA is implemented from scratch using NumPy and SciPy**, including OLS estimation of AR parameters, differencing for stationarity, and confidence interval construction.

---

## Repository Structure

```
.
├── README.md                  # This file
├── io_analysis.py             # Core I-O analysis (Part I and II)
├── data_loader.py             # Data loading utilities for RATS-era data files
├── visualize.py               # Matplotlib visualization module
├── dynamic_io_arima.py        # ARIMA forecast + dynamic I-O simulation (Part III)
└── io_analysis.ipynb          # Jupyter Notebook (all analyses + charts)
```

---

## From RATS to Python: Modernization Notes

| Aspect | Original (RATS, ~1987) | Modern (Python, 2024) |
|---|---|---|
| Language | RATS by Estima | Python 3.x |
| Matrix ops | `MAT`, `DEC REC`, `EVAL` | NumPy (`@`, `np.linalg`) |
| Loops | `DO I=1,28 ... END` | `for i in range(28):` |
| Inverse | `INV(A)` | `np.linalg.inv(A)` |
| Transpose | `TR(A)` | `A.T` |
| Diagonal | `DIAG(v)` | `np.diag(v)` |
| Data I/O | `OPEN DATA` / `READ` | `np.loadtxt()` |
| ARIMA | Built-in RATS procedure | Implemented from scratch (NumPy/SciPy) |
| Visualization | None (tabular output only) | Matplotlib (4-panel charts, dashboards) |
| Reproducibility | Single-machine batch job | Jupyter Notebook, modular Python |

---

## Data Files

The following data files are referenced by the original RATS programs. These are **not included** in this repository (original thesis data from the 1980s):

| File | Description | Shape |
|---|---|---|
| `GULF.DAT` | Input coefficient and Leontief data (oil shock analysis) | 28×28 + vectors |
| `MTO60.DAT` | Input-output table (base year) | 46×44 |
| `MTO65.DAT` | Input-output table (comparison year) | 46×44 |
| `RIO69.N70` | 1970 input-output table (69-sector) | 79×84 |
| `RIO69.N80` | 1980 input-output table (69-sector) | 79×84 |

To use real data, set `USE_REAL_DATA = True` in `io_analysis.ipynb` and point `DATA_DIR` to your data directory. The `data_loader.py` module handles Fortran-style scientific notation (`1.23D+04`) automatically.

---

## Requirements

```
numpy
scipy
matplotlib
jupyter  (for .ipynb)
```

Install:
```bash
pip install numpy scipy matplotlib jupyter
```

---

## Usage

### Run the core analysis:
```bash
python io_analysis.py
```

### Run the ARIMA dynamic extension:
```bash
python dynamic_io_arima.py
```

### Generate all charts:
```bash
python visualize.py
```

### Open the Jupyter Notebook:
```bash
jupyter notebook io_analysis.ipynb
```

---

## Acknowledgements

This research was originally conducted under the supervision of **Professor Shinichiro Nakamura**, Nakamura Laboratory, Faculty of Political Science and Economics, Waseda University (1986–1988). Professor Nakamura is currently Professor Emeritus at Waseda University and is internationally recognized for his foundational contributions to **Waste Input-Output (WIO) analysis** and ecological economics.

- Professor Nakamura's research profile: [researchmap.jp/read0030000](https://researchmap.jp/read0030000?lang=en)

The modernization of this program from RATS to Python was carried out in 2024–2025 with the assistance of **[Claude](https://www.anthropic.com/claude)** (Anthropic), which helped transcribe the original RATS code from scanned thesis pages, convert it to Python, implement ARIMA from scratch, and generate visualizations.

---

## Author

Hiroshi Omata (早稲田大学政治経済学部経済学科 中村研究室, 1986–1988)

---

## License

MIT License — feel free to use, adapt, and build upon this work.

---

## References

- Leontief, W. (1986). *Input-Output Economics* (2nd ed.). Oxford University Press.
- Nakamura, S. & Kondo, Y. (2002). "Input-Output Analysis of Waste Management." *Journal of Industrial Ecology*, 6(1), 39–63.
- Box, G.E.P. & Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Estima. *RATS (Regression Analysis of Time Series)*. [www.estima.com](https://www.estima.com/)
