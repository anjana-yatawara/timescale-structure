# The Timescale Structure of Volatility Memory

**Anjana Yatawara**
Department of Mathematics and Statistics, California State University, Bakersfield

Replication package for "The Timescale Structure of Volatility Memory," submitted to the *Journal of Financial Econometrics*.

---

## Abstract

The autocorrelation of absolute returns decays slowly for most financial assets, yet the structure of this persistence across timescales remains unmeasured. This paper decomposes the sample autocorrelation into a weighted mixture of exponential components for 458 assets spanning 11 classes over 2000–2026, using regularized non-negative least squares on a grid of 150 decay rates. The main finding concerns aggregation. The share of weight at short timescales (half-life below ten trading days) falls from 30% for individual stocks to 9% at portfolio size ten and to zero at portfolio size fifty, using the same underlying stocks. Aggregated null portfolios constructed from standard GARCH and FIGARCH processes do not reproduce this collapse, confirming that it is a distinctive feature of financial data.

## Requirements

Python 3.9 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies: NumPy (>=1.21), SciPy (>=1.7), Pandas (>=1.3), Matplotlib (>=3.5).

## Data

The analysis uses daily adjusted closing prices from Yahoo Finance for 466 financial assets, January 2000 through March 2026. Place the data file in the `data/` directory:

```
data/volatility_memory_500_data.csv
```

Columns: `Date`, `Ticker`, `Name`, `Class`, `Subclass`, `AdjClose`, `Volume`, `LogRet`.

After applying minimum-observation filters (2,500 for most classes, 2,000 for cryptocurrency), 458 assets remain for analysis.

## Replication

All scripts are designed to run from the repository root directory. Results are written to `results/`. Figures are written to `Fig/`.

### Step 1: Validation diagnostics

Complete monotonicity check, block bootstrap, resolution analysis, null simulations (GJR-GARCH, EGARCH, FIGARCH), and clustered inference for the aggregation gradient.

```bash
python run_validation.py data/volatility_memory_500_data.csv --workers 34
```

Output: `results/phase5_validation/`

Estimated runtime: 60 minutes on 34 cores.

### Step 2: Robustness diagnostics

Matched aggregation from US stocks, threshold sensitivity (2, 5, 10, 15, 20 days), and crisis-period exclusion (removing 2008–2009 and February–June 2020).

```bash
python run_robustness.py data/volatility_memory_500_data.csv --workers 34
```

Output: `results/phase6_robustness/`

Estimated runtime: 30 minutes on 34 cores.

### Step 3: Aggregated null portfolios

The central identification test. Simulates 285 heterogeneous GJR-GARCH series and 285 heterogeneous FIGARCH series, aggregates them into portfolios of N = 10, 25, 50, 100, 200, and 285, and runs the decomposition on each portfolio. Also recomputes the corrected N = 1 baseline on the 285-stock matched pool and tests the leverage ratio against a GJR-GARCH null.

```bash
python run_aggregated_nulls.py data/volatility_memory_500_data.csv --workers 34
```

Output: `results/phase7_final/`

Estimated runtime: 45 minutes on 34 cores.

### Step 4: Figures

Generates three publication-quality figures in PDF format:

- **Figure 1**: Estimated exponential-mixture weights for AAPL, SPY, and BTC-USD.
- **Figure 2**: Median fast fraction as a function of portfolio size (empirical vs null).
- **Figure 3**: Resolution analysis detection rate as a function of separation ratio.

```bash
python generate_figures.py data/volatility_memory_500_data.csv
```

Output: `Fig/`

## Repository structure

```
timescale-volatility/
    README.md
    LICENSE
    requirements.txt
    .gitignore
    core_final.py             ARCH(inf) filter, GARCH and kernel estimation,
                              stretched exponential fitting, LR testing,
                              GJR-GARCH / SEARCH / FIGARCH simulation
    mixing_distribution.py    Regularized NNLS decomposition of the sample
                              ACF into exponential-mixture weights, L-curve
                              regularization selection, summary diagnostics
    run2_bimodality.py        Peak extraction and fast/slow classification
    run_validation.py         Complete monotonicity, bootstrap, resolution
                              analysis, null zoo, clustered inference
    run_robustness.py         Matched aggregation, threshold sensitivity,
                              crisis exclusion
    run_aggregated_nulls.py   Aggregated null portfolios, corrected baseline,
                              leverage null
    generate_figures.py       Publication-quality figures
    data/                     Input data (not tracked; see Data section)
    results/                  Output from scripts (generated)
    Fig/                      Figures (generated)
```

## Key results

| Portfolio size | Median fast fraction |
|:--------------:|:--------------------:|
| 1 (individual) | 29.6% |
| 10 | 8.6% |
| 25 | 0.4% |
| 50 | 0.0% |
| 100 | 0.0% |
| 200 | 0.0% |
| 285 (full pool) | 0.0% |

Aggregated null portfolios (GJR-GARCH and FIGARCH) do not reproduce this attenuation. The fast fraction under null portfolios remains high and non-monotonic across all portfolio sizes.

Formal inference: Wilcoxon rank-sum p = 1.32 × 10⁻⁴ (stocks vs indices); Kruskal–Wallis p = 9.42 × 10⁻⁶ across five aggregation levels. The contrast survives crisis-period exclusion (p = 1.12 × 10⁻⁴), all five threshold choices (2, 5, 10, 15, 20 days), and block bootstrap resampling (non-overlapping 95% confidence intervals for stocks and indices).

## Limitations reported in the paper

- The empirical autocorrelations do not satisfy complete monotonicity (order-2 median pass fraction: 0.574). The decomposition is a regularized approximation, not an exact Bernstein inversion.
- Individual-asset bimodality cannot be distinguished from smooth long memory: FIGARCH with d = 0.4 produces a 55% bimodality false positive rate.
- The fast end of the timescale distribution is unresolved at daily frequency.
- The leverage ratio (median 1.68) falls within the range generated by GJR-GARCH (median 2.33 under null), so sign asymmetry does not by itself require a multi-timescale explanation.
- The VIX regime-dependence result is based on 14 assets and is exploratory.

## Citation

```bibtex
@article{yatawara2026timescale,
    author  = {Yatawara, Anjana},
    title   = {The Timescale Structure of Volatility Memory},
    journal = {Journal of Financial Econometrics},
    year    = {2026},
    note    = {Submitted}
}
```

## License

MIT License. See `LICENSE`.

## Contact

Anjana Yatawara
Department of Mathematics and Statistics
California State University, Bakersfield
ayatawara@csub.edu
https://yatawara.com
