"""
FIGURE GENERATION: The Timescale Structure of Volatility Memory
================================================================
Generates three publication-quality figures for the paper.

  Figure 1: Three-panel decomposition (AAPL, SPY, BTC-USD)
  Figure 2: Matched aggregation curve (empirical vs null)
  Figure 3: Resolution analysis curve

Usage:
  python generate_figures.py volatility_memory_500_data.csv

Expects:
  core_final.py, mixing_distribution.py in same directory
  results_500/phase2_mixing/mixing_results.csv  (optional, for Fig 1)
  results_500/phase6_final/matched_aggregation.csv (for Fig 2)
  results_500/phase5_critic/resolution_results.csv (for Fig 3)
  results_500/phase7_final/aggregated_null.csv (for Fig 2 null overlay)

Output:
  Fig/fig_example_decompositions.pdf
  Fig/fig_matched_aggregation.pdf
  Fig/fig_resolution.pdf
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# ---- Publication style ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': False,
    'text.usetex': False,
})

OUTDIR = 'Fig'
os.makedirs(OUTDIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==============================================================
#  FIGURE 1: THREE-PANEL DECOMPOSITION
# ==============================================================

def fig1_decompositions(datafile):
    """
    Three-panel figure showing decomposition weights for
    AAPL (US Stock), SPY (US Index), BTC-USD (Cryptocurrency).
    """
    from mixing_distribution import decompose_acf, make_rate_grid, rate_to_halflife

    RATE_GRID = make_rate_grid(150)
    HL_GRID = rate_to_halflife(RATE_GRID)

    # Load data
    df = pd.read_csv(datafile)
    df['Date'] = pd.to_datetime(df['Date'])

    tickers = ['AAPL', 'SPY', 'BTC-USD']
    labels = ['(a) AAPL (U.S. stock)', '(b) SPY (U.S. index)',
              '(c) BTC-USD (cryptocurrency)']

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=False)

    for i, (tk, label) in enumerate(zip(tickers, labels)):
        sub = df[df['Ticker'] == tk].sort_values('Date')
        r = sub['LogRet'].values
        r = r - np.mean(r)

        result = decompose_acf(r, M=150, max_lag=500)
        w = result['weights']

        # Normalize weights for display
        w_norm = w / np.max(w) if np.max(w) > 0 else w

        ax = axes[i]
        ax.fill_between(HL_GRID, 0, w_norm, alpha=0.3, color='#2C5F8A')
        ax.plot(HL_GRID, w_norm, color='#2C5F8A', linewidth=1.0)

        # 10-day threshold line
        ax.axvline(x=10, color='#B85C3C', linestyle='--',
                   linewidth=0.8, alpha=0.7)

        ax.set_xscale('log')
        ax.set_xlim(0.2, 800)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Half-life (trading days)')
        if i == 0:
            ax.set_ylabel('Normalized weight')
        ax.set_title(label, fontsize=9, pad=6)

        # Clean tick labels
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        ax.xaxis.set_minor_locator(LogLocator(subs='auto', numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())

    plt.tight_layout(w_pad=1.5)
    outpath = os.path.join(OUTDIR, 'fig_example_decompositions.pdf')
    fig.savefig(outpath)
    plt.close()
    print(f"  Figure 1 saved: {outpath}")


# ==============================================================
#  FIGURE 2: MATCHED AGGREGATION CURVE
# ==============================================================

def fig2_matched_aggregation():
    """
    Median fast fraction vs portfolio size N.
    Empirical curve + null overlay from Run 7.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # ---- Empirical data from Run 6 ----
    emp_N = [1, 10, 25, 50, 100, 200, 285]
    emp_FF = [0.296, 0.086, 0.004, 0.000, 0.000, 0.000, 0.000]
    emp_FF_pct = [100 * x for x in emp_FF]

    ax.plot(emp_N, emp_FF_pct, 'o-', color='#2C5F8A',
            markersize=5, linewidth=1.5, label='Empirical', zorder=5)

    # ---- Null curves from Run 7 (if available) ----
    null_path = 'results_500/phase7_final/aggregated_null.csv'
    if os.path.exists(null_path):
        null_df = pd.read_csv(null_path)

        for model, color, marker, lab in [
            ('GJR_GARCH', '#B85C3C', 's', 'GJR-GARCH null'),
            ('FIGARCH', '#6B8E6B', '^', 'FIGARCH null')
        ]:
            sub = null_df[null_df['model'] == model]
            if len(sub) == 0:
                continue
            Ns = sorted(sub['N'].unique())
            medians = [100 * sub[sub['N'] == N]['ff_10d'].median() for N in Ns]
            ax.plot(Ns, medians, marker=marker, color=color,
                    markersize=4, linewidth=1.0, linestyle='--',
                    alpha=0.7, label=lab)
    else:
        # Manual null reference lines if CSV not available
        ax.axhline(y=16.9, color='#B85C3C', linestyle=':',
                   linewidth=0.8, alpha=0.6, label='GJR-GARCH null mean')

    ax.set_xscale('log')
    ax.set_xlabel('Portfolio size $N$')
    ax.set_ylabel('Median fast fraction $F_{10}$ (%)')
    ax.set_xlim(0.8, 400)
    ax.set_ylim(-2, 40)

    ax.set_xticks([1, 10, 25, 50, 100, 285])
    ax.set_xticklabels(['1', '10', '25', '50', '100', '285'])
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='0.8')

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig_matched_aggregation.pdf')
    fig.savefig(outpath)
    plt.close()
    print(f"  Figure 2 saved: {outpath}")


# ==============================================================
#  FIGURE 3: RESOLUTION ANALYSIS
# ==============================================================

def fig3_resolution():
    """
    Detection rate vs separation ratio for simulated
    two-component ACFs.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # ---- From Run 5 results ----
    res_path = 'results_500/phase5_critic/resolution_results.csv'
    if os.path.exists(res_path):
        res_df = pd.read_csv(res_path)
        seps = res_df['separation'].values
        bimod = 100 * res_df['bimodal_rate'].values
        fs = 100 * res_df['fast_slow_rate'].values

        ax.plot(seps, bimod, 'o-', color='#2C5F8A',
                markersize=5, linewidth=1.2, label='Bimodality')
        ax.plot(seps, fs, 's--', color='#B85C3C',
                markersize=5, linewidth=1.2, label='Fast + slow')
    else:
        # Manual data from Run 5 report
        seps = [3, 5, 10, 20, 50, 100, 200, 500]
        bimod = [35, 77, 100, 100, 100, 95, 34, 30]
        fs = [32, 22, 26, 100, 100, 100, 100, 90]

        ax.plot(seps, bimod, 'o-', color='#2C5F8A',
                markersize=5, linewidth=1.2, label='Bimodality')
        ax.plot(seps, fs, 's--', color='#B85C3C',
                markersize=5, linewidth=1.2, label='Fast + slow')

    # Empirical separation line
    ax.axvline(x=172, color='0.4', linestyle=':',
               linewidth=0.8, alpha=0.7)
    ax.annotate('Empirical\n($172\\times$)',
                xy=(172, 50), fontsize=7, color='0.4',
                ha='right', va='center',
                xytext=(-8, 0), textcoords='offset points')

    # 70% detection threshold
    ax.axhline(y=70, color='0.7', linestyle='-',
               linewidth=0.5, alpha=0.5)

    ax.set_xscale('log')
    ax.set_xlabel('Separation ratio')
    ax.set_ylabel('Detection rate (%)')
    ax.set_xlim(2, 700)
    ax.set_ylim(-5, 108)

    ax.set_xticks([3, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(['3', '5', '10', '20', '50', '100', '200', '500'])
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.legend(loc='lower right', frameon=True, framealpha=0.9,
              edgecolor='0.8')

    plt.tight_layout()
    outpath = os.path.join(OUTDIR, 'fig_resolution.pdf')
    fig.savefig(outpath)
    plt.close()
    print(f"  Figure 3 saved: {outpath}")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', nargs='?',
                        default='volatility_memory_500_data.csv')
    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print(f"  GENERATING PUBLICATION FIGURES")
    print(f"{'=' * 50}")

    # Figure 1 needs raw data + decomposition
    if os.path.exists(args.datafile):
        print("\n  Figure 1: Three-panel decomposition...")
        try:
            fig1_decompositions(args.datafile)
        except Exception as e:
            print(f"    ERROR: {e}")
            print("    Skipping Figure 1 (needs mixing_distribution.py)")
    else:
        print(f"\n  Skipping Figure 1 (data file not found: {args.datafile})")

    # Figure 2 uses pre-computed results
    print("\n  Figure 2: Matched aggregation curve...")
    fig2_matched_aggregation()

    # Figure 3 uses pre-computed results
    print("\n  Figure 3: Resolution analysis...")
    fig3_resolution()

    print(f"\n{'=' * 50}")
    print(f"  ALL FIGURES IN: {OUTDIR}/")
    print(f"{'=' * 50}")
