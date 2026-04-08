"""
RUN 7: FINAL PRE-SUBMISSION TESTS
====================================
Three tests to close the last identification gaps:

  TEST 1: Aggregated null portfolios (THE critical test)
          Simulate 285 GJR-GARCH series + 285 FIGARCH series,
          aggregate into portfolios of N=10,25,50,100,200,285,
          run decomposition. Compare N-curves to empirical.

  TEST 2: Corrected N=1 baseline on 285-stock pool
          Recompute fast fraction using only the 285 stocks
          in the matched-aggregation pool, not all 323.

  TEST 3: Leverage ratio under GJR-GARCH null
          Compute leverage ratio on the 200 GJR-GARCH sims
          from Run 5 to check if our empirical 1.68 is real.

Usage:
  python run7_final.py volatility_memory_500_data.csv --workers 34

Expects in same directory:
  core_final.py, mixing_distribution.py

Output:
  results_500/phase7_final/
    REPORT_FINAL.txt
    aggregated_null_garch.csv
    aggregated_null_figarch.csv
    corrected_baseline.csv
    leverage_null.csv
"""

import sys, os, time, io, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_final import simulate_gjr_garch, simulate_figarch, J_MAX
from mixing_distribution import (
    decompose_acf, make_rate_grid, rate_to_halflife,
    sample_acf_abs
)

RATE_GRID = make_rate_grid(150)
HL_GRID = rate_to_halflife(RATE_GRID)
OUTDIR = 'results_500/phase7_final'

MIN_OBS = 2500
MIN_OBS_CRYPTO = 2000


class Report:
    def __init__(self, path):
        self.path = path
        self.buf = io.StringIO()
    def p(self, text="", end="\n"):
        print(text, end=end)
        self.buf.write(text + end)
    def save(self):
        with open(self.path, 'w') as f:
            f.write(self.buf.getvalue())


def load_data(fp):
    """Load CSV, return assets dict AND raw DataFrame."""
    df = pd.read_csv(fp)
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Subclass' in df.columns and 'Sector' not in df.columns:
        df = df.rename(columns={'Subclass': 'Sector'})
    if 'Sector' not in df.columns:
        df['Sector'] = ''

    assets = {}
    for tk in df['Ticker'].unique():
        sub = df[df['Ticker'] == tk].sort_values('Date')
        r = sub['LogRet'].values
        r = r - np.mean(r)
        cls = sub['Class'].iloc[0] if 'Class' in sub.columns else 'Unknown'
        sector = sub['Sector'].iloc[0] if 'Sector' in sub.columns else ''
        if pd.isna(sector):
            sector = ''
        min_obs = MIN_OBS_CRYPTO if cls == 'Cryptocurrency' else MIN_OBS
        if len(r) >= min_obs:
            assets[tk] = {'r': r, 'T': len(r), 'class': cls, 'sector': sector}
    return assets, df


# ==============================================================
#  TEST 1: AGGREGATED NULL PORTFOLIOS
# ==============================================================
#
#  THE critical test. If aggregating null processes also
#  drives fast fraction to zero, then the matched-aggregation
#  curve is a mechanical property of our estimator, not a
#  distinctive feature of financial data.
#
#  If null portfolios stay at ~18% regardless of N, then
#  the empirical collapse is real.
#
#  We simulate 285 series from each null family with
#  HETEROGENEOUS parameters (different beta/d for each),
#  aggregate into portfolios of various sizes, and run
#  the decomposition on each portfolio.
# ==============================================================

def _simulate_null_portfolio(args):
    """Simulate N null series, average them, run decomposition."""
    model, N, trial, T, param_list = args

    rng = np.random.default_rng(hash((model, N, trial)) % (2**31))

    # Simulate N series with heterogeneous parameters
    returns_stack = []
    for i in range(N):
        params = param_list[i % len(param_list)]
        seed = hash((model, N, trial, i)) % (2**31)
        try:
            if model == 'GJR_GARCH':
                r, _ = simulate_gjr_garch(T, **params, seed=seed)
            elif model == 'FIGARCH':
                r, _, _ = simulate_figarch(T, **params, seed=seed)
            else:
                continue
            r = r - np.mean(r)
            returns_stack.append(r)
        except Exception:
            continue

    if len(returns_stack) < max(2, N // 2):
        return None

    # Equal-weighted portfolio
    min_len = min(len(r) for r in returns_stack)
    port = np.mean([r[:min_len] for r in returns_stack], axis=0)
    port = port - np.mean(port)

    if len(port) < 2000:
        return None

    # Run decomposition
    try:
        result = decompose_acf(port, M=150, max_lag=500)
        w = result['weights']
        total = np.sum(w)
        ff = np.sum(w[HL_GRID < 10]) / max(total, 1e-15)
        width = result['diagnostics']['width_log']
        return {
            'model': model, 'N': N, 'trial': trial,
            'ff_10d': ff, 'width': width
        }
    except Exception:
        return None


def run_test1(n_workers, T=6600, n_trials=30):
    """Aggregated null portfolio simulations."""

    Ns = [1, 10, 25, 50, 100, 200, 285]

    # ---- GJR-GARCH with heterogeneous parameters ----
    # Constrained so P = a + gamma/2 + beta < 0.999 always.
    # beta in [0.85, 0.92], gamma in [0.03, 0.10], a in [0.01, 0.04]
    rng = np.random.default_rng(42)
    garch_params = []
    for i in range(285):
        beta = rng.uniform(0.85, 0.92)
        gamma = rng.uniform(0.03, 0.10)
        a = rng.uniform(0.01, 0.04)
        P = a + gamma / 2 + beta
        # Safety: if P >= 0.999, reduce beta
        while P >= 0.999:
            beta -= 0.01
            P = a + gamma / 2 + beta
        omega = 2.0 * (1 - P)
        garch_params.append({
            'omega': omega, 'a': a, 'gamma': gamma, 'beta': beta
        })

    # ---- FIGARCH with heterogeneous volatility levels ----
    # d fixed at 0.40 (same as Run 5, the standard calibration).
    # Heterogeneity through a and gamma (different volatility
    # levels across simulated stocks, same memory structure).
    # Enforce P = (a + gamma/2) * sum(g) < 0.95 for each draw.
    figarch_params = []
    # Pre-compute sum(g) for d=0.4
    _delta = np.zeros(J_MAX + 1)
    _delta[0] = 1.0
    for k in range(1, J_MAX + 1):
        _delta[k] = _delta[k-1] * (k - 1 - 0.4) / k
    _psi = np.zeros(J_MAX)
    _psi[0] = 0.2 - 0.5 + 0.4  # phi1 - beta1 + d
    for k in range(1, J_MAX):
        kk = k + 1
        _psi[k] = 0.5 * _psi[k-1] + (0.2 * _delta[kk-1] - _delta[kk])
    _psi = np.maximum(_psi, 0.0)
    _g_fig = _psi / max(_psi[0], 1e-10)
    _sum_g = np.sum(_g_fig)
    # Max (a + gamma/2) = 0.95 / sum_g
    _max_half_news = 0.95 / _sum_g  # ~ 0.1077

    for i in range(285):
        a_i = rng.uniform(0.03, 0.06)
        gamma_i = rng.uniform(0.04, 0.08)
        # Enforce stationarity
        while (a_i + gamma_i / 2) * _sum_g >= 0.95:
            a_i *= 0.9
            gamma_i *= 0.9
        figarch_params.append({
            'omega': 0.1, 'a': a_i, 'gamma': gamma_i,
            'phi1': 0.2, 'beta1': 0.5, 'd': 0.4
        })

    # Build task list
    args_list = []
    for model, param_list in [('GJR_GARCH', garch_params),
                               ('FIGARCH', figarch_params)]:
        for N in Ns:
            n_runs = 1 if N == 285 else n_trials
            if N == 1:
                # For N=1, just simulate 30 individual series
                for trial in range(n_trials):
                    args_list.append((model, 1, trial, T, param_list))
            else:
                for trial in range(n_runs):
                    args_list.append((model, N, trial, T, param_list))

    print(f"    Total tasks: {len(args_list)}")

    with Pool(processes=n_workers) as pool:
        out = pool.map(_simulate_null_portfolio, args_list)

    rows = [r for r in out if r is not None]
    return pd.DataFrame(rows)


# ==============================================================
#  TEST 2: CORRECTED N=1 BASELINE
# ==============================================================
#
#  The critic caught that Table 3 uses 323 stocks for N=1
#  but 285 for the portfolio exercise. We must recompute
#  the N=1 fast fraction on the SAME 285-stock pool.
# ==============================================================

def run_test2(df_raw, assets):
    """Recompute fast fraction for the 285-stock matched pool."""

    # Identify the 285-stock pool (same logic as Run 6)
    us_stocks = [tk for tk, d in assets.items()
                 if d['class'] == 'US Stock' and d['T'] >= 5000]

    # Build date-aligned matrix to find common-history stocks
    pivot = df_raw[df_raw['Ticker'].isin(us_stocks)].pivot_table(
        index='Date', columns='Ticker', values='LogRet')
    min_count = int(0.8 * len(us_stocks))
    pivot = pivot.dropna(axis=0, thresh=min_count)
    pool_285 = list(pivot.columns)

    print(f"    285-stock pool: {len(pool_285)} stocks")

    # Load mixing results if available
    mixing_csv = 'results_500/phase2_mixing/mixing_results.csv'
    if os.path.exists(mixing_csv):
        df_mix = pd.read_csv(mixing_csv)
        pool_mix = df_mix[df_mix['Ticker'].isin(pool_285)]

        result = {
            'n_pool': len(pool_285),
            'n_matched': len(pool_mix),
            'median_ff_pool285': pool_mix['fast_frac'].median(),
            'mean_ff_pool285': pool_mix['fast_frac'].mean(),
            'median_ff_all323': df_mix[df_mix['Class'] == 'US Stock']['fast_frac'].median(),
        }
        return result, pool_285
    else:
        print("    WARNING: mixing_results.csv not found")
        print("    Will compute from scratch for pool stocks")

        # Compute fast fraction for each stock in pool
        rows = []
        for tk in pool_285:
            if tk in assets:
                try:
                    r = assets[tk]['r']
                    res = decompose_acf(r, M=150, max_lag=500)
                    w = res['weights']
                    total = np.sum(w)
                    ff = np.sum(w[HL_GRID < 10]) / max(total, 1e-15)
                    rows.append({'Ticker': tk, 'fast_frac': ff})
                except Exception:
                    pass

        df_pool = pd.DataFrame(rows)
        result = {
            'n_pool': len(pool_285),
            'n_matched': len(df_pool),
            'median_ff_pool285': df_pool['fast_frac'].median(),
            'mean_ff_pool285': df_pool['fast_frac'].mean(),
            'median_ff_all323': np.nan,
        }
        return result, pool_285


# ==============================================================
#  TEST 3: LEVERAGE RATIO UNDER GJR-GARCH NULL
# ==============================================================
#
#  GJR-GARCH is asymmetric (gamma > 0). What leverage ratio
#  does it produce? If the null ratio is near our empirical
#  1.68, then our finding is not distinctive. If it is near
#  1.0, then 1.68 is a real empirical feature.
# ==============================================================

def _leverage_null_worker(args):
    """Simulate GJR-GARCH, compute leverage ratio."""
    seed, T, params = args
    try:
        r, _ = simulate_gjr_garch(T, **params, seed=seed)
        r = r - np.mean(r)
        T_len = len(r)

        abs_r = np.abs(r)
        neg_mask = (r < 0).astype(float)
        pos_mask = (r >= 0).astype(float)

        x_neg = abs_r * neg_mask
        x_pos = abs_r * pos_mask

        x_neg_dm = x_neg - np.mean(x_neg)
        x_pos_dm = x_pos - np.mean(x_pos)
        y_dm = abs_r - np.mean(abs_r)

        var_neg = np.mean(x_neg_dm**2)
        var_pos = np.mean(x_pos_dm**2)
        var_y = np.mean(y_dm**2)

        if var_neg < 1e-15 or var_pos < 1e-15 or var_y < 1e-15:
            return np.nan

        ccf_neg = np.mean(x_neg_dm[:T_len-1] * y_dm[1:]) / np.sqrt(var_neg * var_y)
        ccf_pos = np.mean(x_pos_dm[:T_len-1] * y_dm[1:]) / np.sqrt(var_pos * var_y)

        if ccf_pos < 1e-10:
            return np.nan

        return ccf_neg / ccf_pos
    except Exception:
        return np.nan


def run_test3(n_workers, n_sims=200, T=6600):
    """Leverage ratio under GJR-GARCH null."""

    # Same parameters as Run 5
    P = 0.03 + 0.07 / 2 + 0.91
    params = {'omega': 2.0 * (1 - P), 'a': 0.03, 'gamma': 0.07, 'beta': 0.91}
    args_list = [(seed, T, params) for seed in range(n_sims)]

    with Pool(processes=n_workers) as pool:
        ratios = pool.map(_leverage_null_worker, args_list)

    ratios = [r for r in ratios if np.isfinite(r)]
    return np.array(ratios)


# ==============================================================
#  REPORT
# ==============================================================

def print_report(rw, agg_df, baseline, lev_ratios):

    rw.p("=" * 72)
    rw.p("  RUN 7: FINAL PRE-SUBMISSION TESTS")
    rw.p("=" * 72)

    # ── TEST 1: AGGREGATED NULL PORTFOLIOS ──
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  TEST 1: AGGREGATED NULL PORTFOLIOS")
    rw.p(f"{'=' * 72}")

    if len(agg_df) > 0:
        for model in ['GJR_GARCH', 'FIGARCH']:
            sub = agg_df[agg_df['model'] == model]
            if len(sub) == 0:
                continue

            rw.p(f"\n  {model}:")
            rw.p(f"  {'N':>5s} {'Trials':>6s} {'Med FF':>8s} {'Mean FF':>8s} "
                 f"{'SD':>6s} {'Med Width':>10s}")
            rw.p(f"  {'~' * 48}")

            for N in sorted(sub['N'].unique()):
                s = sub[sub['N'] == N]
                ff = s['ff_10d'].dropna()
                w = s['width'].dropna()
                if len(ff) > 0:
                    sd_str = f"{ff.std():>6.3f}" if len(ff) > 1 else f"{'---':>6s}"
                    rw.p(f"  {N:>5d} {len(s):>6d} {ff.median():>8.3f} "
                         f"{ff.mean():>8.3f} {sd_str} {w.median():>10.3f}")

        # ── THE CRITICAL COMPARISON ──
        rw.p(f"\n  {'=' * 60}")
        rw.p(f"  CRITICAL COMPARISON: Empirical vs Null N-curves")
        rw.p(f"  {'=' * 60}")

        rw.p(f"\n  {'N':>5s} {'Empirical':>10s} {'GJR null':>10s} {'FIGARCH null':>13s}")
        rw.p(f"  {'~' * 40}")

        # Empirical values from Run 6
        empirical = {1: 0.311, 10: 0.086, 25: 0.004,
                     50: 0.000, 100: 0.000, 200: 0.000, 285: 0.000}

        for N in [1, 10, 25, 50, 100, 200, 285]:
            emp = empirical.get(N, np.nan)
            gjr_sub = agg_df[(agg_df['model'] == 'GJR_GARCH') & (agg_df['N'] == N)]
            fig_sub = agg_df[(agg_df['model'] == 'FIGARCH') & (agg_df['N'] == N)]

            gjr_ff = gjr_sub['ff_10d'].median() if len(gjr_sub) > 0 else np.nan
            fig_ff = fig_sub['ff_10d'].median() if len(fig_sub) > 0 else np.nan

            rw.p(f"  {N:>5d} {100*emp:>9.1f}% {100*gjr_ff:>9.1f}% {100*fig_ff:>12.1f}%")

        # Verdict
        gjr_n1 = agg_df[(agg_df['model'] == 'GJR_GARCH') & (agg_df['N'] == 1)]['ff_10d'].median()
        gjr_n285 = agg_df[(agg_df['model'] == 'GJR_GARCH') & (agg_df['N'] == 285)]['ff_10d'].median()
        fig_n1 = agg_df[(agg_df['model'] == 'FIGARCH') & (agg_df['N'] == 1)]['ff_10d'].median()
        fig_n285 = agg_df[(agg_df['model'] == 'FIGARCH') & (agg_df['N'] == 285)]['ff_10d'].median()

        gjr_drop = gjr_n1 - gjr_n285 if np.isfinite(gjr_n1) and np.isfinite(gjr_n285) else 0
        fig_drop = fig_n1 - fig_n285 if np.isfinite(fig_n1) and np.isfinite(fig_n285) else 0
        emp_drop = 0.311 - 0.000

        rw.p(f"\n  Empirical drop (N=1 to N=285): {100*emp_drop:.1f}pp")
        rw.p(f"  GJR-GARCH null drop:            {100*gjr_drop:.1f}pp")
        rw.p(f"  FIGARCH null drop:              {100*fig_drop:.1f}pp")

        if gjr_drop < emp_drop * 0.5 and fig_drop < emp_drop * 0.5:
            rw.p(f"\n  VERDICT: PASS — null portfolios do NOT reproduce the")
            rw.p(f"  empirical collapse. The matched-aggregation curve is a")
            rw.p(f"  distinctive feature of financial data.")
        elif gjr_drop > emp_drop * 0.8 or fig_drop > emp_drop * 0.8:
            rw.p(f"\n  VERDICT: FAIL — null portfolios reproduce the collapse.")
            rw.p(f"  The N-curve may be mechanical, not distinctive.")
        else:
            rw.p(f"\n  VERDICT: PARTIAL — null shows some attenuation but less")
            rw.p(f"  than the empirical curve. Discuss in the paper.")

    # ── TEST 2: CORRECTED N=1 BASELINE ──
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  TEST 2: CORRECTED N=1 BASELINE (285-stock pool)")
    rw.p(f"{'=' * 72}")

    if baseline:
        rw.p(f"  Pool size: {baseline['n_pool']}")
        rw.p(f"  Assets with mixing results: {baseline['n_matched']}")
        rw.p(f"  Median FF (285 pool):  {100*baseline['median_ff_pool285']:.1f}%")
        rw.p(f"  Mean FF (285 pool):    {100*baseline['mean_ff_pool285']:.1f}%")
        if np.isfinite(baseline.get('median_ff_all323', np.nan)):
            rw.p(f"  Median FF (all 323):   {100*baseline['median_ff_all323']:.1f}%")
        rw.p(f"\n  ACTION: Replace 31.1% with {100*baseline['median_ff_pool285']:.1f}% "
             f"everywhere in matched-aggregation discussion.")

    # ── TEST 3: LEVERAGE UNDER NULL ──
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  TEST 3: LEVERAGE RATIO UNDER GJR-GARCH NULL")
    rw.p(f"{'=' * 72}")

    if len(lev_ratios) > 0:
        rw.p(f"  N simulations: {len(lev_ratios)}")
        rw.p(f"  Median leverage ratio: {np.median(lev_ratios):.3f}")
        rw.p(f"  Mean leverage ratio:   {np.mean(lev_ratios):.3f}")
        rw.p(f"  SD:                    {np.std(lev_ratios):.3f}")
        rw.p(f"  [5%, 95%] CI:          [{np.percentile(lev_ratios, 5):.3f}, "
             f"{np.percentile(lev_ratios, 95):.3f}]")
        rw.p(f"  Empirical median:      1.680")
        rw.p(f"  Empirical > null 95th: {'YES' if 1.68 > np.percentile(lev_ratios, 95) else 'NO'}")

        if 1.68 > np.percentile(lev_ratios, 95):
            rw.p(f"\n  VERDICT: PASS — empirical leverage ratio (1.68) exceeds")
            rw.p(f"  the 95th percentile of the GJR-GARCH null distribution.")
            rw.p(f"  The cross-class leverage ordering is not explained by")
            rw.p(f"  a standard asymmetric GARCH model.")
        else:
            rw.p(f"\n  VERDICT: The empirical leverage ratio falls within the")
            rw.p(f"  null distribution. The finding may reflect standard")
            rw.p(f"  asymmetric GARCH dynamics.")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=OUTDIR)
    parser.add_argument('--n_trials', type=int, default=30)
    args = parser.parse_args()

    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    n_workers = args.workers or max(1, cpu_count() - 2)

    print(f"\n{'=' * 60}")
    print(f"  RUN 7: FINAL PRE-SUBMISSION TESTS")
    print(f"  Workers: {n_workers}")
    print(f"{'=' * 60}")

    t0_total = time.time()

    assets, df_raw = load_data(args.datafile)
    print(f"  Loaded {len(assets)} assets")

    # ── TEST 1: AGGREGATED NULL PORTFOLIOS ──
    print(f"\n  TEST 1: Aggregated null portfolios...")
    t0 = time.time()
    agg_df = run_test1(n_workers, T=6600, n_trials=args.n_trials)
    agg_df.to_csv(os.path.join(OUTDIR, 'aggregated_null.csv'),
                  index=False, float_format='%.4f')
    print(f"    Done ({time.time()-t0:.0f}s)")

    # ── TEST 2: CORRECTED BASELINE ──
    print(f"\n  TEST 2: Corrected N=1 baseline...")
    t0 = time.time()
    baseline, pool_285 = run_test2(df_raw, assets)
    print(f"    Done ({time.time()-t0:.0f}s)")

    # Save pool list
    with open(os.path.join(OUTDIR, 'pool_285_tickers.txt'), 'w') as f:
        for tk in sorted(pool_285):
            f.write(tk + '\n')

    # ── TEST 3: LEVERAGE NULL ──
    print(f"\n  TEST 3: Leverage ratio under GJR-GARCH null...")
    t0 = time.time()
    lev_ratios = run_test3(n_workers, n_sims=200, T=6600)
    pd.DataFrame({'leverage_ratio': lev_ratios}).to_csv(
        os.path.join(OUTDIR, 'leverage_null.csv'),
        index=False, float_format='%.4f')
    print(f"    Done ({time.time()-t0:.0f}s)")

    # ── REPORT ──
    rw = Report(os.path.join(OUTDIR, 'REPORT_FINAL.txt'))
    print_report(rw, agg_df, baseline, lev_ratios)
    rw.save()

    elapsed = time.time() - t0_total
    print(f"\n{'=' * 60}")
    print(f"  ALL 3 TESTS COMPLETE: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
