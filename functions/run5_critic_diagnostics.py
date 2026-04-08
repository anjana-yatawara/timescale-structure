"""
RUN 5: CRITIC DIAGNOSTICS
===========================
Five diagnostics addressing the adversarial audit:

  DIAG 1: Complete monotonicity check (~30 seconds)
  DIAG 2: Block bootstrap for fast fraction and width (~50 min)
  DIAG 3: Resolution analysis (~5 min)
  DIAG 4: Richer null zoo — EGARCH + FIGARCH (~5 min)
  DIAG 5: Clustered inference for class-level gradient (~10 sec)

Usage:
  python run5_critic_diagnostics.py volatility_memory_500_data.csv --workers 34

Expects in same directory:
  core_final.py, mixing_distribution.py, run2_bimodality.py

Output:
  results_500/phase5_critic/
    REPORT_CRITIC_DIAGNOSTICS.txt
    cm_check_results.csv
    bootstrap_results.csv
    resolution_results.csv
    null_zoo_results.csv
"""

import sys, os, time, io, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_final import simulate_gjr_garch, simulate_figarch, J_MAX
from mixing_distribution import (
    decompose_acf, make_rate_grid, rate_to_halflife,
    sample_acf_abs, build_design_matrix, first_difference_matrix,
    nnls_tikhonov, l_curve_select, _detect_bimodality, mixing_diagnostics
)

try:
    from run2_bimodality import extract_peaks, classify_peaks
    HAS_BIMODALITY = True
except ImportError:
    HAS_BIMODALITY = False

RATE_GRID = make_rate_grid(150)
HL_GRID = rate_to_halflife(RATE_GRID)
OUTDIR = 'results_500/phase5_critic'

# Minimum observations (same as master pipeline)
MIN_OBS = 2500
MIN_OBS_CRYPTO = 2000


class Report:
    """Simple report writer that prints AND saves to file."""
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
    """Load CSV with flexible column handling."""
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
    return assets


# ==============================================================
#  DIAGNOSTIC 1: COMPLETE MONOTONICITY CHECK
# ==============================================================

def check_complete_monotonicity(r, max_lag=300, noise_tol=0.005):
    """
    Check whether ACF of |r_t| is approximately completely monotone.

    Returns dict with fraction of lags satisfying each order (0-3).
    noise_tol: allow violations up to this magnitude (sampling noise).
    """
    acf = sample_acf_abs(r, max_lag)
    L = len(acf)

    results = {}

    # Order 0: ACF(l) >= -tol (positive)
    violations_0 = np.sum(acf < -noise_tol)
    results['order0_pass_frac'] = 1.0 - violations_0 / L
    results['order0_first_fail'] = int(np.argmax(acf < -noise_tol)) + 1 if violations_0 > 0 else L

    # Order 1: Delta^1 = ACF(l) - ACF(l+1) >= -tol (non-increasing)
    d1 = acf[:-1] - acf[1:]
    violations_1 = np.sum(d1 < -noise_tol)
    results['order1_pass_frac'] = 1.0 - violations_1 / len(d1)

    # Order 2: Delta^2 = ACF(l) - 2*ACF(l+1) + ACF(l+2) >= -tol (convex)
    d2 = acf[:-2] - 2 * acf[1:-1] + acf[2:]
    violations_2 = np.sum(d2 < -noise_tol)
    results['order2_pass_frac'] = 1.0 - violations_2 / len(d2)

    # Order 3: (-1)^3 * Delta^3 = -ACF(l) + 3*ACF(l+1) - 3*ACF(l+2) + ACF(l+3)
    d3 = -acf[:-3] + 3 * acf[1:-2] - 3 * acf[2:-1] + acf[3:]
    violations_3 = np.sum(d3 < -noise_tol)
    results['order3_pass_frac'] = 1.0 - violations_3 / len(d3)

    results['n_lags'] = L
    return results


def run_diag1(assets):
    """Diagnostic 1: Complete monotonicity for all assets."""
    rows = []
    for tk in assets:
        r = assets[tk]['r']
        cm = check_complete_monotonicity(r, max_lag=300)
        cm['Ticker'] = tk
        cm['Class'] = assets[tk]['class']
        rows.append(cm)
    return pd.DataFrame(rows)


# ==============================================================
#  DIAGNOSTIC 2: BLOCK BOOTSTRAP
# ==============================================================

def block_bootstrap_resample(r, block_len=50, rng=None):
    """
    Draw one block bootstrap resample of the return series.

    Splits r into overlapping blocks of length block_len,
    samples blocks with replacement until we reach len(r).
    """
    if rng is None:
        rng = np.random.default_rng()
    T = len(r)
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T - block_len + 1, size=n_blocks)
    resampled = np.concatenate([r[s:s + block_len] for s in starts])
    resampled = resampled[:T]
    resampled = resampled - np.mean(resampled)
    return resampled


def _bootstrap_one(args):
    """Worker: one bootstrap resample for one asset."""
    tk, r, boot_seed, block_len = args
    rng = np.random.default_rng(boot_seed)
    try:
        r_boot = block_bootstrap_resample(r, block_len=block_len, rng=rng)
        result = decompose_acf(r_boot, M=150, max_lag=400)
        w = result['weights']
        total = np.sum(w)
        fast_frac = np.sum(w[HL_GRID < 10]) / max(total, 1e-15)
        width = result['diagnostics']['width_log']
        is_bimodal = result['diagnostics']['is_bimodal']
        dom_hl = result['diagnostics']['dominant_hl']
        return tk, fast_frac, width, is_bimodal, dom_hl
    except Exception:
        return tk, np.nan, np.nan, False, np.nan


def run_diag2(assets, n_workers, n_boot=200, block_len=50):
    """Diagnostic 2: Block bootstrap for 50 representative assets."""

    # Select 50 representative assets stratified by class
    selected = []
    class_counts = {}
    for tk in sorted(assets.keys(), key=lambda t: -assets[t]['T']):
        cls = assets[tk]['class']
        if cls not in class_counts:
            class_counts[cls] = 0
        max_per = 8 if cls == 'US Stock' else 5
        if class_counts[cls] < max_per:
            selected.append(tk)
            class_counts[cls] += 1
        if len(selected) >= 50:
            break

    print(f"    Selected {len(selected)} assets for bootstrap")
    for cls in sorted(class_counts.keys()):
        print(f"      {cls}: {class_counts[cls]}")

    # Build worker arguments
    args_list = []
    for tk in selected:
        for b in range(n_boot):
            seed = hash((tk, b)) % (2**31)
            args_list.append((tk, assets[tk]['r'], seed, block_len))

    with Pool(processes=n_workers) as pool:
        out = pool.map(_bootstrap_one, args_list)

    # Collect results by asset
    boot_data = {}
    for tk, ff, width, bimod, dom_hl in out:
        if tk not in boot_data:
            boot_data[tk] = {'ff': [], 'width': [], 'bimod': [], 'dom_hl': []}
        if not np.isnan(ff):
            boot_data[tk]['ff'].append(ff)
            boot_data[tk]['width'].append(width)
            boot_data[tk]['bimod'].append(bimod)
            boot_data[tk]['dom_hl'].append(dom_hl)

    # Summarize
    rows = []
    for tk in selected:
        if tk not in boot_data or len(boot_data[tk]['ff']) < 10:
            continue
        ff_arr = np.array(boot_data[tk]['ff'])
        w_arr = np.array(boot_data[tk]['width'])
        bimod_arr = np.array(boot_data[tk]['bimod'])

        rows.append({
            'Ticker': tk,
            'Class': assets[tk]['class'],
            'n_boot': len(ff_arr),
            'ff_mean': np.mean(ff_arr),
            'ff_se': np.std(ff_arr),
            'ff_ci_lo': np.percentile(ff_arr, 5),
            'ff_ci_hi': np.percentile(ff_arr, 95),
            'width_mean': np.mean(w_arr),
            'width_se': np.std(w_arr),
            'width_ci_lo': np.percentile(w_arr, 5),
            'width_ci_hi': np.percentile(w_arr, 95),
            'bimod_freq': np.mean(bimod_arr),
        })

    return pd.DataFrame(rows)


# ==============================================================
#  DIAGNOSTIC 3: RESOLUTION ANALYSIS
# ==============================================================

def _resolution_one(args):
    """Worker: one simulation for resolution analysis."""
    separation, trial_seed, noise_sigma, w_fast = args
    rng = np.random.default_rng(trial_seed)

    u2 = 0.005  # slow rate fixed (HL = 139 days)
    u1 = u2 * separation
    max_lag = 400
    lags = np.arange(1, max_lag + 1, dtype=float)
    acf_true = w_fast * np.exp(-u1 * lags) + (1 - w_fast) * np.exp(-u2 * lags)

    acf_noisy = acf_true + rng.normal(0, noise_sigma, size=max_lag)

    first_neg = np.argmax(acf_noisy <= 0)
    if first_neg > 10:
        acf_noisy = acf_noisy[:first_neg]
    if len(acf_noisy) < 20:
        return separation, trial_seed, False, False, np.nan, np.nan

    lags_used = np.arange(1, len(acf_noisy) + 1, dtype=float)
    A = build_design_matrix(lags_used, RATE_GRID)
    L_reg = first_difference_matrix(len(RATE_GRID))

    try:
        best_lam, w_rec, _ = l_curve_select(A, acf_noisy, L_reg, n_lambda=25)
    except Exception:
        return separation, trial_seed, False, False, np.nan, np.nan

    is_bimodal = _detect_bimodality(w_rec, RATE_GRID)

    has_fast_slow = False
    recovered_fast_hl = np.nan
    recovered_slow_hl = np.nan
    if HAS_BIMODALITY:
        peaks = extract_peaks(w_rec, RATE_GRID)
        cl = classify_peaks(peaks)
        has_fast_slow = cl['has_fast_and_slow']
        recovered_fast_hl = cl['fastest_hl']
        recovered_slow_hl = cl['slowest_hl']

    return separation, trial_seed, is_bimodal, has_fast_slow, recovered_fast_hl, recovered_slow_hl


def run_diag3(n_workers, n_trials=100):
    """Diagnostic 3: Resolution analysis with simulated 2-component mixtures."""

    separations = [3, 5, 10, 20, 50, 100, 200, 500]
    noise_sigma = 0.015
    w_fast = 0.3

    args_list = []
    for sep in separations:
        for trial in range(n_trials):
            seed = sep * 10000 + trial
            args_list.append((sep, seed, noise_sigma, w_fast))

    with Pool(processes=n_workers) as pool:
        out = pool.map(_resolution_one, args_list)

    rows = []
    for sep in separations:
        trials = [o for o in out if o[0] == sep]
        n = len(trials)
        n_bimodal = sum(1 for o in trials if o[2])
        n_fast_slow = sum(1 for o in trials if o[3])

        fast_hls = [o[4] for o in trials if o[3] and np.isfinite(o[4])]
        slow_hls = [o[5] for o in trials if o[3] and np.isfinite(o[5])]

        true_fast_hl = np.log(2) / (0.005 * sep)
        true_slow_hl = np.log(2) / 0.005

        rows.append({
            'separation': sep,
            'n_trials': n,
            'bimodal_rate': n_bimodal / max(n, 1),
            'fast_slow_rate': n_fast_slow / max(n, 1),
            'true_fast_hl': true_fast_hl,
            'true_slow_hl': true_slow_hl,
            'med_recovered_fast_hl': np.median(fast_hls) if fast_hls else np.nan,
            'med_recovered_slow_hl': np.median(slow_hls) if slow_hls else np.nan,
        })

    return pd.DataFrame(rows)


# ==============================================================
#  DIAGNOSTIC 4: RICHER NULL ZOO
# ==============================================================

def simulate_egarch(T, omega, alpha, gamma, beta, seed=None):
    """
    Simulate from EGARCH(1,1) of Nelson (1991).

    log(sigma2_t) = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1}
                    + beta*log(sigma2_{t-1})
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(T)
    E_abs_z = np.sqrt(2.0 / np.pi)

    r = np.zeros(T)
    log_sigma2 = np.zeros(T)

    log_sigma2[0] = omega / (1 - beta)
    r[0] = np.exp(0.5 * log_sigma2[0]) * z[0]

    for t in range(1, T):
        log_sigma2[t] = (omega
                         + alpha * (np.abs(z[t-1]) - E_abs_z)
                         + gamma * z[t-1]
                         + beta * log_sigma2[t-1])
        sigma_t = np.exp(0.5 * log_sigma2[t])
        sigma_t = max(sigma_t, 1e-6)
        r[t] = sigma_t * z[t]
        z[t] = r[t] / sigma_t

    return r


def _null_zoo_worker(args):
    """Worker: simulate one null series, run decomposition."""
    model, seed, params = args
    T = 6600

    try:
        if model == 'EGARCH':
            r = simulate_egarch(T, **params, seed=seed)
        elif model == 'FIGARCH':
            r, _, _ = simulate_figarch(T, **params, seed=seed)
        elif model == 'GJR_GARCH':
            r, _ = simulate_gjr_garch(T, **params, seed=seed)
        else:
            return model, seed, {}

        r = r - np.mean(r)

        result = decompose_acf(r, M=150, max_lag=500)
        w = result['weights']
        total = np.sum(w)
        fast_frac = np.sum(w[HL_GRID < 10]) / max(total, 1e-15)
        is_bimodal = result['diagnostics']['is_bimodal']
        width = result['diagnostics']['width_log']

        has_fast_slow = False
        if HAS_BIMODALITY:
            peaks = extract_peaks(w, RATE_GRID)
            cl = classify_peaks(peaks)
            has_fast_slow = cl['has_fast_and_slow']

        return model, seed, {
            'bimodal': is_bimodal,
            'fast_slow': has_fast_slow,
            'fast_frac': fast_frac,
            'width': width,
        }
    except Exception as e:
        return model, seed, {'bimodal': False, 'fast_slow': False,
                             'fast_frac': 0, 'width': 0, 'error': str(e)}


def run_diag4(n_workers, n_sims=200):
    """Diagnostic 4: False positive rates under EGARCH and FIGARCH nulls."""

    args_list = []

    # (a) EGARCH
    for seed in range(n_sims):
        args_list.append(('EGARCH', seed, {
            'omega': -0.1, 'alpha': 0.1, 'gamma': -0.05, 'beta': 0.98
        }))

    # (b) FIGARCH
    for seed in range(n_sims):
        args_list.append(('FIGARCH', seed + 10000, {
            'omega': 0.1, 'a': 0.05, 'gamma': 0.07,
            'phi1': 0.2, 'beta1': 0.5, 'd': 0.4
        }))

    # (c) GJR-GARCH reference
    for seed in range(n_sims):
        P = 0.03 + 0.07/2 + 0.91
        args_list.append(('GJR_GARCH', seed + 20000, {
            'omega': 2.0 * (1 - P), 'a': 0.03, 'gamma': 0.07, 'beta': 0.91
        }))

    with Pool(processes=n_workers) as pool:
        out = pool.map(_null_zoo_worker, args_list)

    summary = {}
    for model in ['GJR_GARCH', 'EGARCH', 'FIGARCH']:
        trials = [o[2] for o in out if o[0] == model and 'bimodal' in o[2]]
        n = len(trials)
        if n == 0:
            continue
        summary[model] = {
            'n': n,
            'bimodal_rate': sum(t['bimodal'] for t in trials) / n,
            'fast_slow_rate': sum(t['fast_slow'] for t in trials) / n,
            'mean_fast_frac': np.mean([t['fast_frac'] for t in trials]),
            'mean_width': np.mean([t['width'] for t in trials]),
            'n_errors': sum(1 for t in trials if 'error' in t),
        }

    return summary


# ==============================================================
#  DIAGNOSTIC 5: CLUSTERED INFERENCE
# ==============================================================

def run_diag5(mixing_csv_path):
    """Diagnostic 5: Clustered bootstrap + Wilcoxon test for gradient."""

    df = pd.read_csv(mixing_csv_path)
    results = {}

    # (a) Cluster bootstrap
    n_boot = 2000
    rng = np.random.default_rng(42)

    gradient_classes = ['US Stock', 'Factor ETF', 'Sector ETF', 'Thematic ETF', 'US Index']
    boot_medians = {cls: [] for cls in gradient_classes}

    for _ in range(n_boot):
        for cls in gradient_classes:
            sub = df[df['Class'] == cls]['fast_frac'].values
            if len(sub) == 0:
                boot_medians[cls].append(np.nan)
                continue
            resample = rng.choice(sub, size=len(sub), replace=True)
            boot_medians[cls].append(np.median(resample))

    results['bootstrap'] = {}
    for cls in gradient_classes:
        arr = np.array(boot_medians[cls])
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            results['bootstrap'][cls] = {
                'mean': np.mean(arr),
                'se': np.std(arr),
                'ci_lo': np.percentile(arr, 2.5),
                'ci_hi': np.percentile(arr, 97.5),
                'n_assets': len(df[df['Class'] == cls]),
            }

    # (b) Wilcoxon rank-sum test
    from scipy.stats import mannwhitneyu, kruskal

    stocks_ff = df[df['Class'] == 'US Stock']['fast_frac'].values
    indices_ff = df[df['Class'] == 'US Index']['fast_frac'].values

    if len(stocks_ff) > 0 and len(indices_ff) > 0:
        stat, p_wilcox = mannwhitneyu(stocks_ff, indices_ff, alternative='greater')
        results['wilcoxon_stocks_vs_indices'] = {
            'statistic': stat, 'p_value': p_wilcox,
            'n_stocks': len(stocks_ff), 'n_indices': len(indices_ff),
        }

    # (c) Kruskal-Wallis
    groups = [df[df['Class'] == cls]['fast_frac'].dropna().values
              for cls in gradient_classes if len(df[df['Class'] == cls]) > 0]
    if len(groups) >= 2:
        stat_kw, p_kw = kruskal(*groups)
        results['kruskal_wallis'] = {'statistic': stat_kw, 'p_value': p_kw}

    # (d) Permutation test for monotonicity
    n_perm = 10000
    rng_perm = np.random.default_rng(123)
    observed_gradient = [df[df['Class'] == cls]['fast_frac'].median()
                         for cls in gradient_classes
                         if len(df[df['Class'] == cls]) > 0]
    observed_range = max(observed_gradient) - min(observed_gradient)

    all_ff = df[df['Class'].isin(gradient_classes)]['fast_frac'].values
    all_cls = df[df['Class'].isin(gradient_classes)]['Class'].values
    class_sizes = [np.sum(all_cls == cls) for cls in gradient_classes if np.sum(all_cls == cls) > 0]

    n_exceed = 0
    for _ in range(n_perm):
        shuffled = rng_perm.permutation(all_ff)
        idx = 0
        medians = []
        for sz in class_sizes:
            medians.append(np.median(shuffled[idx:idx+sz]))
            idx += sz
        perm_range = max(medians) - min(medians)
        if perm_range >= observed_range:
            n_exceed += 1

    results['permutation_test'] = {
        'observed_range': observed_range,
        'p_value': n_exceed / n_perm,
        'n_permutations': n_perm,
    }

    return results


# ==============================================================
#  REPORT
# ==============================================================

def print_report(rw, cm_df, boot_df, res_df, null_zoo, clustered):

    rw.p("=" * 72)
    rw.p("  CRITIC DIAGNOSTICS: FIVE TESTS")
    rw.p("=" * 72)

    # DIAGNOSTIC 1
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  DIAGNOSTIC 1: COMPLETE MONOTONICITY CHECK")
    rw.p(f"{'=' * 72}")

    n = len(cm_df)
    for order in range(4):
        col = f'order{order}_pass_frac'
        med = cm_df[col].median()
        pct_good = (cm_df[col] > 0.90).sum()
        rw.p(f"  Order {order}: median pass fraction = {med:.3f}, "
             f"assets with >90% pass = {pct_good}/{n} ({100*pct_good/n:.0f}%)")

    rw.p(f"\n  By class (median pass fraction per order):")
    rw.p(f"  {'Class':<20s} {'N':>3s} {'Ord0':>6s} {'Ord1':>6s} {'Ord2':>6s} {'Ord3':>6s}")
    rw.p(f"  {'~'*45}")
    for cls in sorted(cm_df['Class'].unique()):
        sub = cm_df[cm_df['Class'] == cls]
        rw.p(f"  {cls[:19]:<20s} {len(sub):>3d} "
             f"{sub['order0_pass_frac'].median():>6.3f} "
             f"{sub['order1_pass_frac'].median():>6.3f} "
             f"{sub['order2_pass_frac'].median():>6.3f} "
             f"{sub['order3_pass_frac'].median():>6.3f}")

    cm_pass = cm_df['order2_pass_frac'].median() > 0.85
    rw.p(f"\n  VERDICT: {'PASS' if cm_pass else 'CONCERN'} "
         f"(order-2 median = {cm_df['order2_pass_frac'].median():.3f}, "
         f"threshold = 0.85)")

    # DIAGNOSTIC 2
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  DIAGNOSTIC 2: BLOCK BOOTSTRAP (fast fraction and width)")
    rw.p(f"{'=' * 72}")

    if len(boot_df) > 0:
        rw.p(f"  {len(boot_df)} assets, {boot_df['n_boot'].iloc[0]} resamples each")
        rw.p(f"\n  {'Ticker':<8s} {'Class':<15s} {'FF mean':>8s} {'FF SE':>6s} "
             f"{'FF 90%CI':>14s} {'W mean':>7s} {'W SE':>6s} {'Bim%':>5s}")
        rw.p(f"  {'~'*70}")

        for _, row in boot_df.sort_values(['Class', 'Ticker']).iterrows():
            rw.p(f"  {row['Ticker']:<8s} {str(row['Class'])[:14]:<15s} "
                 f"{row['ff_mean']:>8.3f} {row['ff_se']:>6.3f} "
                 f"[{row['ff_ci_lo']:>5.3f},{row['ff_ci_hi']:>5.3f}] "
                 f"{row['width_mean']:>7.3f} {row['width_se']:>6.3f} "
                 f"{100*row['bimod_freq']:>4.0f}%")

        rw.p(f"\n  CLASS-LEVEL BOOTSTRAP SUMMARY:")
        rw.p(f"  {'Class':<20s} {'N':>3s} {'FF mean':>8s} {'FF SE':>6s} {'FF 90%CI':>16s}")
        rw.p(f"  {'~'*55}")
        for cls in ['US Stock', 'Sector ETF', 'US Index', 'Commodity', 'Cryptocurrency']:
            sub = boot_df[boot_df['Class'] == cls]
            if len(sub) > 0:
                rw.p(f"  {cls[:19]:<20s} {len(sub):>3d} "
                     f"{sub['ff_mean'].median():>8.3f} {sub['ff_se'].median():>6.3f} "
                     f"[{sub['ff_ci_lo'].median():>5.3f},{sub['ff_ci_hi'].median():>5.3f}]")

    # DIAGNOSTIC 3
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  DIAGNOSTIC 3: RESOLUTION ANALYSIS")
    rw.p(f"{'=' * 72}")

    if len(res_df) > 0:
        rw.p(f"  {'Sep':>5s} {'True fast':>10s} {'True slow':>10s} "
             f"{'Bimod%':>7s} {'F+S%':>6s} {'Rec fast':>9s} {'Rec slow':>9s}")
        rw.p(f"  {'~'*58}")
        for _, row in res_df.iterrows():
            rw.p(f"  {row['separation']:>5.0f}x "
                 f"{row['true_fast_hl']:>9.1f}d {row['true_slow_hl']:>9.1f}d "
                 f"{100*row['bimodal_rate']:>6.1f}% {100*row['fast_slow_rate']:>5.1f}% "
                 f"{row['med_recovered_fast_hl']:>8.1f}d {row['med_recovered_slow_hl']:>8.1f}d")

        for _, row in res_df.iterrows():
            if row['fast_slow_rate'] >= 0.70:
                rw.p(f"\n  RESOLUTION LIMIT: {row['separation']:.0f}x "
                     f"(first separation with F+S detection >= 70%)")
                rw.p(f"  Our empirical separation: 172x")
                rw.p(f"  VERDICT: {'PASS' if row['separation'] <= 172 else 'CONCERN'}")
                break

    # DIAGNOSTIC 4
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  DIAGNOSTIC 4: NULL ZOO (EGARCH + FIGARCH + GJR-GARCH)")
    rw.p(f"{'=' * 72}")

    if null_zoo:
        rw.p(f"  {'Model':<15s} {'N':>4s} {'Bimod%':>7s} {'F+S%':>6s} "
             f"{'Mean FF':>8s} {'Mean W':>7s} {'Errors':>6s}")
        rw.p(f"  {'~'*55}")
        for model in ['GJR_GARCH', 'EGARCH', 'FIGARCH']:
            if model in null_zoo:
                s = null_zoo[model]
                rw.p(f"  {model:<15s} {s['n']:>4d} "
                     f"{100*s['bimodal_rate']:>6.1f}% {100*s['fast_slow_rate']:>5.1f}% "
                     f"{100*s['mean_fast_frac']:>7.1f}% {s['mean_width']:>7.3f} "
                     f"{s['n_errors']:>6d}")

        if 'FIGARCH' in null_zoo:
            fig_rate = null_zoo['FIGARCH']['bimodal_rate']
            rw.p(f"\n  CRITICAL: FIGARCH bimodal false positive = {100*fig_rate:.1f}%")
            if fig_rate < 0.20:
                rw.p(f"  VERDICT: PASS (detector distinguishes smooth long memory from bimodal)")
            elif fig_rate < 0.30:
                rw.p(f"  VERDICT: MARGINAL (similar to GJR-GARCH null)")
            else:
                rw.p(f"  VERDICT: CONCERN (detector cannot distinguish FIGARCH from bimodal)")

    # DIAGNOSTIC 5
    rw.p(f"\n{'=' * 72}")
    rw.p(f"  DIAGNOSTIC 5: CLUSTERED INFERENCE FOR GRADIENT")
    rw.p(f"{'=' * 72}")

    if clustered:
        if 'bootstrap' in clustered:
            rw.p(f"\n  Cluster bootstrap (2000 resamples):")
            rw.p(f"  {'Class':<20s} {'N':>3s} {'Median FF':>10s} {'SE':>6s} {'95% CI':>18s}")
            rw.p(f"  {'~'*58}")
            for cls in ['US Stock', 'Factor ETF', 'Sector ETF', 'Thematic ETF', 'US Index']:
                if cls in clustered['bootstrap']:
                    b = clustered['bootstrap'][cls]
                    rw.p(f"  {cls:<20s} {b['n_assets']:>3d} "
                         f"{100*b['mean']:>9.1f}% {100*b['se']:>5.1f}% "
                         f"[{100*b['ci_lo']:>5.1f}%, {100*b['ci_hi']:>5.1f}%]")

        if 'wilcoxon_stocks_vs_indices' in clustered:
            w = clustered['wilcoxon_stocks_vs_indices']
            rw.p(f"\n  Wilcoxon rank-sum (stocks > indices): "
                 f"p = {w['p_value']:.2e} "
                 f"(N_stocks={w['n_stocks']}, N_indices={w['n_indices']})")

        if 'kruskal_wallis' in clustered:
            kw = clustered['kruskal_wallis']
            rw.p(f"  Kruskal-Wallis across 5 classes: "
                 f"p = {kw['p_value']:.2e}")

        if 'permutation_test' in clustered:
            pt = clustered['permutation_test']
            rw.p(f"  Permutation test for gradient range: "
                 f"p = {pt['p_value']:.4f} ({pt['n_permutations']} permutations)")
            rw.p(f"  Observed range: {100*pt['observed_range']:.1f}pp")

        if 'wilcoxon_stocks_vs_indices' in clustered:
            p = clustered['wilcoxon_stocks_vs_indices']['p_value']
            rw.p(f"\n  VERDICT: {'PASS' if p < 0.001 else 'CONCERN'} "
                 f"(Wilcoxon p = {p:.2e})")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=OUTDIR)
    parser.add_argument('--n_boot', type=int, default=200)
    parser.add_argument('--mixing_csv', type=str,
                        default='results_500/phase2_mixing/mixing_results.csv')
    args = parser.parse_args()

    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    n_workers = args.workers or max(1, cpu_count() - 2)

    print(f"\n{'=' * 60}")
    print(f"  RUN 5: CRITIC DIAGNOSTICS")
    print(f"  Workers: {n_workers}")
    print(f"{'=' * 60}")

    t0_total = time.time()

    assets = load_data(args.datafile)
    print(f"  Loaded {len(assets)} assets")

    # DIAGNOSTIC 1
    print(f"\n  DIAGNOSTIC 1: Complete monotonicity...")
    t0 = time.time()
    cm_df = run_diag1(assets)
    cm_df.to_csv(os.path.join(OUTDIR, 'cm_check_results.csv'),
                 index=False, float_format='%.4f')
    print(f"    Done ({time.time()-t0:.0f}s)")

    # DIAGNOSTIC 2
    print(f"\n  DIAGNOSTIC 2: Block bootstrap (n_boot={args.n_boot})...")
    t0 = time.time()
    boot_df = run_diag2(assets, n_workers, n_boot=args.n_boot, block_len=50)
    boot_df.to_csv(os.path.join(OUTDIR, 'bootstrap_results.csv'),
                   index=False, float_format='%.4f')
    print(f"    Done ({time.time()-t0:.0f}s)")

    # DIAGNOSTIC 3
    print(f"\n  DIAGNOSTIC 3: Resolution analysis...")
    t0 = time.time()
    res_df = run_diag3(n_workers, n_trials=100)
    res_df.to_csv(os.path.join(OUTDIR, 'resolution_results.csv'),
                  index=False, float_format='%.4f')
    print(f"    Done ({time.time()-t0:.0f}s)")

    # DIAGNOSTIC 4
    print(f"\n  DIAGNOSTIC 4: Null zoo (EGARCH + FIGARCH + GJR)...")
    t0 = time.time()
    null_zoo = run_diag4(n_workers, n_sims=200)
    print(f"    Done ({time.time()-t0:.0f}s)")

    # DIAGNOSTIC 5
    print(f"\n  DIAGNOSTIC 5: Clustered inference...")
    t0 = time.time()
    clustered = run_diag5(args.mixing_csv)
    print(f"    Done ({time.time()-t0:.0f}s)")

    # REPORT
    rw = Report(os.path.join(OUTDIR, 'REPORT_CRITIC_DIAGNOSTICS.txt'))
    print_report(rw, cm_df, boot_df, res_df, null_zoo, clustered)
    rw.save()

    elapsed = time.time() - t0_total
    print(f"\n{'=' * 60}")
    print(f"  ALL 5 DIAGNOSTICS COMPLETE: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
