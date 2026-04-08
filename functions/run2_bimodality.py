"""
RUN 2: BIMODALITY PEAK EXTRACTION & ROBUSTNESS
================================================
Reads run1 outputs. For every bimodal asset:
  - Extracts peak locations (half-lives) and relative weights
  - Tests whether bimodality survives across 5 regularization strengths
  - Tabulates fast-peak vs slow-peak clustering

Usage:
  python run2_bimodality.py --run1dir results/mixing --workers 34

Expects in run1dir:
  mixing_results.csv
  mixing_weights.npz
Also expects shape_of_memory_100.csv in working directory.
"""

import sys, os, time, io, argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixing_distribution import (
    make_rate_grid, build_design_matrix, first_difference_matrix,
    nnls_tikhonov, sample_acf_abs, mixing_diagnostics,
    rate_to_halflife, _detect_bimodality
)

OUTDIR = 'results/bimodality'
RATE_GRID = make_rate_grid(150)
HL_GRID = rate_to_halflife(RATE_GRID)


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


# ==============================================================
#  PEAK EXTRACTION
# ==============================================================

def extract_peaks(w, rates, min_prominence=0.05, smooth_sigma=2):
    """
    Find peaks in the mixing distribution.
    
    Returns list of dicts with:
      rate, halflife, weight (absolute), weight_frac (fraction of total),
      peak_idx, left_idx, right_idx
    """
    if np.max(w) < 1e-15:
        return []
    
    total = np.sum(w)
    w_norm = w / total
    
    # Smooth
    w_smooth = gaussian_filter1d(w_norm, sigma=smooth_sigma)
    
    # Find local maxima
    peaks = []
    for i in range(1, len(w_smooth) - 1):
        if w_smooth[i] > w_smooth[i-1] and w_smooth[i] > w_smooth[i+1]:
            if w_smooth[i] > min_prominence * np.max(w_smooth):
                peaks.append(i)
    
    # Also check endpoints
    if len(w_smooth) > 2:
        if w_smooth[0] > w_smooth[1] and w_smooth[0] > min_prominence * np.max(w_smooth):
            peaks.insert(0, 0)
        if w_smooth[-1] > w_smooth[-2] and w_smooth[-1] > min_prominence * np.max(w_smooth):
            peaks.append(len(w_smooth) - 1)
    
    if len(peaks) == 0:
        # Single dominant component
        dom = np.argmax(w_norm)
        return [{
            'rate': rates[dom],
            'halflife': np.log(2) / rates[dom],
            'weight': w[dom],
            'weight_frac': w_norm[dom],
            'peak_idx': dom,
            'height': w_smooth[dom],
        }]
    
    # Sort by height (descending)
    peaks = sorted(peaks, key=lambda i: -w_smooth[i])
    
    result = []
    for idx in peaks:
        # Compute weight in the peak's neighborhood (±5 grid points)
        lo = max(0, idx - 5)
        hi = min(len(w), idx + 6)
        peak_weight = np.sum(w_norm[lo:hi])
        
        result.append({
            'rate': rates[idx],
            'halflife': np.log(2) / rates[idx],
            'weight': w[idx],
            'weight_frac': peak_weight,
            'peak_idx': idx,
            'height': w_smooth[idx],
        })
    
    return result


def classify_peaks(peaks):
    """
    Classify peaks into fast/slow categories.
    
    Fast: HL < 10 days (u > 0.069)
    Medium: 10 ≤ HL < 50 days
    Slow: HL ≥ 50 days (u < 0.0139)
    """
    fast = [p for p in peaks if p['halflife'] < 10]
    medium = [p for p in peaks if 10 <= p['halflife'] < 50]
    slow = [p for p in peaks if p['halflife'] >= 50]
    
    return {
        'n_peaks': len(peaks),
        'n_fast': len(fast),
        'n_medium': len(medium),
        'n_slow': len(slow),
        'fast_peaks': fast,
        'medium_peaks': medium,
        'slow_peaks': slow,
        'has_fast_and_slow': len(fast) > 0 and len(slow) > 0,
        'fastest_hl': min(p['halflife'] for p in peaks) if peaks else np.nan,
        'slowest_hl': max(p['halflife'] for p in peaks) if peaks else np.nan,
        'dominant_hl': peaks[0]['halflife'] if peaks else np.nan,
        'dominant_frac': peaks[0]['weight_frac'] if peaks else np.nan,
    }


# ==============================================================
#  MULTI-LAMBDA ROBUSTNESS
# ==============================================================

def decompose_at_lambda(acf, lam, rates=None, L_reg=None):
    """Run NNLS decomposition at a specific λ value."""
    if rates is None:
        rates = RATE_GRID
    L_acf = len(acf)
    lags = np.arange(1, L_acf + 1, dtype=float)
    A = build_design_matrix(lags, rates)
    M = len(rates)
    if L_reg is None:
        L_reg = first_difference_matrix(M)
    
    w, dr, rn = nnls_tikhonov(A, acf, lam, L_reg)
    
    # Reconstruction R²
    b_hat = A @ w
    ss_res = np.sum((acf - b_hat)**2)
    ss_tot = np.sum((acf - np.mean(acf))**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return w, R2


def robustness_check(acf, lambda_center, n_lambdas=7):
    """
    Run decomposition at multiple λ values around the L-curve selection.
    Check which features survive.
    """
    # Span 2 orders of magnitude around center
    lambdas = np.logspace(
        np.log10(lambda_center) - 1,
        np.log10(lambda_center) + 1,
        n_lambdas
    )
    
    M = len(RATE_GRID)
    L_reg = first_difference_matrix(M)
    
    results = []
    for lam in lambdas:
        w, R2 = decompose_at_lambda(acf, lam, RATE_GRID, L_reg)
        peaks = extract_peaks(w, RATE_GRID)
        classification = classify_peaks(peaks)
        is_bimodal = _detect_bimodality(w, RATE_GRID)
        
        results.append({
            'lambda': lam,
            'R2': R2,
            'n_peaks': classification['n_peaks'],
            'is_bimodal': is_bimodal,
            'has_fast_and_slow': classification['has_fast_and_slow'],
            'fastest_hl': classification['fastest_hl'],
            'slowest_hl': classification['slowest_hl'],
            'dominant_hl': classification['dominant_hl'],
            'weights': w,
        })
    
    # Count how many λ values give bimodality
    n_bimodal = sum(r['is_bimodal'] for r in results)
    n_fast_slow = sum(r['has_fast_and_slow'] for r in results)
    
    return {
        'lambdas': lambdas,
        'results': results,
        'n_bimodal': n_bimodal,
        'n_fast_slow': n_fast_slow,
        'robust_bimodal': n_bimodal >= 4,  # majority rule
        'robust_fast_slow': n_fast_slow >= 4,
    }


# ==============================================================
#  WORKER
# ==============================================================

def load_data(fp):
    df = pd.read_csv(fp)
    df['Date'] = pd.to_datetime(df['Date'])
    assets = {}
    for tk in df['Ticker'].unique():
        sub = df[df['Ticker'] == tk].sort_values('Date')
        r = sub['LogRet'].values
        r = r - np.mean(r)
        cls = sub['Class'].iloc[0] if 'Class' in sub.columns else 'Unknown'
        min_obs = 2000 if cls == 'Crypto' else 2500
        if len(r) >= min_obs:
            assets[tk] = {'r': r, 'class': cls}
    return assets


def _worker(args):
    tk, r, acf_weights, acf_lambda, meta = args
    t0 = time.time()
    
    try:
        # Step 1: Compute ACF
        acf = sample_acf_abs(r, max_lag=500)
        first_neg = np.argmax(acf <= 0)
        if first_neg > 10:
            acf = acf[:first_neg]
        
        # Step 2: Extract peaks from run1 weights
        peaks = extract_peaks(acf_weights, RATE_GRID)
        classification = classify_peaks(peaks)
        
        # Step 3: Robustness check at multiple λ
        robust = robustness_check(acf, max(acf_lambda, 1e-6), n_lambdas=7)
        
        elapsed = time.time() - t0
        
        # Build result row
        row = {
            'Ticker': tk,
            'Class': meta['class'],
            'n_peaks': classification['n_peaks'],
            'n_fast': classification['n_fast'],
            'n_medium': classification['n_medium'],
            'n_slow': classification['n_slow'],
            'has_fast_and_slow': classification['has_fast_and_slow'],
            'fastest_hl': classification['fastest_hl'],
            'slowest_hl': classification['slowest_hl'],
            'dominant_hl': classification['dominant_hl'],
            'dominant_frac': classification['dominant_frac'],
            'hl_ratio': classification['slowest_hl'] / max(classification['fastest_hl'], 0.1),
            'robust_bimodal': robust['robust_bimodal'],
            'robust_fast_slow': robust['robust_fast_slow'],
            'n_lambda_bimodal': robust['n_bimodal'],
            'n_lambda_fast_slow': robust['n_fast_slow'],
        }
        
        # Add individual peak info (up to 4 peaks)
        for i, p in enumerate(peaks[:4]):
            row[f'peak{i+1}_hl'] = p['halflife']
            row[f'peak{i+1}_rate'] = p['rate']
            row[f'peak{i+1}_frac'] = p['weight_frac']
        
        tag = "BIMODAL" if classification['has_fast_and_slow'] else ""
        robust_tag = f"robust={robust['n_bimodal']}/7" if classification['has_fast_and_slow'] else ""
        
        peak_str = " | ".join(f"HL={p['halflife']:.1f}d({100*p['weight_frac']:.0f}%)" 
                              for p in peaks[:3])
        
        print(f"  {tk:<8s} {meta['class']:<12s} peaks={classification['n_peaks']} "
              f"[{peak_str}] {tag} {robust_tag} ({elapsed:.0f}s)")
        
        return tk, row
        
    except Exception as e:
        print(f"  {tk:<8s} FAILED: {e}")
        return tk, None


# ==============================================================
#  REPORT
# ==============================================================

def print_report(rw, df):
    n = len(df)
    rw.p("=" * 72)
    rw.p("  BIMODALITY ANALYSIS — REPORT")
    rw.p(f"  {n} assets, peak extraction + multi-λ robustness")
    rw.p("=" * 72)
    
    # ── Summary ──
    n_bimodal = df['has_fast_and_slow'].sum()
    n_robust = df['robust_bimodal'].sum()
    n_robust_fs = df['robust_fast_slow'].sum()
    
    rw.p(f"\n  Assets with fast AND slow peaks: {n_bimodal}/{n} ({100*n_bimodal/n:.0f}%)")
    rw.p(f"  Robust bimodal (≥4/7 λ values): {n_robust}/{n} ({100*n_robust/n:.0f}%)")
    rw.p(f"  Robust fast+slow (≥4/7 λ):      {n_robust_fs}/{n} ({100*n_robust_fs/n:.0f}%)")
    
    # ── Peak count distribution ──
    rw.p(f"\n{'─'*60}")
    rw.p(f"  NUMBER OF PEAKS")
    rw.p(f"{'─'*60}")
    for np_ in sorted(df['n_peaks'].unique()):
        c = (df['n_peaks'] == np_).sum()
        rw.p(f"  {np_} peaks: {c} assets")
    
    # ── Fast/slow classification by class ──
    rw.p(f"\n{'─'*72}")
    rw.p(f"  FAST/SLOW PEAK STRUCTURE BY CLASS")
    rw.p(f"{'─'*72}")
    rw.p(f"  {'Class':<16s} {'N':>3s} {'F+S':>4s} {'Robust':>6s} "
         f"{'Med Fast HL':>11s} {'Med Slow HL':>11s} {'Med Ratio':>9s}")
    rw.p(f"  {'─'*65}")
    
    for cls in sorted(df['Class'].unique()):
        sub = df[df['Class'] == cls]
        n_cls = len(sub)
        n_fs = sub['has_fast_and_slow'].sum()
        n_rob = sub['robust_fast_slow'].sum()
        
        fs_sub = sub[sub['has_fast_and_slow']]
        med_fast = fs_sub['fastest_hl'].median() if len(fs_sub) > 0 else np.nan
        med_slow = fs_sub['slowest_hl'].median() if len(fs_sub) > 0 else np.nan
        med_ratio = fs_sub['hl_ratio'].median() if len(fs_sub) > 0 else np.nan
        
        rw.p(f"  {cls[:15]:<16s} {n_cls:>3d} {n_fs:>3d}/{n_cls} {n_rob:>4d}/{n_cls} "
             f"{med_fast:>9.1f}d {med_slow:>9.1f}d {med_ratio:>9.1f}×")
    
    # ── Peak location clustering ──
    rw.p(f"\n{'─'*72}")
    rw.p(f"  PEAK LOCATION CLUSTERING (assets with fast+slow peaks)")
    rw.p(f"{'─'*72}")
    
    fs = df[df['has_fast_and_slow']].copy()
    if len(fs) > 0:
        # Fast peak histogram
        fast_hls = fs['fastest_hl'].dropna()
        rw.p(f"\n  FAST PEAK half-lives (N={len(fast_hls)}):")
        bins = [(0, 1, '<1d'), (1, 3, '1-3d'), (3, 5, '3-5d'), 
                (5, 10, '5-10d')]
        for lo, hi, label in bins:
            c = ((fast_hls >= lo) & (fast_hls < hi)).sum()
            bar = '█' * c
            rw.p(f"    {label:>6s}: {c:>3d} {bar}")
        rw.p(f"    Median: {fast_hls.median():.1f}d, Mean: {fast_hls.mean():.1f}d")
        
        # Slow peak histogram
        slow_hls = fs['slowest_hl'].dropna()
        rw.p(f"\n  SLOW PEAK half-lives (N={len(slow_hls)}):")
        bins = [(10, 30, '10-30d'), (30, 60, '30-60d'), (60, 100, '60-100d'),
                (100, 200, '100-200d'), (200, 500, '200-500d'), (500, 10000, '>500d')]
        for lo, hi, label in bins:
            c = ((slow_hls >= lo) & (slow_hls < hi)).sum()
            bar = '█' * c
            rw.p(f"    {label:>8s}: {c:>3d} {bar}")
        rw.p(f"    Median: {slow_hls.median():.1f}d, Mean: {slow_hls.mean():.1f}d")
        
        # Separation ratio
        rw.p(f"\n  SEPARATION RATIO (slow_HL / fast_HL):")
        rw.p(f"    Median: {fs['hl_ratio'].median():.1f}×")
        rw.p(f"    Mean:   {fs['hl_ratio'].mean():.1f}×")
        rw.p(f"    Range:  {fs['hl_ratio'].min():.1f}× to {fs['hl_ratio'].max():.1f}×")
    
    # ── HAR connection ──
    rw.p(f"\n{'─'*72}")
    rw.p(f"  HAR CONNECTION: Do peaks cluster near 1, 5, 22 days?")
    rw.p(f"{'─'*72}")
    
    if 'peak1_hl' in df.columns:
        all_peak_hls = []
        for col in ['peak1_hl', 'peak2_hl', 'peak3_hl', 'peak4_hl']:
            if col in df.columns:
                all_peak_hls.extend(df[col].dropna().tolist())
        
        if all_peak_hls:
            all_peak_hls = np.array(all_peak_hls)
            har_targets = [1, 5, 22]
            for target in har_targets:
                near = np.sum(np.abs(all_peak_hls - target) < target * 0.5)
                rw.p(f"  Peaks within 50% of {target}d: {near}/{len(all_peak_hls)}")
    
    # ── Full table ──
    rw.p(f"\n{'═'*90}")
    rw.p(f"  FULL RESULTS")
    rw.p(f"{'═'*90}")
    rw.p(f"  {'Ticker':<8s} {'Class':<12s} {'#Pk':>3s} {'Fast HL':>8s} {'Slow HL':>8s} "
         f"{'Ratio':>6s} {'Robust':>6s} {'Peak details'}")
    rw.p(f"  {'─'*85}")
    
    for _, row in df.sort_values(['Class', 'Ticker']).iterrows():
        peak_detail = ""
        for i in range(1, 5):
            hl_col = f'peak{i}_hl'
            frac_col = f'peak{i}_frac'
            if hl_col in row and pd.notna(row.get(hl_col)):
                hl = row[hl_col]
                frac = row.get(frac_col, 0)
                peak_detail += f" {hl:.1f}d({100*frac:.0f}%)"
        
        robust_str = f"{row['n_lambda_bimodal']}/7" if row['has_fast_and_slow'] else "."
        
        rw.p(f"  {row['Ticker']:<8s} {str(row['Class'])[:11]:<12s} "
             f"{row['n_peaks']:>3.0f} "
             f"{row['fastest_hl']:>8.1f} {row['slowest_hl']:>8.1f} "
             f"{row['hl_ratio']:>6.1f} {robust_str:>6s} "
             f"{peak_detail}")


def make_figures(df):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  No matplotlib, skipping figures")
        return
    
    # ── Figure 1: Fast vs Slow peak half-lives ──
    fs = df[df['has_fast_and_slow']].copy()
    if len(fs) > 5:
        fig, ax = plt.subplots(figsize=(8, 6))
        classes = sorted(fs['Class'].unique())
        for cls in classes:
            sub = fs[fs['Class'] == cls]
            ax.scatter(sub['fastest_hl'], sub['slowest_hl'], label=cls, alpha=0.7, s=40)
        
        ax.set_xlabel('Fast peak half-life (days)')
        ax.set_ylabel('Slow peak half-life (days)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Bimodal Assets: Fast vs Slow Timescales')
        ax.plot([0.1, 1000], [0.1, 1000], 'k--', alpha=0.3, label='1:1')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, 'fig_fast_vs_slow.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(OUTDIR, 'fig_fast_vs_slow.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # ── Figure 2: Histogram of all peak locations ──
    all_peaks = []
    for col in ['peak1_hl', 'peak2_hl', 'peak3_hl', 'peak4_hl']:
        if col in df.columns:
            all_peaks.extend(df[col].dropna().tolist())
    
    if all_peaks:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(np.log10(all_peaks), bins=40, color='steelblue', alpha=0.8, edgecolor='white')
        
        # Mark HAR timescales
        for target, label in [(1, '1d'), (5, '5d'), (22, '22d'), (63, '63d'), (252, '252d')]:
            ax.axvline(np.log10(target), color='red', linestyle='--', alpha=0.7)
            ax.text(np.log10(target), ax.get_ylim()[1]*0.95, label, ha='center', 
                    fontsize=9, color='red')
        
        ax.set_xlabel('log₁₀(Half-life in days)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of All Peak Locations Across 100 Assets')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, 'fig_peak_histogram.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(OUTDIR, 'fig_peak_histogram.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # ── Figure 3: Separation ratio by class ──
    if len(fs) > 5:
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = sorted(fs['Class'].unique())
        data = [fs[fs['Class']==c]['hl_ratio'].values for c in classes]
        bp = ax.boxplot(data, labels=classes, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)
        ax.set_ylabel('Slow HL / Fast HL')
        ax.set_title('Timescale Separation Ratio by Asset Class')
        ax.set_yscale('log')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, 'fig_separation_by_class.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(OUTDIR, 'fig_separation_by_class.png'), bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    print("  Figures saved")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default='shape_of_memory_100.csv')
    parser.add_argument('--run1dir', type=str, default='results/mixing')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=OUTDIR)
    args = parser.parse_args()
    
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    
    n_workers = args.workers or max(1, cpu_count() - 2)
    
    print(f"\n{'═'*60}")
    print(f"  RUN 2: BIMODALITY PEAK EXTRACTION")
    print(f"  Workers: {n_workers}")
    print(f"{'═'*60}")
    
    # Load run1 results
    run1_csv = os.path.join(args.run1dir, 'mixing_results.csv')
    run1_npz = os.path.join(args.run1dir, 'mixing_weights.npz')
    
    df_run1 = pd.read_csv(run1_csv)
    npz = np.load(run1_npz)
    
    print(f"  Loaded run1: {len(df_run1)} assets, {len(npz.files)} weight arrays")
    
    # Load raw returns for ACF recomputation
    assets = load_data(args.datafile)
    print(f"  Loaded raw data: {len(assets)} assets")
    
    # Match rates grid
    rates_run1 = npz['rates']
    
    # Build worker args
    args_list = []
    for _, row in df_run1.iterrows():
        tk = row['Ticker']
        if tk not in assets:
            continue
        
        acf_key = f'{tk}_acf'
        if acf_key not in npz:
            continue
        
        acf_weights = npz[acf_key]
        acf_lambda = row.get('ACF_lambda', row.get('K_lambda', 1.0))
        # Use a reasonable default if lambda not in CSV
        if pd.isna(acf_lambda) or acf_lambda <= 0:
            acf_lambda = 1.0
        
        args_list.append((
            tk, 
            assets[tk]['r'], 
            acf_weights, 
            acf_lambda,
            {'class': assets[tk]['class']}
        ))
    
    print(f"  Processing {len(args_list)} assets")
    
    # Run
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        out = pool.map(_worker, args_list)
    elapsed = time.time() - t0
    
    # Collect
    rows = [row for tk, row in out if row is not None]
    df = pd.DataFrame(rows)
    n_ok = len(df)
    
    print(f"\n  {n_ok} assets done, {len(args_list)-n_ok} failed, {elapsed/60:.1f} min")
    
    # Save
    csv_path = os.path.join(OUTDIR, 'bimodality_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Saved: {csv_path}")
    
    # Report
    rw = Report(os.path.join(OUTDIR, 'REPORT_BIMODALITY.txt'))
    print_report(rw, df)
    rw.save()
    
    # Figures
    make_figures(df)
    
    print(f"\n{'═'*60}")
    print(f"  COMPLETE: {elapsed/60:.1f} minutes")
    print(f"{'═'*60}")
