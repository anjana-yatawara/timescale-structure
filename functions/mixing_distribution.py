"""
MIXING DISTRIBUTION RECOVERY
==============================
Recovers the distribution of relaxation rates ρ(u) from a memory kernel
or autocorrelation function via regularized non-negative least squares.

Mathematical basis:
  Bernstein's theorem: f completely monotone ⟺ f(t) = ∫ exp(-tu) dμ(u)
  Pollard (1946): exp(-t^α) = ∫ exp(-tu) f_α(u) du  for 0 < α ≤ 1

The discrete inverse problem:
  b_j = Σ_k w_k exp(-j u_k)  →  A w = b,  w ≥ 0

Solved via Tikhonov-regularized NNLS with L-curve parameter selection.
"""

import numpy as np
from scipy.optimize import nnls
from scipy.interpolate import CubicSpline


# ==============================================================
#  RATE GRID
# ==============================================================

def make_rate_grid(M=150, u_min=0.001, u_max=2.5):
    """
    Log-spaced grid of exponential decay rates.
    
    u = 0.001 → half-life = 693 days (2.7 years)
    u = 0.01  → half-life = 69 days  (quarterly)
    u = 0.1   → half-life = 6.9 days (weekly)
    u = 0.5   → half-life = 1.4 days (daily)
    u = 2.5   → half-life = 0.28 days (intraday)
    """
    return np.logspace(np.log10(u_min), np.log10(u_max), M)


def rate_to_halflife(u):
    """Convert rate to half-life in days."""
    return np.log(2) / np.maximum(u, 1e-15)


# ==============================================================
#  DESIGN MATRIX
# ==============================================================

def build_design_matrix(lags, rates):
    """
    A[j,k] = exp(-lags[j] * rates[k])
    
    lags: array of lag values (1,...,J or 1,...,L)
    rates: array of rate grid values u_1,...,u_M
    """
    return np.exp(-np.outer(lags, rates))


# ==============================================================
#  TIKHONOV-REGULARIZED NNLS
# ==============================================================

def nnls_tikhonov(A, b, lam, L=None):
    """
    Solve: min ||Aw - b||² + λ||Lw||²  subject to w ≥ 0
    
    Transforms to standard NNLS on augmented system:
      [A; √λ L] w = [b; 0]
    """
    m, n = A.shape
    if L is None:
        L = np.eye(n)  # zeroth order (ridge)
    
    # Augmented system
    A_aug = np.vstack([A, np.sqrt(lam) * L])
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])
    
    w, residual = nnls(A_aug, b_aug)
    
    # Compute data residual and regularization norm separately
    data_residual = np.linalg.norm(A @ w - b)
    reg_norm = np.linalg.norm(L @ w)
    
    return w, data_residual, reg_norm


def first_difference_matrix(n):
    """First-difference operator D₁: (D₁w)_k = w_{k+1} - w_k"""
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D


# ==============================================================
#  L-CURVE SELECTION
# ==============================================================

def l_curve_select(A, b, L=None, n_lambda=40, lam_range=(1e-8, 1e2)):
    """
    Select regularization parameter λ by L-curve method.
    
    Computes NNLS for each λ, plots ||Aw-b|| vs ||Lw||,
    selects λ at maximum curvature.
    
    Returns: best_lam, w_best, diagnostics_dict
    """
    lambdas = np.logspace(np.log10(lam_range[0]), np.log10(lam_range[1]), n_lambda)
    
    data_res = np.zeros(n_lambda)
    reg_norms = np.zeros(n_lambda)
    weights_all = []
    
    for i, lam in enumerate(lambdas):
        w, dr, rn = nnls_tikhonov(A, b, lam, L)
        data_res[i] = dr
        reg_norms[i] = rn
        weights_all.append(w)
    
    # Curvature of L-curve in log-log space
    xi = np.log10(np.maximum(data_res, 1e-15))
    eta = np.log10(np.maximum(reg_norms, 1e-15))
    
    # Approximate curvature via finite differences
    if n_lambda >= 5:
        dxi = np.gradient(xi)
        deta = np.gradient(eta)
        ddxi = np.gradient(dxi)
        ddeta = np.gradient(deta)
        
        # Curvature κ = (ξ'η'' - η'ξ'') / (ξ'² + η'²)^{3/2}
        numer = dxi * ddeta - deta * ddxi
        denom = (dxi**2 + deta**2)**1.5
        kappa = np.where(denom > 1e-20, numer / denom, 0.0)
        
        # Maximum curvature (corner of L-curve)
        # Exclude endpoints
        inner = kappa[2:-2]
        if len(inner) > 0:
            best_idx = np.argmax(inner) + 2
        else:
            best_idx = n_lambda // 2
    else:
        best_idx = n_lambda // 2
    
    best_lam = lambdas[best_idx]
    w_best = weights_all[best_idx]
    
    return best_lam, w_best, {
        'lambdas': lambdas,
        'data_residuals': data_res,
        'reg_norms': reg_norms,
        'best_idx': best_idx,
        'best_lam': best_lam,
    }


# ==============================================================
#  MIXING DISTRIBUTION DIAGNOSTICS
# ==============================================================

def mixing_diagnostics(w, rates):
    """
    Compute summary statistics of the recovered mixing distribution.
    
    w: weight vector (non-negative)
    rates: rate grid
    """
    # Normalize to a distribution
    total = np.sum(w)
    if total < 1e-15:
        return {
            'dominant_rate': np.nan, 'dominant_hl': np.nan,
            'mean_rate': np.nan, 'mean_hl': np.nan,
            'width_log': np.nan, 'K_eff': 0,
            'is_bimodal': False, 'entropy': 0.0,
        }
    
    p = w / total  # probability weights
    
    # Dominant component
    dom_idx = np.argmax(w)
    dominant_rate = rates[dom_idx]
    dominant_hl = np.log(2) / dominant_rate
    
    # Width: weighted standard deviation of log-rate
    log_rates = np.log(rates)
    mean_log_rate = np.sum(p * log_rates)
    mean_rate = np.exp(mean_log_rate)
    mean_hl = np.log(2) / mean_rate
    
    var_log_rate = np.sum(p * (log_rates - mean_log_rate)**2)
    width_log = np.sqrt(var_log_rate)  # std dev in log-rate space
    
    # Effective number of components
    active = w > 0.01 * np.max(w)
    K_eff = np.sum(active)
    
    # Shannon entropy (higher = more dispersed)
    p_pos = p[p > 1e-15]
    entropy = -np.sum(p_pos * np.log(p_pos))
    
    # Bimodality detection via Hartigan's dip statistic (simplified)
    # Look for two distinct peaks in the weight distribution
    is_bimodal = _detect_bimodality(w, rates)
    
    # Percentiles in half-life space
    hl_grid = np.log(2) / rates[::-1]  # reversed so HL is increasing
    p_rev = p[::-1]
    cum_rev = np.cumsum(p_rev)
    
    hl_10 = hl_grid[np.searchsorted(cum_rev, 0.10)] if cum_rev[-1] > 0.10 else np.nan
    hl_50 = hl_grid[np.searchsorted(cum_rev, 0.50)] if cum_rev[-1] > 0.50 else np.nan
    hl_90 = hl_grid[np.searchsorted(cum_rev, 0.90)] if cum_rev[-1] > 0.90 else np.nan
    
    return {
        'dominant_rate': dominant_rate,
        'dominant_hl': dominant_hl,
        'mean_rate': mean_rate,
        'mean_hl': mean_hl,
        'width_log': width_log,
        'K_eff': int(K_eff),
        'is_bimodal': is_bimodal,
        'entropy': entropy,
        'hl_10': hl_10,  # 10th percentile half-life (fast component)
        'hl_50': hl_50,  # median half-life
        'hl_90': hl_90,  # 90th percentile half-life (slow component)
    }


def _detect_bimodality(w, rates):
    """
    Simple bimodality detection: find local maxima in smoothed weights.
    Returns True if 2+ distinct peaks with valley between them dropping
    below 50% of the smaller peak.
    """
    if np.max(w) < 1e-15:
        return False
    
    # Smooth with a small kernel
    from scipy.ndimage import gaussian_filter1d
    w_smooth = gaussian_filter1d(w, sigma=2)
    
    # Find local maxima
    peaks = []
    for i in range(1, len(w_smooth) - 1):
        if w_smooth[i] > w_smooth[i-1] and w_smooth[i] > w_smooth[i+1]:
            if w_smooth[i] > 0.05 * np.max(w_smooth):  # threshold
                peaks.append(i)
    
    if len(peaks) < 2:
        return False
    
    # Check if valley between the two tallest peaks is deep enough
    peaks_sorted = sorted(peaks, key=lambda i: -w_smooth[i])
    p1, p2 = sorted(peaks_sorted[:2])
    valley = np.min(w_smooth[p1:p2+1])
    smaller_peak = min(w_smooth[p1], w_smooth[p2])
    
    return valley < 0.5 * smaller_peak


# ==============================================================
#  SAMPLE ACF OF ABSOLUTE RETURNS
# ==============================================================

def sample_acf_abs(r, max_lag=500):
    """
    Compute sample autocorrelation of |r_t| at lags 1,...,max_lag.
    
    ACF(ℓ) = Corr(|r_t|, |r_{t+ℓ}|)
    """
    abs_r = np.abs(r)
    mu = np.mean(abs_r)
    x = abs_r - mu
    var = np.mean(x**2)
    
    T = len(r)
    acf = np.zeros(max_lag)
    
    for lag in range(1, max_lag + 1):
        if lag >= T:
            acf[lag - 1] = 0.0
        else:
            acf[lag - 1] = np.mean(x[:T-lag] * x[lag:]) / var
    
    return acf


# ==============================================================
#  FULL DECOMPOSITION PIPELINE
# ==============================================================

def decompose_kernel(kernel, M=150, u_min=0.001, u_max=2.5,
                     n_lambda=40, max_lag=None):
    """
    Full pipeline: decompose a kernel into mixture of exponentials.
    
    Parameters
    ----------
    kernel : array, shape (J,)
        Memory kernel g(j) for j=1,...,J (not necessarily normalized)
    
    Returns
    -------
    dict with: weights, rates, diagnostics, L-curve info, reconstruction
    """
    # Normalize
    g_norm = kernel / kernel[0]
    J = len(g_norm)
    if max_lag is not None:
        J = min(J, max_lag)
        g_norm = g_norm[:J]
    
    # Setup
    lags = np.arange(1, J + 1, dtype=float)
    rates = make_rate_grid(M, u_min, u_max)
    A = build_design_matrix(lags, rates)
    b = g_norm
    L = first_difference_matrix(M)
    
    # L-curve selection
    best_lam, w_best, lcurve = l_curve_select(A, b, L, n_lambda)
    
    # Reconstruction quality
    b_hat = A @ w_best
    ss_res = np.sum((b - b_hat)**2)
    ss_tot = np.sum((b - np.mean(b))**2)
    R2_recon = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    max_error = np.max(np.abs(b - b_hat))
    
    # Diagnostics
    diag = mixing_diagnostics(w_best, rates)
    
    return {
        'weights': w_best,
        'rates': rates,
        'halflives': rate_to_halflife(rates),
        'lambda': best_lam,
        'R2_recon': R2_recon,
        'max_error': max_error,
        'reconstruction': b_hat,
        'original': b,
        'diagnostics': diag,
        'lcurve': lcurve,
    }


def decompose_acf(r, M=150, u_min=0.001, u_max=2.5,
                  n_lambda=40, max_lag=500):
    """
    Full pipeline: decompose ACF of |r_t| into mixture of exponentials.
    """
    acf = sample_acf_abs(r, max_lag)
    
    # Truncate at first non-positive value
    first_neg = np.argmax(acf <= 0)
    if first_neg > 10:
        acf = acf[:first_neg]
    
    L_acf = len(acf)
    lags = np.arange(1, L_acf + 1, dtype=float)
    rates = make_rate_grid(M, u_min, u_max)
    A = build_design_matrix(lags, rates)
    b = acf
    L_reg = first_difference_matrix(M)
    
    best_lam, w_best, lcurve = l_curve_select(A, b, L_reg, n_lambda)
    
    b_hat = A @ w_best
    ss_res = np.sum((b - b_hat)**2)
    ss_tot = np.sum((b - np.mean(b))**2)
    R2_recon = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    diag = mixing_diagnostics(w_best, rates)
    
    return {
        'weights': w_best,
        'rates': rates,
        'halflives': rate_to_halflife(rates),
        'lambda': best_lam,
        'R2_recon': R2_recon,
        'acf_original': acf,
        'acf_reconstructed': b_hat,
        'n_lags_used': L_acf,
        'diagnostics': diag,
        'lcurve': lcurve,
    }
