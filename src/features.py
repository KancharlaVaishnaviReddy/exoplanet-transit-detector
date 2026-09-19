"""
features.py
------------
Turns a raw flux array (a light curve) into a fixed set of numeric features
that a classical ML model (Random Forest) can learn from. This mirrors what
astronomers actually look at: overall brightness statistics, the depth/shape
of the deepest dip, and how "spiky" vs "smooth" that dip is.
"""

import numpy as np
from scipy import stats


def flatten_curve(flux, window=21):
    """
    Remove slow trends (stellar variability) using a rolling median filter,
    the same normalization trick used in real pipelines (e.g. `lightkurve`'s
    .flatten() method) before searching for transits.
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    if window >= n:
        window = n - 1 if n % 2 == 0 else n - 2
    if window < 3:
        return flux - np.median(flux) + 1.0

    half = window // 2
    trend = np.copy(flux)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        trend[i] = np.median(flux[lo:hi])
    trend[trend == 0] = 1e-8
    flattened = flux / trend
    return flattened


def extract_features(flux):
    """
    Given a raw flux array, return a dict of engineered features.
    """
    flux = np.asarray(flux, dtype=float)
    flat = flatten_curve(flux)

    baseline = np.median(flat)
    min_flux = np.min(flat)
    depth = baseline - min_flux                # how deep the biggest dip is
    dip_idx = np.argmin(flat)

    # width of the dip: count contiguous points below half the depth
    half_depth_level = baseline - depth / 2
    below = flat < half_depth_level
    # find contiguous run containing dip_idx
    width = 0
    if below[dip_idx]:
        i = dip_idx
        while i >= 0 and below[i]:
            width += 1
            i -= 1
        i = dip_idx + 1
        while i < len(below) and below[i]:
            width += 1
            i += 1

    features = {
        "mean_flux": float(np.mean(flat)),
        "std_flux": float(np.std(flat)),
        "median_flux": float(baseline),
        "min_flux": float(min_flux),
        "max_flux": float(np.max(flat)),
        "depth": float(depth),
        "dip_width": float(width),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
        "depth_to_width_ratio": float(depth / (width + 1e-6)),
        "range_flux": float(np.max(flat) - np.min(flat)),
        "mad": float(np.median(np.abs(flat - np.median(flat)))),  # median absolute deviation
    }
    return features


def extract_features_batch(flux_matrix):
    """flux_matrix: 2D array, one light curve per row -> DataFrame of features."""
    import pandas as pd
    rows = [extract_features(row) for row in flux_matrix]
    return pd.DataFrame(rows)


FEATURE_NAMES = [
    "mean_flux", "std_flux", "median_flux", "min_flux", "max_flux",
    "depth", "dip_width", "skewness", "kurtosis",
    "depth_to_width_ratio", "range_flux", "mad",
]
