"""
generate_data.py
-----------------
Generates a realistic, labeled light-curve dataset that mimics NASA Kepler
photometry (brightness vs. time for a star), for exoplanet transit detection.

Why synthetic data?
Real Kepler/TESS FITS files are large and require an internet download from
NASA's MAST archive. To make this project runnable immediately, completely
offline, and reproducible, we simulate light curves using the same physics
astronomers use to describe a transit (a periodic box-shaped dip in flux),
plus realistic noise sources:
  - Photon/instrument noise (Gaussian)
  - Stellar variability (slow sinusoidal drift from starspots)
  - Eclipsing binary "false positives" (deeper, V-shaped dips) -- these are
    the main real-world confuser class astronomers must rule out.

You can later swap this out for real data: drop a CSV with the same column
format into data/real_data.csv (see README) and the rest of the pipeline
works unchanged. Real data sources: NASA Exoplanet Archive, MAST (Kepler/
K2/TESS), or the Kaggle "Kepler Labelled Time Series Data" dataset.
"""

import numpy as np
import pandas as pd
import os

RNG_SEED = 42
N_POINTS = 200          # flux samples per light curve (like a folded/binned curve)
N_SAMPLES = 3000        # total number of light curves to simulate


def add_noise(flux, noise_level, rng):
    return flux + rng.normal(0, noise_level, size=flux.shape)


def add_stellar_variability(flux, t, rng):
    """Slow sinusoidal drift caused by starspots rotating in/out of view."""
    amp = rng.uniform(0.0005, 0.004)
    period = rng.uniform(50, 150)
    phase = rng.uniform(0, 2 * np.pi)
    return flux + amp * np.sin(2 * np.pi * t / period + phase)


def make_transit_curve(t, rng):
    """Class 1: real planet transit -> periodic box-shaped dip."""
    flux = np.ones_like(t)
    depth = rng.uniform(0.005, 0.03)      # 0.5% - 3% dip (typical planet transit)
    duration = rng.uniform(3, 10)         # points in transit
    center = N_POINTS / 2 + rng.uniform(-5, 5)
    in_transit = np.abs(t - center) < (duration / 2)
    flux[in_transit] -= depth
    # slight ingress/egress smoothing (U-shape, not sharp box)
    edge = (np.abs(t - center) >= duration / 2) & (np.abs(t - center) < duration / 2 + 2)
    flux[edge] -= depth * 0.4
    flux = add_stellar_variability(flux, t, rng)
    flux = add_noise(flux, rng.uniform(0.0008, 0.0025), rng)
    return flux, depth, duration


def make_binary_curve(t, rng):
    """False positive: eclipsing binary -> deeper, sharper V-shaped dip."""
    flux = np.ones_like(t)
    depth = rng.uniform(0.04, 0.15)       # much deeper than a planet
    duration = rng.uniform(2, 6)
    center = N_POINTS / 2 + rng.uniform(-5, 5)
    dist = np.abs(t - center)
    in_eclipse = dist < (duration / 2)
    # V-shaped (linear) dip rather than flat-bottomed box
    flux[in_eclipse] -= depth * (1 - dist[in_eclipse] / (duration / 2))
    flux = add_stellar_variability(flux, t, rng)
    flux = add_noise(flux, rng.uniform(0.001, 0.003), rng)
    return flux, depth, duration


def make_noise_curve(t, rng):
    """Class 0: no transit -> just noise + stellar variability."""
    flux = np.ones_like(t)
    flux = add_stellar_variability(flux, t, rng)
    flux = add_noise(flux, rng.uniform(0.0008, 0.0025), rng)
    return flux, 0.0, 0.0


def generate_dataset(n_samples=N_SAMPLES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(N_POINTS, dtype=float)

    rows = []
    for i in range(n_samples):
        r = rng.random()
        if r < 0.35:
            flux, depth, dur = make_transit_curve(t, rng)
            label = 1
            source = "transit"
        elif r < 0.55:
            flux, depth, dur = make_binary_curve(t, rng)
            label = 0
            source = "eclipsing_binary"
        else:
            flux, depth, dur = make_noise_curve(t, rng)
            label = 0
            source = "quiet_star"

        row = {"id": i, "label": label, "source": source,
               "true_depth": depth, "true_duration": dur}
        for j, v in enumerate(flux):
            row[f"flux_{j}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_dataset()
    out_path = os.path.join(out_dir, "light_curves.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} light curves -> {out_path}")
    print(df["label"].value_counts())
    print(df["source"].value_counts())
