"""Raw-EEG data augmentation (jitter, scaling, magnitude-warp).

Confirmed (round 3 of this project's experiments) to improve accuracy for both ATCNet_full and
Mini-ATCNet on this dataset without any architecture change. Applied only to the inner-training
split of each fold, before scaling -- augmented trials must never leak into validation or test.

Reimplemented from scratch here (no external augmentation library) for raw (channels, time) EEG
trials, since ATCNet has no CSP step to augment around, unlike this project's MixNet experiments
where the same three augmentations were applied to the post-CSP spectral-spatial signal instead.

Randomness comes only from an explicit `np.random.Generator` passed in by the caller. Nothing here
touches NumPy's global RNG, so augmentation can never change the draws of any other component
(for example the validation split of a later fold).
"""
import numpy as np
from scipy.interpolate import CubicSpline


def augment_jitter(X, rng, sigma=0.03):
    return X + rng.normal(0, sigma * np.std(X), X.shape)


def augment_scaling(X, rng, sigma=0.1):
    factors = rng.normal(1.0, sigma, size=(X.shape[0], X.shape[1], 1))
    return X * factors


def augment_magnitude_warp(X, rng, sigma=0.1, n_knots=4):
    n_trials, n_chans, n_samples = X.shape
    warped = np.empty_like(X)
    knot_xs = np.linspace(0, n_samples - 1, n_knots)
    sample_xs = np.arange(n_samples)
    for i in range(n_trials):
        for c in range(n_chans):
            knot_ys = rng.normal(1.0, sigma, n_knots)
            curve = CubicSpline(knot_xs, knot_ys)(sample_xs)
            warped[i, c] = X[i, c] * curve
    return warped


def augment_batch(X, y, ratio=1.0, rng=None):
    """Generate `ratio` additional augmented trials per real trial (ratio=1.0 doubles the set),
    each augmented trial randomly picking one of the three augmentations above.

    `rng` is a `np.random.Generator`; pass one derived from a fixed seed for reproducible runs.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_aug = int(round(len(X) * ratio))
    if n_aug == 0:
        return X, y
    idx = rng.choice(len(X), size=n_aug, replace=True)
    X_pick, y_pick = X[idx], y[idx]

    aug_fns = [augment_jitter, augment_scaling, augment_magnitude_warp]
    splits = np.array_split(np.arange(n_aug), len(aug_fns))
    X_aug = np.empty_like(X_pick)
    for fn, split_idx in zip(aug_fns, splits):
        if len(split_idx) == 0:
            continue
        X_aug[split_idx] = fn(X_pick[split_idx], rng)

    X_out = np.concatenate([X, X_aug], axis=0)
    y_out = np.concatenate([y, y_pick], axis=0)
    return X_out, y_out
