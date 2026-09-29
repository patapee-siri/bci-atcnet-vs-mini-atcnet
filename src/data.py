"""Trial-level epoching for BCI Competition IV-2a (no sliding-window leakage).

Each trial becomes exactly one (channels, time) sample spanning the full [0, 4]s motor-imagery
cue window -- no overlapping sliding windows, so no trial's samples can leak across a train/test
split the way they would with the sliding-window approach used in this project's original,
pre-rewrite notebook.
"""
import os

import mne
import numpy as np

from .config import DATA_DIR, EEG_CHS, LABEL_MAP, TMIN, TMAX

mne.set_log_level('ERROR')


def load_subject_epochs(subject, session='T', data_dir=DATA_DIR):
    """Load one subject's session as trial-level epochs.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    y : ndarray, shape (n_trials,) -- integer class labels via LABEL_MAP
    """
    path = os.path.join(data_dir, f'A{subject:02d}{session}.gdf')
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    raw.pick_channels(EEG_CHS)
    raw.filter(l_freq=4., h_freq=40., fir_design='firwin', verbose=False)

    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    wanted = {k: v for k, v in event_id_map.items() if k in LABEL_MAP}
    if not wanted:
        raise RuntimeError(f'No MI events found in {path}')

    epochs = mne.Epochs(raw, events, event_id=wanted, tmin=TMIN, tmax=TMAX,
                         baseline=None, preload=True, on_missing='ignore', verbose=False)
    X = epochs.get_data()  # (n_trials, n_channels, n_samples)
    inv_map = {v: LABEL_MAP[k] for k, v in wanted.items()}
    y = np.array([inv_map[c] for c in epochs.events[:, -1]])
    return X, y
