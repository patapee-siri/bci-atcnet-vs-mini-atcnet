"""Constants and hyperparameters shared across the data, model, and training modules.

Kept as plain module-level values (not a config class/dataclass) since every value here is a
literal used exactly once per run and there is no need for multiple simultaneous configurations
in-process -- the two model configs are already distinguished by name in MODEL_CONFIGS below.
"""

# Kaggle dataset mount path (see README for the dataset source and local-path override).
DATA_DIR = '/kaggle/input/datasets/meen14052547/bciciv-2a-gdf/BCICIV_2a_gdf'
SAVE_DIR_ROOT = '/kaggle/working/saved_models_atcnet_compare'
SUBJECTS = list(range(1, 10))

EEG_CHS = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
    'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
    'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz',
    'EEG-15', 'EEG-16'
]
N_CHANS = len(EEG_CHS)

LABEL_MAP = {'769': 0, '770': 1, '771': 2, '772': 3}
CLASS_NAMES = ['Left', 'Right', 'Foot', 'Tongue']
N_CLASSES = 4

TMIN, TMAX = 0.0, 4.0
SFREQ_TARGET = 250

SEED = 42

# The two configurations under comparison. 'ATCNet_full' uses ATCNet's own published defaults
# (Altaheri et al., IEEE TII 2023 -- models.py: ATCNet_ function signature); 'Mini-ATCNet' is a
# self-designed lightweight variant (not a published architecture) -- see README for the full
# per-hyperparameter rationale.
MODEL_CONFIGS = {
    'ATCNet_full': dict(n_windows=5, eegn_F1=16, eegn_D=2, eegn_kernel_size=64, eegn_pool_size=7, eegn_dropout=0.3,
                         tcn_depth=2, tcn_kernel_size=4, tcn_filters=32, tcn_dropout=0.3,
                         mha_key_dim=8, mha_num_heads=2),
    'Mini-ATCNet': dict(n_windows=3, eegn_F1=8, eegn_D=1, eegn_kernel_size=64, eegn_pool_size=7, eegn_dropout=0.3,
                         tcn_depth=1, tcn_kernel_size=4, tcn_filters=16, tcn_dropout=0.3,
                         mha_key_dim=4, mha_num_heads=1),
}

# Training defaults (all overridable via train.py CLI flags).
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 5e-4
DEFAULT_PATIENCE = 35
DEFAULT_N_FOLDS = 5
DEFAULT_AUGMENT_RATIO = 1.0
DEFAULT_WARMUP_EPOCHS = 10
