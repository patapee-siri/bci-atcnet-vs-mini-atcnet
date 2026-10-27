"""Single-(subject, config) training entrypoint, subject-dependent 5-fold CV.

Run as its own process per (subject, config) pair -- either directly via the CLI below, or
spawned as a subprocess by a driver loop (see notebooks/02_atcnet_vs_mini_atcnet_augment_lrwarmup.ipynb).
The subprocess-per-run pattern exists because a long-running in-process loop over all 90
(subject, config, fold) combinations was found to crash from an unrecoverable TensorFlow/Keras
resource leak around fold ~30 -- see the notebook's fix-history markdown for the full debugging
trail (XLA recompilation cost, batch-size-driven retracing, and the leak itself). Running each
combination in its own process guarantees the OS reclaims all GPU/host memory on exit regardless
of the leak's root cause.

Example:
    python -m src.train --subject 1 --config Mini-ATCNet --save_dir_root ./saved_models
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .augment import augment_batch
from .config import (DATA_DIR, DEFAULT_AUGMENT_RATIO, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR,
                      DEFAULT_N_FOLDS, DEFAULT_PATIENCE, DEFAULT_WARMUP_EPOCHS, MODEL_CONFIGS,
                      N_CLASSES, SAVE_DIR_ROOT, SEED)
from .data import load_subject_epochs
from .model import LRWarmup, build_atcnet

# No global np.random.seed / tf.random.set_seed here on purpose: every fold derives its own
# generators from (SEED, subject, fold) via fold_rngs(), so no fold's randomness depends on
# another fold having run first, and a single fold can be rerun in isolation.
tf.config.optimizer.set_jit(False)  # see README / notebook fix-history: avoids a growing XLA
# recompile cost when input shapes vary slightly across folds/runs.

PROTOCOL_VERSION = 2  # 1 = original Kaggle run (global RNG, unstratified 85/15 split)


def fold_rngs(subject, fold):
    """Independent random sources for one fold, derived only from (SEED, subject, fold).

    Returns (split_rng, aug_rng, tf_seed): generators for the inner validation split and for the
    augmentation, plus an integer seed for TensorFlow/Keras weight initialisation and dropout.
    """
    split_ss, aug_ss, tf_ss = np.random.SeedSequence([SEED, subject, fold]).spawn(3)
    tf_seed = int(tf_ss.generate_state(1)[0] % (2 ** 31 - 1))
    return np.random.default_rng(split_ss), np.random.default_rng(aug_ss), tf_seed


def stratified_inner_split(y, rng, val_frac=0.15):
    """Class-stratified inner train/validation split; returns indices into `y`.

    Stratification keeps every class present in the validation set with near-equal proportions,
    so early stopping on val_loss is not driven by a class-skewed validation sample.
    """
    return train_test_split(np.arange(len(y)), test_size=val_frac, stratify=y,
                            random_state=int(rng.integers(2 ** 31 - 1)))


def run_subject(subject, config_name, save_dir_root=SAVE_DIR_ROOT, epochs=DEFAULT_EPOCHS,
                 batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR, patience=DEFAULT_PATIENCE,
                 n_folds=DEFAULT_N_FOLDS, augment_ratio=DEFAULT_AUGMENT_RATIO,
                 warmup_epochs=DEFAULT_WARMUP_EPOCHS, data_dir=DATA_DIR):
    """Train and evaluate one subject with subject-dependent K-fold CV for one model config.

    Returns a list of per-fold result dicts (also written to
    `{save_dir_root}/{config_name}/S{subject:02d}_results.json`).
    """
    save_dir = os.path.join(save_dir_root, config_name)
    os.makedirs(save_dir, exist_ok=True)
    model_kwargs = MODEL_CONFIGS[config_name]

    X, y = load_subject_epochs(subject, session='T', data_dir=data_dir)
    n_chans, n_samples = X.shape[1], X.shape[2]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        # leakage guard: a trial must never sit in both the training and the test partition
        assert np.intersect1d(train_idx, test_idx).size == 0, 'train/test trial overlap'
        X_tr_raw, y_tr = X[train_idx], y[train_idx]
        X_te_raw, y_te = X[test_idx], y[test_idx]

        split_rng, aug_rng, tf_seed = fold_rngs(subject, fold)
        tr_i, va_i = stratified_inner_split(y_tr, split_rng)
        assert np.intersect1d(tr_i, va_i).size == 0, 'inner train/validation trial overlap'
        X_tr, y_tr_f = X_tr_raw[tr_i], y_tr[tr_i]
        X_va, y_va_f = X_tr_raw[va_i], y_tr[va_i]
        val_class_counts = np.bincount(y_va_f, minlength=N_CLASSES)
        val_trial_idx = train_idx[va_i]

        # augment the inner-training split only, before scaling -- never touches val/test
        X_tr, y_tr_f = augment_batch(X_tr, y_tr_f, ratio=augment_ratio, rng=aug_rng)

        scaler = StandardScaler()
        n_tr = X_tr.shape[0]
        X_tr_flat = scaler.fit_transform(X_tr.reshape(n_tr, -1)).reshape(X_tr.shape)
        X_va_flat = scaler.transform(X_va.reshape(X_va.shape[0], -1)).reshape(X_va.shape)
        X_te_flat = scaler.transform(X_te_raw.reshape(X_te_raw.shape[0], -1)).reshape(X_te_raw.shape)

        X_tr_in = X_tr_flat[..., np.newaxis].astype('float32')
        X_va_in = X_va_flat[..., np.newaxis].astype('float32')
        X_te_in = X_te_flat[..., np.newaxis].astype('float32')

        tf.keras.utils.set_random_seed(tf_seed)  # per-fold weight init / dropout, independent of fold order
        model = build_atcnet(N_CLASSES, n_chans, n_samples, model_name=config_name, **model_kwargs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                        restore_best_weights=True, mode='min')
        # monitors val_loss, not val_accuracy: a collapsed model (predicting one class for every
        # trial) leaves val_accuracy perfectly flat, so early stopping on val_accuracy sees no
        # improvement to react to and gives up almost immediately -- val_loss keeps moving.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
        lr_warmup = LRWarmup(target_lr=lr, warmup_epochs=warmup_epochs)

        train_start = time.perf_counter()
        history = model.fit(X_tr_in, y_tr_f, validation_data=(X_va_in, y_va_f), epochs=epochs,
                             batch_size=batch_size, callbacks=[lr_warmup, early_stop, reduce_lr], verbose=0)
        train_time = time.perf_counter() - train_start
        n_epochs_run = len(history.history['loss'])

        fold_tag = f'S{subject:02d}_fold{fold:02d}'
        model.save(os.path.join(save_dir, f'{fold_tag}_model.keras'))
        with open(os.path.join(save_dir, f'{fold_tag}_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        test_start = time.perf_counter()
        # fixed batch_size (not len(X_te_in)): caps the number of distinct input shapes TF has to
        # handle across all folds, avoiding a per-fold XLA/graph retrace cost.
        y_pred_proba = model.predict(X_te_in, batch_size=batch_size, verbose=0)
        test_time = time.perf_counter() - test_start
        y_pred = np.argmax(y_pred_proba, axis=1)

        pred_counts = np.bincount(y_pred, minlength=N_CLASSES)
        collapsed = bool((pred_counts.max() / len(y_pred)) > 0.9)

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='macro')
        kappa = cohen_kappa_score(y_te, y_pred)
        fold_results.append({
            'config': config_name, 'subject': subject, 'fold': fold, 'protocol_version': PROTOCOL_VERSION,
            'val_size': int(len(va_i)), 'val_class_counts': val_class_counts.tolist(),
            'val_trial_idx': val_trial_idx.tolist(),
            'accuracy': float(acc), 'f1_macro': float(f1), 'kappa': float(kappa),
            'train_time_s': train_time, 'n_epochs': n_epochs_run,
            'test_time_s_total': test_time, 'test_time_ms_per_trial': (test_time / len(y_te)) * 1000,
            'pred_class_counts': pred_counts.tolist(), 'collapsed': collapsed,
            'y_true': y_te.tolist(), 'y_pred': y_pred.tolist(),
        })
        collapse_flag = '  [COLLAPSED]' if collapsed else ''
        print(f'  [{config_name}] Subject {subject} fold {fold}: acc={acc:.3f} f1={f1:.3f} '
              f'train={train_time:.1f}s ({n_epochs_run} epochs) '
              f'test={(test_time / len(y_te)) * 1000:.2f}ms/trial pred_counts={pred_counts.tolist()} '
              f'val_counts={val_class_counts.tolist()}{collapse_flag}')

    out_path = os.path.join(save_dir, f'S{subject:02d}_results.json')
    with open(out_path, 'w') as f:
        json.dump(fold_results, f)
    print(f'saved {out_path}')
    return fold_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--config', type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--save_dir_root', type=str, default=SAVE_DIR_ROOT)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--n_folds', type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument('--augment_ratio', type=float, default=DEFAULT_AUGMENT_RATIO)
    parser.add_argument('--warmup_epochs', type=int, default=DEFAULT_WARMUP_EPOCHS)
    args = parser.parse_args()

    run_subject(subject=args.subject, config_name=args.config, save_dir_root=args.save_dir_root,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                n_folds=args.n_folds, augment_ratio=args.augment_ratio, warmup_epochs=args.warmup_epochs,
                data_dir=args.data_dir)


if __name__ == '__main__':
    main()
