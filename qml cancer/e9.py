"""Experiment 9: head-development quantum circuit/encoder ablations (NO test access).

Vast.ai notebook: %run e9.py
Diagnostic only: %env E9_SEEDS=42  then %run e9.py
Official comparison requires E9_SEEDS=42,43,44 (default).
MIMIC_CXR_ROOT overrides dataset discovery.

Requires the working Torch/torchvision environment used for E6, plus numpy,
pandas, Pillow, scikit-learn, matplotlib, tqdm, and pennylane==0.45.1.
Install pennylane==0.45.1 separately before running. No package is installed
by this script. Does not reinstall Torch. Heads use CPU; frozen images use CUDA.

Design references (motivation, not evidence of improvement on this dataset):
https://quantum-journal.org/papers/q-2020-02-06-226/
https://pennylane.ai/demos/tutorial_expressivity_fourier_series
https://docs.pennylane.ai/en/stable/code/api/pennylane.IsingZZ.html
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageFile, ImageOps
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import densenet121
from tqdm.auto import tqdm

# One encoder, one learned circuit, and two matched controls.
ROOT_OVERRIDE = os.environ.get('MIMIC_CXR_ROOT', '').strip()
SEEDS = [int(s) for s in os.environ.get('E9_SEEDS', '42,43,44').split(',')]
if len(SEEDS) != len(set(SEEDS)) or not SEEDS:
    raise ValueError('E9_SEEDS must be a nonempty list of distinct integers')
IMAGE_SIZE, FEATURE_DIM = 320, 1024
N_QUBITS, Q_LAYERS = 8, 3
ANGLE_DIM, OBSERVABLE_DIM = 2 * N_QUBITS * Q_LAYERS, 4 * N_QUBITS
EXTRACT_BATCH_SIZE, HEAD_BATCH_SIZE = 24, 128
NUM_WORKERS = 4 if sys.platform.startswith('linux') else 0
EXTRACT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_HEAD_THREADS = 4
BASE_EPOCHS, MAX_EPOCHS = 40, 40
WARMUP_EPOCHS, MIN_EPOCHS, PATIENCE = 1, 8, 7
STOP_MIN_DELTA = 0.0001
BASE_LR, CORE_LR, READOUT_LR = 1e-5, 1e-4, 1e-4
WEIGHT_DECAY, RESIDUAL_L2 = 1e-3, 0.05
DROPOUT, BRANCH_DROPOUT = 0.30, 0.10
RESIDUAL_LIMIT, POS_WEIGHT_CAP = 0.20, 5.0
DEV_FRACTION, DEV_SPLIT_SEED = 0.20, 2026
MEANINGFUL_AP_GAIN, MAX_AUROC_DROP = 0.003, 0.002
MIN_POSITIVES_FOR_THRESHOLD = 30
BOOTSTRAP_SAMPLES, BOOTSTRAP_ALPHA, BOOTSTRAP_SEED = 1000, 0.05, 12345
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
          'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
METRIC_NAMES = ['macro_AUROC', 'macro_AUPRC', 'macro_AUROC_50plus', 'macro_AUPRC_50plus']
FIXED_THRESHOLDS = {label: 0.5 for label in LABELS}
ARMS = ['linear_control', 'classical_residual', 'quantum_residual', 'frozen_quantum']
QUANTUM_ARMS = ['quantum_residual', 'frozen_quantum']
PREPROCESSING = 'E2 RGB -> aspect-preserving bilinear resize/pad320 -> ImageNet normalization'
STAGE = 'develop'
PROTOCOL_PATH = None
CONFIRMATION_RUN = None
qml = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def clean_json(value):
    """Write real JSON null, never the nonstandard NaN / Infinity tokens."""
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(payload), indent=2,
                                    allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_index(path):
    frame = pd.read_csv(path, dtype={'subject_id': 'string', 'study_id': 'string',
                                     'dicom_id': 'string'})
    required = ['image_path', 'dicom_id', 'subject_id', 'study_id', 'ViewPosition'] + LABELS
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f'Index missing columns: {missing}')
    for column in ['image_path', 'dicom_id', 'subject_id', 'study_id']:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'Index contains missing/empty {column}.')
    if frame['dicom_id'].duplicated().any():
        raise ValueError('Duplicate dicom_id values in the image index.')
    # Filter views before any JPEG decoding.
    frame = frame.loc[frame['ViewPosition'].isin(['AP', 'PA'])].copy()
    if frame.empty:
        raise ValueError('No AP/PA images in the index.')
    for label in LABELS:
        frame[label] = pd.to_numeric(frame[label], errors='raise')
        invalid = frame[label].notna() & ~frame[label].isin([-1, 0, 1])
        if invalid.any():
            raise ValueError(f'Unexpected values in {label}: {frame.loc[invalid, label].unique()}')
    # Relative paths, if present, are relative to the index CSV's directory.
    frame['image_path'] = frame['image_path'].map(
        lambda value: str(Path(value) if Path(value).is_absolute() else path.parent / value))
    return frame.reset_index(drop=True)

def read_patient_assignments(path):
    frame = pd.read_csv(path, dtype={'subject_id': 'string'})
    if not {'subject_id', 'split'}.issubset(frame.columns):
        raise ValueError(f'{path} must contain subject_id and split.')
    if frame[['subject_id', 'split']].isna().any().any():
        raise ValueError(f'{path} contains missing split assignments.')
    if not frame['split'].isin(['train', 'val', 'test']).all():
        raise ValueError(f'{path} has unknown split names; expected train / val / test.')
    assignments = frame[['subject_id', 'split']].drop_duplicates()
    if assignments['subject_id'].duplicated().any():
        raise ValueError(f'Patient leakage: multiple splits assigned in {path}.')
    return frame, assignments.set_index('subject_id')['split']

def validate_jpegs(frame):
    audit_rows = []
    for image_path in tqdm(frame['image_path'].drop_duplicates(), desc='Strict JPEG validation'):
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        status, strict_message, permissive_message = 'intact', '', ''
        if not Path(image_path).is_file():
            status, strict_message = 'missing', 'File not found'
        else:
            try:
                with Image.open(image_path) as image:
                    image.load()  # verify() alone does not force full decoding
                    image.convert('L').load()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as strict_error:
                strict_message = repr(strict_error)
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    with Image.open(image_path) as image:
                        image.load()
                        image.convert('L').load()
                    status = 'recoverable_truncated'
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as permissive_error:
                    status, permissive_message = 'fatal', repr(permissive_error)
                finally:
                    ImageFile.LOAD_TRUNCATED_IMAGES = False
        audit_rows.append({'image_path': image_path, 'status': status,
                           'strict_error': strict_message, 'permissive_error': permissive_message})
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return pd.DataFrame(audit_rows)

def encode_labels(raw):
    raw = np.asarray(raw, dtype=np.float32)
    return (np.where(raw == 1, 1.0, 0.0).astype(np.float32), (raw != -1).astype(np.float32))

def calculate_metrics(targets, probabilities, masks, macro_indices=None, stable_indices=None):
    if targets.shape != probabilities.shape or targets.shape != masks.shape:
        raise ValueError('Metric arrays must have identical shapes.')
    if targets.ndim != 2 or targets.shape[1] != len(LABELS):
        raise ValueError('Expected [n_images, 13] metric arrays.')
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Non-finite predicted probabilities.')
    results = {}
    for label_index, label in enumerate(LABELS):
        valid = masks[:, label_index] == 1
        truth = targets[valid, label_index]
        probability = probabilities[valid, label_index]
        positives = int((truth == 1).sum())
        negatives = int((truth == 0).sum())
        prevalence = positives / len(truth) if len(truth) else np.nan
        # Match Exp3's evaluability policy for both metrics.
        auroc = roc_auc_score(truth, probability) if positives and negatives else np.nan
        auprc = average_precision_score(truth, probability) if positives and negatives else np.nan
        results[label] = {'AUROC': float(auroc), 'AUPRC': float(auprc),
                          'prevalence': prevalence,
                          'AUPRC_lift_over_prevalence': auprc / prevalence if prevalence > 0 else np.nan,
                          'positives': positives, 'negatives': negatives, 'n_valid': len(truth)}
    eligible_indices = [i for i, label in enumerate(LABELS)
                        if results[label]['positives'] > 0 and results[label]['negatives'] > 0]
    if macro_indices is None:
        macro_indices = eligible_indices
    if stable_indices is None:
        stable_indices = [i for i in eligible_indices if results[LABELS[i]]['positives'] >= 50]
    for metric in ['AUROC', 'AUPRC']:
        # Ordinary mean deliberately propagates NaN for a degenerate bootstrap
        # draw instead of quietly changing which labels define its macro metric.
        results[f'macro_{metric}'] = (float(np.mean([results[LABELS[i]][metric] for i in macro_indices]))
                                     if len(macro_indices) else np.nan)
        results[f'macro_{metric}_50plus'] = (float(np.mean([results[LABELS[i]][metric] for i in stable_indices]))
                                            if len(stable_indices) else np.nan)
    results['macro_label_count'] = len(macro_indices)
    results['stable_label_count'] = len(stable_indices)
    return results

def threshold_metrics(targets, probabilities, masks, thresholds):
    """None excludes a label entirely; never silently replace it by 0.5."""
    chosen = [i for i, label in enumerate(LABELS) if thresholds.get(label) is not None]
    if not chosen:
        # Preserve table schema when no tune label meets threshold-fit criteria.
        return {'evaluated_labels': 0, 'valid_label_entries': 0}, pd.DataFrame(
            columns=['label', 'threshold', 'accuracy', 'balanced_accuracy', 'precision',
                     'recall_sensitivity', 'specificity', 'F1', 'MCC', 'Jaccard',
                     'TP', 'TN', 'FP', 'FN', 'n_valid', 'NPV'])
    truth = targets[:, chosen].astype(bool)
    valid = masks[:, chosen].astype(bool)
    probs = probabilities[:, chosen]
    predicted = probs >= np.array([thresholds[LABELS[i]] for i in chosen])
    tp = (valid & truth & predicted).sum(0).astype(float)
    tn = (valid & ~truth & ~predicted).sum(0).astype(float)
    fp = (valid & ~truth & predicted).sum(0).astype(float)
    fn = (valid & truth & ~predicted).sum(0).astype(float)
    def divide(a, b):
        return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=np.asarray(b) > 0)
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    accuracy = divide(tp + tn, tp + tn + fp + fn)
    balanced = (recall + specificity) / 2
    mcc = divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    jaccard = divide(tp, tp + fp + fn)
    rows = []
    for offset, label_index in enumerate(chosen):
        rows.append({'label': LABELS[label_index], 'threshold': thresholds[LABELS[label_index]],
                     'accuracy': accuracy[offset], 'balanced_accuracy': balanced[offset],
                     'precision': precision[offset], 'recall_sensitivity': recall[offset],
                     'specificity': specificity[offset], 'F1': f1[offset], 'MCC': mcc[offset],
                     'Jaccard': jaccard[offset], 'TP': int(tp[offset]), 'TN': int(tn[offset]),
                     'FP': int(fp[offset]), 'FN': int(fn[offset]), 'n_valid': int(valid[:, offset].sum())})
    def finite_mean(values):
        values = np.asarray(values)
        return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan
    total = int(valid.sum())
    active_rows = valid.any(1)
    complete_rows = valid.all(1)
    exact_on_observed = ((predicted == truth) | ~valid).all(1)
    row_tp = (valid & predicted & truth).sum(1)
    row_fp = (valid & predicted & ~truth).sum(1)
    row_fn = (valid & ~predicted & truth).sum(1)
    # Samples with empty true/predicted positive union have undefined Jaccard/F1;
    # exclude them from those sample means and expose counts.
    sample_f1 = divide(2 * row_tp, 2 * row_tp + row_fp + row_fn)
    sample_jaccard = divide(row_tp, row_tp + row_fp + row_fn)
    micro_tp, micro_tn, micro_fp, micro_fn = tp.sum(), tn.sum(), fp.sum(), fn.sum()
    report = {
        'evaluated_labels': len(chosen), 'label_names': [LABELS[i] for i in chosen],
        'valid_label_entries': total, 'images_with_observed_labels': int(active_rows.sum()),
        'complete_label_images': int(complete_rows.sum()),
        'label_accuracy': float(divide(micro_tp + micro_tn, total)),
        'hamming_loss': float(divide(micro_fp + micro_fn, total)),
        'all_negative_label_accuracy_baseline': float(divide(micro_tn + micro_fp, total)),
        'subset_accuracy_complete_labels': float(exact_on_observed[complete_rows].mean()) if complete_rows.any() else np.nan,
        'exact_match_observed_labels': float(exact_on_observed[active_rows].mean()) if active_rows.any() else np.nan,
        'micro_precision': float(divide(micro_tp, micro_tp + micro_fp)),
        'micro_recall': float(divide(micro_tp, micro_tp + micro_fn)),
        'micro_F1': float(divide(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn)),
        'micro_Jaccard': float(divide(micro_tp, micro_tp + micro_fp + micro_fn)),
        'samples_F1': finite_mean(sample_f1[active_rows]),
        'samples_Jaccard': finite_mean(sample_jaccard[active_rows]),
        'samples_F1_defined': int(np.isfinite(sample_f1[active_rows]).sum()),
        'macro_accuracy': finite_mean(accuracy), 'macro_balanced_accuracy': finite_mean(balanced),
        'macro_precision': finite_mean(precision), 'macro_recall': finite_mean(recall),
        'macro_specificity': finite_mean(specificity), 'macro_F1': finite_mean(f1),
        'macro_MCC': finite_mean(mcc), 'macro_Jaccard': finite_mean(jaccard),
        'macro_F1_defined_labels': int(np.isfinite(f1).sum()),
        'macro_precision_defined_labels': int(np.isfinite(precision).sum()),
        'brier_score': float(((probs - truth.astype(float))**2)[valid].mean()) if total else np.nan,
        'undefined_policy': 'NaN/null and omitted from macro means; counts/coverage reported',
    }
    if total and truth[valid].any() and (~truth[valid]).any():
        report['micro_AUROC'] = float(roc_auc_score(truth[valid], probs[valid]))
        report['micro_AUPRC'] = float(average_precision_score(truth[valid], probs[valid]))
    else:
        report.update(micro_AUROC=np.nan, micro_AUPRC=np.nan)
    return report, pd.DataFrame(rows)

def fit_validation_thresholds(result):
    thresholds, threshold_rows = {}, []
    for label_index, label in enumerate(LABELS):
        valid = result['masks'][:, label_index] == 1
        truth = result['targets'][valid, label_index]
        probability = result['probabilities'][valid, label_index]
        positives, negatives = int((truth == 1).sum()), int((truth == 0).sum())
        threshold, best_f1 = None, np.nan
        status = 'too_few_positives' if positives < MIN_POSITIVES_FOR_THRESHOLD else 'one_class_only'
        if positives >= MIN_POSITIVES_FOR_THRESHOLD and negatives > 0:
            precision, recall, candidates = precision_recall_curve(truth, probability)
            # Last PR point has no threshold; exclude it. Search all observed
            # cutoffs, fixing the old arbitrary 0.05..0.95 search range.
            scores = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                               out=np.zeros_like(precision[:-1]), where=(precision[:-1] + recall[:-1]) > 0)
            best_index = int(np.argmax(scores))  # stable tie break: lowest threshold
            threshold, best_f1, status = float(candidates[best_index]), float(scores[best_index]), 'fitted'
        thresholds[label] = threshold
        threshold_rows.append({'label': label, 'validation_positives': positives,
                               'validation_negatives': negatives, 'threshold': threshold,
                               'validation_F1': best_f1, 'status': status})
    return thresholds, pd.DataFrame(threshold_rows)


# ResizeAndPad below is taken directly from the supplied Experiment 2.
class ResizeAndPad:

    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        scale = min(self.size / width, self.size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        pad_width = self.size - new_width
        pad_height = self.size - new_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)
        return img


def prepare_data():
    """Reuse E6's exact cohort, label policy, preprocessing, and cache signature."""
    root_candidates = [Path('/workspace/private/unzippedarchive'),
                       Path('/workspace/unzippedarchive/unzippedarchive'),
                       Path('/workspace/unzippedarchive'), Path.cwd()]
    roots = list(dict.fromkeys(p.resolve() for p in root_candidates
                if (p / 'xray_image_label_index.csv').is_file()
                and (p / 'xray_training_experiment_2/best_model.pt').is_file()))
    if ROOT_OVERRIDE:
        ROOT = Path(ROOT_OVERRIDE).expanduser().resolve()
    elif len(roots) == 1:
        ROOT = roots[0]
    else:
        raise RuntimeError(f'Set MIMIC_CXR_ROOT to your dataset root; found {roots}')
    INDEX_CSV = ROOT / 'xray_image_label_index.csv'
    SPLIT_CSV = ROOT / 'xray_training_output/patient_level_splits.csv'
    EXP2_DIR = ROOT / 'xray_training_experiment_2'
    SOURCE_CHECKPOINT, SOURCE_SUMMARY = EXP2_DIR / 'best_model.pt', EXP2_DIR / 'summary.json'
    CACHE_ROOT = ROOT / 'xray_exp2_frozen_feature_cache'
    for path in [INDEX_CSV, SPLIT_CSV, SOURCE_CHECKPOINT, SOURCE_SUMMARY,
                 EXP2_DIR / 'patient_level_splits.csv']:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_DIR = ROOT / 'xray_training_experiment_9_quantum_value' / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    source = torch.load(SOURCE_CHECKPOINT, map_location='cpu', weights_only=True)
    source_summary = json.loads(SOURCE_SUMMARY.read_text())
    assert source['labels'] == LABELS and source['image_size'] == IMAGE_SIZE
    assert source_summary['labels'] == LABELS
    assert source['epoch'] == source_summary['best_epoch']
    for key in ['macro_AUROC', 'macro_AUPRC']:
        assert np.isclose(source[key], source_summary[f'best_validation_{key}']), key
    backbone = densenet121(weights=None)  # checkpoint supplies every weight; no download
    backbone.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(FEATURE_DIM, len(LABELS)))
    backbone.load_state_dict(source['model_state_dict'], strict=True)
    backbone.requires_grad_(False).eval().to(EXTRACT_DEVICE)
    original_weight = backbone.classifier[1].weight.detach().cpu().clone()
    original_bias = backbone.classifier[1].bias.detach().cpu().clone()
    source_metrics = {key: float(source[key]) for key in ['macro_AUROC', 'macro_AUPRC']}
    source_epoch = int(source['epoch'])
    source_weights_for_export = {k: v.detach().cpu().clone() for k,v in source['model_state_dict'].items() if k.startswith('features.')}
    source_state_digest = sha256_file(SOURCE_CHECKPOINT)
    del source

    eval_transform = transforms.Compose([
        ResizeAndPad(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Deterministic features for every arm. No image augmentation in this frozen
    # feature pilot; no augmented validation/test inputs or cached random images.
    df = read_index(INDEX_CSV)
    original_splits, assignments = read_patient_assignments(SPLIT_CSV)
    if not set(df.subject_id).issubset(assignments.index):
        raise ValueError('Image index contains patients absent from the original split.')
    df['split'] = df.subject_id.map(assignments)
    train_subject_ids = set(assignments.index[assignments == 'train'])
    val_subject_ids = set(assignments.index[assignments == 'val'])
    test_subject_ids = set(assignments.index[assignments == 'test'])
    assert train_subject_ids.isdisjoint(val_subject_ids)
    assert train_subject_ids.isdisjoint(test_subject_ids)
    assert val_subject_ids.isdisjoint(test_subject_ids)
    jpeg_audit = validate_jpegs(df.loc[df.split == 'train'])
    jpeg_audit.to_csv(OUTPUT_DIR / 'jpeg_validation_report.csv', index=False)
    unusable = set(jpeg_audit.loc[jpeg_audit.status.isin(['fatal', 'missing']), 'image_path'])
    # E2's complete cohort comparison uses its original split manifest; never decode/extract val/test.
    df = df.loc[~df.image_path.isin(unusable)].reset_index(drop=True)
    print('JPEG audit:', jpeg_audit.status.value_counts().to_dict())
    reference = pd.read_csv(EXP2_DIR / 'patient_level_splits.csv', dtype={'subject_id': 'string', 'dicom_id': 'string'})
    assert not reference.dicom_id.duplicated().any()
    assert set(reference.dicom_id) == set(df.dicom_id), 'Image cohort differs from Experiment 2.'
    aligned = reference.set_index('dicom_id').loc[df.dicom_id]
    for column in ['subject_id', 'split', 'ViewPosition']:
        assert np.array_equal(aligned[column].astype(str), df[column].astype(str)), column
    assert np.array_equal(aligned[LABELS].to_numpy(float), df[LABELS].to_numpy(float), equal_nan=True), 'Labels changed.'
    frames = {name: df.loc[df.split == name].reset_index(drop=True) for name in ['train', 'val', 'test']}
    for name, frame in frames.items():
        assert len(frame), f'Empty split: {name}'
        print(name, len(frame), 'images;', frame.subject_id.nunique(), 'patients')
    for column in ['subject_id', 'study_id', 'dicom_id', 'image_path']:
        sets = [set(frames[name][column]) for name in ['train', 'val', 'test']]
        assert all(sets[i].isdisjoint(sets[j]) for i, j in [(0,1), (0,2), (1,2)]), column
    patients = np.sort(frames['train'].subject_id.astype(str).unique())
    rng = np.random.default_rng(2027)
    rng.shuffle(patients)
    n_fit, n_tune = int(round(.70 * len(patients))), int(round(.15 * len(patients)))
    inner = {str(p): ('fit' if i < n_fit else 'tune' if i < n_fit+n_tune else 'report')
             for i, p in enumerate(patients)}
    assignments_inner = pd.DataFrame({'subject_id': sorted(inner),
                                      'inner_split': [inner[p] for p in sorted(inner)]})
    assignments_inner.to_csv(OUTPUT_DIR / 'e9_inner_patient_splits.csv', index=False)
    frames['train'].loc[:, 'inner_split'] = frames['train'].subject_id.astype(str).map(inner)
    indices = {}
    for name in ['fit', 'tune', 'report']:
        indices[name] = np.flatnonzero(frames['train'].inner_split.to_numpy() == name)
        frames[name] = frames['train'].iloc[indices[name]].reset_index(drop=True)
        assert len(frames[name]), f'Empty inner split: {name}'
    for column in ['subject_id', 'study_id', 'dicom_id', 'image_path']:
        sets = [set(frames[name][column]) for name in ['fit', 'tune', 'report']]
        assert all(sets[i].isdisjoint(sets[j]) for i,j in [(0,1),(0,2),(1,2)]), column
    original_splits.to_csv(OUTPUT_DIR / 'patient_level_splits.csv', index=False)
    # Only original training population appears in E9's saved modeling manifest.
    frames['train'].to_csv(OUTPUT_DIR / 'dataset_manifest.csv', index=False)


    class ExtractionDataset(Dataset):
        def __init__(self, frame): self.paths = frame.image_path.tolist()
        def __len__(self): return len(self.paths)
        def __getitem__(self, index):
            with Image.open(self.paths[index]) as image:
                tensor = eval_transform(image.convert('RGB'))
            return tensor


    @torch.no_grad()
    def extract_features(frame):
        generator = torch.Generator().manual_seed(42)
        kwargs = dict(batch_size=EXTRACT_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=EXTRACT_DEVICE.type == 'cuda', worker_init_fn=seed_worker, generator=generator)
        if NUM_WORKERS: kwargs['multiprocessing_context'] = 'fork'
        loader = DataLoader(ExtractionDataset(frame), **kwargs)
        arrays = []
        backbone.eval()
        for images in tqdm(loader, desc='Frozen DenseNet feature extraction'):
            images = images.to(EXTRACT_DEVICE, non_blocking=True)
            features = F.adaptive_avg_pool2d(F.relu(backbone.features(images), inplace=True), (1,1)).flatten(1)
            arrays.append(features.cpu().numpy())
        result = np.concatenate(arrays).astype(np.float32)
        assert result.shape == (len(frame), FEATURE_DIM) and np.isfinite(result).all()
        return result


    # Preserve E8's train+val cache signature for verified reuse; never load test.
    active_splits = ['train']
    source_hash = sha256_file(SOURCE_CHECKPOINT)
    assert source_hash == source_state_digest, 'Source checkpoint changed while reading.'
    cache_spec = {'revision': 'exp2-rgb320-pad-bilinear-imagenet-fp32-v1', 'checkpoint_sha256': source_hash,
                  'torch': str(torch.__version__), 'torchvision': torchvision.__version__, 'pillow': PIL.__version__,
                  'extraction_device': str(EXTRACT_DEVICE),
                  'gpu': torch.cuda.get_device_name(0) if EXTRACT_DEVICE.type == 'cuda' else None,
                  'batch_size': EXTRACT_BATCH_SIZE,
                  'records': [[str(row.dicom_id), row.split, str(row.image_path), Path(row.image_path).stat().st_size,
                               Path(row.image_path).stat().st_mtime_ns]
                              for row in df.loc[df.split.isin(['train', 'val'])].itertuples(index=False)]}
    cache_key = hashlib.sha256(json.dumps(cache_spec, sort_keys=True).encode()).hexdigest()
    cache_dir = CACHE_ROOT / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_arrays = {}
    cache_start = time.perf_counter()
    for split in active_splits:
        path, manifest_path = cache_dir / f'{split}.npy', cache_dir / f'{split}.json'
        reusable = False
        if path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            reusable = (manifest.get('sha256') == sha256_file(path)
                        and manifest.get('dicom_ids') == frames[split].dicom_id.astype(str).tolist()
                        and manifest.get('cache_spec') == cache_spec)
        if reusable:
            array = np.load(path, allow_pickle=False)
            assert array.shape == (len(frames[split]), FEATURE_DIM) and array.dtype == np.float32
            assert np.isfinite(array).all()
            print('Reusing features:', split, array.shape)
        else:
            manifest_path.unlink(missing_ok=True)
            array = extract_features(frames[split])
            temporary = path.with_suffix('.tmp')
            with open(temporary, 'wb') as stream: np.save(stream, array, allow_pickle=False)
            os.replace(temporary, path)
            save_json(manifest_path, {'sha256': sha256_file(path), 'cache_spec': cache_spec,
                                     'dicom_ids': frames[split].dicom_id.astype(str).tolist()})
        feature_arrays[split] = array
    for name in ['fit', 'tune', 'report']:
        feature_arrays[name] = feature_arrays['train'][indices[name]]
        active_splits.append(name)
    feature_seconds = time.perf_counter() - cache_start
    print(f'Feature cache: {sum(a.nbytes for a in feature_arrays.values()) / 2**20:.1f} MiB; loading/extraction {feature_seconds:.1f}s')
    # Independently re-extract a small sample: detects ordering/stale feature issues.
    spot = extract_features(frames['train'].iloc[:min(3, len(frames['train']))])
    assert np.allclose(spot, feature_arrays['train'][:len(spot)], atol=2e-4, rtol=2e-4), 'Feature-cache verification failed.'

    encoded = {name: encode_labels(frames[name][LABELS].to_numpy(np.float32)) for name in active_splits}

    def make_result(split, probabilities):
        targets, masks = encoded[split]
        return {'targets': targets, 'masks': masks, 'probabilities': probabilities,
                'dicom_ids': frames[split].dicom_id.astype(str).to_numpy(),
                'subject_ids': frames[split].subject_id.astype(str).to_numpy(),
                'views': frames[split].ViewPosition.to_numpy(),
                'metrics': calculate_metrics(targets, probabilities, masks)}


    eval_split = 'tune'
    with torch.no_grad():
        z = torch.from_numpy(feature_arrays[eval_split])
        baseline_probabilities = torch.sigmoid(F.linear(z, original_weight, original_bias)).numpy()
    baseline_result = make_result(eval_split, baseline_probabilities)
    print('Saved E2 checkpoint metrics:', source_metrics)
    print(eval_split, 'source metrics:', {key: baseline_result['metrics'][key] for key in METRIC_NAMES})
    # E2's saved validation metrics are NOT comparable to inner tune metrics.
    # No backbone gradient or BatchNorm updates in any head run.
    del backbone, z
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Train-only standardization, shared by the full classifier and both encoders.
    feature_mean = feature_arrays['fit'].mean(0, dtype=np.float64).astype(np.float32)
    feature_scale = feature_arrays['fit'].std(0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    np.savez(OUTPUT_DIR / 'feature_standardization.npz', mean=feature_mean, scale=feature_scale)
    standardized = {name: np.ascontiguousarray((array - feature_mean) / feature_scale, dtype=np.float32)
                    for name, array in feature_arrays.items()}
    train_targets, train_masks = encoded['fit']
    positives = (train_targets * train_masks).sum(0)
    negatives = ((1-train_targets) * train_masks).sum(0)
    pos_weights = np.clip(np.divide(negatives, positives, out=np.ones(len(LABELS), dtype=np.float32),
                                   where=positives > 0), 1, POS_WEIGHT_CAP)
    patient_counts = frames['fit'].subject_id.value_counts()
    patient_inverse_weight = frames['fit'].subject_id.map(patient_counts).rdiv(1).to_numpy(np.float32)
    patient_inverse_weight /= patient_inverse_weight.mean()
    return SimpleNamespace(
        root=ROOT, output=OUTPUT_DIR, frames=frames, active_splits=active_splits,
        eval_split=eval_split, patient_weights=torch.from_numpy(patient_inverse_weight),
        features={k: torch.from_numpy(v) for k, v in standardized.items()},
        targets={k: torch.from_numpy(encoded[k][0]) for k in active_splits},
        masks={k: torch.from_numpy(encoded[k][1]) for k in active_splits},
        feature_mean=feature_mean, feature_scale=feature_scale,
        original_weight=original_weight, original_bias=original_bias,
        pos_weights=pos_weights, baseline=baseline_result, result=make_result,
        source_hash=source_hash, source_epoch=source_epoch, source_metrics=source_metrics,
        source_path=SOURCE_CHECKPOINT, cache_dir=cache_dir, feature_seconds=feature_seconds,
        backbone_state={k.removeprefix('features.'): v for k, v in source_weights_for_export.items()},
        transform=eval_transform)


# Predeclared E9 design: tune-only selection, report evaluated once after all
# arms, epochs, limits, thresholds and architecture have been fixed.
CORE_LR, READOUT_LR, WARMUP_EPOCHS = 3e-5, 1e-4, 2
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 40, 8, 7
RESIDUAL_LIMIT = 0.10
MEANINGFUL_AP_GAIN, MAX_AUROC_DROP = .003, .002
CANDIDATES = ('Q0', 'Q1', 'Q2')
OBS_DIM = 32


def base_layer(data):
    layer = nn.Linear(FEATURE_DIM, len(LABELS))
    with torch.no_grad():
        layer.weight.copy_(data.original_weight * torch.from_numpy(data.feature_scale)[None])
        layer.bias.copy_(data.original_bias + F.linear(torch.from_numpy(data.feature_mean),
                                                       data.original_weight).flatten())
    layer.requires_grad_(False)
    return layer


def encoder_fit(data, dim, random_projection=False):
    """Only fit features determine the orthogonal, standardized and bounded map."""
    x = data.features['fit'].numpy()
    base = base_layer(data)
    _, singular, vt = np.linalg.svd(base.weight.detach().numpy(), full_matrices=False)
    rank = int((singular > singular.max() * 1e-6).sum())
    basis = vt[:rank].T
    centered = x - x.mean(0)
    residual = centered - (centered @ basis) @ basis.T
    if random_projection:
        gen = np.random.default_rng(2027)
        raw = gen.standard_normal((FEATURE_DIM, dim))
        raw -= basis @ (basis.T @ raw)
        components = np.linalg.qr(raw)[0].T.astype(np.float32)
        explained = None
    else:
        pca = PCA(n_components=dim, svd_solver='randomized', random_state=42)
        pca.fit(residual)
        components = pca.components_.astype(np.float32)
        explained = pca.explained_variance_ratio_.tolist()
    scores = residual @ components.T
    if dim == 48 and not random_projection:
        # Match E8: interleave the 3 PCA blocks wire-by-wire.
        order = np.arange(dim).reshape(-1, 3).T.flatten()
        components, scores = components[order], scores[:, order]
    scale = np.maximum(scores.std(0), 1e-4)
    weight = torch.tensor(components / scale[:, None], dtype=torch.float32)
    bias = -weight @ torch.from_numpy(x.mean(0).astype(np.float32))
    overlap = float(np.max(np.abs(weight.numpy() @ basis)))
    assert overlap < 1e-3, f'Encoder overlaps anchor: {overlap}'
    with torch.no_grad():
        angles = (2 * torch.atan(.5 * F.linear(data.features['fit'], weight, bias))).numpy()
    info = dict(dim=dim, rank=rank, max_anchor_overlap=overlap, explained_variance=explained,
                angle_mean=angles.mean(0).tolist(), angle_std=angles.std(0).tolist(),
                angle_min=angles.min(0).tolist(), angle_max=angles.max(0).tolist(),
                saturation_fraction=(np.abs(angles) > 0.95 * math.pi).mean(0).tolist(),
                output_variance=angles.var(0).tolist(), random_projection=random_projection)
    return (weight, bias), info


class FixedEncoder(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.register_buffer('weight', mapping[0].clone())
        self.register_buffer('bias', mapping[1].clone())

    def forward(self, x):
        return 2 * torch.atan(.5 * F.linear(x, self.weight, self.bias))


def quantum_core(candidate, entangled=True, correlations=True, reupload=True):
    """Q0 matches E8: 48 values in three blocks; Q1/Q2: 16 repeated 4 times."""
    global qml
    if qml is None:
        import pennylane as qml
    layers = 3 if candidate == 'Q0' else 4
    dev = qml.device('default.qubit', wires=8, shots=None)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(inputs, rotations, couplings, final_y):
        expected_dim = 48 if candidate == 'Q0' else 16
        if inputs.shape[-1] != expected_dim:
            raise ValueError(f'{candidate} requires {expected_dim} angles, got {inputs.shape[-1]}')
        if candidate == 'Q0':
            # Reproduce the E8 reference circuit's initial mixer.
            for wire in range(8):
                qml.RY(math.pi / 4, wires=wire)
        for layer in range(layers):
            if candidate == 'Q0' or reupload or layer == 0:
                for wire in range(8):
                    offset = layer * 16 if candidate == 'Q0' else 0
                    # TorchLayer passes [batch, angles]. Index the LAST axis so
                    # PennyLane broadcasts one angle per example for every gate.
                    qml.RY(inputs[..., offset + wire], wires=wire)
                    qml.RZ(inputs[..., offset + 8 + wire], wires=wire)
            for wire in range(8):
                qml.Rot(*[rotations[layer, wire, k] for k in range(3)], wires=wire)
            if entangled:
                # Full alternating ring matchings; Q2 alternates ZZ and XX gates.
                for parity in (0, 1):
                    for wire in range(parity, 8, 2):
                        gate = qml.IsingXX if candidate == 'Q2' and layer % 2 else qml.IsingZZ
                        gate(couplings[layer, wire], wires=(wire, (wire+1) % 8))
        for wire in range(8):
            qml.RY(math.pi/4 + final_y[wire], wires=wire)
        observables = ([qml.expval(qml.PauliZ(w)) for w in range(8)]
                       + [qml.expval(qml.PauliX(w)) for w in range(8)])
        if correlations:
            observables += ([qml.expval(qml.PauliZ(w) @ qml.PauliZ((w+1)%8)) for w in range(8)]
                            + [qml.expval(qml.PauliX(w) @ qml.PauliX((w+1)%8)) for w in range(8)])
        return observables

    module = qml.qnn.TorchLayer(circuit, {'rotations': (layers,8,3),
                                         'couplings': (layers,8), 'final_y': (8,)},
                                init_method={'rotations': lambda t: nn.init.normal_(t, std=.05),
                                             'couplings': lambda t: nn.init.constant_(t, .15),
                                             'final_y': lambda t: nn.init.zeros_(t)})
    if not entangled:
        module.couplings.requires_grad_(False)
    return module, circuit


class MatchedCore(nn.Module):
    """Tiny tanh MLP, exactly the same encoded input and observation dimension.

    Choose hidden width minimizing |(input_dim+out_dim+1)*width+out_dim
    - VQC_parameter_count|. Q0: 113 vs 104; Q1/Q2: 130 vs 136.
    """
    def __init__(self, dim, core_count, out_dim):
        super().__init__()
        # A shared low-rank periodic perceptron; closest possible width by count.
        widths = range(1, 17)
        hidden = min(widths, key=lambda h: abs((dim + out_dim + 1)*h + out_dim - core_count))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out_dim), nn.Tanh())

    def forward(self, angles):
        return self.net(angles)


class StrongCore(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim*2, 64), nn.GELU(),
                                 nn.Linear(64, out_dim), nn.Tanh())

    def forward(self, angles):
        return self.net(torch.cat([angles.sin(), angles.cos()], dim=-1))


class Head(nn.Module):
    def __init__(self, data, kind, seed, candidate='Q2', mapping=None,
                 limit=RESIDUAL_LIMIT, entangled=True, correlations=True, reupload=True):
        super().__init__()
        seed_everything(seed)
        if candidate not in CANDIDATES:
            raise ValueError(f'Unknown quantum candidate: {candidate}')
        if kind not in ('linear_control', 'quantum', 'frozen_quantum', 'no_entanglement',
                        'local_only', 'single_encode', 'random_encoder',
                        'matched_classical', 'strong_classical'):
            raise ValueError(f'Unknown head kind: {kind}')
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError('Residual limit must be finite and positive')
        self.kind, self.candidate, self.limit = kind, candidate, limit
        self.base = base_layer(data)
        self.circuit = None
        if kind != 'linear_control':
            expected_dim = 48 if candidate == 'Q0' else 16
            if (mapping is None or len(mapping) != 2
                    or tuple(mapping[0].shape) != (expected_dim, FEATURE_DIM)
                    or tuple(mapping[1].shape) != (expected_dim,)):
                raise ValueError(f'{kind}/{candidate}: expected encoder weight '
                                 f'[{expected_dim}, {FEATURE_DIM}] and bias [{expected_dim}]')
            if not all(torch.isfinite(tensor).all() for tensor in mapping):
                raise ValueError('Encoder contains nonfinite values')
            self.encoder = FixedEncoder(mapping)
            dim = len(mapping[1])
            out_dim = 32 if correlations else 16
            self.is_quantum = kind in ('quantum', 'frozen_quantum', 'no_entanglement',
                                       'local_only', 'single_encode', 'random_encoder')
            if self.is_quantum:
                self.core, self.circuit = quantum_core(candidate, entangled, correlations, reupload)
            else:
                quantum_count = (3 if candidate == 'Q0' else 4)*8*4+8
                self.core = MatchedCore(dim, quantum_count, out_dim) if kind == 'matched_classical' else StrongCore(dim, out_dim)
            self.readout = nn.Linear(out_dim, len(LABELS))
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
            self.register_buffer('observation_mean', torch.zeros(out_dim))
            self.register_buffer('observation_scale', torch.ones(out_dim))
            if kind == 'frozen_quantum':
                self.core.requires_grad_(False)

    def observations(self, x):
        return self.core(self.encoder(x)).float()

    def components(self, x):
        base = self.base(x)
        if self.kind == 'linear_control':
            return base, torch.zeros_like(base)
        obs = (self.observations(x) - self.observation_mean) / self.observation_scale
        raw = self.readout(obs)
        return base, self.limit * torch.tanh(raw / self.limit)

    def forward(self, x):
        a, b = self.components(x)
        return a + b


@torch.no_grad()
def init_scaling(model, x):
    if model.kind == 'linear_control': return
    model.eval()
    # Fit only; online means avoid holding the full N x 32 matrix.
    n = 0
    total = torch.zeros_like(model.observation_mean, dtype=torch.float64)
    square = total.clone()
    for batch in x.split(HEAD_BATCH_SIZE):
        obs = model.observations(batch).double()
        total += obs.sum(0); square += obs.square().sum(0); n += len(batch)
    mean = total/n
    model.observation_mean.copy_(mean.float())
    model.observation_scale.copy_((square/n - mean.square()).clamp_min(0).sqrt().clamp_min(.1).float())


def core_vector(model):
    return torch.cat([p.detach().flatten().cpu() for p in model.core.parameters()])


def inspect_model(model, data, split):
    if model.kind == 'linear_control': return {}
    model.eval()
    with torch.no_grad():
        obs = np.concatenate([model.observations(x).numpy() for x in data.features[split].split(HEAD_BATCH_SIZE)])
    return {'observable_mean': obs.mean(0).tolist(), 'observable_std': obs.std(0).tolist(),
            'observable_variance': obs.var(0).tolist(), 'observable_covariance': np.cov(obs, rowvar=False).tolist(),
            'readout_weight_norm': float(model.readout.weight.norm())}


@torch.no_grad()
def evaluate(model, data, split):
    model.eval()
    probs, logits_all, stats = [], [], []
    for x in data.features[split].split(HEAD_BATCH_SIZE):
        base, residual = model.components(x)
        logits = base + residual
        if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite logits')
        probs.append(logits.sigmoid().numpy()); logits_all.append(logits.numpy())
        stats.append(torch.stack([base, residual], dim=-1).numpy())
    result = data.result(split, np.concatenate(probs))
    result['logits'] = np.concatenate(logits_all)
    values = np.concatenate(stats)
    b, r = values[...,0], values[...,1]
    result['residual_rms'] = float(np.sqrt(np.mean(r*r)))
    result['residual_mean'] = float(r.mean()); result['residual_std'] = float(r.std())
    result['residual_max_abs'] = float(np.abs(r).max())
    result['base_logit_rms'] = float(np.sqrt(np.mean(b*b)))
    result['residual_base_ratio'] = result['residual_rms']/max(1e-12, result['base_logit_rms'])
    return result


def train(model, data, seed, directory):
    """Early stop on TUNE AUPRC only. Save final and selected core separately."""
    directory.mkdir(parents=True, exist_ok=False)
    if model.kind == 'linear_control':
        result = evaluate(model, data, 'tune')
        save_checkpoint(model, data, seed, 0, directory)
        return {'epoch': 0, 'history': [], 'initial': None, 'final': None, 'gradient_nonzero': False}
    init_scaling(model, data.features['fit'])
    initial = core_vector(model).clone()
    optimizer = torch.optim.AdamW([
        {'params': model.readout.parameters(), 'lr': READOUT_LR, 'name': 'readout'},
        {'params': list(p for p in model.core.parameters() if p.requires_grad),
         'lr': CORE_LR, 'name': 'core'}], weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(reduction='none',
                                     pos_weight=torch.as_tensor(data.pos_weights))
    # First two epochs: frozen quantum/classical core; readout only.
    best, best_epoch, best_state, history = -float('inf'), 0, None, []
    best_grad = False
    for epoch in range(1, MAX_EPOCHS+1):
        model.train(); seed_everything(seed*1000+epoch)
        permutation = torch.randperm(len(data.features['fit']))
        losses, grad_l2, grad_max = [], 0., 0.
        for idx in permutation.split(HEAD_BATCH_SIZE):
            x = data.features['fit'][idx]
            mask = data.masks['fit'][idx] * data.patient_weights[idx,None]
            if mask.sum().item() == 0: continue
            optimizer.zero_grad(set_to_none=True)
            # no_grad prevents huge quantum autograd graphs during warmup.
            base = model.base(x)
            if epoch <= WARMUP_EPOCHS:
                with torch.no_grad(): observed = model.observations(x)
                obs = (observed - model.observation_mean)/model.observation_scale
                residual = model.limit * torch.tanh(model.readout(obs)/model.limit)
            else:
                residual = model.components(x)[1]
            logits = base + residual
            loss = (criterion(logits, data.targets['fit'][idx])*mask).sum()/mask.sum()
            loss = loss + RESIDUAL_L2 * residual.square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if epoch > WARMUP_EPOCHS:
                grads = [p.grad.detach() for p in model.core.parameters() if p.grad is not None]
                if grads:
                    grad_l2 = max(grad_l2, math.sqrt(sum(float(g.square().sum()) for g in grads)))
                    grad_max = max(grad_max, max(float(g.abs().max()) for g in grads))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          1., error_if_nonfinite=True)
            if epoch <= WARMUP_EPOCHS:
                optimizer.param_groups[1]['lr'] = 0.
            else:
                optimizer.param_groups[1]['lr'] = CORE_LR
            optimizer.step()
            losses.append(float(loss))
        outcome = evaluate(model, data, 'tune')
        change = float((core_vector(model)-initial).norm())
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                   macro_AUPRC=outcome['metrics']['macro_AUPRC'],
                   macro_AUROC=outcome['metrics']['macro_AUROC'],
                   grad_L2=grad_l2, grad_max=grad_max, core_change_L2=change,
                   core_LR=optimizer.param_groups[1]['lr'], readout_LR=READOUT_LR,
                   **{key: outcome[key] for key in ('residual_rms','residual_mean','residual_std',
                                                    'residual_max_abs','base_logit_rms','residual_base_ratio')})
        history.append(row)
        score = outcome['metrics']['macro_AUPRC']
        if np.isfinite(score) and score > best + STOP_MIN_DELTA:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch >= MIN_EPOCHS and epoch-best_epoch >= PATIENCE: break
    final = core_vector(model).clone()
    if best_state is None: raise RuntimeError('No finite tune score')
    model.load_state_dict(best_state, strict=True)
    save_checkpoint(model, data, seed, best_epoch, directory)
    pd.DataFrame(history).to_csv(directory/'training_history.csv', index=False)
    torch.save({'initial_core': initial, 'selected_core': core_vector(model), 'final_core': final},
               directory/'quantum_parameters.pt')
    return {'epoch': best_epoch, 'history': history, 'initial': initial,
            'final': final, 'gradient_nonzero': any(r['grad_L2'] > 0 for r in history[WARMUP_EPOCHS:])}


def save_checkpoint(model, data, seed, epoch, directory):
    payload = {'experiment': 9, 'kind': model.kind, 'candidate': model.candidate,
               'seed': seed, 'epoch': epoch, 'labels': LABELS, 'limit': model.limit,
               'source_sha256': data.source_hash, 'feature_mean': torch.as_tensor(data.feature_mean),
               'feature_scale': torch.as_tensor(data.feature_scale), 'preprocessing': PREPROCESSING,
               'state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               'source_backbone_state': data.backbone_state}
    path = directory/'best_model.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)


def load_inference_bundle(path, image_device='cpu'):
    """Load selected arm as an image-to-probabilities model without E2 files.

    Returns (nn.Module, torchvision image transform, ordered labels, metadata).
    Image input is a batch of preprocessed RGB 320x320 tensors.
    """
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle['labels'] != LABELS or bundle['preprocessing'] != PREPROCESSING:
        raise ValueError('Incompatible label order or image preprocessing')
    state = bundle['state_dict']; kind=bundle['kind']
    mapping = ((torch.zeros_like(state['encoder.weight']),
                torch.zeros_like(state['encoder.bias'])) if kind!='linear_control' else None)
    dummy = SimpleNamespace(original_weight=torch.zeros(len(LABELS),FEATURE_DIM),
                            original_bias=torch.zeros(len(LABELS)),
                            feature_mean=np.zeros(FEATURE_DIM,dtype=np.float32),
                            feature_scale=np.ones(FEATURE_DIM,dtype=np.float32))
    head = Head(dummy,kind,bundle['seed'],bundle['candidate'],mapping,bundle['limit'],
                entangled=kind!='no_entanglement',correlations=kind!='local_only',
                reupload=kind!='single_encode')
    head.load_state_dict(state,strict=True)
    backbone=densenet121(weights=None).features
    backbone.load_state_dict(bundle['source_backbone_state'],strict=True)
    class Export(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=backbone.eval().requires_grad_(False).to(image_device)
            self.head=head.eval().requires_grad_(False)
            self.register_buffer('mean',bundle['feature_mean'].clone())
            self.register_buffer('scale',bundle['feature_scale'].clone())
        @torch.no_grad()
        def forward(self,images):
            x=F.adaptive_avg_pool2d(F.relu(self.backbone(images.to(image_device)),inplace=False),(1,1)).flatten(1)
            x=((x.cpu()-self.mean)/self.scale).float()
            return self.head(x).sigmoid()
    transform=transforms.Compose([ResizeAndPad(IMAGE_SIZE),transforms.ToTensor(),
                 transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return Export(),transform,LABELS,bundle


def ensemble(results, data, split):
    result = data.result(split, np.mean([r['probabilities'] for r in results], axis=0))
    result['logits'] = np.mean([r['logits'] for r in results], axis=0)
    result['residual_rms'] = float(np.mean([r['residual_rms'] for r in results]))
    return result


def patient_metrics(result):
    """E8 sensitivity definition: mean observed probability, any observed positive."""
    patients, inverse = np.unique(result['subject_ids'], return_inverse=True)
    targets = np.zeros((len(patients), len(LABELS)), np.float32)
    masks = np.zeros_like(targets)
    probs = np.full_like(targets, .5)
    for p in range(len(patients)):
        indices = inverse == p
        for j in range(len(LABELS)):
            observed = indices & (result['masks'][:, j] == 1)
            if observed.any():
                masks[p,j] = 1
                targets[p,j] = result['targets'][observed,j].max()
                probs[p,j] = result['probabilities'][observed,j].mean()
    return calculate_metrics(targets, probs, masks)


def metric_block(result, thresholds):
    summary, rows = threshold_metrics(result['targets'], result['probabilities'], result['masks'], thresholds)
    summary.update({k:v for k,v in result['metrics'].items() if not isinstance(v,dict)})
    return summary, rows


def extended(result, thresholds):
    summary, rows = metric_block(result, thresholds)
    for index, row in rows.iterrows():
        tp,tn,fp,fn = (row[k] for k in ('TP','TN','FP','FN'))
        rows.loc[index,'NPV'] = tn/(tn+fn) if tn+fn else np.nan
    valid = result['masks'].astype(bool)
    probs = result['probabilities'][valid]; truth = result['targets'][valid]
    if len(probs):
        bins = np.minimum((probs*10).astype(int), 9)
        summary['ECE'] = sum(float(np.abs(probs[bins == b].mean()-truth[bins == b].mean()) * (bins == b).mean())
                             for b in range(10) if (bins == b).any())
    tp,tn,fp,fn = [float(rows[k].sum()) for k in ('TP','TN','FP','FN')]
    summary['micro_specificity'] = tn/(tn+fp) if tn+fp else np.nan
    summary['micro_MCC'] = ((tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
                            if min(tp+fp,tp+fn,tn+fp,tn+fn)>0 else np.nan)
    summary['micro_NPV'] = tn/(tn+fn) if tn+fn else np.nan
    return summary, rows


def patient_bootstrap(results, data, samples=BOOTSTRAP_SAMPLES):
    """Identical resampled PATIENT indices for every arm and metric; fixed label set."""
    first = next(iter(results.values()))
    ids = first['subject_ids']; patients, inverse = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(patients))]
    baseline = first['metrics']
    eligible = [i for i,l in enumerate(LABELS) if baseline[l]['positives'] and baseline[l]['negatives']]
    stable = [i for i in eligible if baseline[LABELS[i]]['positives'] >= 50]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {name: {key: [] for key in METRIC_NAMES} for name in results}
    for _ in tqdm(range(samples), desc='Patient bootstrap'):
        chosen = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        for name, result in results.items():
            m = calculate_metrics(result['targets'][idx], result['probabilities'][idx],
                                  result['masks'][idx], eligible, stable)
            for key in METRIC_NAMES: draws[name][key].append(m[key])
    ci, contrasts = [], []
    for name, result in results.items():
        for key in METRIC_NAMES:
            arr = np.asarray(draws[name][key]); arr = arr[np.isfinite(arr)]
            ci.append(dict(arm=name, metric=key, point=result['metrics'][key],
                           lower=np.percentile(arr,2.5) if len(arr) else np.nan,
                           upper=np.percentile(arr,97.5) if len(arr) else np.nan,
                           finite_draws=len(arr)))
    for name in results:
        if name == 'quantum': continue
        for key in METRIC_NAMES:
            delta = np.array(draws['quantum'][key]) - np.array(draws[name][key])
            finite = delta[np.isfinite(delta)]
            contrasts.append(dict(contrast=f'quantum - {name}', metric=key,
                                  point=results['quantum']['metrics'][key]-results[name]['metrics'][key],
                                  lower=np.percentile(finite,2.5) if len(finite) else np.nan,
                                  upper=np.percentile(finite,97.5) if len(finite) else np.nan,
                                  finite_draws=len(finite)))
    return pd.DataFrame(ci), pd.DataFrame(contrasts)


def plot_results(output, report, histories, comparisons, chosen):
    major = [a for a in ('source','linear_control','matched_classical','strong_classical',
                          'frozen_quantum','quantum') if a in report]
    def savefig(name):
        plt.tight_layout(); plt.savefig(output/name, dpi=160); plt.close()
    fig,ax = plt.subplots(figsize=(11,5))
    ax.bar(major,[report[a]['metrics']['macro_AUPRC'] for a in major]); ax.tick_params(axis='x',rotation=30)
    ax.set_ylabel('Report macro AUPRC'); savefig('model_comparison.png')
    for key, name in [('AUPRC','per_class_AUPRC.png'),('AUROC','per_class_AUROC.png')]:
        fig,ax=plt.subplots(figsize=(14,5))
        matrix = [[report[a]['metrics'][l][key] for l in LABELS] for a in major]
        im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1)
        ax.set_xticks(range(len(LABELS)),LABELS,rotation=70,ha='right'); ax.set_yticks(range(len(major)),major)
        fig.colorbar(im,ax=ax); savefig(name)
    for metric, filename in [('macro_AUPRC','training_curves.png'),
                             ('macro_AUPRC','auprc_vs_epoch.png'),('macro_AUROC','auroc_vs_epoch.png'),
                             ('grad_L2','quantum_gradient.png'),('core_change_L2','quantum_change.png'),
                             ('residual_rms','quantum_training_diagnostics.png')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for (arm, seed), h in histories.items():
            if len(h) and (arm == 'quantum' or metric.startswith('macro_')):
                ax.plot([r['epoch'] for r in h],[r[metric] for r in h],label=f'{arm} {seed}')
        ax.set(xlabel='Epoch', ylabel=metric); ax.legend(fontsize=6); savefig(filename)
    for curve, filename in [('roc','roc_curves.png'),('pr','pr_curves.png'),
                            ('calibration','calibration_curves.png')]:
        fig, axes=plt.subplots(4,4,figsize=(17,15)); axes=axes.ravel()
        for i,label in enumerate(LABELS):
            ax=axes[i]
            for arm in major:
                r=report[arm]; valid=r['masks'][:,i].astype(bool)
                y=r['targets'][valid,i]; p=r['probabilities'][valid,i]
                if not len(y) or not y.any() or y.all(): continue
                if curve=='roc':
                    from sklearn.metrics import roc_curve
                    x,v,_=roc_curve(y,p)
                elif curve=='pr':
                    precision,recall,_=precision_recall_curve(y,p)
                    x,v=recall,precision
                else:
                    from sklearn.calibration import calibration_curve
                    v,x=calibration_curve(y,p,n_bins=10,strategy='uniform')
                ax.plot(x,v,label=arm,linewidth=1)
            ax.set_title(label,fontsize=9)
        axes[-1].axis('off'); axes[0].legend(fontsize=5); savefig(filename)
    for normalized, filename in [(False,'confusion_matrices_counts.png'),
                                 (True,'confusion_matrices_normalized.png')]:
        rows = comparisons['quantum']['05'][1].set_index('label')
        fig,axes=plt.subplots(4,4,figsize=(15,14))
        for ax,label in zip(axes.ravel(),LABELS):
            row=rows.loc[label]
            matrix=np.array([[row.TN,row.FP],[row.FN,row.TP]],dtype=float)
            if normalized:
                matrix/=np.maximum(matrix.sum(1,keepdims=True),1)
            ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1 if normalized else None)
            for i in range(2):
                for j in range(2): ax.text(j,i,f'{matrix[i,j]:.2f}' if normalized else f'{matrix[i,j]:.0f}',ha='center',va='center')
            ax.set_title(label,fontsize=9); ax.set_xticks([0,1],['Neg','Pos']); ax.set_yticks([0,1],['Neg','Pos'])
        axes.ravel()[-1].axis('off'); savefig(filename)


def validate_quantum_batching():
    """Fail before JPEG auditing if a circuit does not broadcast over examples."""
    for candidate in CANDIDATES:
        seed_everything(42)
        layer, _ = quantum_core(candidate)
        dim = 48 if candidate == 'Q0' else 16
        angles = torch.zeros(2, dim, requires_grad=True)
        observations = layer(angles)
        if observations.shape != (2, OBS_DIM) or not torch.isfinite(observations).all():
            raise RuntimeError(f'{candidate}: expected finite [2, {OBS_DIM}] observations, '
                               f'got {tuple(observations.shape)}')
        observations.sum().backward()
        if angles.grad is None or not torch.isfinite(angles.grad).all():
            raise RuntimeError(f'{candidate}: invalid input gradients')


def main():
    global qml
    if importlib.util.find_spec('pennylane') is None:
        raise RuntimeError('Install pennylane==0.45.1 on Vast.ai before running e9.py')
    import pennylane as quantum_library
    qml = quantum_library
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    seed_everything(42)
    validate_quantum_batching()
    data=prepare_data(); out=data.output
    import shutil
    source_path = None
    if '__file__' in globals():
        candidate = Path(__file__).expanduser()
        if candidate.is_file():
            source_path = candidate
    if source_path is None:
        # A notebook cell has no __file__. Prefer an explicitly supplied script,
        # then an uploaded e9.py in the notebook's working directory.
        for candidate in (os.environ.get('E9_SOURCE_FILE'), 'e9.py'):
            if candidate and Path(candidate).expanduser().is_file():
                source_path = Path(candidate).expanduser()
                break
    if source_path is not None:
        shutil.copy2(source_path, out/'e9.py')
    else:
        # When the entire script was pasted into one cell, preserve that cell.
        try:
            shell = get_ipython()
        except NameError:
            shell = None
        cells = shell.history_manager.input_hist_raw if shell is not None else []
        full_script = next((cell for cell in reversed(cells)
                            if 'def prepare_data(' in cell and 'def main(' in cell
                            and "if __name__ == '__main__':" in cell), None)
        if full_script is None:
            raise RuntimeError('Cannot locate the E9 source in this notebook. Set E9_SOURCE_FILE '
                               'to the uploaded e9.py path, or run the entire script in one cell.')
        (out/'e9.py').write_text(full_script, encoding='utf-8')
    save_json(out/'configuration.json',dict(seeds=SEEDS,candidates=CANDIDATES,inner_split_seed=2027,
              ratios=[.70,.15,.15],epochs=MAX_EPOCHS,patience=PATIENCE,warmup=WARMUP_EPOCHS,
              core_lr=CORE_LR,readout_lr=READOUT_LR,weight_decay=WEIGHT_DECAY,
              residual_l2=RESIDUAL_L2,limit=RESIDUAL_LIMIT,bootstrap=BOOTSTRAP_SAMPLES,
              no_test_evaluation=True, gate={'min_AUPRC_gain':MEANINGFUL_AP_GAIN,
              'max_AUROC_drop':MAX_AUROC_DROP,'bootstrap_lower_gt_zero':True}))
    save_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,
              torchvision=torchvision.__version__,pennylane=qml.__version__,platform=platform.platform()))
    save_json(out/'source_checkpoint_hash.json',dict(path=str(data.source_path),sha256=data.source_hash,
                                                       epoch=data.source_epoch,cache=str(data.cache_dir),
                                                       train_cache_sha256=sha256_file(data.cache_dir/'train.npy')))
    encoders={}
    for candidate,dim in [('Q0',48),('Q1',16),('Q2',16)]:
        encoders[candidate], info = encoder_fit(data,dim)
        save_json(out/f'encoder_{candidate}.json',info)
        seed_everything(42)
        drawing = Head(data,'quantum',42,candidate,encoders[candidate])
        (out/f'quantum_circuit_{candidate}.txt').write_text(
            qml.draw(drawing.circuit)(torch.zeros(dim),
              drawing.core.rotations, drawing.core.couplings, drawing.core.final_y),encoding='utf-8')
        del drawing
    candidate_runs={}; candidates=[]; histories={}
    # Architecture search uses ONLY fit/tune; not even source report predictions yet.
    for candidate in CANDIDATES:
        for seed in SEEDS:
            model=Head(data,'quantum',seed,candidate,encoders[candidate])
            record=train(model,data,seed,out/f'candidate_{candidate}_seed_{seed}')
            score=evaluate(model,data,'tune')
            candidates.append(dict(candidate=candidate,seed=seed,selected_epoch=record['epoch'],
                                   tune_AUPRC=score['metrics']['macro_AUPRC'],
                                   tune_AUROC=score['metrics']['macro_AUROC'],
                                   core_change=float((core_vector(model)-record['initial']).norm()),
                                   gradient_nonzero=record['gradient_nonzero']))
            candidate_runs[candidate,seed]=(model,record)
            histories[f'candidate_{candidate}',seed]=record['history']
    table=pd.DataFrame(candidates); table.to_csv(out/'architecture_candidates.csv',index=False)
    baseline_tune=data.baseline['metrics']['macro_AUROC']
    eligible=table.groupby('candidate').agg({'tune_AUPRC':'mean','tune_AUROC':'mean',
                                               'gradient_nonzero':'all','core_change':'min'})
    good=eligible[(eligible.tune_AUROC>=baseline_tune-MAX_AUROC_DROP)&
                  eligible.gradient_nonzero&(eligible.core_change>0)]
    # If none meet safeguards, choose best AUPRC and explicitly label ineligible.
    selected=(good if len(good) else eligible).sort_values('tune_AUPRC',ascending=False).index[0]
    selection=dict(selected=selected,eligible=selected in good.index,table=eligible.reset_index().to_dict('records'),
                   selection_policy='max mean tune AUPRC subject to AUROC tolerance, gradients and change; fallback if none')
    save_json(out/'architecture_selection.json',selection)
    # Tune-only residual diagnostic. Separate models, no report scores used.
    limits=[]
    for limit in (.05,.10,.20):
        if limit == RESIDUAL_LIMIT:
            m,rec=candidate_runs[selected,SEEDS[0]]
        else:
            m=Head(data,'quantum',SEEDS[0],selected,encoders[selected],limit)
            rec=train(m,data,SEEDS[0],out/f'limit_{limit:.2f}_seed_{SEEDS[0]}')
        limits.append(dict(limit=limit,tune_AUPRC=evaluate(m,data,'tune')['metrics']['macro_AUPRC']))
    pd.DataFrame(limits).to_csv(out/'residual_limit_diagnostic.csv',index=False)
    limit=sorted(limits,key=lambda r:(-r['tune_AUPRC'],r['limit']))[0]['limit']
    # If limit changed, retrain all seeds of the selected architecture, fit/tune only.
    if limit != RESIDUAL_LIMIT:
        for seed in SEEDS:
            m=Head(data,'quantum',seed,selected,encoders[selected],limit)
            rec=train(m,data,seed,out/f'selected_quantum_seed_{seed}')
            candidate_runs[selected,seed]=(m,rec)
    selection['selected_limit']=limit; save_json(out/'architecture_selection.json',selection)
    # Change only the projection, never the selected circuit's input dimension.
    random_encoder, random_info = encoder_fit(data, len(encoders[selected][1]), True)
    save_json(out/'encoder_random.json', random_info)
    models={'quantum':{},'frozen_quantum':{},'matched_classical':{},'strong_classical':{},
            'no_entanglement':{},'local_only':{},'random_encoder':{}}
    if selected != 'Q0':models['single_encode']={}
    for seed in SEEDS:
        models['quantum'][seed]=candidate_runs[selected,seed]
        for kind in models:
            if kind=='quantum':continue
            mapping=random_encoder if kind=='random_encoder' else encoders[selected]
            model=Head(data,kind,seed,selected,mapping,limit,
                       entangled=kind!='no_entanglement',correlations=kind!='local_only',
                       reupload=kind!='single_encode')
            rec=train(model,data,seed,out/f'{kind}_seed_{seed}')
            models[kind][seed]=(model,rec); histories[kind,seed]=rec['history']
        histories['quantum',seed]=models['quantum'][seed][1]['history']
    linear=Head(data,'linear_control',42)
    with torch.no_grad():
        source_logits=F.linear(data.features['fit'][:32]*torch.from_numpy(data.feature_scale)+
                 torch.from_numpy(data.feature_mean),data.original_weight,data.original_bias)
        assert torch.allclose(linear(data.features['fit'][:32]),source_logits,atol=3e-5,rtol=3e-5)
    train(linear,data,42,out/'linear_control_seed_42')
    # Freeze selection NOW. All report evaluations happen only below.
    report={'source':evaluate(linear,data,'report'),'linear_control':evaluate(linear,data,'report')}
    tune=ensemble([evaluate(models['quantum'][seed][0],data,'tune') for seed in SEEDS],data,'tune')
    thresholds,threshold_rows=fit_validation_thresholds(tune)
    threshold_rows.to_csv(out/'tune_thresholds.csv',index=False)
    save_json(out/'thresholds.json',thresholds)
    report_seeds={}; diagnostics=[]; ablations=[]; rows=[]
    for kind,series in models.items():
        report_seeds[kind]={}
        for seed,(model,record) in series.items():
            r=evaluate(model,data,'report'); report_seeds[kind][seed]=r
            change=core_vector(model)-record['initial']
            diag=inspect_model(model,data,'fit')
            if model.is_quantum:
                save_json(out/f'{kind}_seed_{seed}_observables.json',diag)
            diagnostics.append(dict(arm=kind,seed=seed,epoch=record['epoch'],
                parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                core_count=sum(p.numel() for p in model.core.parameters() if p.requires_grad),
                core_change_L2=float(change.norm()),
                core_change_relative=float(change.norm()/(record['initial'].norm()+1e-12)),
                gradient_nonzero=record['gradient_nonzero'],
                max_gradient=max([h['grad_L2'] for h in record['history']],default=0),
                readout_weight_norm=diag['readout_weight_norm'],
                residual_rms=r['residual_rms']))
            seed_metrics,_=extended(r,FIXED_THRESHOLDS)
            rows.append(dict(arm=kind,seed=seed,epoch=record['epoch'],
                             parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             core_parameters=sum(p.numel() for p in model.core.parameters() if p.requires_grad),
                             **{key:r['metrics'][key] for key in METRIC_NAMES},
                             **{key:seed_metrics[key] for key in ('micro_AUROC','micro_AUPRC','label_accuracy',
                               'macro_balanced_accuracy','micro_recall','micro_specificity','micro_precision',
                               'micro_F1','micro_MCC','brier_score')},
                             residual_rms=r['residual_rms'],core_change_L2=float(change.norm())))
        report[kind]=ensemble(list(report_seeds[kind].values()),data,'report')
    # Circuit reset retains trained readout and normalization; no re-fitting.
    reset_results=[]; disabled_results=[]
    for seed in SEEDS:
        model,record=models['quantum'][seed]
        model=copy.deepcopy(model)
        # TorchLayer parameter iteration order is the saved initial order.
        with torch.no_grad():
            position=0
            for param in model.core.parameters():
                count=param.numel()
                param.copy_(record['initial'][position:position+count].reshape_as(param))
                position+=count
        reset_results.append(evaluate(model,data,'report'))
        # Gate suppression of all entanglers at inference, keeping readout fixed.
        model=copy.deepcopy(models['quantum'][seed][0])
        with torch.no_grad(): model.core.couplings.zero_()
        disabled_results.append(evaluate(model,data,'report'))
    report['circuit_reset']=ensemble(reset_results,data,'report')
    report['inference_no_entanglement']=ensemble(disabled_results,data,'report')
    for kind in ('circuit_reset','inference_no_entanglement','no_entanglement',
                 'local_only','single_encode','random_encoder','frozen_quantum'):
        if kind in report:
            ablations.append(dict(ablation=kind,
                macro_AUPRC=report[kind]['metrics']['macro_AUPRC'],
                macro_AUROC=report[kind]['metrics']['macro_AUROC'],
                AUPRC_delta=report['quantum']['metrics']['macro_AUPRC']-report[kind]['metrics']['macro_AUPRC'],
                AUROC_delta=report['quantum']['metrics']['macro_AUROC']-report[kind]['metrics']['macro_AUROC'],
                probability_RMS=float(np.sqrt(np.mean((report['quantum']['probabilities']-report[kind]['probabilities'])**2))),
                logit_RMS=float(np.sqrt(np.mean((report['quantum']['logits']-report[kind]['logits'])**2)))))
    pd.DataFrame(ablations).to_csv(out/'quantum_ablation_metrics.csv',index=False)
    pd.DataFrame([dict(ablation=k,label=l,
                       AUPRC_delta=report['quantum']['metrics'][l]['AUPRC']-report[k]['metrics'][l]['AUPRC'],
                       AUROC_delta=report['quantum']['metrics'][l]['AUROC']-report[k]['metrics'][l]['AUROC'])
                  for k in ('circuit_reset','frozen_quantum','no_entanglement','inference_no_entanglement')
                  for l in LABELS]).to_csv(out/'quantum_ablation_per_class.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'quantum_diagnostics.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'experiment_comparison.csv',index=False)
    pd.DataFrame([dict(arm=kind,metric=key,mean=float(np.mean([r['metrics'][key] for r in series.values()])),
                       std=float(np.std([r['metrics'][key] for r in series.values()],ddof=1)) if len(series)>1 else np.nan)
                  for kind,series in report_seeds.items() for key in METRIC_NAMES]).to_csv(out/'seed_summary.csv',index=False)
    comparisons={}; class_rows=[]; predictions=[]; confusion={'05':[],'fitted':[]}
    classifications={'05':[],'fitted':[]}
    for kind,r in report.items():
        comparisons[kind]={}
        for t_name, thresholds_here in [('05',FIXED_THRESHOLDS),('fitted',thresholds)]:
            summary, classes=extended(r,thresholds_here)
            classes.insert(0,'arm',kind)
            comparisons[kind][t_name]=(summary,classes)
            classes.to_csv(out/f'{kind}_classification_per_class_{t_name}.csv',index=False)
            save_json(out/f'{kind}_threshold_metrics_{t_name}.json',summary)
            classifications[t_name].append(classes)
            confusion[t_name].append(classes[['arm','label','TP','TN','FP','FN']])
            confusion[t_name].append(pd.DataFrame([dict(arm=kind,label='MICRO',
                **{k:int(classes[k].sum()) for k in ('TP','TN','FP','FN')})]))
        for label in LABELS: class_rows.append(dict(arm=kind,label=label,**r['metrics'][label]))
        for i in range(len(r['subject_ids'])):
            row=dict(arm=kind,subject_id=r['subject_ids'][i],dicom_id=r['dicom_ids'][i])
            for j,l in enumerate(LABELS):
                row[f'{l}_prob']=r['probabilities'][i,j]
                row[f'{l}_target']=r['targets'][i,j] if r['masks'][i,j] else np.nan
            predictions.append(row)
    pd.DataFrame([dict(arm=kind, **{key:value for key,value in patient_metrics(r).items()
                                   if not isinstance(value,dict)}) for kind,r in report.items()]).to_csv(
                                   out/'report_patient_metrics.csv',index=False)
    pd.DataFrame(class_rows).to_csv(out/'report_per_class_metrics.csv',index=False)
    pd.DataFrame(class_rows).to_csv(out/'per_class_model_comparison.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'report_predictions.csv',index=False)
    for key in confusion:
        pd.concat(confusion[key],ignore_index=True).to_csv(out/f'report_confusion_matrices_{key}.csv',index=False)
        pd.concat(classifications[key],ignore_index=True).to_csv(
            out/f'report_classification_per_class_{key}.csv',index=False)
    save_json(out/'threshold_metrics.json', {k:v['fitted'][0] for k,v in comparisons.items()})
    ci,paired=patient_bootstrap(report,data)
    ci.to_csv(out/'bootstrap_ci.csv',index=False)
    paired.to_csv(out/'paired_patient_bootstrap_differences.csv',index=False)
    plot_results(out,report,histories,comparisons,selected)
    # Gate is predeclared above; no post-hoc tuning from report.
    qml_ap=report['quantum']['metrics']['macro_AUPRC']; qml_auc=report['quantum']['metrics']['macro_AUROC']
    def delta(name): return qml_ap-report[name]['metrics']['macro_AUPRC']
    def paired_lower(name):
        row=paired[(paired.contrast==f'quantum - {name}')&(paired.metric=='macro_AUPRC')]
        return float(row.iloc[0]['lower'])
    gates={
        'parameters_changed':all(float((core_vector(models['quantum'][s][0])-models['quantum'][s][1]['initial']).norm())>0 for s in SEEDS),
        'nonzero_gradients':all(models['quantum'][s][1]['gradient_nonzero'] for s in SEEDS),
        'reset_reduces_AUPRC':delta('circuit_reset')>0,
        'beats_frozen':delta('frozen_quantum')>0,
        'beats_matched_classical':delta('matched_classical')>0,
        'meaningful_source_gain':delta('source')>=MEANINGFUL_AP_GAIN,
        'AUROC_tolerance':qml_auc>=report['source']['metrics']['macro_AUROC']-MAX_AUROC_DROP,
        'paired_CI_source':paired_lower('source')>0,
        'paired_CI_frozen':paired_lower('frozen_quantum')>0,
        'paired_CI_matched':paired_lower('matched_classical')>0,
        'paired_CI_strong_classical':paired_lower('strong_classical')>0,
        'paired_CI_reset':paired_lower('circuit_reset')>0,
        'entanglement_contribution':delta('no_entanglement')>0,
        'paired_CI_no_entanglement':paired_lower('no_entanglement')>0,
        'seed_consistency':all(report_seeds['quantum'][s]['metrics']['macro_AUPRC'] >
                           report_seeds['frozen_quantum'][s]['metrics']['macro_AUPRC'] for s in SEEDS)}
    summary=dict(selected=selected,limit=limit,official=SEEDS==[42,43,44],
                 candidate_for_E10=all(gates.values()) and SEEDS==[42,43,44],gates=gates,
                 report_note='Head-development holdout only: E2 DenseNet already trained on all original training patients. Not independent clinical validation.',
                 not_a_quantum_speedup=True,ensemble_metrics={k:r['metrics'] for k,r in report.items()})
    save_json(out/'summary.json',summary)
    protocol=dict(architecture=selected,qubits=8,layers=3 if selected=='Q0' else 4,
                  entanglement='ZZ alternating ring' if selected!='Q2' else 'alternating ZZ/XX ring',
                  encoding='fit-only fixed complementary PCA',angle_dim=48 if selected=='Q0' else 16,
                  observables=['Z','X','ZZ','XX'],residual_limit=limit,
                  core_LR=CORE_LR,readout_LR=READOUT_LR,warmup=WARMUP_EPOCHS,
                  epoch_policy='best tune macro AUPRC per seed',epochs={str(s):models['quantum'][s][1]['epoch'] for s in SEEDS},
                  seeds=SEEDS,thresholds=thresholds,preprocessing=PREPROCESSING,
                  source_checkpoint_sha256=data.source_hash,feature_mean=data.feature_mean,
                  feature_scale=data.feature_scale,weights_checkpoint_pattern='candidate_<Q>_seed_<seed>/best_model.pt (or selected_quantum_seed_<seed>)')
    save_json(out/'protocol.json',protocol)
    save_json(out/'e10_handoff.json',dict(**protocol,config_sha256=sha256_file(out/'configuration.json'),
            e9_source_sha256=sha256_file(out/'e9.py'),candidate_for_E10=summary['candidate_for_E10']))
    (out/'RESULTS.md').write_text('# Experiment 9\n\n'+summary['report_note']+'\n\n'
       +f"Selected: {selected}; residual bound: {limit}; E10 gate: {summary['candidate_for_E10']}\n\n"
       +'Gate results:\n'+''.join(f'- {k}: {v}\n' for k,v in gates.items())
       +'\nThese head-development findings cannot establish hardware or computational quantum advantage.\n',encoding='utf-8')
    print('\nE9 report by seed:')
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nE9 report ensembles:')
    print(pd.DataFrame([dict(arm=k,macro_AUROC=r['metrics']['macro_AUROC'],
                      macro_AUPRC=r['metrics']['macro_AUPRC'],
                      **{m:comparisons[k]['05'][0].get(m) for m in ('micro_AUROC','micro_AUPRC',
                        'label_accuracy','macro_balanced_accuracy','micro_recall','micro_specificity',
                        'micro_precision','micro_F1','micro_MCC','brier_score','ECE')})
                        for k,r in report.items()]).to_string(index=False))
    print('\nPaired bootstrap contrasts:'); print(paired.to_string(index=False))
    print('EXPERIMENT 9 E10 CANDIDATE:', 'YES' if summary['candidate_for_E10'] else 'NO')
    print('Gate:',gates,'\nOutput:',out)


if __name__ == '__main__':
    main()
