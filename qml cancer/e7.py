"""Experiment 7: controlled quantum encoding and circuit improvements.

Vast.ai notebook (upload this file; E6's a.py is NOT required):
    %run b.py
Quick execution check with synthetic data, no dataset needed:
    %run b.py --self-check
One-seed exploratory run:
    %env E7_SEEDS=42
    %run b.py
Set MIMIC_CXR_ROOT if automatic dataset discovery is ambiguous.

Requires the working Torch/torchvision environment used for E6, plus numpy,
pandas, Pillow, scikit-learn, matplotlib, tqdm, and pennylane==0.45.1.
Missing PennyLane is installed only when running, never on import.
Does not reinstall Torch. Heads use CPU; frozen image extraction uses CUDA.

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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import densenet121
from tqdm.auto import tqdm

# Configuration: deliberately two encoders, one main circuit, two targeted controls.
ROOT_OVERRIDE = os.environ.get('MIMIC_CXR_ROOT', '').strip()
SEEDS = [int(s) for s in os.environ.get('E7_SEEDS', '42,43,44').split(',')]
IMAGE_SIZE, FEATURE_DIM = 320, 1024
N_QUBITS, Q_LAYERS = 8, 3
ANGLE_DIM, OBSERVABLE_DIM = 2 * N_QUBITS * Q_LAYERS, 4 * N_QUBITS
EXTRACT_BATCH_SIZE, HEAD_BATCH_SIZE = 24, 128
NUM_WORKERS = 4 if sys.platform.startswith('linux') else 0
EXTRACT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_HEAD_THREADS = 4
BASE_EPOCHS, MAX_EPOCHS = 40, 50
WARMUP_EPOCHS, MIN_EPOCHS, PATIENCE = 3, 15, 12
STOP_MIN_DELTA = 0.0001
BASE_LR, ENCODER_LR, CORE_LR, READOUT_LR = 1e-5, 5e-5, 1e-3, 2e-4
WEIGHT_DECAY, RESIDUAL_L2 = 1e-3, 0.05
DROPOUT, BRANCH_DROPOUT = 0.30, 0.10
RESIDUAL_LIMIT, POS_WEIGHT_CAP = 0.75, 5.0
# Validation is already used across E2/E6/E7: results remain exploratory.
# Leave test off until the complete procedure is chosen in advance.
RUN_TEST = False
MIN_POSITIVES_FOR_THRESHOLD = 30
BOOTSTRAP_SAMPLES, BOOTSTRAP_ALPHA, BOOTSTRAP_SEED = 500, 0.05, 12345
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
          'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
METRIC_NAMES = ['macro_AUROC', 'macro_AUPRC', 'macro_AUROC_50plus', 'macro_AUPRC_50plus']
FIXED_THRESHOLDS = {label: 0.5 for label in LABELS}
ARMS = ['linear_control', 'pca_classical', 'pca_quantum',
        'task_classical', 'task_quantum', 'task_frozen', 'task_no_entanglement']
QUANTUM_ARMS = [arm for arm in ARMS if arm.endswith(('quantum', 'frozen', 'entanglement'))]
PREPROCESSING = 'E2 RGB -> aspect-preserving bilinear resize/pad320 -> ImageNet normalization'
qml = None


def ensure_quantum_dependency():
    global qml
    if importlib.util.find_spec('pennylane') is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pennylane==0.45.1'])
    import pennylane
    qml = pennylane


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
        return {'evaluated_labels': 0, 'valid_label_entries': 0}, pd.DataFrame()
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
    OUTPUT_DIR = ROOT / 'xray_training_experiment_7_encoding' / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
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
    jpeg_audit = validate_jpegs(df)
    jpeg_audit.to_csv(OUTPUT_DIR / 'jpeg_validation_report.csv', index=False)
    unusable = set(jpeg_audit.loc[jpeg_audit.status.isin(['fatal', 'missing']), 'image_path'])
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
    original_splits.to_csv(OUTPUT_DIR / 'patient_level_splits.csv', index=False)
    df.to_csv(OUTPUT_DIR / 'dataset_manifest.csv', index=False)


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


    active_splits = ['train', 'val'] + (['test'] if RUN_TEST else [])
    source_hash = sha256_file(SOURCE_CHECKPOINT)
    assert source_hash == source_state_digest, 'Source checkpoint changed while reading.'
    cache_spec = {'revision': 'exp2-rgb320-pad-bilinear-imagenet-fp32-v1', 'checkpoint_sha256': source_hash,
                  'torch': str(torch.__version__), 'torchvision': torchvision.__version__, 'pillow': PIL.__version__,
                  'extraction_device': str(EXTRACT_DEVICE),
                  'gpu': torch.cuda.get_device_name(0) if EXTRACT_DEVICE.type == 'cuda' else None,
                  'batch_size': EXTRACT_BATCH_SIZE,
                  'records': [[str(row.dicom_id), row.split, str(row.image_path), Path(row.image_path).stat().st_size,
                               Path(row.image_path).stat().st_mtime_ns]
                              for row in df.loc[df.split.isin(active_splits)].itertuples(index=False)]}
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
    feature_seconds = time.perf_counter() - cache_start
    print(f'Feature cache: {sum(a.nbytes for a in feature_arrays.values()) / 2**20:.1f} MiB; loading/extraction {feature_seconds:.1f}s')
    # Independently re-extract a small sample: detects ordering/stale feature issues.
    spot = extract_features(frames['val'].iloc[:min(3, len(frames['val']))])
    assert np.allclose(spot, feature_arrays['val'][:len(spot)], atol=2e-4, rtol=2e-4), 'Feature-cache verification failed.'

    encoded = {name: encode_labels(frames[name][LABELS].to_numpy(np.float32)) for name in active_splits}

    def make_result(split, probabilities):
        targets, masks = encoded[split]
        return {'targets': targets, 'masks': masks, 'probabilities': probabilities,
                'dicom_ids': frames[split].dicom_id.astype(str).to_numpy(),
                'subject_ids': frames[split].subject_id.astype(str).to_numpy(),
                'views': frames[split].ViewPosition.to_numpy(),
                'metrics': calculate_metrics(targets, probabilities, masks)}


    with torch.no_grad():
        z = torch.from_numpy(feature_arrays['val'])
        baseline_probabilities = torch.sigmoid(F.linear(z, original_weight, original_bias)).numpy()
    baseline_result = make_result('val', baseline_probabilities)
    print('Saved E2 checkpoint metrics:', source_metrics)
    print('Reproduced FP32 E2 metrics:', {key: baseline_result['metrics'][key] for key in METRIC_NAMES})
    for key in ['macro_AUROC', 'macro_AUPRC']:
        if abs(baseline_result['metrics'][key] - source_metrics[key]) > .003:
            raise RuntimeError(f'Exp2 reproduction differs by >0.003 for {key}. Check preprocessing/checkpoint before head training.')
    # No backbone gradient or BatchNorm updates in any head run.
    del backbone, z
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Train-only standardization, shared by the full classifier and both encoders.
    feature_mean = feature_arrays['train'].mean(0, dtype=np.float64).astype(np.float32)
    feature_scale = feature_arrays['train'].std(0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    np.savez(OUTPUT_DIR / 'feature_standardization.npz', mean=feature_mean, scale=feature_scale)
    standardized = {name: np.ascontiguousarray((array - feature_mean) / feature_scale, dtype=np.float32)
                    for name, array in feature_arrays.items()}
    train_targets, train_masks = encoded['train']
    positives = (train_targets * train_masks).sum(0)
    negatives = ((1-train_targets) * train_masks).sum(0)
    pos_weights = np.clip(np.divide(negatives, positives, out=np.ones(len(LABELS), dtype=np.float32),
                                   where=positives > 0), 1, POS_WEIGHT_CAP)
    return SimpleNamespace(
        root=ROOT, output=OUTPUT_DIR, frames=frames, active_splits=active_splits,
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


def quantum_core(entangled=True):
    """48 angles in three blocks; 104 trainable circuit parameters; 32 observables.

    The final RY mixer ensures the last diagonal ZZ gates can affect Z readout.
    No-entanglement removes ZZ and freezes its unused parameters.
    Exact analytic state-vector simulation, not a hardware advantage claim.
    """
    device = qml.device('default.qubit', wires=N_QUBITS, shots=None)

    @qml.qnode(device, interface='torch', diff_method='backprop')
    def circuit(inputs, rotations, couplings, final_y):
        for wire in range(N_QUBITS):
            qml.RY(math.pi / 4, wires=wire)
        for layer in range(Q_LAYERS):
            offset = 2 * N_QUBITS * layer
            for wire in range(N_QUBITS):
                qml.RY(inputs[..., offset + wire], wires=wire)
                qml.RZ(inputs[..., offset + N_QUBITS + wire], wires=wire)
                qml.Rot(*[rotations[layer, wire, i] for i in range(3)], wires=wire)
            if entangled:
                # Alternating nearest-neighbor matchings, including the wrap edge.
                for parity in (0, 1):
                    for wire in range(parity, N_QUBITS, 2):
                        qml.IsingZZ(couplings[layer, wire], wires=[wire, (wire + 1) % N_QUBITS])
        for wire in range(N_QUBITS):
            qml.RY(math.pi / 4 + final_y[wire], wires=wire)
        return ([qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]
                + [qml.expval(qml.PauliX(w)) for w in range(N_QUBITS)]
                + [qml.expval(qml.PauliZ(w) @ qml.PauliZ((w + 1) % N_QUBITS)) for w in range(N_QUBITS)]
                + [qml.expval(qml.PauliX(w) @ qml.PauliX((w + 1) % N_QUBITS)) for w in range(N_QUBITS)])

    layer = qml.qnn.TorchLayer(circuit, {
        'rotations': (Q_LAYERS, N_QUBITS, 3),
        'couplings': (Q_LAYERS, N_QUBITS), 'final_y': (N_QUBITS,)},
        init_method={'rotations': lambda t: nn.init.normal_(t, std=0.05),
                     'couplings': lambda t: nn.init.constant_(t, 0.15),
                     'final_y': lambda t: nn.init.zeros_(t)})
    if not entangled:
        layer.couplings.requires_grad_(False)
    return layer


class ClassicalCore(nn.Module):
    """Classical replacement receives the EXACT same angles as its VQC partner.

    A substantial Fourier-feature MLP control: 4,160 parameters versus the
    circuit's 104. This is input/output matched, NOT parameter matched; the
    larger control avoids favoring QML through an artificially tiny baseline.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * ANGLE_DIM, 32), nn.GELU(),
                                 nn.Linear(32, OBSERVABLE_DIM), nn.Tanh())

    def forward(self, angles):
        return self.net(torch.cat([angles.sin(), angles.cos()], dim=-1))


class AngleEncoder(nn.Module):
    """Frozen 1024->48 map, followed by 96 mildly adaptable scale/shift values."""
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.zeros(ANGLE_DIM, FEATURE_DIM))
        self.register_buffer('bias', torch.zeros(ANGLE_DIM))
        self.gain_raw = nn.Parameter(torch.full((ANGLE_DIM,), -math.log(2)))
        self.shift_raw = nn.Parameter(torch.zeros(ANGLE_DIM))

    def forward(self, x):
        z = F.linear(x, self.weight, self.bias)
        gain = 0.5 + 1.5 * self.gain_raw.sigmoid()  # initially 1, bounded [0.5, 2]
        # Slower saturation than pi*tanh; preserves signs and feature magnitudes.
        return 2 * torch.atan(0.5 * gain * z + 0.25 * self.shift_raw.tanh())


class ResidualHead(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        seed_everything(seed)
        self.base = nn.Linear(FEATURE_DIM, len(LABELS))
        self.base_dropout = nn.Dropout(DROPOUT)
        if arm != 'linear_control':
            self.encoder = AngleEncoder()
            self.branch_dropout = nn.Dropout(BRANCH_DROPOUT)
            # Create shared objects before arm-specific random initializers.
            self.readout = nn.Linear(OBSERVABLE_DIM, len(LABELS))
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
            self.core = ClassicalCore() if arm.endswith('classical') else quantum_core(arm != 'task_no_entanglement')
            self.base.requires_grad_(False)
            if arm == 'task_frozen':
                self.core.requires_grad_(False)
            self.register_buffer('observation_mean', torch.zeros(OBSERVABLE_DIM))
            self.register_buffer('observation_scale', torch.ones(OBSERVABLE_DIM))

    def components(self, x, base_only=False):
        if self.arm == 'linear_control':
            base = self.base(self.base_dropout(x))
            return base, torch.zeros_like(base)
        base = self.base(x)  # fixed deterministic anchor even during training
        if base_only:
            return base, torch.zeros_like(base)
        angles = self.branch_dropout(self.encoder(x))
        observed = self.core(angles).to(dtype=x.dtype)
        observed = (observed - self.observation_mean) / self.observation_scale
        raw = self.readout(observed)
        residual = RESIDUAL_LIMIT * torch.tanh(raw / RESIDUAL_LIMIT)
        return base, residual

    def forward(self, x):
        base, residual = self.components(x)
        return base + residual


def fit_encoder_maps(data, pca, base):
    """Both maps use training images only. Task logits come from the frozen anchor."""
    x = data.features['train']
    components = torch.from_numpy(pca.components_.astype(np.float32))
    center = torch.from_numpy(pca.mean_.astype(np.float32))
    scores = F.linear(x - center, components)
    scale = scores.std(dim=0, unbiased=False).clamp_min(1e-4)
    pca_weight = components / scale[:, None]
    pca_bias = -pca_weight @ center
    with torch.no_grad():
        logits = base(x)
        logit_scale = logits.std(dim=0, unbiased=False).clamp_min(0.1)
        task_weight = base.weight.detach() / logit_scale[:, None]
        task_bias = (base.bias.detach() - logits.mean(0)) / logit_scale
    remaining = ANGLE_DIM - len(LABELS)
    # Spread leading PCs/task logits across layers and both rotation axes.
    order = torch.arange(ANGLE_DIM).reshape(-1, Q_LAYERS).T.flatten()
    maps = {
        'pca': (pca_weight[order], pca_bias[order]),
        'task': (torch.cat([task_weight, pca_weight[:remaining]])[order],
                 torch.cat([task_bias, pca_bias[:remaining]])[order])}
    return maps


@torch.no_grad()
def initialize_observation_scaling(model, train_x):
    # All training images, deterministic, with dropout disabled. This is per-core
    # readout normalization, not a change to the common input encoder.
    model.eval()
    total = torch.zeros(OBSERVABLE_DIM, dtype=torch.float64)
    square = torch.zeros_like(total)
    for batch in train_x.split(HEAD_BATCH_SIZE):
        obs = model.core(model.encoder(batch)).double()
        total += obs.sum(0)
        square += obs.square().sum(0)
    mean = total / len(train_x)
    std = (square / len(train_x) - mean.square()).clamp_min(0).sqrt().clamp_min(0.1)
    model.observation_mean.copy_(mean.float())
    model.observation_scale.copy_(std.float())


def make_loss(data):
    return nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.as_tensor(data.pos_weights))


@torch.no_grad()
def evaluate_head(model, data, split='val', base_only=False):
    model.eval()
    probabilities, loss_sum, valid_count, residual_sq, count = [], 0., 0., 0., 0
    criterion = make_loss(data)
    for start in range(0, len(data.features[split]), HEAD_BATCH_SIZE):
        selected = slice(start, start + HEAD_BATCH_SIZE)
        base, residual = model.components(data.features[split][selected], base_only)
        logits = base + residual
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite evaluation logits')
        mask = data.masks[split][selected]
        loss_sum += (criterion(logits, data.targets[split][selected]) * mask).sum().item()
        valid_count += mask.sum().item()
        residual_sq += residual.square().sum().item()
        count += residual.numel()
        probabilities.append(logits.sigmoid().numpy())
    result = data.result(split, np.concatenate(probabilities))
    result.update(loss=loss_sum / max(valid_count, 1), residual_logit_rms=math.sqrt(residual_sq / count))
    return result


def atomic_torch_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, data, seed, epoch, result):
    return {'experiment': 'Experiment 7 encoding', 'arm': model.arm, 'seed': seed, 'epoch': epoch,
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'labels': LABELS, 'n_qubits': N_QUBITS, 'q_layers': Q_LAYERS,
            'feature_dim': FEATURE_DIM, 'image_size': IMAGE_SIZE, 'residual_limit': RESIDUAL_LIMIT,
            'feature_mean': torch.as_tensor(data.feature_mean), 'feature_scale': torch.as_tensor(data.feature_scale),
            'source_checkpoint_sha256': data.source_hash, 'source_epoch': data.source_epoch,
            'macro_AUROC': float(result['metrics']['macro_AUROC']),
            'macro_AUPRC': float(result['metrics']['macro_AUPRC']),
            'pennylane_version': str(qml.__version__),
            'note': 'Inference checkpoint; epoch 0 residual is inactive. Not an optimizer resume file.'}


def train_head(model, data, seed, directory):
    """First train an anchor; then train every residual with that anchor frozen."""
    arm = model.arm
    is_base = arm == 'linear_control'
    if is_base:
        groups = [{'params': list(model.base.parameters()), 'lr': BASE_LR, 'name': 'base'}]
    else:
        model.encoder.requires_grad_(False)
        model.core.requires_grad_(False)
        groups = [{'params': list(model.readout.parameters()), 'lr': READOUT_LR, 'name': 'readout'},
                  {'params': list(model.encoder.parameters()), 'lr': 0., 'name': 'encoder'}]
        if arm != 'task_frozen':
            core_params = [p for name, p in model.core.named_parameters()
                           if not (arm == 'task_no_entanglement' and name == 'couplings')]
            groups.append({'params': core_params, 'lr': 0., 'name': 'core'})
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    generator = torch.Generator().manual_seed(seed + 1000)
    seed_everything(seed + 2000)
    criterion = make_loss(data)
    initial = evaluate_head(model, data)
    best_score = initial['metrics']['macro_AUPRC']
    if not math.isfinite(best_score):
        raise RuntimeError('No evaluable validation macro AUPRC')
    best_epoch, stale, stopping_reference = 0, 0, best_score
    initial_state = copy.deepcopy(model.state_dict())
    atomic_torch_save(checkpoint_payload(model, data, seed, 0, initial), directory / 'best_model.pt')
    history = [{'epoch': 0, 'train_loss': np.nan, 'validation_loss': initial['loss'],
                **{k: initial['metrics'][k] for k in METRIC_NAMES}, 'residual_logit_rms': initial['residual_logit_rms'],
                'encoder_gradient_max': 0., 'core_gradient_max': 0.}]
    max_epochs = BASE_EPOCHS if is_base else MAX_EPOCHS
    started_total = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        if not is_base and epoch == WARMUP_EPOCHS + 1:
            model.encoder.requires_grad_(True)
            if arm != 'task_frozen':
                model.core.requires_grad_(True)
                if arm == 'task_no_entanglement':
                    model.core.couplings.requires_grad_(False)
            for group in optimizer.param_groups:
                if group['name'] == 'encoder':
                    group['lr'] = ENCODER_LR
                elif group['name'] == 'core':
                    group['lr'] = CORE_LR
            stale, stopping_reference = 0, best_score
        model.train()
        order = torch.randperm(len(data.features['train']), generator=generator)
        loss_sum, valid_count, encoder_grad, core_grad = 0., 0., 0., 0.
        started = time.perf_counter()
        for selected in order.split(HEAD_BATCH_SIZE):
            mask = data.masks['train'][selected]
            denominator = mask.sum()
            if denominator.item() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            base, residual = model.components(data.features['train'][selected])
            masked_sum = (criterion(base + residual, data.targets['train'][selected]) * mask).sum()
            loss = masked_sum / denominator + RESIDUAL_L2 * residual.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{arm}: non-finite loss')
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            if not is_base:
                def grad_norm(module):
                    return math.sqrt(sum(p.grad.square().sum().item() for p in module.parameters() if p.grad is not None))
                encoder_grad = max(encoder_grad, grad_norm(model.encoder))
                core_grad = max(core_grad, grad_norm(model.core))
            optimizer.step()
            loss_sum += masked_sum.detach().item()
            valid_count += denominator.item()
        result = evaluate_head(model, data)
        score = result['metrics']['macro_AUPRC']
        if not math.isfinite(score):
            raise FloatingPointError('Non-finite validation score')
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            atomic_torch_save(checkpoint_payload(model, data, seed, epoch, result), directory / 'best_model.pt')
        if is_base or epoch > WARMUP_EPOCHS:
            scheduler.step(score)
            if score > stopping_reference + STOP_MIN_DELTA:
                stale, stopping_reference = 0, score
            else:
                stale += 1
        row = {'epoch': epoch, 'train_loss': loss_sum / max(valid_count, 1),
               'validation_loss': result['loss'], **{k: result['metrics'][k] for k in METRIC_NAMES},
               'residual_logit_rms': result['residual_logit_rms'],
               'encoder_gradient_max': encoder_grad, 'core_gradient_max': core_grad,
               'seconds': time.perf_counter() - started,
               **{f'lr_{g["name"]}': g['lr'] for g in optimizer.param_groups}}
        history.append(row)
        pd.DataFrame(history).to_csv(directory / 'training_history.csv', index=False)
        print(f'{arm} seed={seed} {epoch:02d}/{max_epochs} | AUROC {result["metrics"]["macro_AUROC"]:.5f} '
              f'AP {score:.5f} | residual {row["residual_logit_rms"]:.3f} | '
              f'core grad {core_grad:.3g} | {row["seconds"]:.1f}s' + (' | best' if improved else ''), flush=True)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if not is_base and arm != 'task_frozen' and max(r['core_gradient_max'] for r in history) <= 0:
        raise RuntimeError(f'{arm}: core did not receive gradients')
    best = torch.load(directory / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'], strict=True)
    result = evaluate_head(model, data)
    assert abs(result['metrics']['macro_AUPRC'] - best_score) < 1e-7
    diagnostics = {}
    if not is_base:
        for name in ('base', 'encoder', 'core', 'readout'):
            diagnostics[f'{name}_parameter_change_l2'] = math.sqrt(sum(
                (p.detach() - initial_state[f'{name}.{key}']).square().sum().item()
                for key, p in getattr(model, name).named_parameters()))
        assert diagnostics['base_parameter_change_l2'] == 0, 'Frozen anchor changed'
        if arm == 'task_frozen':
            assert diagnostics['core_parameter_change_l2'] == 0
    info = {'arm': arm, 'seed': seed, 'checkpoint_epoch': best_epoch,
            'epochs_completed': len(history) - 1, 'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters_final_stage': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'core_parameters': 0 if is_base else sum(p.numel() for p in model.core.parameters()),
            'residual_logit_rms': result['residual_logit_rms'],
            'seconds': time.perf_counter() - started_total, **diagnostics}
    save_json(directory / 'training_summary.json', info)
    return result, info, initial_state


def bootstrap_report(result):
    """Exact patient-cluster bootstrap using multiplicity weights.

    Sort each label's scores once, group score ties, then evaluate all patient
    draws in NumPy. Patient multiplicities are mathematically equivalent to
    concatenating their images repeatedly. Undefined fixed-label draws stay NaN.
    """
    subjects, inverse = np.unique(np.asarray(result['subject_ids']), return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patient_counts = np.stack([np.bincount(rng.choice(len(subjects),size=len(subjects),replace=True),
                                         minlength=len(subjects)) for _ in range(BOOTSTRAP_SAMPLES)])
    image_weights = patient_counts[:,inverse].astype(np.float64)
    point = result['metrics']
    eligible = [i for i,label in enumerate(LABELS) if point[label]['positives'] and point[label]['negatives']]
    stable = [i for i in eligible if point[LABELS[i]]['positives']>=50]
    per_label = np.full((BOOTSTRAP_SAMPLES,len(LABELS),2),np.nan)
    for column in range(len(LABELS)):
        valid = np.flatnonzero(result['masks'][:,column]==1)
        if not len(valid): continue
        order = valid[np.argsort(-result['probabilities'][valid,column],kind='mergesort')]
        scores = result['probabilities'][order,column]
        truth = result['targets'][order,column].astype(np.float64)
        # End of each equal-score group; weighted ties match sklearn definitions.
        ends = np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
        weights = image_weights[:,order]
        tp = np.cumsum(weights*truth[None,:],axis=1)[:,ends]
        fp = np.cumsum(weights*(1-truth)[None,:],axis=1)[:,ends]
        positive,negative = tp[:,-1],fp[:,-1]
        good = (positive>0)&(negative>0)
        tpr = np.divide(tp,positive[:,None],out=np.zeros_like(tp),where=positive[:,None]>0)
        fpr = np.divide(fp,negative[:,None],out=np.zeros_like(fp),where=negative[:,None]>0)
        previous_tpr = np.concatenate([np.zeros((BOOTSTRAP_SAMPLES,1)),tpr[:,:-1]],axis=1)
        auc = np.sum((tpr+previous_tpr)*np.diff(fpr,axis=1,prepend=0),axis=1)/2
        precision = np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        ap = np.sum(np.diff(tpr,axis=1,prepend=0)*precision,axis=1)
        per_label[good,column,0],per_label[good,column,1] = auc[good],ap[good]
    macros = np.full((BOOTSTRAP_SAMPLES,len(METRIC_NAMES)),np.nan)
    if eligible:
        macros[:,0]=per_label[:,eligible,0].mean(axis=1)
        macros[:,1]=per_label[:,eligible,1].mean(axis=1)
    if stable:
        macros[:,2]=per_label[:,stable,0].mean(axis=1)
        macros[:,3]=per_label[:,stable,1].mean(axis=1)
    keys=[('macro',metric) for metric in METRIC_NAMES]+[(label,metric) for label in LABELS for metric in ['AUROC','AUPRC']]
    draws=np.concatenate([macros,per_label.reshape(BOOTSTRAP_SAMPLES,-1)],axis=1)
    rows=[]
    for column,(label,metric) in enumerate(keys):
        finite=draws[:,column][np.isfinite(draws[:,column])]
        lower,upper=np.quantile(finite,[BOOTSTRAP_ALPHA/2,1-BOOTSTRAP_ALPHA/2]) if len(finite) else [np.nan,np.nan]
        rows.append({'label':label,'metric':metric,'point_estimate':point[metric] if label=='macro' else point[label][metric],
                     'ci_lower':lower,'ci_upper':upper,'confidence_level':1-BOOTSTRAP_ALPHA,
                     'bootstrap_samples_requested':BOOTSTRAP_SAMPLES,'bootstrap_samples_valid':len(finite),
                     'bootstrap_samples_undefined':BOOTSTRAP_SAMPLES-len(finite)})
    return pd.DataFrame(rows),macros

def save_evaluation(result, directory, prefix, fitted_thresholds=None, fit_thresholds=False):
    per_class = pd.DataFrame([{'label': label, **result['metrics'][label]} for label in LABELS])
    per_class.to_csv(directory / f'{prefix}_per_class_metrics.csv', index=False)
    predictions = pd.DataFrame({'subject_id': result['subject_ids'], 'dicom_id': result['dicom_ids'],
                                'ViewPosition': result['views']})
    for i, label in enumerate(LABELS):
        predictions[f'{label}_target'] = result['targets'][:, i]
        predictions[f'{label}_mask'] = result['masks'][:, i].astype(int)
        predictions[f'{label}_prob'] = result['probabilities'][:, i]
    predictions.to_csv(directory / f'{prefix}_predictions.csv', index=False)
    fixed, fixed_per_class = threshold_metrics(result['targets'], result['probabilities'], result['masks'], FIXED_THRESHOLDS)
    fixed.update(threshold_source='fixed_0.5', threshold_fitting_on_evaluation_data=False,
                 subset_accuracy_definition='Exact match only on images with all 13 labels observed.')
    fixed_per_class.to_csv(directory / f'{prefix}_classification_per_class_05.csv', index=False)
    if fit_thresholds:
        assert prefix == 'validation', 'Threshold fitting is validation-only.'
        fitted_thresholds, threshold_table = fit_validation_thresholds(result)
        threshold_table.to_csv(directory / 'validation_thresholds.csv', index=False)
        save_json(directory / 'thresholds.json', fitted_thresholds)
    reports = {'fixed_05': fixed}
    if fitted_thresholds is not None:
        fitted, fitted_per_class = threshold_metrics(result['targets'], result['probabilities'], result['masks'], fitted_thresholds)
        fitted.update(threshold_source='validation_fitted', threshold_fitting_on_evaluation_data=fit_thresholds,
                      subset_accuracy_definition='Exact match only on images with all selected labels observed.',
                      note='Labels below 30 validation positives excluded. Validation-fitted scores are in-sample and optimistic.')
        reports['validation_fitted'] = fitted
        fitted_per_class.to_csv(directory / f'{prefix}_classification_per_class_fitted.csv', index=False)
    save_json(directory / f'{prefix}_classification_metrics.json', reports)
    view_rows = []
    for view in ['AP', 'PA']:
        chosen = result['views'] == view
        if not chosen.any(): continue
        ranking = calculate_metrics(result['targets'][chosen], result['probabilities'][chosen], result['masks'][chosen])
        classification, _ = threshold_metrics(result['targets'][chosen], result['probabilities'][chosen],
                                               result['masks'][chosen], FIXED_THRESHOLDS)
        view_rows.append({'view': view, 'images': int(chosen.sum()),
                          'patients': len(np.unique(result['subject_ids'][chosen])),
                          **{key: ranking[key] for key in METRIC_NAMES},
                          'macro_label_count': ranking['macro_label_count'], 'stable_label_count': ranking['stable_label_count'],
                          'label_accuracy_05': classification['label_accuracy'], 'micro_F1_05': classification['micro_F1']})
    pd.DataFrame(view_rows).to_csv(directory / f'{prefix}_view_breakdown.csv', index=False)
    ci, draws = bootstrap_report(result)
    ci.to_csv(directory / f'{prefix}_bootstrap_ci.csv', index=False)
    np.save(directory / f'{prefix}_macro_bootstrap_draws.npy', draws, allow_pickle=False)
    return fixed, fitted_thresholds, draws

def paired_rows_for(left,right,left_draws,right_draws,comparison,seed):
    for key in ['dicom_ids','subject_ids','targets','masks']:
        assert np.array_equal(left[key],right[key]),f'Pairing mismatch: {key}'
    output=[]
    for column,metric in enumerate(METRIC_NAMES):
        delta=left_draws[:,column]-right_draws[:,column]
        finite=delta[np.isfinite(delta)]
        low,high=np.quantile(finite,[BOOTSTRAP_ALPHA/2,1-BOOTSTRAP_ALPHA/2]) if len(finite) else [np.nan,np.nan]
        output.append({'seed':seed,'comparison':comparison,'metric':metric,
                       'point_delta':left['metrics'][metric]-right['metrics'][metric],
                       'ci_lower':low,'ci_upper':high,'confidence_level':1-BOOTSTRAP_ALPHA,
                       'bootstrap_samples_requested':BOOTSTRAP_SAMPLES,'bootstrap_samples_valid':len(finite)})
    return output


def export_bundle(model, data, seed, directory, thresholds):
    bundle = {'head_checkpoint': torch.load(directory / 'best_model.pt', map_location='cpu', weights_only=True),
              'backbone_features_state_dict': data.backbone_state, 'preprocessing': PREPROCESSING,
              'thresholds': thresholds, 'observable_order': 'Z, X, neighboring ZZ, neighboring XX'}
    atomic_torch_save(bundle, directory / 'inference_bundle.pt')


class ExportedHybrid(nn.Module):
    """Images on image_device, head on CPU. Use loader's device argument, not .cuda()."""
    def __init__(self, bundle, image_device):
        super().__init__()
        checkpoint = bundle['head_checkpoint']
        assert checkpoint['labels'] == LABELS
        assert checkpoint['n_qubits'] == N_QUBITS and checkpoint['q_layers'] == Q_LAYERS
        assert checkpoint['residual_limit'] == RESIDUAL_LIMIT
        self.image_device = torch.device(image_device)
        self.cnn = densenet121(weights=None).features.to(self.image_device)
        self.cnn.load_state_dict(bundle['backbone_features_state_dict'], strict=True)
        self.head = ResidualHead(checkpoint['arm'], checkpoint['seed'])
        self.head.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.register_buffer('feature_mean', checkpoint['feature_mean'].cpu())
        self.register_buffer('feature_scale', checkpoint['feature_scale'].cpu())
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        pooled = F.adaptive_avg_pool2d(F.relu(self.cnn(images.to(self.image_device))), (1, 1)).flatten(1).cpu()
        return self.head((pooled - self.feature_mean) / self.feature_scale)


def load_exported_hybrid(path, image_device=None):
    ensure_quantum_dependency()
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    return ExportedHybrid(bundle, image_device or ('cuda' if torch.cuda.is_available() else 'cpu')), bundle['thresholds']


def run_experiment():
    data = prepare_data()
    pca = PCA(n_components=ANGLE_DIM, svd_solver='randomized', random_state=42)
    pca.fit(data.features['train'].numpy())
    np.savez(data.output / 'pca_train_only.npz', mean=pca.mean_, components=pca.components_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    config_names = ('SEEDS', 'ARMS', 'LABELS', 'IMAGE_SIZE', 'FEATURE_DIM', 'N_QUBITS',
                    'Q_LAYERS', 'ANGLE_DIM', 'OBSERVABLE_DIM', 'EXTRACT_BATCH_SIZE',
                    'HEAD_BATCH_SIZE', 'BASE_EPOCHS', 'MAX_EPOCHS', 'WARMUP_EPOCHS',
                    'MIN_EPOCHS', 'PATIENCE', 'STOP_MIN_DELTA', 'BASE_LR', 'ENCODER_LR',
                    'CORE_LR', 'READOUT_LR', 'WEIGHT_DECAY', 'RESIDUAL_L2', 'DROPOUT',
                    'BRANCH_DROPOUT', 'RESIDUAL_LIMIT', 'POS_WEIGHT_CAP', 'RUN_TEST',
                    'MIN_POSITIVES_FOR_THRESHOLD', 'BOOTSTRAP_SAMPLES', 'BOOTSTRAP_ALPHA',
                    'BOOTSTRAP_SEED', 'PREPROCESSING')
    configuration = {name: globals()[name] for name in config_names}
    configuration.update(experiment='Experiment 7 encoding', source_checkpoint_sha256=data.source_hash,
                         source_epoch=data.source_epoch, feature_cache=str(data.cache_dir),
                         pca_variance_48=float(pca.explained_variance_ratio_.sum()),
                         pca_variance_35=float(pca.explained_variance_ratio_[:ANGLE_DIM-len(LABELS)].sum()),
                         label_policy='Same as E6: NaN/0 negative, 1 positive, -1 masked',
                         selection='Validation macro AUPRC; include epoch 0; all seeds reported',
                         controls='Identical input encoders; MLP has larger core parameter budget',
                         torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                         pennylane=str(qml.__version__), python=platform.python_version())
    save_json(data.output / 'configuration.json', configuration)
    drawing_model = ResidualHead('task_quantum', 42)
    drawing = qml.draw(drawing_model.core.qnode, decimals=2)(
        torch.zeros(ANGLE_DIM), drawing_model.core.rotations,
        drawing_model.core.couplings, drawing_model.core.final_y)
    (data.output / 'quantum_circuit.txt').write_text(drawing + '\n', encoding='utf-8')
    del drawing_model
    # Preserve the exact runnable source beside the results, if run from a file.
    source_file = Path(globals().get('__file__', 'b.py'))
    if source_file.is_file():
        (data.output / 'b.py').write_bytes(source_file.read_bytes())
    atomic_torch_save({'features': data.backbone_state, 'source_sha256': data.source_hash}, data.output / 'frozen_backbone.pt')
    print('PCA variance retained by 48 PCs:', configuration['pca_variance_48'], flush=True)
    print('All residual arms start from the same selected linear anchor per seed.', flush=True)
    results, draws_by_run, test_results, paired, rows, ablation_rows = {}, {}, {}, [], [], []

    def record(arm, seed, result, directory, info):
        fixed, thresholds, draws = save_evaluation(result, directory, 'validation', fit_thresholds=True)
        results[(arm, seed)], draws_by_run[(arm, seed)] = result, draws
        rows.append({**info, 'arm': arm, 'seed': seed,
                     **{f'validation_{k}': result['metrics'][k] for k in METRIC_NAMES},
                     'validation_label_accuracy_05': fixed['label_accuracy'],
                     'validation_micro_F1_05': fixed['micro_F1']})
        pd.DataFrame(rows).to_csv(data.output / 'experiment_comparison.csv', index=False)
        return thresholds

    base_directory = data.output / 'experiment_2_reproduced'
    base_directory.mkdir()
    record('experiment_2_reproduced', 'source', data.baseline, base_directory, {'checkpoint_epoch': data.source_epoch})
    for seed in SEEDS:
        anchor = ResidualHead('linear_control', seed)
        with torch.no_grad():
            anchor.base.weight.copy_(data.original_weight * torch.as_tensor(data.feature_scale))
            anchor.base.bias.copy_(data.original_bias + data.original_weight @ torch.as_tensor(data.feature_mean))
        initial = evaluate_head(anchor, data)
        assert np.allclose(initial['probabilities'], data.baseline['probabilities'], atol=3e-5, rtol=3e-5)
        directory = data.output / f'linear_control_seed_{seed}'
        directory.mkdir()
        anchor_result, info, _ = train_head(anchor, data, seed, directory)
        thresholds = record('linear_control', seed, anchor_result, directory, info)
        if RUN_TEST:
            test_results[('linear_control', seed)] = evaluate_head(anchor, data, 'test')
            save_evaluation(test_results[('linear_control', seed)], directory, 'test', fitted_thresholds=thresholds)
        maps = fit_encoder_maps(data, pca, anchor.base)
        atomic_torch_save({k: {'weight': v[0], 'bias': v[1]} for k, v in maps.items()}, data.output / f'encoder_maps_seed_{seed}.pt')
        for arm in ARMS[1:]:
            model = ResidualHead(arm, seed)
            model.base.load_state_dict(anchor.base.state_dict())
            weight, bias = maps[arm.split('_')[0]]
            model.encoder.weight.copy_(weight)
            model.encoder.bias.copy_(bias)
            initialize_observation_scaling(model, data.features['train'])
            initial = evaluate_head(model, data)
            assert np.array_equal(initial['probabilities'], anchor_result['probabilities'])
            directory = data.output / f'{arm}_seed_{seed}'
            directory.mkdir()
            result, info, initial_state = train_head(model, data, seed, directory)
            thresholds = record(arm, seed, result, directory, info)
            if arm in QUANTUM_ARMS:
                # Selected model, zero branch: should reproduce the common anchor.
                removed = evaluate_head(model, data, base_only=True)
                assert np.array_equal(removed['probabilities'], anchor_result['probabilities'])
                # Reset circuit only; keep trained encoder, normalization and readout.
                saved = copy.deepcopy(model.core.state_dict())
                model.core.load_state_dict({k.removeprefix('core.'): v for k, v in initial_state.items() if k.startswith('core.')})
                reset = evaluate_head(model, data)
                model.core.load_state_dict(saved)
                ablation_dir = directory / 'circuit_reset'
                ablation_dir.mkdir()
                _, _, reset_draws = save_evaluation(reset, ablation_dir, 'validation', fitted_thresholds=thresholds)
                paired.extend(paired_rows_for(result, reset, draws_by_run[(arm, seed)], reset_draws,
                                               arm + ' minus circuit_reset', seed))
                ablation_rows.append({'arm': arm, 'seed': seed, 'ablation': 'circuit_reset',
                                      **{k: reset['metrics'][k] for k in METRIC_NAMES}})
                export_bundle(model, data, seed, directory, thresholds)
            if RUN_TEST:
                test_results[(arm, seed)] = evaluate_head(model, data, 'test')
                save_evaluation(test_results[(arm, seed)], directory, 'test', fitted_thresholds=thresholds)
            del model
            gc.collect()
        del anchor

    comparisons = [('pca_quantum', 'pca_classical'), ('task_quantum', 'task_classical'),
                   ('task_quantum', 'pca_quantum'), ('task_quantum', 'task_frozen'),
                   ('task_quantum', 'task_no_entanglement')]
    comparisons += [(arm, 'linear_control') for arm in ARMS[1:]]
    for seed in SEEDS:
        for left, right in comparisons:
            paired.extend(paired_rows_for(results[(left, seed)], results[(right, seed)],
                           draws_by_run[(left, seed)], draws_by_run[(right, seed)], left + ' minus ' + right, seed))
    if len(SEEDS) > 1:
        for arm in ARMS:
            directory = data.output / f'{arm}_ensemble'
            directory.mkdir()
            result = data.result('val', np.mean([results[(arm, seed)]['probabilities'] for seed in SEEDS], axis=0))
            thresholds = record(arm, 'ensemble', result, directory, {'checkpoint_epoch': None})
            save_json(directory / 'ensemble_manifest.json', {
                'members': [str(data.output / f'{arm}_seed_{seed}' / 'best_model.pt') for seed in SEEDS],
                'weights': [1 / len(SEEDS)] * len(SEEDS)})
            if RUN_TEST:
                result = data.result('test', np.mean([test_results[(arm, seed)]['probabilities'] for seed in SEEDS], axis=0))
                save_evaluation(result, directory, 'test', fitted_thresholds=thresholds)
        for left, right in comparisons:
            paired.extend(paired_rows_for(results[(left, 'ensemble')], results[(right, 'ensemble')],
                           draws_by_run[(left, 'ensemble')], draws_by_run[(right, 'ensemble')],
                           left + ' minus ' + right, 'ensemble'))
    comparison = pd.DataFrame(rows)
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(data.output / 'paired_patient_bootstrap_differences.csv', index=False)
    pd.DataFrame(ablation_rows).to_csv(data.output / 'quantum_ablation_metrics.csv', index=False)
    seed_summary = []
    for arm in ARMS:
        selected = comparison.loc[(comparison.arm == arm) & comparison.seed.isin(SEEDS)]
        for metric in METRIC_NAMES:
            values = selected[f'validation_{metric}'].to_numpy(float)
            seed_summary.append({'arm': arm, 'metric': metric, 'mean': values.mean(),
                                 'seed_sd': values.std(ddof=1) if len(values) > 1 else np.nan})
    pd.DataFrame(seed_summary).to_csv(data.output / 'seed_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, METRIC_NAMES[:2]):
        for i, arm in enumerate(ARMS):
            values = [results[(arm, seed)]['metrics'][metric] for seed in SEEDS]
            ax.scatter([i] * len(values), values)
            ax.plot([i - .2, i + .2], [np.mean(values)] * 2, color='black')
        ax.set_xticks(range(len(ARMS)), ARMS, rotation=30, ha='right')
        ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(data.output / 'model_comparison.png', dpi=160)
    plt.close(fig)
    notes = [
        'Validation reused from earlier experiments: exploratory results, not an independent performance estimate.',
        'Patient bootstrap is conditional on fitted/selected models; it excludes model-selection and training uncertainty.',
        'Classical MLP sees the identical encoder output; its larger parameter budget is reported explicitly.',
        'The no-entanglement control still has a classical readout of local/product measurements.',
        'Epoch 0 retention means the quantum branch did not improve validation AUPRC.',
        'A useful quantum contribution needs improvement over the common anchor AND relevant controls across seeds.',
        'Simulation results do not establish quantum computational advantage.']
    save_json(data.output / 'summary.json', {'configuration': configuration, 'comparison': rows,
              'seed_summary': seed_summary, 'paired_comparisons': paired, 'interpretation': notes})
    lines = ['# Experiment 7 results', '', 'Validation results; test evaluated: ' + str(RUN_TEST), '',
             '| Arm | Seed | Epoch | AUROC | AUPRC |', '|---|---|---|---|---|']
    for row in rows:
        lines.append(f'| {row["arm"]} | {row["seed"]} | {row.get("checkpoint_epoch")} | '
                     f'{row["validation_macro_AUROC"]:.6f} | {row["validation_macro_AUPRC"]:.6f} |')
    lines += ['', *['- ' + note for note in notes]]
    (data.output / 'b.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n' + comparison[['arm', 'seed', 'checkpoint_epoch', 'validation_macro_AUROC', 'validation_macro_AUPRC']].to_string(index=False))
    print('\nPaired macro AUPRC differences (positive favors left):')
    print(paired_df.loc[paired_df.metric == 'macro_AUPRC'].to_string(index=False))
    print('Saved to:', data.output)
    if not RUN_TEST:
        print('Test features, predictions and metrics were not computed.')
    return data.output


def self_check():
    """Small synthetic checks; no dataset, images, or cloud resources required."""
    seed_everything(42)
    x = torch.randn(5, FEATURE_DIM)
    q = ResidualHead('task_quantum', 42)
    c = ResidualHead('task_classical', 42)
    with torch.no_grad():
        mapping = torch.randn(ANGLE_DIM, FEATURE_DIM) * 0.03
        for model in (q, c):
            model.encoder.weight.copy_(mapping)
    assert torch.equal(q.encoder(x), c.encoder(x)), 'Classical/quantum encoder mismatch'
    assert torch.equal(q.base.weight, c.base.weight)
    assert torch.equal(q.readout.weight, c.readout.weight)
    angles = q.encoder(x).detach().requires_grad_(True)
    batched = q.core(angles)
    single = torch.cat([q.core(a[None]) for a in angles])
    assert batched.shape == (len(x), OBSERVABLE_DIM)
    assert torch.allclose(batched, single, atol=2e-6, rtol=2e-6)
    (batched * torch.linspace(0.1, 1., OBSERVABLE_DIM)).sum().backward()
    assert angles.grad is not None and torch.isfinite(angles.grad).all() and angles.grad.abs().sum() > 0
    for name, p in q.core.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    for arm in ARMS[1:]:
        model = ResidualHead(arm, 42)
        model.encoder.weight.copy_(mapping)
        initialize_observation_scaling(model, x)
        model.eval()
        assert torch.equal(model(x), model.base(x)), 'Zero residual must preserve anchor'
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        truth = torch.randint(0, 2, (len(x), len(LABELS))).float()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x), truth)
            loss.backward()
            if step == 1:
                assert model.encoder.gain_raw.grad.abs().sum() > 0
                if arm != 'task_frozen':
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.parameters())
            optimizer.step()
        assert all(p.grad is None for p in model.base.parameters())
        clone = ResidualHead(arm, 42)
        clone.load_state_dict(model.state_dict(), strict=True)
        clone.eval()
        assert torch.equal(model(x), clone(x)), 'Checkpoint round trip mismatch'
        print(f'PASS {arm}: forward, backward, frozen base, checkpoint round trip')
    print('PASS batched circuit, all circuit gradients, identical encoders, and zero-residual initialization')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-check', action='store_true', help='Synthetic checks only; no dataset needed')
    # Accept Jupyter's kernel -f argument when pasted as one notebook cell.
    parser.add_argument('-f', help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert len(SEEDS) > 0 and len(SEEDS) == len(set(SEEDS))
    assert ANGLE_DIM == 48 and N_QUBITS == 8 and Q_LAYERS == 3
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(min(CPU_HEAD_THREADS, os.cpu_count() or 1))
    ensure_quantum_dependency()
    seed_everything(42)
    print('EXPERIMENT 7 | extraction:', EXTRACT_DEVICE, '| heads: CPU | seeds:', SEEDS)
    print('Torch:', torch.__version__, '| torchvision:', torchvision.__version__, '| PennyLane:', qml.__version__)
    if args.self_check:
        self_check()
    else:
        self_check()  # fail early on incompatible circuit environments before image work
        OUTPUT_DIR = run_experiment()
