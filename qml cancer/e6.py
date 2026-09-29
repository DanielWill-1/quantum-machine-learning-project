# ============================================================
# EXPERIMENT 6 — RESIDUAL QUANTUM TRANSFER MODEL
# One complete notebook cell / standalone Python script.
# Run in a fresh kernel with the working Torch + torchvision installation.
# Actual analytic quantum-circuit simulation; no quantum hardware/API required.
# Quantum branch plus retained 1024-feature classifier; full controls and ablations.
# Do not interpret simulator performance as quantum hardware advantage.
# ============================================================
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, gc, json, math, time, random, hashlib, platform, subprocess, importlib.util
from pathlib import Path
from datetime import datetime, timezone
# If PennyLane is missing, install only it; keep your working CUDA Torch build.
if importlib.util.find_spec('pennylane') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pennylane==0.45.1'])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageFile, ImageOps
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import densenet121
from tqdm.auto import tqdm
import pennylane as qml

# ============================================================
# 0. CONFIGURATION
# ============================================================
# Set MIMIC_CXR_ROOT, or edit ROOT_OVERRIDE if your dataset moved.
ROOT_OVERRIDE = os.environ.get('MIMIC_CXR_ROOT', '').strip()
root_candidates = [Path('/workspace/private/unzippedarchive'),
                   Path('/workspace/unzippedarchive/unzippedarchive'), Path('/workspace/unzippedarchive'),
                   Path('workspace/unzippedarchive/unzippedarchive').resolve(), Path.cwd()]
if ROOT_OVERRIDE:
    ROOT = Path(ROOT_OVERRIDE).expanduser().resolve()
else:
    roots = list(dict.fromkeys(path.resolve() for path in root_candidates
                  if (path / 'xray_image_label_index.csv').is_file()
                  and (path / 'xray_training_experiment_2/best_model.pt').is_file()))
    if len(roots) != 1:
        raise RuntimeError(f'Set ROOT_OVERRIDE to your dataset root; found candidates: {roots}')
    ROOT = roots[0]
INDEX_CSV = ROOT / 'xray_image_label_index.csv'
SPLIT_CSV = ROOT / 'xray_training_output/patient_level_splits.csv'
EXP2_DIR = ROOT / 'xray_training_experiment_2'
SOURCE_CHECKPOINT = EXP2_DIR / 'best_model.pt'
SOURCE_SUMMARY = EXP2_DIR / 'summary.json'
OUTPUT_ROOT = ROOT / 'xray_training_experiment_6_residual_qml'
CACHE_ROOT = ROOT / 'xray_exp2_frozen_feature_cache'
IMAGE_SIZE = 320
FEATURE_DIM = 1024
EXTRACT_BATCH_SIZE = 24
NUM_WORKERS = 4 if sys.platform.startswith('linux') else 0
EXTRACT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# FP32 once-only extraction, used identically by all heads.
SEEDS = [42, 43, 44]  # set [42] for a quick exploratory smoke run only
N_QUBITS = 6
Q_LAYERS = 2
HEAD_BATCH_SIZE = 128
MAX_EPOCHS = 40
WARMUP_EPOCHS = 5
MIN_EPOCHS = 15
PATIENCE = 10
STOP_MIN_DELTA = 0.0002
PROJECT_LR = 3e-4
READOUT_LR = 1e-3
CORE_LR = 1e-3
BASE_LR = 1e-5
RESIDUAL_L2 = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.30
POS_WEIGHT_CAP = 5.0
HEAD_DEVICE = 'cpu'  # explicit CPU circuit: avoids the E5 CPU/CUDA mismatch
# The DenseNet feature extraction still runs on CUDA.
CPU_HEAD_THREADS = 4
RUN_TEST = False
MIN_POSITIVES_FOR_THRESHOLD = 30
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 12345
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
          'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
METRIC_NAMES = ['macro_AUROC', 'macro_AUPRC', 'macro_AUROC_50plus', 'macro_AUPRC_50plus']
ARMS = ['linear_control', 'classical_residual', 'frozen_quantum', 'quantum_residual']
FIXED_THRESHOLDS = {label: 0.5 for label in LABELS}
assert N_QUBITS == 6 and Q_LAYERS == 2, 'The matched control is specified for 6 qubits / 2 re-upload layers.'
assert MAX_EPOCHS >= MIN_EPOCHS > WARMUP_EPOCHS
assert len(SEEDS) == len(set(SEEDS)) and len(SEEDS) > 0


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ImageFile.LOAD_TRUNCATED_IMAGES = True


torch.set_default_dtype(torch.float32)
seed_everything(42)
torch.set_num_threads(min(CPU_HEAD_THREADS, os.cpu_count() or 1))
for path in [INDEX_CSV, SPLIT_CSV, SOURCE_CHECKPOINT, SOURCE_SUMMARY, EXP2_DIR / 'patient_level_splits.csv']:
    if not path.is_file(): raise FileNotFoundError(path)
OUTPUT_DIR = OUTPUT_ROOT / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
print('EXPERIMENT 6 RESIDUAL QML | feature extraction:', EXTRACT_DEVICE)
print('Torch:', torch.__version__, '| torchvision:', torchvision.__version__, '| PennyLane:', qml.__version__)
print('Seeds:', SEEDS, '| RUN_TEST:', RUN_TEST)

# Shared mask-aware data, ranking, classification, and bootstrap helpers.
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

# ============================================================
# 1. EXACT EXP2 ARCHITECTURE, PREPROCESSING AND PATIENT SPLITS
# ============================================================
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

# Train-only standardization. No PCA is needed: the shared trainable projection
# performs the 1024 -> 4 compression. Neither validation nor test fits statistics.
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



# ============================================================
# 2. RESIDUAL HEADS: FULL CLASSIFIER + SMALL LEARNED CORRECTION
# ============================================================
assert HEAD_DEVICE == 'cpu', 'This version deliberately keeps all heads/circuits on CPU.'
head_device = torch.device(HEAD_DEVICE)
OBSERVABLE_DIM = 3 * N_QUBITS
ANGLE_DIM = Q_LAYERS * N_QUBITS


def quantum_core():
    device = qml.device('default.qubit', wires=N_QUBITS, shots=None)
    @qml.qnode(device, interface='torch', diff_method='backprop')
    def circuit(inputs, weights):
        for wire in range(N_QUBITS):
            qml.RY(math.pi / 4, wires=wire)
        # Each round receives its own six projected features (data re-uploading).
        for layer in range(Q_LAYERS):
            for wire in range(N_QUBITS):
                qml.RY(inputs[..., layer * N_QUBITS + wire], wires=wire)
                qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
            for wire in range(N_QUBITS):
                qml.CZ(wires=[wire, (wire + 1) % N_QUBITS])
        # Z, X and neighboring ZZ provide 18 readout features, not just six.
        return ([qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)] +
                [qml.expval(qml.PauliX(w)) for w in range(N_QUBITS)] +
                [qml.expval(qml.PauliZ(w) @ qml.PauliZ((w+1) % N_QUBITS)) for w in range(N_QUBITS)])
    return qml.qnn.TorchLayer(circuit, {'weights': (Q_LAYERS, N_QUBITS, 3)},
                             init_method=lambda t: nn.init.uniform_(t, -.1, .1))


class ClassicalCore(nn.Module):
    """36 shared mixing weights, matching 2*6*3 circuit parameters.

    Two recurrent updates use the same mixing matrix. Fixed output transforms
    produce 18 bounded features. Equal parameter counts do not imply identical
    expressivity or compute; this is a specified small nonlinear control.
    """
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(N_QUBITS, N_QUBITS, bias=False)
        nn.init.orthogonal_(self.mix.weight)
    def forward(self, angles):
        state = torch.zeros_like(angles[..., :N_QUBITS])
        for layer in range(Q_LAYERS):
            state = torch.tanh(self.mix(state + angles[..., layer*N_QUBITS:(layer+1)*N_QUBITS] / math.pi))
        return torch.cat([state, torch.sin(math.pi/2 * state), state * torch.roll(state, 1, -1)], dim=-1)


class ResidualHead(nn.Module):
    def __init__(self, arm, seed, initialize_from_source=True):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        seed_everything(seed)
        self.arm = arm
        self.base = nn.Linear(FEATURE_DIM, len(LABELS))
        if initialize_from_source:
            with torch.no_grad():
                self.base.weight.copy_(original_weight * torch.from_numpy(feature_scale))
                self.base.bias.copy_(original_bias + original_weight @ torch.from_numpy(feature_mean))
        # Buffers document the source classifier for parameter-change diagnostics.
        self.register_buffer('source_base_weight', self.base.weight.detach().clone())
        self.register_buffer('source_base_bias', self.base.bias.detach().clone())
        self.dropout = nn.Dropout(DROPOUT)
        if arm != 'linear_control':
            self.project = nn.Linear(FEATURE_DIM, ANGLE_DIM)
            nn.init.xavier_uniform_(self.project.weight, gain=.1)
            nn.init.zeros_(self.project.bias)
            self.readout = nn.Linear(OBSERVABLE_DIM, len(LABELS))
            # Exact initial E2 function; quantum gradients begin after the first
            # readout update. Do not also zero the projection or circuit weights.
            nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)
            self.core = ClassicalCore() if arm == 'classical_residual' else quantum_core()
            if arm == 'frozen_quantum': self.core.requires_grad_(False)
            if arm in ['frozen_quantum', 'quantum_residual']:
                self.register_buffer('initial_circuit_weights', self.core.weights.detach().clone())

    def components(self, x, base_only=False):
        dropped = self.dropout(x)
        base = self.base(dropped)
        if self.arm == 'linear_control' or base_only:
            return base, torch.zeros_like(base)
        angles = math.pi * torch.tanh(self.project(dropped))
        measured = self.core(angles).to(dtype=x.dtype)
        return base, self.readout(measured)

    def forward(self, x):
        base, residual = self.components(x)
        return base + residual


# Shared initializers and identical total trainable counts for learned residual arms.
check_classical = ResidualHead('classical_residual', 42)
check_quantum = ResidualHead('quantum_residual', 42)
for key in ['base.weight','base.bias','project.weight','project.bias','readout.weight','readout.bias']:
    assert torch.equal(check_classical.state_dict()[key], check_quantum.state_dict()[key]), key
assert sum(p.numel() for p in check_classical.parameters()) == sum(p.numel() for p in check_quantum.parameters())
print('Learned residual model parameters:', sum(p.numel() for p in check_quantum.parameters()))
angles_check = torch.randn(3, ANGLE_DIM, requires_grad=True)
q_batch = check_quantum.core(angles_check)
q_single = torch.cat([check_quantum.core(angles_check[i:i+1]) for i in range(3)], dim=0)
assert q_batch.shape == (3, OBSERVABLE_DIM)
assert torch.allclose(q_batch, q_single, atol=2e-6, rtol=2e-6)
(q_batch * torch.linspace(.1, 1., OBSERVABLE_DIM)).sum().backward()
assert angles_check.grad is not None and torch.isfinite(angles_check.grad).all() and angles_check.grad.abs().sum() > 0
assert check_quantum.core.weights.grad is not None and torch.isfinite(check_quantum.core.weights.grad).all()
assert check_quantum.core.weights.grad.abs().sum() > 0
print('Circuit batch and gradient checks passed; head device: CPU, extraction device:', EXTRACT_DEVICE)
del check_classical, check_quantum, angles_check, q_batch, q_single
features = {name: torch.from_numpy(value) for name, value in standardized.items()}
targets = {name: torch.from_numpy(encoded[name][0]) for name in active_splits}
masks = {name: torch.from_numpy(encoded[name][1]) for name in active_splits}
criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(pos_weights))


@torch.no_grad()
def evaluate_head(model, split, base_only=False):
    model.eval()
    probabilities, loss_sum, valid_count, residual_square, residual_abs, elements = [], 0., 0., 0., 0., 0
    for start in range(0, len(features[split]), HEAD_BATCH_SIZE):
        base, residual = model.components(features[split][start:start+HEAD_BATCH_SIZE], base_only=base_only)
        logits = base + residual
        truth, mask = targets[split][start:start+HEAD_BATCH_SIZE], masks[split][start:start+HEAD_BATCH_SIZE]
        if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite logits.')
        loss_sum += (criterion(logits, truth) * mask).sum().item()
        valid_count += mask.sum().item()
        residual_square += residual.square().sum().item()
        residual_abs += residual.abs().sum().item(); elements += residual.numel()
        probabilities.append(torch.sigmoid(logits).numpy())
    result = make_result(split, np.concatenate(probabilities))
    result['loss'] = loss_sum / valid_count if valid_count else np.nan
    result['residual_logit_rms'] = math.sqrt(residual_square / elements)
    result['residual_logit_mean_absolute'] = residual_abs / elements
    return result


def parameter_diagnostics(model):
    result = {'base_weight_change_l2': (model.base.weight-model.source_base_weight).norm().item(),
              'base_bias_change_l2': (model.base.bias-model.source_base_bias).norm().item()}
    if model.arm != 'linear_control':
        result['readout_weight_l2'] = model.readout.weight.norm().item()
    if model.arm in ['quantum_residual', 'frozen_quantum']:
        result['circuit_parameter_change_l2'] = (model.core.weights-model.initial_circuit_weights).norm().item()
    return result


# ============================================================
# 3. WARMUP THEN DIFFERENTIAL-LR JOINT HEAD TRAINING
# ============================================================
def train_head(arm, seed, run_dir):
    model = ResidualHead(arm, seed)
    model.base.requires_grad_(False)
    groups = [{'params': list(model.base.parameters()), 'lr': 0., 'name': 'base'}]
    if arm != 'linear_control':
        groups += [{'params': list(model.project.parameters()), 'lr': PROJECT_LR, 'name': 'projection'},
                   {'params': list(model.readout.parameters()), 'lr': READOUT_LR, 'name': 'readout'}]
        core_parameters = [p for p in model.core.parameters() if p.requires_grad]
        if core_parameters: groups.append({'params': core_parameters, 'lr': CORE_LR, 'name': 'core'})
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3, min_lr=1e-7)
    order_generator = torch.Generator(device='cpu').manual_seed(seed+1000)
    seed_everything(seed+2000)
    initial = evaluate_head(model, 'val')
    assert np.allclose(initial['probabilities'], baseline_probabilities, atol=3e-5, rtol=3e-5), 'Initial residual model must reproduce E2.'
    best_score = float(initial['metrics']['macro_AUPRC'])
    if not math.isfinite(best_score): raise RuntimeError('No evaluable validation score.')
    best_epoch, stopping_reference, stale = 0, best_score, 0
    total_parameters = sum(p.numel() for p in model.parameters())
    # Base is trainable in stage 2; circuit of frozen_quantum stays frozen.
    trainable_stage2 = total_parameters - (model.core.weights.numel() if arm == 'frozen_quantum' else 0)

    def save_checkpoint(epoch, result):
        checkpoint = {'experiment': 'Experiment 6 residual QML', 'arm': arm, 'seed': seed, 'epoch': epoch,
            'model_state_dict': {k:v.detach().clone() for k,v in model.state_dict().items()},
            'macro_AUROC': float(result['metrics']['macro_AUROC']), 'macro_AUPRC': float(result['metrics']['macro_AUPRC']),
            'labels': LABELS, 'source_checkpoint': str(SOURCE_CHECKPOINT), 'source_checkpoint_sha256': source_hash,
            'source_epoch': source_epoch, 'feature_mean': torch.from_numpy(feature_mean), 'feature_scale': torch.from_numpy(feature_scale),
            'n_qubits': N_QUBITS, 'q_layers': Q_LAYERS, 'dropout': DROPOUT, 'feature_dim': FEATURE_DIM,
            'image_size': IMAGE_SIZE, 'observable_order': 'Z_0..Z_5, X_0..X_5, Z_i Z_(i+1 mod 6)',
            'torch_version': str(torch.__version__), 'pennylane_version': str(qml.__version__),
            'diagnostics': parameter_diagnostics(model), 'residual_logit_rms': result['residual_logit_rms'],
            'note': 'Inference checkpoint, not optimizer resume. Epoch 0 means unchanged E2 and inactive residual.'}
        temporary = run_dir / 'best_model.tmp'
        torch.save(checkpoint, temporary); os.replace(temporary, run_dir / 'best_model.pt')

    save_checkpoint(0, initial)
    history = [{'epoch': 0, 'stage': 'source', 'train_loss': np.nan, 'validation_loss': initial['loss'],
                **{k:initial['metrics'][k] for k in METRIC_NAMES}, 'train_seconds': 0., 'validation_seconds': 0.,
                'residual_logit_rms': 0.}]
    started_total = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS+1):
        if epoch == WARMUP_EPOCHS+1:
            model.base.requires_grad_(True)
            optimizer.param_groups[0]['lr'] = BASE_LR
            stopping_reference, stale = best_score, 0
        model.train()
        indices = torch.randperm(len(features['train']), generator=order_generator)
        loss_sum, valid_count, penalty_sum, batch_count, core_gradient_max, projection_gradient_max = 0., 0., 0., 0, 0., 0.
        learning_rates = {group['name']:group['lr'] for group in optimizer.param_groups}
        started = time.perf_counter()
        for selected in indices.split(HEAD_BATCH_SIZE):
            # Warmup intentionally leaves the linear control at the source checkpoint.
            if arm == 'linear_control' and epoch <= WARMUP_EPOCHS: continue
            mask = masks['train'][selected]; denominator = mask.sum()
            if denominator.item() == 0: continue
            optimizer.zero_grad(set_to_none=True)
            base, residual = model.components(features['train'][selected])
            masked_sum = (criterion(base+residual, targets['train'][selected])*mask).sum()
            penalty = RESIDUAL_L2 * residual.square().mean()
            loss = masked_sum / denominator + penalty
            if not torch.isfinite(loss): raise FloatingPointError(f'{arm}: non-finite loss')
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
            if not torch.isfinite(gradient_norm): raise FloatingPointError('Non-finite gradient')
            if arm != 'linear_control':
                projection_gradient_max = max(projection_gradient_max, model.project.weight.grad.norm().item())
                grads = [p.grad for p in model.core.parameters() if p.requires_grad and p.grad is not None]
                if grads: core_gradient_max = max(core_gradient_max, math.sqrt(sum(g.square().sum().item() for g in grads)))
            optimizer.step()
            loss_sum += masked_sum.detach().item(); valid_count += denominator.item()
            penalty_sum += penalty.detach().item(); batch_count += 1
        train_seconds = time.perf_counter()-started
        started = time.perf_counter(); result = evaluate_head(model, 'val'); validation_seconds = time.perf_counter()-started
        score = float(result['metrics']['macro_AUPRC'])
        if not math.isfinite(score): raise FloatingPointError('Non-finite validation metric')
        fixed, _ = threshold_metrics(result['targets'], result['probabilities'], result['masks'], FIXED_THRESHOLDS)
        row = {'epoch': epoch, 'stage': 'warmup' if epoch<=WARMUP_EPOCHS else 'joint_head',
               'train_loss': loss_sum/valid_count if valid_count else np.nan,
               'residual_penalty': penalty_sum/max(batch_count,1), 'validation_loss': result['loss'],
               **{k:result['metrics'][k] for k in METRIC_NAMES},
               'label_accuracy_05': fixed['label_accuracy'], 'micro_F1_05': fixed['micro_F1'],
               'residual_logit_rms': result['residual_logit_rms'], 'core_gradient_max': core_gradient_max,
               'projection_gradient_max': projection_gradient_max, 'train_seconds': train_seconds,
               'validation_seconds': validation_seconds, **{f'lr_{k}':v for k,v in learning_rates.items()}}
        history.append(row)
        improved = score > best_score
        if improved: best_score, best_epoch = score, epoch; save_checkpoint(epoch, result)
        if epoch > WARMUP_EPOCHS:
            if score > stopping_reference+STOP_MIN_DELTA: stopping_reference, stale = score, 0
            else: stale += 1
            scheduler.step(score)
        pd.DataFrame(history).to_csv(run_dir/'training_history.csv', index=False)
        print(f'{arm} seed={seed} {epoch:02d}/{MAX_EPOCHS} {row["stage"]} | '
              f'AUROC {row["macro_AUROC"]:.5f} AP {score:.5f} | acc {fixed["label_accuracy"]:.4f} '
              f'F1 {fixed["micro_F1"]:.4f} | residual RMS {row["residual_logit_rms"]:.3f} | '
              f'train {train_seconds:.2f}s val {validation_seconds:.2f}s' + (' | best' if improved else ''))
        if epoch>=MIN_EPOCHS and stale>=PATIENCE: break
    elapsed = time.perf_counter()-started_total
    if arm == 'quantum_residual' and max(row.get('core_gradient_max',0) for row in history) <= 0:
        raise RuntimeError('Quantum circuit received no gradient during training.')
    if arm == 'frozen_quantum': assert torch.equal(model.core.weights, model.initial_circuit_weights)
    best = torch.load(run_dir/'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'], strict=True)
    result = evaluate_head(model, 'val')
    assert abs(result['metrics']['macro_AUPRC']-best_score)<1e-7
    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for column in ['train_loss','validation_loss']: axes[0].plot(history_df.epoch,history_df[column],label=column)
    for column in ['macro_AUROC','macro_AUPRC']: axes[1].plot(history_df.epoch,history_df[column],label=column)
    axes[2].plot(history_df.epoch, history_df.residual_logit_rms, label='Residual RMS')
    for ax in axes: ax.axvline(best_epoch,color='gray',linestyle='--'); ax.set_xlabel('Epoch'); ax.legend()
    fig.suptitle(f'{arm}, seed {seed}'); fig.tight_layout(); fig.savefig(run_dir/'training_curves.png',dpi=150);plt.close(fig)
    info = {'arm':arm, 'seed':seed, 'checkpoint_epoch':best_epoch, 'epochs_completed':len(history)-1,
            'parameters':total_parameters, 'trainable_parameters_stage2':trainable_stage2,
            'head_training_and_validation_seconds':elapsed, 'residual_logit_rms':result['residual_logit_rms'],
            'quantum_branch_active': bool(arm=='quantum_residual' and result['residual_logit_rms']>1e-8),
            **parameter_diagnostics(model)}
    return model, result, info

# ============================================================
# 4. PATIENT BOOTSTRAPS, THRESHOLDS, AND DETAILED REPORTING
# ============================================================


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

# ============================================================
# 5. RUN EVERY DECLARED ARM/SEED; DO NOT SELECT A LUCKY SEED
# ============================================================
configuration = {
    'experiment':'Experiment 6 residual QML', 'root':ROOT, 'output_directory':OUTPUT_DIR,
    'source_checkpoint':SOURCE_CHECKPOINT, 'source_checkpoint_sha256':source_hash,
    'source_epoch':source_epoch, 'source_saved_metrics':source_metrics,
    'source_reproduced_metrics':{k:baseline_result['metrics'][k] for k in METRIC_NAMES},
    'source_reference_warning':'This source may differ from earlier E2 runs: compare hashes/epochs, not experiment names.',
    'seeds':SEEDS, 'arms':ARMS, 'n_qubits':N_QUBITS, 'layers':Q_LAYERS, 'observable_dimension':OBSERVABLE_DIM,
    'circuit':'Initial RY(pi/4); two RY data uploads + trainable Rot + CZ rings; Z/X/neighbor-ZZ expectations',
    'classical_control':'Two recurrent updates with a shared bias-free 6x6 matrix; 18 fixed nonlinear outputs',
    'head_device':'cpu', 'extraction_device':str(EXTRACT_DEVICE), 'quantum_backend':'default.qubit, shots=None, torch backprop',
    'backbone_frozen':True, 'image_augmentation':'None; deterministic frozen features for all controls',
    'preprocessing':'Exact E2 RGB -> aspect-preserving bilinear resize and pad320 -> ImageNet normalization',
    'feature_cache':cache_dir, 'cache_load_or_extract_seconds':feature_seconds,
    'feature_dimension':FEATURE_DIM, 'standardization':'Training mean/std only',
    'warmup_epochs':WARMUP_EPOCHS, 'max_epochs':MAX_EPOCHS, 'minimum_epochs':MIN_EPOCHS,
    'patience':PATIENCE, 'stopping_min_delta':STOP_MIN_DELTA, 'batch_size':HEAD_BATCH_SIZE,
    'base_lr':BASE_LR, 'projection_lr':PROJECT_LR, 'readout_lr':READOUT_LR, 'core_lr':CORE_LR,
    'weight_decay':WEIGHT_DECAY, 'dropout':DROPOUT, 'residual_l2':RESIDUAL_L2,
    'pos_weight_cap':POS_WEIGHT_CAP, 'pos_weights':pos_weights, 'labels':LABELS,
    'uncertainty_policy':'NaN/0 negative; 1 positive; -1 masked',
    'selection':'Every true validation macro-AUPRC maximum, including epoch0; no test selection',
    'bootstrap_samples':BOOTSTRAP_SAMPLES, 'bootstrap_seed':BOOTSTRAP_SEED, 'bootstrap_unit':'Patient, all images with multiplicity',
    'bootstrap_implementation':'Vectorized weighted ranking with exact score-tie handling; same as image repetition',
    'bootstrap_caveat':'Conditional on selected checkpoints; does not include training or validation-selection uncertainty',
    'ensemble':'Equal probability average of ALL three specified seeds; reported separately from individual models',
    'run_test':RUN_TEST, 'torch':str(torch.__version__), 'torchvision':str(torchvision.__version__),
    'pennylane':str(qml.__version__), 'numpy':np.__version__, 'python':platform.python_version(),
}
save_json(OUTPUT_DIR/'configuration.json',configuration)
print('Source epoch/hash:',source_epoch,source_hash)
print('E6 preserves the entire 1024-feature base and trains a residual correction. Test evaluation:',RUN_TEST)

baseline_dir = OUTPUT_DIR/'experiment_2_reproduced'; baseline_dir.mkdir()
baseline_fixed, baseline_thresholds, baseline_draws = save_evaluation(baseline_result,baseline_dir,'validation',fit_thresholds=True)
rows = [
    {'arm':'experiment_2_saved', 'seed':None, 'checkpoint_epoch':source_epoch,
     'validation_macro_AUROC':source_metrics['macro_AUROC'],'validation_macro_AUPRC':source_metrics['macro_AUPRC'],
     'metrics_source':str(SOURCE_SUMMARY)},
    {'arm':'experiment_2_reproduced', 'seed':None,'checkpoint_epoch':source_epoch,
     **{f'validation_{k}':baseline_result['metrics'][k] for k in METRIC_NAMES},
     'validation_label_accuracy_05':baseline_fixed['label_accuracy'],'validation_micro_F1_05':baseline_fixed['micro_F1']}
]
results_by_run, draws_by_run, test_by_run, ablation_results, ablation_draws, ablation_rows = {},{},{},{},{},[]
for seed in SEEDS:
    for arm in ARMS:
        directory = OUTPUT_DIR/f'{arm}_seed_{seed}';directory.mkdir()
        model,result,info = train_head(arm,seed,directory)
        fixed,thresholds,draws = save_evaluation(result,directory,'validation',fit_thresholds=True)
        results_by_run[(arm,seed)],draws_by_run[(arm,seed)] = result,draws
        row = {**info,**{f'validation_{k}':result['metrics'][k] for k in METRIC_NAMES},
               'validation_loss':result['loss'],'validation_label_accuracy_05':fixed['label_accuracy'],
               'validation_micro_F1_05':fixed['micro_F1'],'validation_macro_F1_05':fixed['macro_F1'],
               'validation_balanced_accuracy_05':fixed['macro_balanced_accuracy'],
               'validation_macro_MCC_05':fixed['macro_MCC'],
               'all_negative_accuracy_baseline':fixed['all_negative_label_accuracy_baseline'],
               'metrics_source':str(directory/'best_model.pt')}
        rows.append(row)
        save_json(directory/'summary.json',{**configuration,**row,'classification_fixed_05':fixed})
        if arm == 'quantum_residual':
            # Within-model inference ablations are diagnostics, not retrained controls.
            for ablation in ['branch_removed','circuit_reset']:
                if ablation == 'branch_removed':
                    altered = evaluate_head(model,'val',base_only=True)
                else:
                    saved = model.core.weights.detach().clone()
                    try:
                        with torch.no_grad(): model.core.weights.copy_(model.initial_circuit_weights)
                        altered = evaluate_head(model,'val')
                    finally:
                        with torch.no_grad(): model.core.weights.copy_(saved)
                ablation_dir=directory/ablation;ablation_dir.mkdir()
                _,_,altered_draws=save_evaluation(altered,ablation_dir,'validation',fitted_thresholds=thresholds)
                ablation_results[(ablation,seed)],ablation_draws[(ablation,seed)]=altered,altered_draws
                ablation_rows.append({'seed':seed,'ablation':ablation,
                                      **{k:altered['metrics'][k] for k in METRIC_NAMES},
                                      'residual_logit_rms':altered['residual_logit_rms']})
            # Self-contained inference bundle: original frozen CNN + selected head.
            bundle = {'head_checkpoint':torch.load(directory/'best_model.pt',map_location='cpu',weights_only=True),
                      'backbone_features_state_dict':{k.removeprefix('features.'):v for k,v in source_weights_for_export.items()},
                      'preprocessing':configuration['preprocessing'],'configuration':clean_json(configuration),
                      'thresholds':thresholds}
            torch.save(bundle,OUTPUT_DIR/f'hybrid_inference_seed_{seed}.pt')
        if RUN_TEST:
            test_result=evaluate_head(model,'test');test_by_run[(arm,seed)]=test_result
            test_fixed,_,_=save_evaluation(test_result,directory,'test',fitted_thresholds=thresholds)
            save_json(directory/'test_summary.json',{'ranking':test_result['metrics'],'classification':test_fixed})
        del model;gc.collect()
        pd.DataFrame(rows).to_csv(OUTPUT_DIR/'experiment_comparison.csv',index=False)

# Equal-weight probability ensembles are predeclared for ALL arms. Never use a
# best-seed selection or optimize ensemble weights on the validation set.
ensemble_results,ensemble_draws={},{}
if len(SEEDS)>1:
    for arm in ARMS:
        directory=OUTPUT_DIR/f'{arm}_seed_ensemble';directory.mkdir()
        probability=np.mean([results_by_run[(arm,seed)]['probabilities'] for seed in SEEDS],axis=0)
        result=make_result('val',probability)
        fixed,thresholds,draws=save_evaluation(result,directory,'validation',fit_thresholds=True)
        ensemble_results[arm],ensemble_draws[arm]=result,draws
        rows.append({'arm':arm+'_ensemble','seed':'all','checkpoint_epoch':None,
                     **{f'validation_{k}':result['metrics'][k] for k in METRIC_NAMES},
                     'validation_label_accuracy_05':fixed['label_accuracy'],'validation_micro_F1_05':fixed['micro_F1'],
                     'metrics_source':'Equal probability mean of all declared seed checkpoints'})
        save_json(directory/'ensemble_manifest.json',{'members':[str(OUTPUT_DIR/f'{arm}_seed_{seed}'/'best_model.pt') for seed in SEEDS],
                                                    'weights':[1/len(SEEDS)]*len(SEEDS),'validation_metrics':result['metrics']})
        if RUN_TEST:
            test_probability=np.mean([test_by_run[(arm,seed)]['probabilities'] for seed in SEEDS],axis=0)
            save_evaluation(make_result('test',test_probability),directory,'test',fitted_thresholds=thresholds)


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


paired=[]
for seed in SEEDS:
    result,draws=results_by_run[('quantum_residual',seed)],draws_by_run[('quantum_residual',seed)]
    for comparator in ['linear_control','classical_residual','frozen_quantum','experiment_2_reproduced','branch_removed','circuit_reset']:
        if comparator=='experiment_2_reproduced': other,other_draws=baseline_result,baseline_draws
        elif comparator in ['branch_removed','circuit_reset']:
            other,other_draws=ablation_results[(comparator,seed)],ablation_draws[(comparator,seed)]
        else:other,other_draws=results_by_run[(comparator,seed)],draws_by_run[(comparator,seed)]
        paired.extend(paired_rows_for(result,other,draws,other_draws,'quantum_residual minus '+comparator,seed))
if len(SEEDS)>1:
    for comparator in ['linear_control','classical_residual','frozen_quantum']:
        paired.extend(paired_rows_for(ensemble_results['quantum_residual'],ensemble_results[comparator],
                       ensemble_draws['quantum_residual'],ensemble_draws[comparator],
                       'quantum_ensemble minus '+comparator+'_ensemble','all'))
    paired.extend(paired_rows_for(ensemble_results['quantum_residual'],baseline_result,ensemble_draws['quantum_residual'],
                                 baseline_draws,'quantum_ensemble minus experiment_2_reproduced','all'))
paired_df=pd.DataFrame(paired);paired_df.to_csv(OUTPUT_DIR/'paired_patient_bootstrap_differences.csv',index=False)
pd.DataFrame(ablation_rows).to_csv(OUTPUT_DIR/'quantum_ablation_metrics.csv',index=False)
comparison=pd.DataFrame(rows);comparison.to_csv(OUTPUT_DIR/'experiment_comparison.csv',index=False)
seed_rows=[]
for arm in ARMS:
    chosen=comparison.loc[comparison.arm==arm]
    for metric in [f'validation_{k}' for k in METRIC_NAMES]+['validation_label_accuracy_05','validation_micro_F1_05']:
        values=chosen[metric].to_numpy(float)
        seed_rows.append({'arm':arm,'metric':metric,'seeds':len(values),'mean':values.mean(),
                          'standard_deviation_across_seeds':values.std(ddof=1) if len(values)>1 else np.nan,
                          'minimum':values.min(),'maximum':values.max()})
seed_summary=pd.DataFrame(seed_rows);seed_summary.to_csv(OUTPUT_DIR/'seed_summary.csv',index=False)
if RUN_TEST:
    with torch.no_grad():
        probability=torch.sigmoid(F.linear(torch.from_numpy(feature_arrays['test']),original_weight,original_bias)).numpy()
    save_evaluation(make_result('test',probability),baseline_dir,'test',fitted_thresholds=baseline_thresholds)

# Compact comparison chart: seed variability and equal-weight ensemble separate.
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric in zip(axes,['validation_macro_AUROC','validation_macro_AUPRC']):
    for index,arm in enumerate(ARMS):
        values=comparison.loc[comparison.arm==arm,metric].to_numpy(float)
        ax.scatter(np.full(len(values),index),values,s=30)
        ax.plot([index-.2,index+.2],[values.mean()]*2,color='black')
    ax.axhline(baseline_result['metrics'][metric.removeprefix('validation_')],color='gray',linestyle='--',label='E2 reproduced')
    ax.set_xticks(range(len(ARMS)),ARMS,rotation=20,ha='right');ax.set_ylabel(metric);ax.legend()
fig.tight_layout();fig.savefig(OUTPUT_DIR/'model_comparison.png',dpi=150);plt.close(fig)
summary={**configuration,'comparisons':comparison.to_dict('records'),'seed_summary':seed_summary.to_dict('records'),
         'quantum_ablation_metrics':ablation_rows,
         'interpretation':[
             'A high hybrid score does not by itself establish a useful quantum contribution.',
             'Epoch0 retention or zero residual means this checkpoint does not demonstrate learned QML contribution.',
             'Branch removal and circuit reset are inference ablations; the frozen-circuit and classical residual arms are retrained controls.',
             'A trainable quantum branch may work without outperforming every classical model; report actual comparisons.',
             'All validation results remain exploratory after repeated model selection on this split.',
             'Patient bootstrap intervals exclude retraining/model-selection uncertainty; seed SD is not a confidence interval.',
             'The hybrid uses a frozen CNN. The projection, circuit, readout and retained classifier train jointly after warmup.',
             'Simulation does not establish quantum speedup or hardware advantage.'
         ]}
save_json(OUTPUT_DIR/'summary.json',summary)
print('\nCHECKPOINT AND ENSEMBLE COMPARISON:')
print(comparison[['arm','seed','checkpoint_epoch','validation_macro_AUROC','validation_macro_AUPRC']].to_string(index=False))
print('\nQUANTUM CONTRIBUTION DIAGNOSTIC (positive favors intact hybrid):')
print(paired_df.loc[(paired_df.metric=='macro_AUPRC') & paired_df.comparison.str.contains('branch_removed|frozen_quantum|classical_residual')].to_string(index=False))
print('Test untouched: no test features, predictions or metrics computed.' if not RUN_TEST else 'Test metrics reported for all predeclared models with validation-only thresholds.')
print('Saved to:',OUTPUT_DIR)

# ============================================================
# 6. LOAD A SELF-CONTAINED EXPORTED HYBRID FOR IMAGE INFERENCE
# ============================================================
class ExportedHybrid(nn.Module):
    """Forward receives RGB320 ImageNet-normalized tensors and returns CPU logits.

    Keep the quantum head on CPU. This wrapper explicitly moves images to the
    frozen CNN device and its pooled features back to CPU. Use .eval(). Do not
    move the entire wrapper with .cuda(); select image_device in the loader.
    """
    def __init__(self,bundle,image_device):
        super().__init__()
        self.image_device=torch.device(image_device)
        self.cnn=densenet121(weights=None).features.to(self.image_device)
        self.cnn.load_state_dict(bundle['backbone_features_state_dict'],strict=True)
        checkpoint=bundle['head_checkpoint']
        assert checkpoint['n_qubits']==N_QUBITS and checkpoint['q_layers']==Q_LAYERS
        assert checkpoint['labels']==LABELS
        # Every head value is loaded below; no training source globals are needed.
        self.head=ResidualHead(checkpoint['arm'],checkpoint['seed'],initialize_from_source=False)
        self.head.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.register_buffer('feature_mean',checkpoint['feature_mean'].cpu())
        self.register_buffer('feature_scale',checkpoint['feature_scale'].cpu())
        self.requires_grad_(False);self.eval()
    def forward(self,images):
        pooled=F.adaptive_avg_pool2d(F.relu(self.cnn(images.to(self.image_device)),inplace=False),(1,1)).flatten(1).cpu()
        return self.head((pooled-self.feature_mean)/self.feature_scale)


def load_exported_hybrid(path,image_device=None):
    bundle=torch.load(path,map_location='cpu',weights_only=True)
    image_device=image_device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return ExportedHybrid(bundle,image_device),bundle['thresholds']

# After training, for example:
# inference_model, thresholds = load_exported_hybrid(OUTPUT_DIR/'hybrid_inference_seed_42.pt')
# with Image.open('/path/to/image.jpg') as image:
#     tensor = eval_transform(image.convert('RGB')).unsqueeze(0)
# with torch.inference_mode(): probabilities = inference_model(tensor).sigmoid()[0]
# LABELS fixes the output order; thresholds may contain None for rare labels.
