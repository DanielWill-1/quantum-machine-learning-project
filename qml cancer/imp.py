# ============================================================
# EXPERIMENT 5 — FROZEN EXP2 DENSENET + HYBRID QML PILOT
# One complete notebook cell / standalone Python script.
# Run in a fresh kernel with the working Torch + torchvision installation.
# Actual analytic quantum-circuit simulation; no quantum hardware/API required.
# This is a pilot comparison, not evidence of quantum advantage.
# ============================================================
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, gc, json, math, time, random, hashlib, platform, subprocess, importlib.util
from pathlib import Path
from datetime import datetime, timezone
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

# If PennyLane is missing, install only it; keep your working CUDA Torch build.
if importlib.util.find_spec('pennylane') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pennylane>=0.40,<0.46'])
import pennylane as qml

# ============================================================
# 0. CONFIGURATION
# ============================================================
ROOT = Path('workspace/unzippedarchive/unzippedarchive')
INDEX_CSV = ROOT / 'xray_image_label_index.csv'
SPLIT_CSV = ROOT / 'xray_training_output/patient_level_splits.csv'
EXP2_DIR = ROOT / 'xray_training_experiment_2'
SOURCE_CHECKPOINT = EXP2_DIR / 'best_model.pt'
SOURCE_SUMMARY = EXP2_DIR / 'summary.json'
OUTPUT_ROOT = ROOT / 'xray_training_experiment_5_qml_pilot'
CACHE_ROOT = ROOT / 'xray_exp2_frozen_feature_cache'
IMAGE_SIZE = 320
FEATURE_DIM = 1024
EXTRACT_BATCH_SIZE = 24
NUM_WORKERS = 4 if sys.platform.startswith('linux') else 0
EXTRACT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# FP32 once-only extraction, used identically by all heads.
SEEDS = [42, 43, 44]  # set [42] for a quick exploratory smoke run only
N_QUBITS = 4
Q_LAYERS = 2
HEAD_BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 4
STOP_MIN_DELTA = 0.001
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.30
POS_WEIGHT_CAP = 5.0
HEAD_DEVICE = 'auto'  # auto benchmarks CPU/CUDA quantum forward+backward
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
ARMS = ['linear_probe', 'classical_bottleneck', 'quantum_hybrid']
FIXED_THRESHOLDS = {label: 0.5 for label in LABELS}
assert N_QUBITS == 4 and Q_LAYERS == 2, 'This matched pilot is specified for 4 qubits / 2 layers.'
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
print('EXPERIMENT 5 PILOT | feature extraction:', EXTRACT_DEVICE)
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
# 2. MATCHED CLASSICAL AND QUANTUM HEADS
# ============================================================
class ClassicalCore(nn.Module):
    # 16 mixing weights + 4 scales + 4 biases = 24 trainable parameters.
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(N_QUBITS, N_QUBITS, bias=False)
        nn.init.orthogonal_(self.mix.weight)
        self.scale = nn.Parameter(torch.ones(N_QUBITS))
        self.bias = nn.Parameter(torch.zeros(N_QUBITS))
    def forward(self, angles):
        return torch.tanh(self.mix(angles / math.pi) * self.scale + self.bias)


def quantum_core():
    device = qml.device('default.qubit', wires=N_QUBITS, shots=None)
    @qml.qnode(device, interface='torch', diff_method='backprop')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]
    # 2 layers * 4 qubits * 3 rotation angles = 24 parameters.
    return qml.qnn.TorchLayer(circuit, {'weights': (Q_LAYERS, N_QUBITS, 3)},
                             init_method=lambda tensor: nn.init.uniform_(tensor, -0.1, 0.1))


class PilotHead(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        if arm not in ARMS: raise ValueError(f'Unknown arm: {arm}')
        self.arm = arm
        self.dropout = nn.Dropout(DROPOUT)
        if arm == 'linear_probe':
            self.readout = nn.Linear(FEATURE_DIM, len(LABELS))
            # Algebraic conversion restores the original head on standardized z.
            with torch.no_grad():
                self.readout.weight.copy_(original_weight * torch.from_numpy(feature_scale))
                self.readout.bias.copy_(original_bias + original_weight @ torch.from_numpy(feature_mean))
        else:
            seed_everything(seed)
            self.project = nn.Linear(FEATURE_DIM, N_QUBITS)
            nn.init.xavier_uniform_(self.project.weight, gain=.1)
            nn.init.zeros_(self.project.bias)
            self.readout = nn.Linear(N_QUBITS, len(LABELS))
            # Create the core only after shared modules: same starting projection/readout.
            self.core = quantum_core() if arm == 'quantum_hybrid' else ClassicalCore()
    def forward(self, features):
        features = self.dropout(features)
        if self.arm == 'linear_probe': return self.readout(features)
        angles = math.pi * torch.tanh(self.project(features))
        values = self.core(angles).to(dtype=features.dtype)
        return self.readout(values)


classical_check = PilotHead('classical_bottleneck', 42)
quantum_check = PilotHead('quantum_hybrid', 42)
for key in ['project.weight', 'project.bias', 'readout.weight', 'readout.bias']:
    assert torch.equal(classical_check.state_dict()[key], quantum_check.state_dict()[key]), key
assert sum(p.numel() for p in classical_check.parameters()) == sum(p.numel() for p in quantum_check.parameters())
print('Matched trainable parameters:', sum(p.numel() for p in quantum_check.parameters()))
# Real runtime checks: batched circuit == per-sample circuit; gradients must flow
# through both the input encoding and the variational circuit weights.
core = quantum_check.core
angles = torch.randn(3, N_QUBITS, requires_grad=True)
batched = core(angles)
singles = torch.cat([core(angles[i:i+1]) for i in range(3)], 0)
assert batched.shape == (3, N_QUBITS)
assert torch.allclose(batched, singles, atol=1e-6, rtol=1e-6)
(batched * torch.arange(1, N_QUBITS+1)).sum().backward()
assert angles.grad is not None and torch.isfinite(angles.grad).all() and angles.grad.abs().sum() > 0
qgrads = [p.grad for p in core.parameters()]
assert all(g is not None and torch.isfinite(g).all() for g in qgrads)
assert sum(g.abs().sum().item() for g in qgrads) > 0
print('Quantum batched output and gradient smoke checks passed.')
del core, angles, batched, singles, classical_check, quantum_check


def benchmark_device(name):
    model = PilotHead('quantum_hybrid', 42).to(name).eval()
    batch = torch.from_numpy(standardized['train'][:HEAD_BATCH_SIZE]).to(name)
    times = []
    for repetition in range(4):
        if name == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        output = model(batch)
        output.square().mean().backward()
        if name == 'cuda': torch.cuda.synchronize()
        if repetition: times.append(time.perf_counter()-started)
    del model, batch
    if name == 'cuda': torch.cuda.empty_cache()
    return float(np.median(times))


benchmark_rows = []
candidates = ['cpu'] + (['cuda'] if torch.cuda.is_available() else []) if HEAD_DEVICE == 'auto' else [HEAD_DEVICE]
for name in candidates:
    try:
        seconds = benchmark_device(name)
        benchmark_rows.append({'device': name, 'batch_forward_backward_seconds': seconds, 'error': None})
    except (RuntimeError, TypeError, ValueError, NotImplementedError) as error:
        benchmark_rows.append({'device': name, 'batch_forward_backward_seconds': np.nan, 'error': str(error)})
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
valid_devices = [row for row in benchmark_rows if math.isfinite(row['batch_forward_backward_seconds'])]
if not valid_devices: raise RuntimeError(f'No working quantum backend: {benchmark_rows}')
head_device = torch.device(min(valid_devices, key=lambda row: row['batch_forward_backward_seconds'])['device'])
pd.DataFrame(benchmark_rows).to_csv(OUTPUT_DIR / 'head_device_benchmark.csv', index=False)
print('Head device selected:', head_device, '| benchmarking:', benchmark_rows)
# Both matched arms use the same device, dtype, minibatch order, LR and budget.
features = {name: torch.from_numpy(value).to(head_device) for name, value in standardized.items()}
targets = {name: torch.from_numpy(encoded[name][0]).to(head_device) for name in active_splits}
masks = {name: torch.from_numpy(encoded[name][1]).to(head_device) for name in active_splits}
criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(pos_weights, device=head_device))

# ============================================================
# 3. IDENTICAL HEAD-TRAINING BUDGETS; BEST VALIDATION AUPRC
# ============================================================
@torch.no_grad()
def evaluate_head(model, split):
    model.eval()
    probabilities, loss_sum, valid_count = [], 0.0, 0.0
    for start in range(0, len(features[split]), HEAD_BATCH_SIZE):
        logits = model(features[split][start:start+HEAD_BATCH_SIZE])
        truth = targets[split][start:start+HEAD_BATCH_SIZE]
        mask = masks[split][start:start+HEAD_BATCH_SIZE]
        if not torch.isfinite(logits).all(): raise FloatingPointError('Non-finite logits.')
        loss_sum += float((criterion(logits, truth) * mask).sum().item())
        valid_count += float(mask.sum().item())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    result = make_result(split, np.concatenate(probabilities))
    result['loss'] = loss_sum / valid_count if valid_count else float('nan')
    return result


def sync_device():
    if head_device.type == 'cuda': torch.cuda.synchronize()


def train_head(arm, seed, run_dir):
    seed_everything(seed)
    model = PilotHead(arm, seed).to(head_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    # Same scheduler as the supplied Experiment 2, applied equally to each head.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=.5, patience=2, min_lr=1e-7)
    order_generator = torch.Generator(device='cpu').manual_seed(seed + 1000)
    # Reset after construction so differing core initializers do not shift dropout RNG.
    seed_everything(seed + 2000)
    initial = evaluate_head(model, 'val')
    if arm == 'linear_probe':
        assert np.allclose(initial['probabilities'], baseline_probabilities, atol=3e-5, rtol=3e-5), \
            'Standardized linear head does not reproduce the source classifier.'
    best_score = float(initial['metrics']['macro_AUPRC'])
    if not math.isfinite(best_score): raise RuntimeError('No evaluable validation macro AUPRC.')
    best_epoch, stopping_reference, stale_epochs = 0, best_score, 0
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_checkpoint(epoch, result):
        checkpoint = {
            'experiment': 'Experiment 5 QML pilot', 'arm': arm, 'seed': seed, 'epoch': epoch,
            'model_state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            'macro_AUROC': float(result['metrics']['macro_AUROC']),
            'macro_AUPRC': float(result['metrics']['macro_AUPRC']),
            'labels': LABELS, 'source_checkpoint': str(SOURCE_CHECKPOINT),
            'source_checkpoint_sha256': source_hash, 'feature_dim': FEATURE_DIM,
            'feature_mean': torch.from_numpy(feature_mean), 'feature_scale': torch.from_numpy(feature_scale),
            'n_qubits': N_QUBITS, 'q_layers': Q_LAYERS, 'dropout': DROPOUT,
            'trainable_parameters': parameter_count,
            'quantum_backend': 'default.qubit / backprop / analytic shots=None',
            'torch_version': str(torch.__version__), 'pennylane_version': qml.__version__,
            'note': 'Head inference checkpoint; load the frozen E2 source backbone separately. Not an optimizer-resume checkpoint.',
        }
        temporary = run_dir / 'best_model.tmp'
        torch.save(checkpoint, temporary)
        os.replace(temporary, run_dir / 'best_model.pt')

    save_checkpoint(0, initial)
    history = [{'epoch': 0, 'train_loss': np.nan, 'validation_loss': initial['loss'],
                **{key: initial['metrics'][key] for key in METRIC_NAMES},
                'learning_rate': HEAD_LR, 'train_seconds': 0.0, 'validation_seconds': 0.0}]
    total_started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(features['train']), generator=order_generator).to(head_device)
        loss_sum, valid_count = 0.0, 0.0
        learning_rate = optimizer.param_groups[0]['lr']
        sync_device(); started = time.perf_counter()
        for selected in indices.split(HEAD_BATCH_SIZE):
            mask = masks['train'][selected]
            denominator = mask.sum()
            if denominator.item() == 0: continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(features['train'][selected])
            total_loss = (criterion(logits, targets['train'][selected]) * mask).sum()
            loss = total_loss / denominator
            if not torch.isfinite(loss): raise FloatingPointError(f'{arm}: non-finite loss.')
            loss.backward()
            # Fail immediately if a backend produces invalid gradients.
            if epoch == 1 and valid_count == 0:
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f'Missing/non-finite gradient: {arm}/{name}')
            optimizer.step()
            loss_sum += total_loss.detach().item()
            valid_count += denominator.item()
        sync_device(); train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = evaluate_head(model, 'val')
        sync_device(); validation_seconds = time.perf_counter() - started
        score = float(result['metrics']['macro_AUPRC'])
        if not math.isfinite(score): raise FloatingPointError('Non-finite validation AUPRC.')
        fixed, _ = threshold_metrics(result['targets'], result['probabilities'], result['masks'], FIXED_THRESHOLDS)
        row = {'epoch': epoch, 'train_loss': loss_sum / valid_count if valid_count else np.nan,
               'validation_loss': result['loss'], **{key: result['metrics'][key] for key in METRIC_NAMES},
               'label_accuracy_05': fixed['label_accuracy'], 'micro_F1_05': fixed['micro_F1'],
               'learning_rate': learning_rate, 'train_seconds': train_seconds, 'validation_seconds': validation_seconds}
        history.append(row)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            save_checkpoint(epoch, result)
        # Save EVERY true maximum; min_delta applies only to the stopping clock.
        if score > stopping_reference + STOP_MIN_DELTA:
            stopping_reference, stale_epochs = score, 0
        else:
            stale_epochs += 1
        scheduler.step(score)
        pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)
        print(f'{arm} seed={seed} {epoch:02d}/{MAX_EPOCHS} | loss {row["train_loss"]:.4f} '
              f'val {row["validation_loss"]:.4f} | AUROC {row["macro_AUROC"]:.5f} '
              f'AUPRC {score:.5f} | acc@.5 {fixed["label_accuracy"]:.4f} '
              f'microF1@.5 {fixed["micro_F1"]:.4f} | train {train_seconds:.2f}s '
              f'val {validation_seconds:.2f}s' + (' | best saved' if improved else ''))
        if stale_epochs >= PATIENCE:
            print('Early stopping:', PATIENCE, 'epochs without cumulative gain >', STOP_MIN_DELTA)
            break
    elapsed = time.perf_counter() - total_started
    checkpoint = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    best_result = evaluate_head(model, 'val')
    assert abs(best_result['metrics']['macro_AUPRC'] - best_score) < 1e-7
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / 'training_history.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_df.epoch, history_df.train_loss, label='train')
    axes[0].plot(history_df.epoch, history_df.validation_loss, label='validation')
    axes[0].set(title='Masked weighted BCE', xlabel='Epoch'); axes[0].legend()
    for metric in ['macro_AUROC', 'macro_AUPRC']:
        axes[1].plot(history_df.epoch, history_df[metric], label=metric)
    axes[1].axvline(best_epoch, color='gray', linestyle='--')
    axes[1].set(title=f'{arm}, seed {seed}', xlabel='Epoch'); axes[1].legend()
    fig.tight_layout(); fig.savefig(run_dir / 'training_curves.png', dpi=150); plt.close(fig)
    return model, best_result, {'arm': arm, 'seed': seed, 'checkpoint_epoch': best_epoch,
                               'epochs_completed': len(history)-1, 'trainable_parameters': parameter_count,
                               'head_training_and_validation_seconds': elapsed}

# ============================================================
# 4. MASK-AWARE REPORTS AND PAIRED PATIENT BOOTSTRAPS
# ============================================================
def bootstrap_report(result):
    # Keep NaNs at their original draw positions. All runs use identical patient
    # draws, allowing paired differences without doing a second bootstrap pass.
    subjects = np.asarray(result['subject_ids'])
    unique_subjects = np.unique(subjects)
    groups = {subject: np.flatnonzero(subjects == subject) for subject in unique_subjects}
    point = result['metrics']
    eligible = [i for i, label in enumerate(LABELS) if point[label]['positives'] and point[label]['negatives']]
    stable = [i for i in eligible if point[LABELS[i]]['positives'] >= 50]
    keys = [('macro', metric) for metric in METRIC_NAMES] + [(label, metric) for label in LABELS for metric in ['AUROC', 'AUPRC']]
    draws = np.full((BOOTSTRAP_SAMPLES, len(keys)), np.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for draw in tqdm(range(BOOTSTRAP_SAMPLES), desc='Patient-cluster CIs'):
        sample = rng.choice(unique_subjects, len(unique_subjects), replace=True)
        selected = np.concatenate([groups[subject] for subject in sample])
        values = calculate_metrics(result['targets'][selected], result['probabilities'][selected],
                                   result['masks'][selected], eligible, stable)
        draws[draw] = [values[metric] if label == 'macro' else values[label][metric] for label, metric in keys]
    rows = []
    for column, (label, metric) in enumerate(keys):
        finite = draws[:, column][np.isfinite(draws[:, column])]
        lo, hi = np.quantile(finite, [BOOTSTRAP_ALPHA/2, 1-BOOTSTRAP_ALPHA/2]) if len(finite) else [np.nan, np.nan]
        rows.append({'label': label, 'metric': metric,
                     'point_estimate': point[metric] if label == 'macro' else point[label][metric],
                     'ci_lower': lo, 'ci_upper': hi, 'confidence_level': 1-BOOTSTRAP_ALPHA,
                     'bootstrap_samples_requested': BOOTSTRAP_SAMPLES, 'bootstrap_samples_valid': len(finite),
                     'bootstrap_samples_undefined': BOOTSTRAP_SAMPLES-len(finite)})
    return pd.DataFrame(rows), draws[:, :len(METRIC_NAMES)]


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


configuration = {
    'experiment': 'Experiment 5 hybrid QML pilot', 'source_checkpoint': SOURCE_CHECKPOINT,
    'source_checkpoint_sha256': source_hash, 'source_checkpoint_epoch': source_epoch,
    'source_saved_validation_metrics': source_metrics, 'source_summary': SOURCE_SUMMARY,
    'output_directory': OUTPUT_DIR, 'feature_cache': cache_dir, 'feature_cache_seconds': feature_seconds,
    'feature_preprocessing': 'Exact E2: RGB -> bilinear aspect resize + black pad 320 -> tensor -> ImageNet normalization',
    'feature_dimension': FEATURE_DIM, 'feature_standardization': 'Mean/std from training only',
    'image_augmentation': 'None: deterministic frozen-feature pilot, identical for all arms',
    'backbone_frozen': True, 'backbone_batchnorm': 'eval', 'seeds': SEEDS, 'arms': ARMS,
    'head_device': str(head_device), 'feature_extraction_device': str(EXTRACT_DEVICE),
    'n_qubits': N_QUBITS, 'quantum_layers': Q_LAYERS, 'shots': None,
    'quantum_backend': 'PennyLane default.qubit', 'differentiation': 'Torch backprop',
    'encoding': 'pi*tanh(trainable projection); AngleEmbedding Y',
    'variational_circuit': 'StronglyEntanglingLayers; per-wire PauliZ expectation readout',
    'head_batch_size': HEAD_BATCH_SIZE, 'learning_rate': HEAD_LR, 'weight_decay': WEIGHT_DECAY,
    'dropout': DROPOUT, 'pos_weight_cap': POS_WEIGHT_CAP, 'pos_weights': pos_weights,
    'max_epochs': MAX_EPOCHS, 'early_stopping_patience': PATIENCE, 'stopping_min_delta': STOP_MIN_DELTA,
    'scheduler': 'ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-7)',
    'checkpoint_criterion': 'Validation macro AUPRC; save each true maximum, including epoch 0',
    'labels': LABELS, 'label_policy': '1 positive; 0/NaN negative; -1 ignored with mask',
    'bootstrap_samples': BOOTSTRAP_SAMPLES, 'bootstrap_seed': BOOTSTRAP_SEED,
    'bootstrap_unit': 'patient with all images; repeated patients retain multiplicity',
    'bootstrap_interpretation': 'Conditional on fitted checkpoints; excludes training/model-selection uncertainty',
    'threshold_fit_min_validation_positives': MIN_POSITIVES_FOR_THRESHOLD,
    'run_test': RUN_TEST, 'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
    'pennylane': qml.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
}
save_json(OUTPUT_DIR / 'configuration.json', configuration)

baseline_dir = OUTPUT_DIR / 'experiment_2_reproduced'
baseline_dir.mkdir()
baseline_fixed, baseline_thresholds, baseline_draws = save_evaluation(
    baseline_result, baseline_dir, 'validation', fit_thresholds=True)
comparison_rows = [
    {'arm': 'experiment_2_saved_checkpoint', 'seed': None, 'checkpoint_epoch': source_epoch,
     'validation_macro_AUROC': source_metrics['macro_AUROC'], 'validation_macro_AUPRC': source_metrics['macro_AUPRC'],
     'metrics_source': str(SOURCE_SUMMARY)},
    {'arm': 'experiment_2_reproduced_fp32', 'seed': None, 'checkpoint_epoch': source_epoch,
     **{f'validation_{key}': baseline_result['metrics'][key] for key in METRIC_NAMES},
     'validation_label_accuracy_05': baseline_fixed['label_accuracy'],
     'validation_micro_F1_05': baseline_fixed['micro_F1'], 'metrics_source': 'recomputed on identical frozen features'},
]
results_by_run, draws_by_run = {}, {}
for seed in SEEDS:
    for arm in ARMS:
        run_dir = OUTPUT_DIR / f'{arm}_seed_{seed}'
        run_dir.mkdir()
        model, result, run_info = train_head(arm, seed, run_dir)
        fixed, fitted_thresholds, draws = save_evaluation(result, run_dir, 'validation', fit_thresholds=True)
        results_by_run[(arm, seed)], draws_by_run[(arm, seed)] = result, draws
        row = {**run_info, **{f'validation_{key}': result['metrics'][key] for key in METRIC_NAMES},
               'validation_loss': result['loss'], 'validation_label_accuracy_05': fixed['label_accuracy'],
               'validation_micro_F1_05': fixed['micro_F1'], 'validation_macro_F1_05': fixed['macro_F1'],
               'validation_macro_balanced_accuracy_05': fixed['macro_balanced_accuracy'],
               'metrics_source': str(run_dir / 'best_model.pt')}
        comparison_rows.append(row)
        save_json(run_dir / 'summary.json', {**configuration, **row, 'classification_fixed_05': fixed})
        # Test is only accessed if the user explicitly enables it. Every declared
        # seed/arm is then reported; never select a seed or threshold using test.
        if RUN_TEST:
            test_result = evaluate_head(model, 'test')
            test_fixed, _, _ = save_evaluation(test_result, run_dir, 'test', fitted_thresholds=fitted_thresholds)
            save_json(run_dir / 'test_summary.json', {'ranking': test_result['metrics'], 'classification_fixed_05': test_fixed})
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / 'experiment_comparison.csv', index=False)

# Paired bootstrap deltas use the exact same patient draws and image ordering.
# No p-values/multiple-comparison claims; CIs are exploratory after validation selection.
paired_rows = []
for seed in SEEDS:
    quantum = results_by_run[('quantum_hybrid', seed)]
    quantum_draws = draws_by_run[('quantum_hybrid', seed)]
    for comparator in ['classical_bottleneck', 'linear_probe', 'experiment_2_reproduced_fp32']:
        other = baseline_result if comparator.startswith('experiment_2') else results_by_run[(comparator, seed)]
        other_draws = baseline_draws if comparator.startswith('experiment_2') else draws_by_run[(comparator, seed)]
        for key in ['dicom_ids', 'subject_ids', 'targets', 'masks']:
            assert np.array_equal(quantum[key], other[key]), f'Paired comparison mismatch: {key}'
        for column, metric in enumerate(METRIC_NAMES):
            difference = quantum_draws[:, column] - other_draws[:, column]
            finite = difference[np.isfinite(difference)]
            lower, upper = np.quantile(finite, [BOOTSTRAP_ALPHA/2, 1-BOOTSTRAP_ALPHA/2]) if len(finite) else [np.nan, np.nan]
            paired_rows.append({'seed': seed, 'comparison': f'quantum_hybrid minus {comparator}', 'metric': metric,
                                'point_delta': quantum['metrics'][metric] - other['metrics'][metric],
                                'ci_lower': lower, 'ci_upper': upper, 'confidence_level': 1-BOOTSTRAP_ALPHA,
                                'bootstrap_samples_valid': len(finite), 'bootstrap_samples_requested': BOOTSTRAP_SAMPLES})
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(OUTPUT_DIR / 'paired_patient_bootstrap_differences.csv', index=False)
comparison = pd.DataFrame(comparison_rows)
seed_rows = []
for arm in ARMS:
    rows = comparison.loc[comparison.arm == arm]
    for metric in [f'validation_{name}' for name in METRIC_NAMES] + ['validation_label_accuracy_05', 'validation_micro_F1_05']:
        values = rows[metric].to_numpy(float)
        seed_rows.append({'arm': arm, 'metric': metric, 'seeds': len(values), 'mean': values.mean(),
                          'standard_deviation_across_seeds': values.std(ddof=1) if len(values)>1 else np.nan,
                          'minimum': values.min(), 'maximum': values.max()})
seed_summary = pd.DataFrame(seed_rows)
seed_summary.to_csv(OUTPUT_DIR / 'seed_summary.csv', index=False)
paired_seed_summary = paired_df.groupby(['comparison', 'metric']).point_delta.agg(['count', 'mean', 'std']).reset_index()
paired_seed_summary.to_csv(OUTPUT_DIR / 'paired_seed_summary.csv', index=False)
if RUN_TEST:
    with torch.no_grad():
        test_baseline_probabilities = torch.sigmoid(F.linear(torch.from_numpy(feature_arrays['test']), original_weight, original_bias)).numpy()
    save_evaluation(make_result('test', test_baseline_probabilities), baseline_dir, 'test', fitted_thresholds=baseline_thresholds)

summary = {**configuration, 'experiment_comparison': comparison.to_dict('records'),
           'seed_summary': seed_summary.to_dict('records'), 'paired_seed_summary': paired_seed_summary.to_dict('records'),
           'interpretation': [
               'Primary pilot contrast: quantum versus matched classical bottleneck at each seed.',
               'The source backbone was already selected on this validation set; these are exploratory validation results.',
               'A seed standard deviation measures initialization variability, not a confidence interval.',
               'Patient bootstrap intervals condition on fitted checkpoints and do not remove validation-selection bias.',
               'Accuracy must be read alongside its all-negative baseline, AUPRC, recall and F1.',
               'A simulated small circuit does not establish quantum computational advantage.',
               'Keep test untouched until the final experimental design is fixed.'
           ]}
save_json(OUTPUT_DIR / 'summary.json', summary)
(OUTPUT_DIR / 'README.txt').write_text(
    'Experiment 5 frozen-feature QML pilot\n'
    'Use experiment_comparison.csv for each saved checkpoint; seed_summary.csv for mean/std across ALL seeds.\n'
    'Primary comparison: paired_patient_bootstrap_differences.csv, quantum_hybrid minus classical_bottleneck.\n'
    'Positive delta favors QML; validation-selected bootstrap intervals remain exploratory.\n'
    'Each arm/seed folder contains best_model.pt, history, predictions, threshold metrics, ranking metrics, CIs and curves.\n'
    'Load source E2 backbone with its exact RGB320 preprocessing, pool 1024 features, then standardize using checkpoint mean/scale.\n'
    'Recreate the saved arm, load its model_state_dict strictly, set eval(), and sigmoid its 13 raw output logits.\n'
    'Frozen feature extraction is cached; no image decoding or CNN forward pass occurs during head training.\n', encoding='utf-8')
print('\nSAVED CHECKPOINT COMPARISON:')
print(comparison[['arm', 'seed', 'checkpoint_epoch', 'validation_macro_AUROC', 'validation_macro_AUPRC']].to_string(index=False))
print('\nPAIRED QUANTUM MINUS CLASSICAL BOTTLENECK:')
print(paired_df.loc[(paired_df.comparison == 'quantum_hybrid minus classical_bottleneck') &
                    (paired_df.metric == 'macro_AUPRC')].to_string(index=False))
print('RUN_TEST=False: no test features/predictions/metrics computed.' if not RUN_TEST else 'Test evaluated for all declared arms/seeds with validation-only thresholds.')
print('Outputs:', OUTPUT_DIR)
EXPERIMENT 5 PILOT | feature extraction: cuda
Torch: 2.14.0+cu126 | torchvision: 0.29.0+cu126 | PennyLane: 0.45.1
Seeds: [42, 43, 44] | RUN_TEST: False
Strict JPEG validation: 100%
 6723/6723 [01:46<00:00, 64.16it/s]
JPEG audit: {'intact': 6665, 'recoverable_truncated': 58}
train 4830 images; 1231 patients
val 970 images; 264 patients
test 923 images; 264 patients
Frozen DenseNet feature extraction: 100%
 202/202 [00:48<00:00,  3.93it/s]
Frozen DenseNet feature extraction: 100%
 41/41 [00:10<00:00,  6.02it/s]
Feature cache: 22.7 MiB; loading/extraction 99.4s
Frozen DenseNet feature extraction: 100%
 1/1 [00:00<00:00,  2.64it/s]
Saved E2 checkpoint metrics: {'macro_AUROC': 0.7303071009041724, 'macro_AUPRC': 0.2930734458959053}
Reproduced FP32 E2 metrics: {'macro_AUROC': 0.7307837861407264, 'macro_AUPRC': 0.29433588632475105, 'macro_AUROC_50plus': 0.7711530780574897, 'macro_AUPRC_50plus': 0.399016943438275}
Matched trainable parameters: 4189
Quantum batched output and gradient smoke checks passed.
Head device selected: cpu | benchmarking: [{'device': 'cpu', 'batch_forward_backward_seconds': 0.01051478274166584, 'error': None}, {'device': 'cuda', 'batch_forward_backward_seconds': nan, 'error': 'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!'}]
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 43.42it/s]
linear_probe seed=42 01/15 | loss 0.5070 val 0.6412 | AUROC 0.73346 AUPRC 0.29738 | acc@.5 0.8453 microF1@.5 0.5127 | train 0.06s val 0.03s | best saved
linear_probe seed=42 02/15 | loss 0.4776 val 0.6441 | AUROC 0.73143 AUPRC 0.29500 | acc@.5 0.8409 microF1@.5 0.5082 | train 0.06s val 0.03s
linear_probe seed=42 03/15 | loss 0.4650 val 0.6499 | AUROC 0.73147 AUPRC 0.29459 | acc@.5 0.8399 microF1@.5 0.5014 | train 0.06s val 0.03s
linear_probe seed=42 04/15 | loss 0.4572 val 0.6539 | AUROC 0.72392 AUPRC 0.29021 | acc@.5 0.8434 microF1@.5 0.5034 | train 0.06s val 0.03s
linear_probe seed=42 05/15 | loss 0.4454 val 0.6543 | AUROC 0.72072 AUPRC 0.29438 | acc@.5 0.8426 microF1@.5 0.5056 | train 0.06s val 0.03s
Early stopping: 4 epochs without cumulative gain > 0.001
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 43.66it/s]
classical_bottleneck seed=42 01/15 | loss 0.8908 val 0.8953 | AUROC 0.59165 AUPRC 0.22469 | acc@.5 0.5640 microF1@.5 0.3127 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=42 02/15 | loss 0.8624 val 0.8712 | AUROC 0.62498 AUPRC 0.23864 | acc@.5 0.6048 microF1@.5 0.3285 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=42 03/15 | loss 0.8374 val 0.8495 | AUROC 0.63396 AUPRC 0.24242 | acc@.5 0.6340 microF1@.5 0.3385 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=42 04/15 | loss 0.8124 val 0.8305 | AUROC 0.64427 AUPRC 0.24760 | acc@.5 0.6813 microF1@.5 0.3599 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=42 05/15 | loss 0.7890 val 0.8126 | AUROC 0.65025 AUPRC 0.25106 | acc@.5 0.7046 microF1@.5 0.3659 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=42 06/15 | loss 0.7684 val 0.7947 | AUROC 0.65339 AUPRC 0.24708 | acc@.5 0.7149 microF1@.5 0.3718 | train 0.07s val 0.03s
classical_bottleneck seed=42 07/15 | loss 0.7477 val 0.7833 | AUROC 0.66877 AUPRC 0.24544 | acc@.5 0.7366 microF1@.5 0.3696 | train 0.07s val 0.03s
classical_bottleneck seed=42 08/15 | loss 0.7307 val 0.7704 | AUROC 0.66576 AUPRC 0.24839 | acc@.5 0.7596 microF1@.5 0.3927 | train 0.07s val 0.03s
classical_bottleneck seed=42 09/15 | loss 0.7168 val 0.7646 | AUROC 0.66306 AUPRC 0.24606 | acc@.5 0.7758 microF1@.5 0.4008 | train 0.07s val 0.03s
Early stopping: 4 epochs without cumulative gain > 0.001
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 44.21it/s]
quantum_hybrid seed=42 01/15 | loss 0.9257 val 0.9289 | AUROC 0.50409 AUPRC 0.14850 | acc@.5 0.5065 microF1@.5 0.2545 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 02/15 | loss 0.9037 val 0.9117 | AUROC 0.53026 AUPRC 0.16836 | acc@.5 0.5452 microF1@.5 0.2680 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 03/15 | loss 0.8852 val 0.8933 | AUROC 0.55331 AUPRC 0.17444 | acc@.5 0.5754 microF1@.5 0.2743 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 04/15 | loss 0.8659 val 0.8707 | AUROC 0.57439 AUPRC 0.17399 | acc@.5 0.6186 microF1@.5 0.2764 | train 0.40s val 0.06s
quantum_hybrid seed=42 05/15 | loss 0.8455 val 0.8522 | AUROC 0.57747 AUPRC 0.18115 | acc@.5 0.6553 microF1@.5 0.2981 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 06/15 | loss 0.8248 val 0.8340 | AUROC 0.59010 AUPRC 0.18600 | acc@.5 0.6857 microF1@.5 0.3305 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 07/15 | loss 0.8042 val 0.8108 | AUROC 0.60623 AUPRC 0.19973 | acc@.5 0.7350 microF1@.5 0.3658 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 08/15 | loss 0.7916 val 0.7944 | AUROC 0.61975 AUPRC 0.19964 | acc@.5 0.7422 microF1@.5 0.3887 | train 0.40s val 0.06s
quantum_hybrid seed=42 09/15 | loss 0.7736 val 0.7888 | AUROC 0.61905 AUPRC 0.18533 | acc@.5 0.7385 microF1@.5 0.3884 | train 0.40s val 0.06s
quantum_hybrid seed=42 10/15 | loss 0.7588 val 0.7632 | AUROC 0.66175 AUPRC 0.21344 | acc@.5 0.7534 microF1@.5 0.4135 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 11/15 | loss 0.7461 val 0.7548 | AUROC 0.65895 AUPRC 0.21417 | acc@.5 0.7501 microF1@.5 0.4078 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 12/15 | loss 0.7398 val 0.7519 | AUROC 0.64683 AUPRC 0.20744 | acc@.5 0.7688 microF1@.5 0.4149 | train 0.40s val 0.06s
quantum_hybrid seed=42 13/15 | loss 0.7265 val 0.7317 | AUROC 0.65319 AUPRC 0.20982 | acc@.5 0.7975 microF1@.5 0.4593 | train 0.40s val 0.06s
quantum_hybrid seed=42 14/15 | loss 0.7147 val 0.7245 | AUROC 0.67813 AUPRC 0.21763 | acc@.5 0.8008 microF1@.5 0.4655 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=42 15/15 | loss 0.7082 val 0.7275 | AUROC 0.67461 AUPRC 0.21215 | acc@.5 0.8096 microF1@.5 0.4569 | train 0.40s val 0.06s
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 43.16it/s]
linear_probe seed=43 01/15 | loss 0.5071 val 0.6419 | AUROC 0.73145 AUPRC 0.29404 | acc@.5 0.8400 microF1@.5 0.5087 | train 0.06s val 0.03s
linear_probe seed=43 02/15 | loss 0.4784 val 0.6454 | AUROC 0.72579 AUPRC 0.29913 | acc@.5 0.8411 microF1@.5 0.5028 | train 0.06s val 0.03s | best saved
linear_probe seed=43 03/15 | loss 0.4631 val 0.6495 | AUROC 0.72681 AUPRC 0.29390 | acc@.5 0.8416 microF1@.5 0.5068 | train 0.06s val 0.03s
linear_probe seed=43 04/15 | loss 0.4548 val 0.6544 | AUROC 0.72633 AUPRC 0.29681 | acc@.5 0.8437 microF1@.5 0.5021 | train 0.06s val 0.03s
linear_probe seed=43 05/15 | loss 0.4481 val 0.6592 | AUROC 0.71491 AUPRC 0.29581 | acc@.5 0.8433 microF1@.5 0.5058 | train 0.06s val 0.03s
linear_probe seed=43 06/15 | loss 0.4388 val 0.6577 | AUROC 0.72206 AUPRC 0.29441 | acc@.5 0.8436 microF1@.5 0.5041 | train 0.06s val 0.02s
Early stopping: 4 epochs without cumulative gain > 0.001
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 43.80it/s]
classical_bottleneck seed=43 01/15 | loss 0.8735 val 0.8753 | AUROC 0.59448 AUPRC 0.19385 | acc@.5 0.6473 microF1@.5 0.3095 | train 0.07s val 0.02s | best saved
classical_bottleneck seed=43 02/15 | loss 0.8379 val 0.8494 | AUROC 0.63376 AUPRC 0.22093 | acc@.5 0.6686 microF1@.5 0.3261 | train 0.07s val 0.02s | best saved
classical_bottleneck seed=43 03/15 | loss 0.8085 val 0.8249 | AUROC 0.64376 AUPRC 0.23048 | acc@.5 0.7019 microF1@.5 0.3486 | train 0.07s val 0.02s | best saved
classical_bottleneck seed=43 04/15 | loss 0.7822 val 0.8028 | AUROC 0.65186 AUPRC 0.23594 | acc@.5 0.7449 microF1@.5 0.3789 | train 0.07s val 0.02s | best saved
classical_bottleneck seed=43 05/15 | loss 0.7590 val 0.7841 | AUROC 0.65435 AUPRC 0.23314 | acc@.5 0.7606 microF1@.5 0.3970 | train 0.07s val 0.02s
classical_bottleneck seed=43 06/15 | loss 0.7369 val 0.7686 | AUROC 0.65335 AUPRC 0.22781 | acc@.5 0.7836 microF1@.5 0.4219 | train 0.07s val 0.02s
classical_bottleneck seed=43 07/15 | loss 0.7171 val 0.7559 | AUROC 0.64432 AUPRC 0.22308 | acc@.5 0.8023 microF1@.5 0.4368 | train 0.07s val 0.02s
classical_bottleneck seed=43 08/15 | loss 0.7026 val 0.7477 | AUROC 0.65407 AUPRC 0.22510 | acc@.5 0.8084 microF1@.5 0.4472 | train 0.07s val 0.02s
Early stopping: 4 epochs without cumulative gain > 0.001
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 43.52it/s]
quantum_hybrid seed=43 01/15 | loss 0.8772 val 0.8584 | AUROC 0.51732 AUPRC 0.13201 | acc@.5 0.6707 microF1@.5 0.3221 | train 0.40s val 0.06s
quantum_hybrid seed=43 02/15 | loss 0.8282 val 0.8351 | AUROC 0.49071 AUPRC 0.13976 | acc@.5 0.6751 microF1@.5 0.3375 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 03/15 | loss 0.8046 val 0.8174 | AUROC 0.51342 AUPRC 0.15157 | acc@.5 0.7192 microF1@.5 0.3225 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 04/15 | loss 0.7872 val 0.8066 | AUROC 0.50733 AUPRC 0.14984 | acc@.5 0.7440 microF1@.5 0.3084 | train 0.40s val 0.06s
quantum_hybrid seed=43 05/15 | loss 0.7749 val 0.7989 | AUROC 0.53187 AUPRC 0.15442 | acc@.5 0.7562 microF1@.5 0.3021 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 06/15 | loss 0.7608 val 0.7926 | AUROC 0.54704 AUPRC 0.15521 | acc@.5 0.7791 microF1@.5 0.2682 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 07/15 | loss 0.7503 val 0.7830 | AUROC 0.53075 AUPRC 0.17242 | acc@.5 0.8113 microF1@.5 0.2559 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 08/15 | loss 0.7413 val 0.7763 | AUROC 0.57596 AUPRC 0.17720 | acc@.5 0.8113 microF1@.5 0.2785 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 09/15 | loss 0.7324 val 0.7670 | AUROC 0.58148 AUPRC 0.18967 | acc@.5 0.8295 microF1@.5 0.3086 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 10/15 | loss 0.7224 val 0.7578 | AUROC 0.60315 AUPRC 0.19466 | acc@.5 0.8328 microF1@.5 0.3351 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 11/15 | loss 0.7155 val 0.7523 | AUROC 0.59946 AUPRC 0.19730 | acc@.5 0.8311 microF1@.5 0.3272 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 12/15 | loss 0.7079 val 0.7425 | AUROC 0.60841 AUPRC 0.20675 | acc@.5 0.8362 microF1@.5 0.3585 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 13/15 | loss 0.6993 val 0.7357 | AUROC 0.61747 AUPRC 0.20447 | acc@.5 0.8292 microF1@.5 0.3826 | train 0.40s val 0.06s
quantum_hybrid seed=43 14/15 | loss 0.6992 val 0.7325 | AUROC 0.62295 AUPRC 0.21196 | acc@.5 0.8356 microF1@.5 0.3775 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=43 15/15 | loss 0.6887 val 0.7340 | AUROC 0.63544 AUPRC 0.20725 | acc@.5 0.8301 microF1@.5 0.3759 | train 0.40s val 0.06s
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 43.15it/s]
linear_probe seed=44 01/15 | loss 0.5085 val 0.6415 | AUROC 0.73107 AUPRC 0.29408 | acc@.5 0.8419 microF1@.5 0.5100 | train 0.06s val 0.03s
linear_probe seed=44 02/15 | loss 0.4779 val 0.6468 | AUROC 0.72852 AUPRC 0.29681 | acc@.5 0.8429 microF1@.5 0.5051 | train 0.06s val 0.03s | best saved
linear_probe seed=44 03/15 | loss 0.4640 val 0.6501 | AUROC 0.71995 AUPRC 0.29543 | acc@.5 0.8406 microF1@.5 0.5011 | train 0.06s val 0.03s
linear_probe seed=44 04/15 | loss 0.4564 val 0.6537 | AUROC 0.72636 AUPRC 0.29354 | acc@.5 0.8384 microF1@.5 0.4958 | train 0.06s val 0.03s
linear_probe seed=44 05/15 | loss 0.4486 val 0.6594 | AUROC 0.72804 AUPRC 0.29387 | acc@.5 0.8404 microF1@.5 0.5004 | train 0.06s val 0.03s
linear_probe seed=44 06/15 | loss 0.4412 val 0.6575 | AUROC 0.72375 AUPRC 0.29636 | acc@.5 0.8423 microF1@.5 0.5018 | train 0.06s val 0.03s
Early stopping: 4 epochs without cumulative gain > 0.001
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 42.95it/s]
classical_bottleneck seed=44 01/15 | loss 0.9069 val 0.9038 | AUROC 0.62832 AUPRC 0.22532 | acc@.5 0.4877 microF1@.5 0.2512 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=44 02/15 | loss 0.8792 val 0.8861 | AUROC 0.62636 AUPRC 0.22833 | acc@.5 0.5347 microF1@.5 0.2641 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=44 03/15 | loss 0.8570 val 0.8669 | AUROC 0.63239 AUPRC 0.23529 | acc@.5 0.5732 microF1@.5 0.2865 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=44 04/15 | loss 0.8338 val 0.8448 | AUROC 0.65349 AUPRC 0.24515 | acc@.5 0.6157 microF1@.5 0.3029 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=44 05/15 | loss 0.8094 val 0.8247 | AUROC 0.67152 AUPRC 0.24834 | acc@.5 0.6716 microF1@.5 0.3323 | train 0.07s val 0.03s | best saved
classical_bottleneck seed=44 06/15 | loss 0.7861 val 0.8068 | AUROC 0.66342 AUPRC 0.24703 | acc@.5 0.7249 microF1@.5 0.3540 | train 0.07s val 0.03s
classical_bottleneck seed=44 07/15 | loss 0.7628 val 0.7884 | AUROC 0.66305 AUPRC 0.24388 | acc@.5 0.7387 microF1@.5 0.3693 | train 0.07s val 0.03s
classical_bottleneck seed=44 08/15 | loss 0.7410 val 0.7737 | AUROC 0.66108 AUPRC 0.23969 | acc@.5 0.7784 microF1@.5 0.3959 | train 0.07s val 0.03s
classical_bottleneck seed=44 09/15 | loss 0.7234 val 0.7651 | AUROC 0.66304 AUPRC 0.24118 | acc@.5 0.7982 microF1@.5 0.4014 | train 0.07s val 0.03s
Early stopping: 4 epochs without cumulative gain > 0.001
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 43.14it/s]
quantum_hybrid seed=44 01/15 | loss 0.9372 val 0.9382 | AUROC 0.51916 AUPRC 0.15691 | acc@.5 0.4843 microF1@.5 0.2135 | train 0.41s val 0.06s | best saved
quantum_hybrid seed=44 02/15 | loss 0.9138 val 0.9193 | AUROC 0.53531 AUPRC 0.16902 | acc@.5 0.5444 microF1@.5 0.2206 | train 0.41s val 0.06s | best saved
quantum_hybrid seed=44 03/15 | loss 0.8913 val 0.8943 | AUROC 0.57594 AUPRC 0.17986 | acc@.5 0.6207 microF1@.5 0.2394 | train 0.41s val 0.06s | best saved
quantum_hybrid seed=44 04/15 | loss 0.8685 val 0.8731 | AUROC 0.56908 AUPRC 0.18331 | acc@.5 0.6375 microF1@.5 0.2533 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=44 05/15 | loss 0.8443 val 0.8519 | AUROC 0.56663 AUPRC 0.18912 | acc@.5 0.6885 microF1@.5 0.2586 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=44 06/15 | loss 0.8149 val 0.8263 | AUROC 0.59341 AUPRC 0.20011 | acc@.5 0.7224 microF1@.5 0.2943 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=44 07/15 | loss 0.7910 val 0.8005 | AUROC 0.61394 AUPRC 0.21311 | acc@.5 0.7707 microF1@.5 0.3454 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=44 08/15 | loss 0.7754 val 0.7871 | AUROC 0.60939 AUPRC 0.21946 | acc@.5 0.7880 microF1@.5 0.3577 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=44 09/15 | loss 0.7569 val 0.7761 | AUROC 0.61933 AUPRC 0.21772 | acc@.5 0.8045 microF1@.5 0.3756 | train 0.40s val 0.06s
quantum_hybrid seed=44 10/15 | loss 0.7395 val 0.7735 | AUROC 0.61707 AUPRC 0.21496 | acc@.5 0.8249 microF1@.5 0.3593 | train 0.40s val 0.06s
quantum_hybrid seed=44 11/15 | loss 0.7279 val 0.7525 | AUROC 0.61238 AUPRC 0.22745 | acc@.5 0.8016 microF1@.5 0.4066 | train 0.40s val 0.06s | best saved
quantum_hybrid seed=44 12/15 | loss 0.7164 val 0.7437 | AUROC 0.62689 AUPRC 0.22738 | acc@.5 0.8180 microF1@.5 0.3992 | train 0.40s val 0.06s
quantum_hybrid seed=44 13/15 | loss 0.7110 val 0.7425 | AUROC 0.62105 AUPRC 0.21830 | acc@.5 0.8272 microF1@.5 0.4112 | train 0.40s val 0.06s
quantum_hybrid seed=44 14/15 | loss 0.7017 val 0.7316 | AUROC 0.63643 AUPRC 0.22308 | acc@.5 0.8205 microF1@.5 0.4186 | train 0.40s val 0.06s
quantum_hybrid seed=44 15/15 | loss 0.6918 val 0.7225 | AUROC 0.64587 AUPRC 0.23062 | acc@.5 0.8232 microF1@.5 0.4356 | train 0.40s val 0.06s | best saved
Patient-cluster CIs: 100%
 500/500 [00:11<00:00, 44.77it/s]

SAVED CHECKPOINT COMPARISON:
                          arm  seed  checkpoint_epoch  validation_macro_AUROC  validation_macro_AUPRC
experiment_2_saved_checkpoint   NaN                 5                0.730307                0.293073
 experiment_2_reproduced_fp32   NaN                 5                0.730784                0.294336
                 linear_probe  42.0                 1                0.733460                0.297381
         classical_bottleneck  42.0                 5                0.650249                0.251064
               quantum_hybrid  42.0                14                0.678131                0.217628
                 linear_probe  43.0                 2                0.725787                0.299127
         classical_bottleneck  43.0                 4                0.651857                0.235936
               quantum_hybrid  43.0                14                0.622947                0.211960
                 linear_probe  44.0                 2                0.728522                0.296808
         classical_bottleneck  44.0                 5                0.671518                0.248342
               quantum_hybrid  44.0                15                0.645872                0.230621

PAIRED QUANTUM MINUS CLASSICAL BOTTLENECK:
 seed                                comparison      metric  point_delta  ci_lower  ci_upper  confidence_level  bootstrap_samples_valid  bootstrap_samples_requested
   42 quantum_hybrid minus classical_bottleneck macro_AUPRC    -0.033436 -0.048077 -0.015657              0.95                      493                          500
   43 quantum_hybrid minus classical_bottleneck macro_AUPRC    -0.023976 -0.038368 -0.010123              0.95                      493                          500
   44 quantum_hybrid minus classical_bottleneck macro_AUPRC    -0.017721 -0.030294 -0.005066              0.95                      493                          500
RUN_TEST=False: no test features/predictions/metrics computed.
Outputs: workspace/unzippedarchive/unzippedarchive/xray_training_experiment_5_qml_pilot/run_20260929T155742_882354Z

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
EXPERIMENT 6 RESIDUAL QML | feature extraction: cuda
Torch: 2.14.0+cu126 | torchvision: 0.29.0+cu126 | PennyLane: 0.45.1
Seeds: [42, 43, 44] | RUN_TEST: False
Strict JPEG validation: 100%
 6723/6723 [01:42<00:00, 63.40it/s]
JPEG audit: {'intact': 6665, 'recoverable_truncated': 58}
train 4830 images; 1231 patients
val 970 images; 264 patients
test 923 images; 264 patients
Reusing features: train (4830, 1024)
Reusing features: val (970, 1024)
Feature cache: 22.7 MiB; loading/extraction 0.0s
Frozen DenseNet feature extraction: 100%
 1/1 [00:00<00:00,  2.93it/s]
Saved E2 checkpoint metrics: {'macro_AUROC': 0.7303071009041724, 'macro_AUPRC': 0.2930734458959053}
Reproduced FP32 E2 metrics: {'macro_AUROC': 0.7307837861407264, 'macro_AUPRC': 0.29433588632475105, 'macro_AUROC_50plus': 0.7711530780574897, 'macro_AUPRC_50plus': 0.399016943438275}
Learned residual model parameters: 25908
Circuit batch and gradient checks passed; head device: CPU, extraction device: cuda
Source epoch/hash: 5 9818603cbc002b3f1b038e09ca35175bf6f7610f699cbf7994eeab2a9a3efab7
E6 preserves the entire 1024-feature base and trains a residual correction. Test evaluation: False
linear_control seed=42 01/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.02s
linear_control seed=42 02/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.02s
linear_control seed=42 03/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.02s
linear_control seed=42 04/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.02s
linear_control seed=42 05/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.02s
linear_control seed=42 06/40 joint_head | AUROC 0.73251 AP 0.29493 | acc 0.8462 F1 0.5187 | residual RMS 0.000 | train 0.06s val 0.03s | best
linear_control seed=42 07/40 joint_head | AUROC 0.73374 AP 0.29587 | acc 0.8462 F1 0.5198 | residual RMS 0.000 | train 0.06s val 0.03s | best
linear_control seed=42 08/40 joint_head | AUROC 0.73527 AP 0.29635 | acc 0.8457 F1 0.5197 | residual RMS 0.000 | train 0.06s val 0.02s | best
linear_control seed=42 09/40 joint_head | AUROC 0.73630 AP 0.29806 | acc 0.8454 F1 0.5196 | residual RMS 0.000 | train 0.06s val 0.02s | best
linear_control seed=42 10/40 joint_head | AUROC 0.73717 AP 0.29845 | acc 0.8453 F1 0.5191 | residual RMS 0.000 | train 0.06s val 0.03s | best
linear_control seed=42 11/40 joint_head | AUROC 0.73800 AP 0.29854 | acc 0.8452 F1 0.5187 | residual RMS 0.000 | train 0.06s val 0.02s | best
linear_control seed=42 12/40 joint_head | AUROC 0.73900 AP 0.29871 | acc 0.8451 F1 0.5186 | residual RMS 0.000 | train 0.06s val 0.02s | best
linear_control seed=42 13/40 joint_head | AUROC 0.73939 AP 0.29892 | acc 0.8453 F1 0.5191 | residual RMS 0.000 | train 0.06s val 0.02s | best
linear_control seed=42 14/40 joint_head | AUROC 0.73993 AP 0.29895 | acc 0.8451 F1 0.5186 | residual RMS 0.000 | train 0.06s val 0.02s | best
linear_control seed=42 15/40 joint_head | AUROC 0.74036 AP 0.29892 | acc 0.8453 F1 0.5193 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 16/40 joint_head | AUROC 0.74076 AP 0.29847 | acc 0.8453 F1 0.5196 | residual RMS 0.000 | train 0.06s val 0.02s
linear_control seed=42 17/40 joint_head | AUROC 0.74116 AP 0.29885 | acc 0.8448 F1 0.5185 | residual RMS 0.000 | train 0.06s val 0.02s
linear_control seed=42 18/40 joint_head | AUROC 0.74146 AP 0.29929 | acc 0.8446 F1 0.5178 | residual RMS 0.000 | train 0.06s val 0.02s | best
linear_control seed=42 19/40 joint_head | AUROC 0.74194 AP 0.29956 | acc 0.8449 F1 0.5190 | residual RMS 0.000 | train 0.06s val 0.03s | best
linear_control seed=42 20/40 joint_head | AUROC 0.74219 AP 0.29986 | acc 0.8446 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.02s | best
linear_control seed=42 21/40 joint_head | AUROC 0.74250 AP 0.29951 | acc 0.8444 F1 0.5184 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 22/40 joint_head | AUROC 0.74270 AP 0.29958 | acc 0.8440 F1 0.5172 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 23/40 joint_head | AUROC 0.74272 AP 0.29955 | acc 0.8438 F1 0.5172 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 24/40 joint_head | AUROC 0.74300 AP 0.29963 | acc 0.8437 F1 0.5173 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 25/40 joint_head | AUROC 0.74294 AP 0.29987 | acc 0.8436 F1 0.5172 | residual RMS 0.000 | train 0.06s val 0.03s | best
linear_control seed=42 26/40 joint_head | AUROC 0.74300 AP 0.29998 | acc 0.8439 F1 0.5178 | residual RMS 0.000 | train 0.06s val 0.03s | best
linear_control seed=42 27/40 joint_head | AUROC 0.74299 AP 0.29985 | acc 0.8440 F1 0.5180 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 28/40 joint_head | AUROC 0.74328 AP 0.29990 | acc 0.8438 F1 0.5175 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 29/40 joint_head | AUROC 0.74332 AP 0.29994 | acc 0.8436 F1 0.5170 | residual RMS 0.000 | train 0.06s val 0.03s
linear_control seed=42 30/40 joint_head | AUROC 0.74331 AP 0.30014 | acc 0.8437 F1 0.5171 | residual RMS 0.000 | train 0.06s val 0.03s | best
linear_control seed=42 31/40 joint_head | AUROC 0.74333 AP 0.30020 | acc 0.8433 F1 0.5160 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=42 32/40 joint_head | AUROC 0.74337 AP 0.30016 | acc 0.8436 F1 0.5163 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=42 33/40 joint_head | AUROC 0.74339 AP 0.30023 | acc 0.8437 F1 0.5171 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=42 34/40 joint_head | AUROC 0.74330 AP 0.30019 | acc 0.8436 F1 0.5170 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=42 35/40 joint_head | AUROC 0.74332 AP 0.30017 | acc 0.8437 F1 0.5173 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=42 36/40 joint_head | AUROC 0.74337 AP 0.30024 | acc 0.8437 F1 0.5171 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=42 37/40 joint_head | AUROC 0.74326 AP 0.30038 | acc 0.8438 F1 0.5172 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=42 38/40 joint_head | AUROC 0.74325 AP 0.30036 | acc 0.8439 F1 0.5176 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=42 39/40 joint_head | AUROC 0.74331 AP 0.30036 | acc 0.8439 F1 0.5176 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=42 40/40 joint_head | AUROC 0.74330 AP 0.30032 | acc 0.8439 F1 0.5176 | residual RMS 0.000 | train 0.07s val 0.03s
classical_residual seed=42 01/40 warmup | AUROC 0.73727 AP 0.29596 | acc 0.8472 F1 0.5205 | residual RMS 0.079 | train 0.09s val 0.03s | best
classical_residual seed=42 02/40 warmup | AUROC 0.73988 AP 0.29754 | acc 0.8474 F1 0.5201 | residual RMS 0.167 | train 0.09s val 0.03s | best
classical_residual seed=42 03/40 warmup | AUROC 0.73891 AP 0.29660 | acc 0.8497 F1 0.5215 | residual RMS 0.247 | train 0.09s val 0.03s
classical_residual seed=42 04/40 warmup | AUROC 0.73695 AP 0.29540 | acc 0.8491 F1 0.5189 | residual RMS 0.311 | train 0.09s val 0.03s
classical_residual seed=42 05/40 warmup | AUROC 0.73563 AP 0.29293 | acc 0.8499 F1 0.5162 | residual RMS 0.384 | train 0.09s val 0.03s
classical_residual seed=42 06/40 joint_head | AUROC 0.73575 AP 0.29123 | acc 0.8476 F1 0.5160 | residual RMS 0.438 | train 0.09s val 0.03s
classical_residual seed=42 07/40 joint_head | AUROC 0.73553 AP 0.28919 | acc 0.8489 F1 0.5123 | residual RMS 0.483 | train 0.09s val 0.03s
classical_residual seed=42 08/40 joint_head | AUROC 0.73285 AP 0.28783 | acc 0.8486 F1 0.5154 | residual RMS 0.531 | train 0.09s val 0.03s
classical_residual seed=42 09/40 joint_head | AUROC 0.73304 AP 0.28710 | acc 0.8486 F1 0.5090 | residual RMS 0.576 | train 0.09s val 0.03s
classical_residual seed=42 10/40 joint_head | AUROC 0.72720 AP 0.28618 | acc 0.8461 F1 0.5107 | residual RMS 0.604 | train 0.09s val 0.03s
classical_residual seed=42 11/40 joint_head | AUROC 0.73230 AP 0.28658 | acc 0.8469 F1 0.5093 | residual RMS 0.634 | train 0.09s val 0.03s
classical_residual seed=42 12/40 joint_head | AUROC 0.73129 AP 0.28689 | acc 0.8483 F1 0.5084 | residual RMS 0.657 | train 0.09s val 0.03s
classical_residual seed=42 13/40 joint_head | AUROC 0.72894 AP 0.28505 | acc 0.8481 F1 0.5061 | residual RMS 0.674 | train 0.09s val 0.03s
classical_residual seed=42 14/40 joint_head | AUROC 0.72955 AP 0.28533 | acc 0.8459 F1 0.5056 | residual RMS 0.688 | train 0.09s val 0.03s
classical_residual seed=42 15/40 joint_head | AUROC 0.73010 AP 0.28629 | acc 0.8461 F1 0.5056 | residual RMS 0.707 | train 0.09s val 0.03s
frozen_quantum seed=42 01/40 warmup | AUROC 0.73144 AP 0.29396 | acc 0.8507 F1 0.5214 | residual RMS 0.101 | train 0.98s val 0.15s
frozen_quantum seed=42 02/40 warmup | AUROC 0.73259 AP 0.29477 | acc 0.8521 F1 0.5222 | residual RMS 0.170 | train 0.98s val 0.15s | best
frozen_quantum seed=42 03/40 warmup | AUROC 0.73346 AP 0.29372 | acc 0.8535 F1 0.5184 | residual RMS 0.261 | train 0.99s val 0.15s
frozen_quantum seed=42 04/40 warmup | AUROC 0.73149 AP 0.29326 | acc 0.8553 F1 0.5220 | residual RMS 0.318 | train 0.98s val 0.15s
frozen_quantum seed=42 05/40 warmup | AUROC 0.73548 AP 0.29211 | acc 0.8570 F1 0.5129 | residual RMS 0.374 | train 0.98s val 0.15s
frozen_quantum seed=42 06/40 joint_head | AUROC 0.72967 AP 0.28975 | acc 0.8530 F1 0.5153 | residual RMS 0.405 | train 0.99s val 0.15s
frozen_quantum seed=42 07/40 joint_head | AUROC 0.73396 AP 0.28976 | acc 0.8545 F1 0.5143 | residual RMS 0.452 | train 0.99s val 0.15s
frozen_quantum seed=42 08/40 joint_head | AUROC 0.73532 AP 0.28893 | acc 0.8532 F1 0.5139 | residual RMS 0.510 | train 0.99s val 0.15s
frozen_quantum seed=42 09/40 joint_head | AUROC 0.72762 AP 0.28639 | acc 0.8531 F1 0.5102 | residual RMS 0.551 | train 1.00s val 0.15s
frozen_quantum seed=42 10/40 joint_head | AUROC 0.72874 AP 0.28733 | acc 0.8494 F1 0.5077 | residual RMS 0.549 | train 0.99s val 0.15s
frozen_quantum seed=42 11/40 joint_head | AUROC 0.73341 AP 0.28514 | acc 0.8543 F1 0.5074 | residual RMS 0.649 | train 0.99s val 0.15s
frozen_quantum seed=42 12/40 joint_head | AUROC 0.73258 AP 0.28725 | acc 0.8548 F1 0.5061 | residual RMS 0.659 | train 0.99s val 0.15s
frozen_quantum seed=42 13/40 joint_head | AUROC 0.73242 AP 0.28505 | acc 0.8570 F1 0.5041 | residual RMS 0.700 | train 1.01s val 0.15s
frozen_quantum seed=42 14/40 joint_head | AUROC 0.73509 AP 0.28535 | acc 0.8528 F1 0.4986 | residual RMS 0.736 | train 0.99s val 0.15s
frozen_quantum seed=42 15/40 joint_head | AUROC 0.73475 AP 0.28698 | acc 0.8536 F1 0.5045 | residual RMS 0.742 | train 0.99s val 0.15s
quantum_residual seed=42 01/40 warmup | AUROC 0.73143 AP 0.29395 | acc 0.8507 F1 0.5214 | residual RMS 0.101 | train 1.15s val 0.15s
quantum_residual seed=42 02/40 warmup | AUROC 0.73266 AP 0.29476 | acc 0.8523 F1 0.5225 | residual RMS 0.171 | train 1.13s val 0.15s | best
quantum_residual seed=42 03/40 warmup | AUROC 0.73337 AP 0.29379 | acc 0.8537 F1 0.5194 | residual RMS 0.263 | train 1.12s val 0.15s
quantum_residual seed=42 04/40 warmup | AUROC 0.73147 AP 0.29337 | acc 0.8554 F1 0.5223 | residual RMS 0.320 | train 1.13s val 0.15s
quantum_residual seed=42 05/40 warmup | AUROC 0.73575 AP 0.29219 | acc 0.8565 F1 0.5126 | residual RMS 0.372 | train 1.14s val 0.15s
quantum_residual seed=42 06/40 joint_head | AUROC 0.72985 AP 0.28953 | acc 0.8521 F1 0.5130 | residual RMS 0.410 | train 1.14s val 0.15s
quantum_residual seed=42 07/40 joint_head | AUROC 0.73359 AP 0.28965 | acc 0.8545 F1 0.5160 | residual RMS 0.452 | train 1.15s val 0.15s
quantum_residual seed=42 08/40 joint_head | AUROC 0.73510 AP 0.28851 | acc 0.8533 F1 0.5148 | residual RMS 0.503 | train 1.15s val 0.15s
quantum_residual seed=42 09/40 joint_head | AUROC 0.72765 AP 0.28639 | acc 0.8533 F1 0.5084 | residual RMS 0.564 | train 1.14s val 0.15s
quantum_residual seed=42 10/40 joint_head | AUROC 0.72993 AP 0.28671 | acc 0.8489 F1 0.5077 | residual RMS 0.560 | train 1.15s val 0.15s
quantum_residual seed=42 11/40 joint_head | AUROC 0.73512 AP 0.28739 | acc 0.8538 F1 0.5039 | residual RMS 0.625 | train 1.14s val 0.15s
quantum_residual seed=42 12/40 joint_head | AUROC 0.73560 AP 0.28608 | acc 0.8560 F1 0.5079 | residual RMS 0.680 | train 1.14s val 0.15s
quantum_residual seed=42 13/40 joint_head | AUROC 0.73385 AP 0.28456 | acc 0.8569 F1 0.5056 | residual RMS 0.723 | train 1.14s val 0.15s
quantum_residual seed=42 14/40 joint_head | AUROC 0.73462 AP 0.28462 | acc 0.8530 F1 0.5003 | residual RMS 0.750 | train 1.15s val 0.15s
quantum_residual seed=42 15/40 joint_head | AUROC 0.73432 AP 0.28384 | acc 0.8532 F1 0.5003 | residual RMS 0.761 | train 1.13s val 0.15s
linear_control seed=43 01/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=43 02/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=43 03/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=43 04/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=43 05/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=43 06/40 joint_head | AUROC 0.73295 AP 0.29477 | acc 0.8461 F1 0.5182 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 07/40 joint_head | AUROC 0.73414 AP 0.29580 | acc 0.8461 F1 0.5194 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 08/40 joint_head | AUROC 0.73578 AP 0.29640 | acc 0.8455 F1 0.5190 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 09/40 joint_head | AUROC 0.73647 AP 0.29682 | acc 0.8451 F1 0.5186 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 10/40 joint_head | AUROC 0.73759 AP 0.29849 | acc 0.8450 F1 0.5184 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 11/40 joint_head | AUROC 0.73851 AP 0.29860 | acc 0.8453 F1 0.5191 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 12/40 joint_head | AUROC 0.73919 AP 0.29874 | acc 0.8451 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 13/40 joint_head | AUROC 0.73963 AP 0.29897 | acc 0.8453 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 14/40 joint_head | AUROC 0.73996 AP 0.29893 | acc 0.8452 F1 0.5189 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 15/40 joint_head | AUROC 0.74046 AP 0.29918 | acc 0.8454 F1 0.5201 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 16/40 joint_head | AUROC 0.74087 AP 0.29929 | acc 0.8450 F1 0.5192 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 17/40 joint_head | AUROC 0.74127 AP 0.29919 | acc 0.8449 F1 0.5187 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 18/40 joint_head | AUROC 0.74164 AP 0.29941 | acc 0.8448 F1 0.5185 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 19/40 joint_head | AUROC 0.74220 AP 0.29963 | acc 0.8446 F1 0.5185 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 20/40 joint_head | AUROC 0.74232 AP 0.29987 | acc 0.8444 F1 0.5184 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 21/40 joint_head | AUROC 0.74249 AP 0.29993 | acc 0.8445 F1 0.5185 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 22/40 joint_head | AUROC 0.74278 AP 0.30003 | acc 0.8441 F1 0.5176 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 23/40 joint_head | AUROC 0.74269 AP 0.30016 | acc 0.8441 F1 0.5176 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 24/40 joint_head | AUROC 0.74287 AP 0.29980 | acc 0.8439 F1 0.5178 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 25/40 joint_head | AUROC 0.74301 AP 0.29974 | acc 0.8438 F1 0.5175 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 26/40 joint_head | AUROC 0.74324 AP 0.30003 | acc 0.8437 F1 0.5171 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 27/40 joint_head | AUROC 0.74332 AP 0.30024 | acc 0.8440 F1 0.5177 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 28/40 joint_head | AUROC 0.74331 AP 0.30018 | acc 0.8436 F1 0.5170 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 29/40 joint_head | AUROC 0.74331 AP 0.30038 | acc 0.8436 F1 0.5167 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 30/40 joint_head | AUROC 0.74331 AP 0.30038 | acc 0.8438 F1 0.5172 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 31/40 joint_head | AUROC 0.74329 AP 0.30034 | acc 0.8441 F1 0.5176 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 32/40 joint_head | AUROC 0.74324 AP 0.30037 | acc 0.8442 F1 0.5181 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 33/40 joint_head | AUROC 0.74323 AP 0.30041 | acc 0.8445 F1 0.5185 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 34/40 joint_head | AUROC 0.74313 AP 0.30022 | acc 0.8449 F1 0.5194 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 35/40 joint_head | AUROC 0.74312 AP 0.30025 | acc 0.8448 F1 0.5193 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 36/40 joint_head | AUROC 0.74298 AP 0.30048 | acc 0.8445 F1 0.5180 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 37/40 joint_head | AUROC 0.74303 AP 0.30045 | acc 0.8446 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=43 38/40 joint_head | AUROC 0.74291 AP 0.30055 | acc 0.8446 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=43 39/40 joint_head | AUROC 0.74298 AP 0.30052 | acc 0.8445 F1 0.5186 | residual RMS 0.000 | train 0.07s val 0.03s
classical_residual seed=43 01/40 warmup | AUROC 0.73447 AP 0.29597 | acc 0.8469 F1 0.5195 | residual RMS 0.077 | train 0.09s val 0.03s | best
classical_residual seed=43 02/40 warmup | AUROC 0.73336 AP 0.29517 | acc 0.8485 F1 0.5229 | residual RMS 0.182 | train 0.09s val 0.03s
classical_residual seed=43 03/40 warmup | AUROC 0.73241 AP 0.29307 | acc 0.8489 F1 0.5233 | residual RMS 0.271 | train 0.09s val 0.03s
classical_residual seed=43 04/40 warmup | AUROC 0.73499 AP 0.29001 | acc 0.8505 F1 0.5227 | residual RMS 0.339 | train 0.09s val 0.03s
classical_residual seed=43 05/40 warmup | AUROC 0.73272 AP 0.28798 | acc 0.8469 F1 0.5185 | residual RMS 0.411 | train 0.09s val 0.03s
classical_residual seed=43 06/40 joint_head | AUROC 0.72680 AP 0.28968 | acc 0.8471 F1 0.5131 | residual RMS 0.455 | train 0.09s val 0.03s
classical_residual seed=43 07/40 joint_head | AUROC 0.72891 AP 0.28809 | acc 0.8465 F1 0.5137 | residual RMS 0.513 | train 0.10s val 0.03s
classical_residual seed=43 08/40 joint_head | AUROC 0.72702 AP 0.28629 | acc 0.8473 F1 0.5122 | residual RMS 0.558 | train 0.09s val 0.03s
classical_residual seed=43 09/40 joint_head | AUROC 0.72848 AP 0.28536 | acc 0.8419 F1 0.5031 | residual RMS 0.611 | train 0.10s val 0.03s
classical_residual seed=43 10/40 joint_head | AUROC 0.72898 AP 0.28498 | acc 0.8423 F1 0.5067 | residual RMS 0.661 | train 0.09s val 0.03s
classical_residual seed=43 11/40 joint_head | AUROC 0.72818 AP 0.28475 | acc 0.8433 F1 0.5027 | residual RMS 0.667 | train 0.10s val 0.03s
classical_residual seed=43 12/40 joint_head | AUROC 0.72540 AP 0.28426 | acc 0.8449 F1 0.5031 | residual RMS 0.680 | train 0.09s val 0.03s
classical_residual seed=43 13/40 joint_head | AUROC 0.72907 AP 0.28500 | acc 0.8449 F1 0.5025 | residual RMS 0.709 | train 0.10s val 0.03s
classical_residual seed=43 14/40 joint_head | AUROC 0.72746 AP 0.28532 | acc 0.8443 F1 0.5025 | residual RMS 0.728 | train 0.09s val 0.03s
classical_residual seed=43 15/40 joint_head | AUROC 0.72556 AP 0.28456 | acc 0.8443 F1 0.5012 | residual RMS 0.735 | train 0.10s val 0.03s
frozen_quantum seed=43 01/40 warmup | AUROC 0.73241 AP 0.29380 | acc 0.8504 F1 0.5201 | residual RMS 0.094 | train 0.99s val 0.15s
frozen_quantum seed=43 02/40 warmup | AUROC 0.73263 AP 0.29368 | acc 0.8527 F1 0.5211 | residual RMS 0.176 | train 0.99s val 0.15s
frozen_quantum seed=43 03/40 warmup | AUROC 0.73506 AP 0.29071 | acc 0.8534 F1 0.5188 | residual RMS 0.249 | train 0.99s val 0.15s
frozen_quantum seed=43 04/40 warmup | AUROC 0.73377 AP 0.28819 | acc 0.8545 F1 0.5158 | residual RMS 0.288 | train 0.99s val 0.15s
frozen_quantum seed=43 05/40 warmup | AUROC 0.73059 AP 0.28504 | acc 0.8542 F1 0.5147 | residual RMS 0.359 | train 0.99s val 0.15s
frozen_quantum seed=43 06/40 joint_head | AUROC 0.72648 AP 0.28764 | acc 0.8565 F1 0.5126 | residual RMS 0.399 | train 1.00s val 0.15s
frozen_quantum seed=43 07/40 joint_head | AUROC 0.72617 AP 0.28846 | acc 0.8570 F1 0.5134 | residual RMS 0.449 | train 1.00s val 0.15s
frozen_quantum seed=43 08/40 joint_head | AUROC 0.73040 AP 0.28591 | acc 0.8568 F1 0.5158 | residual RMS 0.494 | train 1.00s val 0.15s
frozen_quantum seed=43 09/40 joint_head | AUROC 0.73137 AP 0.28628 | acc 0.8485 F1 0.5099 | residual RMS 0.490 | train 1.00s val 0.15s
frozen_quantum seed=43 10/40 joint_head | AUROC 0.72928 AP 0.28605 | acc 0.8506 F1 0.5108 | residual RMS 0.503 | train 1.00s val 0.15s
frozen_quantum seed=43 11/40 joint_head | AUROC 0.72384 AP 0.28063 | acc 0.8522 F1 0.5054 | residual RMS 0.607 | train 1.00s val 0.15s
frozen_quantum seed=43 12/40 joint_head | AUROC 0.72964 AP 0.28415 | acc 0.8514 F1 0.5046 | residual RMS 0.629 | train 1.00s val 0.15s
frozen_quantum seed=43 13/40 joint_head | AUROC 0.73172 AP 0.28260 | acc 0.8534 F1 0.5051 | residual RMS 0.672 | train 1.00s val 0.15s
frozen_quantum seed=43 14/40 joint_head | AUROC 0.73060 AP 0.28508 | acc 0.8539 F1 0.5083 | residual RMS 0.696 | train 1.02s val 0.15s
frozen_quantum seed=43 15/40 joint_head | AUROC 0.73102 AP 0.28381 | acc 0.8526 F1 0.5023 | residual RMS 0.687 | train 1.00s val 0.15s
quantum_residual seed=43 01/40 warmup | AUROC 0.73242 AP 0.29378 | acc 0.8503 F1 0.5199 | residual RMS 0.094 | train 1.13s val 0.15s
quantum_residual seed=43 02/40 warmup | AUROC 0.73294 AP 0.29385 | acc 0.8528 F1 0.5212 | residual RMS 0.179 | train 1.13s val 0.15s
quantum_residual seed=43 03/40 warmup | AUROC 0.73510 AP 0.28994 | acc 0.8535 F1 0.5190 | residual RMS 0.252 | train 1.14s val 0.15s
quantum_residual seed=43 04/40 warmup | AUROC 0.73462 AP 0.28872 | acc 0.8545 F1 0.5160 | residual RMS 0.294 | train 1.13s val 0.15s
quantum_residual seed=43 05/40 warmup | AUROC 0.73073 AP 0.28617 | acc 0.8540 F1 0.5138 | residual RMS 0.363 | train 1.13s val 0.15s
quantum_residual seed=43 06/40 joint_head | AUROC 0.72603 AP 0.28820 | acc 0.8565 F1 0.5118 | residual RMS 0.404 | train 1.14s val 0.15s
quantum_residual seed=43 07/40 joint_head | AUROC 0.72696 AP 0.28585 | acc 0.8563 F1 0.5163 | residual RMS 0.456 | train 1.14s val 0.15s
quantum_residual seed=43 08/40 joint_head | AUROC 0.73071 AP 0.28670 | acc 0.8572 F1 0.5155 | residual RMS 0.492 | train 1.14s val 0.15s
quantum_residual seed=43 09/40 joint_head | AUROC 0.73225 AP 0.28486 | acc 0.8488 F1 0.5121 | residual RMS 0.500 | train 1.14s val 0.15s
quantum_residual seed=43 10/40 joint_head | AUROC 0.73012 AP 0.28535 | acc 0.8509 F1 0.5109 | residual RMS 0.510 | train 1.14s val 0.15s
quantum_residual seed=43 11/40 joint_head | AUROC 0.72914 AP 0.28413 | acc 0.8528 F1 0.5037 | residual RMS 0.623 | train 1.14s val 0.15s
quantum_residual seed=43 12/40 joint_head | AUROC 0.73223 AP 0.28388 | acc 0.8520 F1 0.5070 | residual RMS 0.632 | train 1.14s val 0.15s
quantum_residual seed=43 13/40 joint_head | AUROC 0.73306 AP 0.28303 | acc 0.8544 F1 0.5076 | residual RMS 0.665 | train 1.14s val 0.15s
quantum_residual seed=43 14/40 joint_head | AUROC 0.73125 AP 0.28604 | acc 0.8529 F1 0.5048 | residual RMS 0.691 | train 1.14s val 0.15s
quantum_residual seed=43 15/40 joint_head | AUROC 0.73191 AP 0.28471 | acc 0.8521 F1 0.5001 | residual RMS 0.699 | train 1.14s val 0.15s
linear_control seed=44 01/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=44 02/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=44 03/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=44 04/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=44 05/40 warmup | AUROC 0.73078 AP 0.29434 | acc 0.8465 F1 0.5184 | residual RMS 0.000 | train 0.00s val 0.03s
linear_control seed=44 06/40 joint_head | AUROC 0.73261 AP 0.29513 | acc 0.8463 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 07/40 joint_head | AUROC 0.73401 AP 0.29581 | acc 0.8460 F1 0.5193 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 08/40 joint_head | AUROC 0.73535 AP 0.29614 | acc 0.8456 F1 0.5194 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 09/40 joint_head | AUROC 0.73640 AP 0.29801 | acc 0.8451 F1 0.5186 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 10/40 joint_head | AUROC 0.73752 AP 0.29843 | acc 0.8453 F1 0.5191 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 11/40 joint_head | AUROC 0.73817 AP 0.29861 | acc 0.8451 F1 0.5183 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 12/40 joint_head | AUROC 0.73884 AP 0.29866 | acc 0.8451 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 13/40 joint_head | AUROC 0.73935 AP 0.29884 | acc 0.8454 F1 0.5191 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 14/40 joint_head | AUROC 0.73988 AP 0.29890 | acc 0.8453 F1 0.5188 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 15/40 joint_head | AUROC 0.74034 AP 0.29897 | acc 0.8456 F1 0.5196 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 16/40 joint_head | AUROC 0.74074 AP 0.29848 | acc 0.8451 F1 0.5191 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 17/40 joint_head | AUROC 0.74125 AP 0.29875 | acc 0.8448 F1 0.5183 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 18/40 joint_head | AUROC 0.74137 AP 0.29940 | acc 0.8445 F1 0.5170 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 19/40 joint_head | AUROC 0.74174 AP 0.29964 | acc 0.8448 F1 0.5185 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 20/40 joint_head | AUROC 0.74205 AP 0.29981 | acc 0.8445 F1 0.5186 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 21/40 joint_head | AUROC 0.74244 AP 0.29988 | acc 0.8443 F1 0.5178 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 22/40 joint_head | AUROC 0.74260 AP 0.29982 | acc 0.8438 F1 0.5165 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 23/40 joint_head | AUROC 0.74254 AP 0.29987 | acc 0.8438 F1 0.5172 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 24/40 joint_head | AUROC 0.74292 AP 0.30006 | acc 0.8438 F1 0.5175 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 25/40 joint_head | AUROC 0.74284 AP 0.29991 | acc 0.8439 F1 0.5176 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 26/40 joint_head | AUROC 0.74299 AP 0.29988 | acc 0.8437 F1 0.5171 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 27/40 joint_head | AUROC 0.74338 AP 0.30009 | acc 0.8436 F1 0.5167 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 28/40 joint_head | AUROC 0.74337 AP 0.30035 | acc 0.8436 F1 0.5166 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 29/40 joint_head | AUROC 0.74314 AP 0.30027 | acc 0.8437 F1 0.5171 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 30/40 joint_head | AUROC 0.74330 AP 0.30035 | acc 0.8436 F1 0.5167 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 31/40 joint_head | AUROC 0.74321 AP 0.30038 | acc 0.8441 F1 0.5177 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 32/40 joint_head | AUROC 0.74324 AP 0.30028 | acc 0.8444 F1 0.5186 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 33/40 joint_head | AUROC 0.74325 AP 0.30040 | acc 0.8445 F1 0.5189 | residual RMS 0.000 | train 0.07s val 0.03s | best
linear_control seed=44 34/40 joint_head | AUROC 0.74323 AP 0.30037 | acc 0.8445 F1 0.5189 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 35/40 joint_head | AUROC 0.74318 AP 0.30027 | acc 0.8445 F1 0.5189 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 36/40 joint_head | AUROC 0.74311 AP 0.30031 | acc 0.8445 F1 0.5187 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 37/40 joint_head | AUROC 0.74306 AP 0.30037 | acc 0.8447 F1 0.5194 | residual RMS 0.000 | train 0.07s val 0.03s
linear_control seed=44 38/40 joint_head | AUROC 0.74299 AP 0.30034 | acc 0.8446 F1 0.5192 | residual RMS 0.000 | train 0.07s val 0.03s
classical_residual seed=44 01/40 warmup | AUROC 0.73310 AP 0.29539 | acc 0.8468 F1 0.5196 | residual RMS 0.093 | train 0.09s val 0.03s | best
classical_residual seed=44 02/40 warmup | AUROC 0.73155 AP 0.29282 | acc 0.8486 F1 0.5213 | residual RMS 0.199 | train 0.09s val 0.03s
classical_residual seed=44 03/40 warmup | AUROC 0.73058 AP 0.29236 | acc 0.8502 F1 0.5224 | residual RMS 0.287 | train 0.09s val 0.03s
classical_residual seed=44 04/40 warmup | AUROC 0.72915 AP 0.29356 | acc 0.8504 F1 0.5223 | residual RMS 0.356 | train 0.09s val 0.03s
classical_residual seed=44 05/40 warmup | AUROC 0.73044 AP 0.28935 | acc 0.8502 F1 0.5214 | residual RMS 0.413 | train 0.09s val 0.03s
classical_residual seed=44 06/40 joint_head | AUROC 0.72677 AP 0.28720 | acc 0.8462 F1 0.5131 | residual RMS 0.472 | train 0.10s val 0.03s
classical_residual seed=44 07/40 joint_head | AUROC 0.72940 AP 0.28887 | acc 0.8463 F1 0.5119 | residual RMS 0.514 | train 0.09s val 0.03s
classical_residual seed=44 08/40 joint_head | AUROC 0.72676 AP 0.28603 | acc 0.8460 F1 0.5083 | residual RMS 0.561 | train 0.09s val 0.03s
classical_residual seed=44 09/40 joint_head | AUROC 0.72878 AP 0.28470 | acc 0.8451 F1 0.5040 | residual RMS 0.590 | train 0.10s val 0.03s
classical_residual seed=44 10/40 joint_head | AUROC 0.73111 AP 0.28519 | acc 0.8446 F1 0.5096 | residual RMS 0.659 | train 0.09s val 0.03s
classical_residual seed=44 11/40 joint_head | AUROC 0.72547 AP 0.28414 | acc 0.8419 F1 0.5019 | residual RMS 0.693 | train 0.09s val 0.03s
classical_residual seed=44 12/40 joint_head | AUROC 0.72672 AP 0.28315 | acc 0.8437 F1 0.5008 | residual RMS 0.700 | train 0.09s val 0.03s
classical_residual seed=44 13/40 joint_head | AUROC 0.72695 AP 0.28411 | acc 0.8443 F1 0.5017 | residual RMS 0.713 | train 0.10s val 0.03s
classical_residual seed=44 14/40 joint_head | AUROC 0.72861 AP 0.28227 | acc 0.8445 F1 0.5001 | residual RMS 0.734 | train 0.09s val 0.03s
classical_residual seed=44 15/40 joint_head | AUROC 0.72695 AP 0.28369 | acc 0.8437 F1 0.5021 | residual RMS 0.756 | train 0.10s val 0.03s
frozen_quantum seed=44 01/40 warmup | AUROC 0.73094 AP 0.29287 | acc 0.8513 F1 0.5213 | residual RMS 0.108 | train 0.99s val 0.15s
frozen_quantum seed=44 02/40 warmup | AUROC 0.73090 AP 0.29195 | acc 0.8525 F1 0.5206 | residual RMS 0.201 | train 1.00s val 0.15s
frozen_quantum seed=44 03/40 warmup | AUROC 0.72468 AP 0.29114 | acc 0.8544 F1 0.5184 | residual RMS 0.267 | train 0.99s val 0.15s
frozen_quantum seed=44 04/40 warmup | AUROC 0.72784 AP 0.29012 | acc 0.8523 F1 0.5256 | residual RMS 0.309 | train 0.99s val 0.15s
frozen_quantum seed=44 05/40 warmup | AUROC 0.72892 AP 0.28628 | acc 0.8549 F1 0.5176 | residual RMS 0.365 | train 0.99s val 0.15s
frozen_quantum seed=44 06/40 joint_head | AUROC 0.72984 AP 0.29084 | acc 0.8546 F1 0.5217 | residual RMS 0.449 | train 0.99s val 0.15s
frozen_quantum seed=44 07/40 joint_head | AUROC 0.72525 AP 0.28834 | acc 0.8506 F1 0.5211 | residual RMS 0.455 | train 1.00s val 0.15s
frozen_quantum seed=44 08/40 joint_head | AUROC 0.72592 AP 0.28869 | acc 0.8523 F1 0.5179 | residual RMS 0.522 | train 1.00s val 0.15s
frozen_quantum seed=44 09/40 joint_head | AUROC 0.73387 AP 0.28654 | acc 0.8524 F1 0.5141 | residual RMS 0.569 | train 0.99s val 0.15s
frozen_quantum seed=44 10/40 joint_head | AUROC 0.73009 AP 0.28904 | acc 0.8556 F1 0.5161 | residual RMS 0.597 | train 1.00s val 0.15s
frozen_quantum seed=44 11/40 joint_head | AUROC 0.72588 AP 0.29163 | acc 0.8536 F1 0.5139 | residual RMS 0.638 | train 1.00s val 0.15s
frozen_quantum seed=44 12/40 joint_head | AUROC 0.72652 AP 0.28866 | acc 0.8510 F1 0.5067 | residual RMS 0.679 | train 1.00s val 0.15s
frozen_quantum seed=44 13/40 joint_head | AUROC 0.72803 AP 0.28916 | acc 0.8553 F1 0.5113 | residual RMS 0.699 | train 1.00s val 0.15s
frozen_quantum seed=44 14/40 joint_head | AUROC 0.73083 AP 0.28818 | acc 0.8547 F1 0.5108 | residual RMS 0.721 | train 1.00s val 0.15s
frozen_quantum seed=44 15/40 joint_head | AUROC 0.72783 AP 0.29081 | acc 0.8498 F1 0.5081 | residual RMS 0.747 | train 1.00s val 0.15s
quantum_residual seed=44 01/40 warmup | AUROC 0.73094 AP 0.29284 | acc 0.8513 F1 0.5213 | residual RMS 0.108 | train 1.13s val 0.15s
quantum_residual seed=44 02/40 warmup | AUROC 0.73099 AP 0.29204 | acc 0.8527 F1 0.5214 | residual RMS 0.201 | train 1.14s val 0.15s
quantum_residual seed=44 03/40 warmup | AUROC 0.72459 AP 0.29109 | acc 0.8547 F1 0.5198 | residual RMS 0.267 | train 1.14s val 0.15s
quantum_residual seed=44 04/40 warmup | AUROC 0.72794 AP 0.29179 | acc 0.8528 F1 0.5272 | residual RMS 0.305 | train 1.13s val 0.15s
quantum_residual seed=44 05/40 warmup | AUROC 0.72823 AP 0.28662 | acc 0.8557 F1 0.5196 | residual RMS 0.361 | train 1.13s val 0.15s
quantum_residual seed=44 06/40 joint_head | AUROC 0.72837 AP 0.28886 | acc 0.8546 F1 0.5214 | residual RMS 0.447 | train 1.14s val 0.15s
quantum_residual seed=44 07/40 joint_head | AUROC 0.72605 AP 0.28755 | acc 0.8494 F1 0.5195 | residual RMS 0.466 | train 1.14s val 0.15s
quantum_residual seed=44 08/40 joint_head | AUROC 0.72769 AP 0.28890 | acc 0.8506 F1 0.5162 | residual RMS 0.515 | train 1.14s val 0.15s
quantum_residual seed=44 09/40 joint_head | AUROC 0.73242 AP 0.28314 | acc 0.8529 F1 0.5138 | residual RMS 0.567 | train 1.14s val 0.15s
quantum_residual seed=44 10/40 joint_head | AUROC 0.72924 AP 0.28522 | acc 0.8541 F1 0.5108 | residual RMS 0.601 | train 1.14s val 0.15s
quantum_residual seed=44 11/40 joint_head | AUROC 0.71907 AP 0.28696 | acc 0.8515 F1 0.5145 | residual RMS 0.613 | train 1.14s val 0.15s
quantum_residual seed=44 12/40 joint_head | AUROC 0.73008 AP 0.28835 | acc 0.8495 F1 0.5057 | residual RMS 0.680 | train 1.14s val 0.15s
quantum_residual seed=44 13/40 joint_head | AUROC 0.73036 AP 0.29060 | acc 0.8549 F1 0.5122 | residual RMS 0.726 | train 1.14s val 0.15s
quantum_residual seed=44 14/40 joint_head | AUROC 0.73325 AP 0.29088 | acc 0.8536 F1 0.5083 | residual RMS 0.740 | train 1.14s val 0.15s
quantum_residual seed=44 15/40 joint_head | AUROC 0.72857 AP 0.28996 | acc 0.8490 F1 0.5083 | residual RMS 0.764 | train 1.14s val 0.15s

CHECKPOINT AND ENSEMBLE COMPARISON:
                        arm seed  checkpoint_epoch  validation_macro_AUROC  validation_macro_AUPRC
         experiment_2_saved None               5.0                0.730307                0.293073
    experiment_2_reproduced None               5.0                0.730784                0.294336
             linear_control   42              37.0                0.743262                0.300378
         classical_residual   42               2.0                0.739878                0.297540
             frozen_quantum   42               2.0                0.732588                0.294767
           quantum_residual   42               2.0                0.732664                0.294761
             linear_control   43              38.0                0.742906                0.300553
         classical_residual   43               1.0                0.734474                0.295970
             frozen_quantum   43               0.0                0.730784                0.294336
           quantum_residual   43               0.0                0.730784                0.294336
             linear_control   44              33.0                0.743250                0.300399
         classical_residual   44               1.0                0.733098                0.295389
             frozen_quantum   44               0.0                0.730784                0.294336
           quantum_residual   44               0.0                0.730784                0.294336
    linear_control_ensemble  all               NaN                0.743133                0.300150
classical_residual_ensemble  all               NaN                0.736058                0.296362
    frozen_quantum_ensemble  all               NaN                0.731501                0.294311
  quantum_residual_ensemble  all               NaN                0.731536                0.294317

QUANTUM CONTRIBUTION DIAGNOSTIC (positive favors intact hybrid):
seed                                         comparison      metric  point_delta  ci_lower  ci_upper  confidence_level  bootstrap_samples_requested  bootstrap_samples_valid
  42          quantum_residual minus classical_residual macro_AUPRC    -0.002779 -0.007229  0.001143              0.95                          500                      493
  42              quantum_residual minus frozen_quantum macro_AUPRC    -0.000006 -0.000320  0.000304              0.95                          500                      493
  42              quantum_residual minus branch_removed macro_AUPRC     0.000425 -0.002068  0.003590              0.95                          500                      493
  43          quantum_residual minus classical_residual macro_AUPRC    -0.001634 -0.005631  0.001200              0.95                          500                      493
  43              quantum_residual minus frozen_quantum macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
  43              quantum_residual minus branch_removed macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
  44          quantum_residual minus classical_residual macro_AUPRC    -0.001053 -0.006117  0.002198              0.95                          500                      493
  44              quantum_residual minus frozen_quantum macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
  44              quantum_residual minus branch_removed macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
 all quantum_ensemble minus classical_residual_ensemble macro_AUPRC    -0.002045 -0.006799  0.001253              0.95                          500                      493
 all     quantum_ensemble minus frozen_quantum_ensemble macro_AUPRC     0.000006 -0.000093  0.000129              0.95                          500                      493
Test untouched: no test features, predictions or metrics computed.
Saved to: /workspace/unzippedarchive/unzippedarchive/xray_training_experiment_6_residual_qml/run_20260929T164056_477274Z

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
EXPERIMENT 7 | extraction: cuda | heads: CPU | seeds: [42, 43, 44]
Torch: 2.14.0+cu126 | torchvision: 0.29.0+cu126 | PennyLane: 0.45.1
PASS pca_classical: forward, backward, frozen base, checkpoint round trip
PASS pca_quantum: forward, backward, frozen base, checkpoint round trip
PASS task_classical: forward, backward, frozen base, checkpoint round trip
PASS task_quantum: forward, backward, frozen base, checkpoint round trip
PASS task_frozen: forward, backward, frozen base, checkpoint round trip
PASS task_no_entanglement: forward, backward, frozen base, checkpoint round trip
PASS batched circuit, all circuit gradients, identical encoders, and zero-residual initialization
Strict JPEG validation: 100%
 6723/6723 [01:44<00:00, 61.61it/s]
JPEG audit: {'intact': 6665, 'recoverable_truncated': 58}
train 4830 images; 1231 patients
val 970 images; 264 patients
test 923 images; 264 patients
Reusing features: train (4830, 1024)
Reusing features: val (970, 1024)
Feature cache: 22.7 MiB; loading/extraction 0.0s
Frozen DenseNet feature extraction: 100%
 1/1 [00:00<00:00,  2.89it/s]
Saved E2 checkpoint metrics: {'macro_AUROC': 0.7303071009041724, 'macro_AUPRC': 0.2930734458959053}
Reproduced FP32 E2 metrics: {'macro_AUROC': 0.7307837861407264, 'macro_AUPRC': 0.29433588632475105, 'macro_AUROC_50plus': 0.7711530780574897, 'macro_AUPRC_50plus': 0.399016943438275}
PCA variance retained by 48 PCs: 0.7503564357757568
All residual arms start from the same selected linear anchor per seed.
linear_control seed=42 01/40 | AUROC 0.73231 AP 0.29483 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 02/40 | AUROC 0.73372 AP 0.29569 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 03/40 | AUROC 0.73527 AP 0.29644 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 04/40 | AUROC 0.73630 AP 0.29676 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 05/40 | AUROC 0.73723 AP 0.29838 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 06/40 | AUROC 0.73806 AP 0.29852 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 07/40 | AUROC 0.73851 AP 0.29874 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 08/40 | AUROC 0.73947 AP 0.29895 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 09/40 | AUROC 0.73990 AP 0.29886 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 10/40 | AUROC 0.74034 AP 0.29911 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 11/40 | AUROC 0.74062 AP 0.29937 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 12/40 | AUROC 0.74112 AP 0.29913 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 13/40 | AUROC 0.74130 AP 0.29933 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 14/40 | AUROC 0.74185 AP 0.29954 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 15/40 | AUROC 0.74216 AP 0.29975 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 16/40 | AUROC 0.74237 AP 0.29986 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 17/40 | AUROC 0.74257 AP 0.29989 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 18/40 | AUROC 0.74265 AP 0.29937 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 19/40 | AUROC 0.74291 AP 0.29948 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 20/40 | AUROC 0.74286 AP 0.29996 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 21/40 | AUROC 0.74301 AP 0.30003 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 22/40 | AUROC 0.74331 AP 0.30019 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 23/40 | AUROC 0.74320 AP 0.30037 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 24/40 | AUROC 0.74329 AP 0.30056 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 25/40 | AUROC 0.74314 AP 0.30061 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 26/40 | AUROC 0.74311 AP 0.30061 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 27/40 | AUROC 0.74307 AP 0.30057 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 28/40 | AUROC 0.74313 AP 0.30048 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 29/40 | AUROC 0.74325 AP 0.30065 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 30/40 | AUROC 0.74310 AP 0.30071 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 31/40 | AUROC 0.74308 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 32/40 | AUROC 0.74302 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 33/40 | AUROC 0.74314 AP 0.30073 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 34/40 | AUROC 0.74310 AP 0.30067 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 35/40 | AUROC 0.74307 AP 0.30072 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 36/40 | AUROC 0.74313 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 37/40 | AUROC 0.74297 AP 0.30084 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 38/40 | AUROC 0.74302 AP 0.30089 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 39/40 | AUROC 0.74303 AP 0.30097 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 40/40 | AUROC 0.74300 AP 0.30103 | residual 0.000 | core grad 0 | 0.1s | best
pca_classical seed=42 01/50 | AUROC 0.74289 AP 0.30115 | residual 0.013 | core grad 0 | 0.1s | best
pca_classical seed=42 02/50 | AUROC 0.74299 AP 0.30162 | residual 0.023 | core grad 0 | 0.1s | best
pca_classical seed=42 03/50 | AUROC 0.74293 AP 0.30159 | residual 0.032 | core grad 0 | 0.1s
pca_classical seed=42 04/50 | AUROC 0.74220 AP 0.30151 | residual 0.175 | core grad 0.0282 | 0.1s
pca_classical seed=42 05/50 | AUROC 0.74058 AP 0.30115 | residual 0.156 | core grad 0.043 | 0.1s
pca_classical seed=42 06/50 | AUROC 0.73911 AP 0.30087 | residual 0.218 | core grad 0.0649 | 0.1s
pca_classical seed=42 07/50 | AUROC 0.73795 AP 0.30001 | residual 0.270 | core grad 0.062 | 0.1s
pca_classical seed=42 08/50 | AUROC 0.73637 AP 0.29865 | residual 0.288 | core grad 0.084 | 0.1s
pca_classical seed=42 09/50 | AUROC 0.73627 AP 0.29859 | residual 0.323 | core grad 0.0915 | 0.1s
pca_classical seed=42 10/50 | AUROC 0.73561 AP 0.29862 | residual 0.321 | core grad 0.106 | 0.1s
pca_classical seed=42 11/50 | AUROC 0.73519 AP 0.29863 | residual 0.325 | core grad 0.078 | 0.1s
pca_classical seed=42 12/50 | AUROC 0.73515 AP 0.29869 | residual 0.335 | core grad 0.0894 | 0.1s
pca_classical seed=42 13/50 | AUROC 0.73531 AP 0.29894 | residual 0.360 | core grad 0.0851 | 0.1s
pca_classical seed=42 14/50 | AUROC 0.73515 AP 0.29933 | residual 0.353 | core grad 0.0948 | 0.1s
pca_classical seed=42 15/50 | AUROC 0.73475 AP 0.29925 | residual 0.350 | core grad 0.0961 | 0.1s
pca_quantum seed=42 01/50 | AUROC 0.74295 AP 0.30113 | residual 0.009 | core grad 0 | 2.3s | best
pca_quantum seed=42 02/50 | AUROC 0.74293 AP 0.30132 | residual 0.016 | core grad 0 | 2.3s | best
pca_quantum seed=42 03/50 | AUROC 0.74269 AP 0.30102 | residual 0.023 | core grad 0 | 2.2s
pca_quantum seed=42 04/50 | AUROC 0.74249 AP 0.30089 | residual 0.029 | core grad 0.00182 | 4.8s
pca_quantum seed=42 05/50 | AUROC 0.74250 AP 0.30081 | residual 0.036 | core grad 0.00183 | 4.8s
pca_quantum seed=42 06/50 | AUROC 0.74239 AP 0.30080 | residual 0.043 | core grad 0.00264 | 4.8s
pca_quantum seed=42 07/50 | AUROC 0.74243 AP 0.30054 | residual 0.049 | core grad 0.00245 | 4.8s
pca_quantum seed=42 08/50 | AUROC 0.74228 AP 0.30051 | residual 0.055 | core grad 0.00312 | 4.8s
pca_quantum seed=42 09/50 | AUROC 0.74210 AP 0.30036 | residual 0.060 | core grad 0.00297 | 4.8s
pca_quantum seed=42 10/50 | AUROC 0.74193 AP 0.30034 | residual 0.063 | core grad 0.0038 | 4.8s
pca_quantum seed=42 11/50 | AUROC 0.74195 AP 0.30031 | residual 0.066 | core grad 0.00344 | 4.8s
pca_quantum seed=42 12/50 | AUROC 0.74188 AP 0.30026 | residual 0.069 | core grad 0.00355 | 4.8s
pca_quantum seed=42 13/50 | AUROC 0.74179 AP 0.30034 | residual 0.072 | core grad 0.00374 | 5.0s
pca_quantum seed=42 14/50 | AUROC 0.74175 AP 0.30010 | residual 0.075 | core grad 0.00419 | 4.8s
pca_quantum seed=42 15/50 | AUROC 0.74174 AP 0.30046 | residual 0.076 | core grad 0.0043 | 5.0s
task_classical seed=42 01/50 | AUROC 0.74253 AP 0.30054 | residual 0.014 | core grad 0 | 0.1s
task_classical seed=42 02/50 | AUROC 0.74189 AP 0.30035 | residual 0.026 | core grad 0 | 0.1s
task_classical seed=42 03/50 | AUROC 0.74145 AP 0.30004 | residual 0.037 | core grad 0 | 0.1s
task_classical seed=42 04/50 | AUROC 0.73922 AP 0.29903 | residual 0.198 | core grad 0.0334 | 0.1s
task_classical seed=42 05/50 | AUROC 0.73586 AP 0.29586 | residual 0.226 | core grad 0.0539 | 0.1s
task_classical seed=42 06/50 | AUROC 0.73723 AP 0.29577 | residual 0.275 | core grad 0.0769 | 0.1s
task_classical seed=42 07/50 | AUROC 0.73713 AP 0.29748 | residual 0.303 | core grad 0.0672 | 0.1s
task_classical seed=42 08/50 | AUROC 0.73773 AP 0.29799 | residual 0.319 | core grad 0.0946 | 0.1s
task_classical seed=42 09/50 | AUROC 0.73828 AP 0.29834 | residual 0.337 | core grad 0.0888 | 0.1s
task_classical seed=42 10/50 | AUROC 0.73794 AP 0.29924 | residual 0.345 | core grad 0.0962 | 0.1s
task_classical seed=42 11/50 | AUROC 0.73783 AP 0.29893 | residual 0.341 | core grad 0.0778 | 0.1s
task_classical seed=42 12/50 | AUROC 0.73798 AP 0.29943 | residual 0.351 | core grad 0.0864 | 0.1s
task_classical seed=42 13/50 | AUROC 0.73808 AP 0.29960 | residual 0.367 | core grad 0.088 | 0.1s
task_classical seed=42 14/50 | AUROC 0.73817 AP 0.29977 | residual 0.364 | core grad 0.09 | 0.1s
task_classical seed=42 15/50 | AUROC 0.73835 AP 0.30022 | residual 0.370 | core grad 0.0847 | 0.1s
task_quantum seed=42 01/50 | AUROC 0.74257 AP 0.30102 | residual 0.011 | core grad 0 | 2.3s
task_quantum seed=42 02/50 | AUROC 0.74214 AP 0.30069 | residual 0.021 | core grad 0 | 2.3s
task_quantum seed=42 03/50 | AUROC 0.74202 AP 0.30082 | residual 0.029 | core grad 0 | 2.3s
task_quantum seed=42 04/50 | AUROC 0.74174 AP 0.30068 | residual 0.037 | core grad 0.0021 | 5.0s
task_quantum seed=42 05/50 | AUROC 0.74155 AP 0.30075 | residual 0.045 | core grad 0.00234 | 5.1s
task_quantum seed=42 06/50 | AUROC 0.74126 AP 0.30052 | residual 0.053 | core grad 0.00291 | 5.1s
task_quantum seed=42 07/50 | AUROC 0.74093 AP 0.30043 | residual 0.060 | core grad 0.00326 | 5.1s
task_quantum seed=42 08/50 | AUROC 0.74064 AP 0.30020 | residual 0.067 | core grad 0.00347 | 5.1s
task_quantum seed=42 09/50 | AUROC 0.74072 AP 0.30027 | residual 0.073 | core grad 0.00369 | 5.1s
task_quantum seed=42 10/50 | AUROC 0.74082 AP 0.30019 | residual 0.080 | core grad 0.0044 | 5.2s
task_quantum seed=42 11/50 | AUROC 0.74067 AP 0.30024 | residual 0.084 | core grad 0.00406 | 5.2s
task_quantum seed=42 12/50 | AUROC 0.74075 AP 0.30012 | residual 0.087 | core grad 0.00473 | 5.2s
task_quantum seed=42 13/50 | AUROC 0.74070 AP 0.29997 | residual 0.090 | core grad 0.00461 | 5.4s
task_quantum seed=42 14/50 | AUROC 0.74065 AP 0.29994 | residual 0.093 | core grad 0.00547 | 5.2s
task_quantum seed=42 15/50 | AUROC 0.74069 AP 0.29969 | residual 0.097 | core grad 0.00474 | 5.2s
task_frozen seed=42 01/50 | AUROC 0.74257 AP 0.30102 | residual 0.011 | core grad 0 | 2.4s
task_frozen seed=42 02/50 | AUROC 0.74214 AP 0.30069 | residual 0.021 | core grad 0 | 2.4s
task_frozen seed=42 03/50 | AUROC 0.74202 AP 0.30082 | residual 0.029 | core grad 0 | 2.4s
task_frozen seed=42 04/50 | AUROC 0.74171 AP 0.30073 | residual 0.037 | core grad 0 | 4.4s
task_frozen seed=42 05/50 | AUROC 0.74139 AP 0.30071 | residual 0.045 | core grad 0 | 4.4s
task_frozen seed=42 06/50 | AUROC 0.74113 AP 0.30045 | residual 0.052 | core grad 0 | 4.4s
task_frozen seed=42 07/50 | AUROC 0.74076 AP 0.30067 | residual 0.059 | core grad 0 | 4.4s
task_frozen seed=42 08/50 | AUROC 0.74036 AP 0.30021 | residual 0.066 | core grad 0 | 4.4s
task_frozen seed=42 09/50 | AUROC 0.74027 AP 0.30021 | residual 0.072 | core grad 0 | 4.4s
task_frozen seed=42 10/50 | AUROC 0.74025 AP 0.29996 | residual 0.075 | core grad 0 | 4.4s
task_frozen seed=42 11/50 | AUROC 0.73999 AP 0.30006 | residual 0.078 | core grad 0 | 4.4s
task_frozen seed=42 12/50 | AUROC 0.74010 AP 0.30014 | residual 0.081 | core grad 0 | 4.4s
task_frozen seed=42 13/50 | AUROC 0.74002 AP 0.30008 | residual 0.084 | core grad 0 | 4.6s
task_frozen seed=42 14/50 | AUROC 0.73992 AP 0.30007 | residual 0.086 | core grad 0 | 4.4s
task_frozen seed=42 15/50 | AUROC 0.73986 AP 0.30008 | residual 0.088 | core grad 0 | 4.4s
task_no_entanglement seed=42 01/50 | AUROC 0.74268 AP 0.30100 | residual 0.011 | core grad 0 | 2.0s
task_no_entanglement seed=42 02/50 | AUROC 0.74245 AP 0.30092 | residual 0.020 | core grad 0 | 2.0s
task_no_entanglement seed=42 03/50 | AUROC 0.74228 AP 0.30078 | residual 0.029 | core grad 0 | 2.0s
task_no_entanglement seed=42 04/50 | AUROC 0.74209 AP 0.30048 | residual 0.037 | core grad 0.00191 | 4.3s
task_no_entanglement seed=42 05/50 | AUROC 0.74193 AP 0.30061 | residual 0.045 | core grad 0.00229 | 4.3s
task_no_entanglement seed=42 06/50 | AUROC 0.74180 AP 0.30069 | residual 0.052 | core grad 0.00257 | 4.3s
task_no_entanglement seed=42 07/50 | AUROC 0.74163 AP 0.30050 | residual 0.060 | core grad 0.00311 | 4.3s
task_no_entanglement seed=42 08/50 | AUROC 0.74152 AP 0.30015 | residual 0.067 | core grad 0.00315 | 4.3s
task_no_entanglement seed=42 09/50 | AUROC 0.74151 AP 0.30015 | residual 0.074 | core grad 0.0033 | 4.3s
task_no_entanglement seed=42 10/50 | AUROC 0.74171 AP 0.30023 | residual 0.082 | core grad 0.00427 | 4.3s
task_no_entanglement seed=42 11/50 | AUROC 0.74162 AP 0.30019 | residual 0.089 | core grad 0.00434 | 4.3s
task_no_entanglement seed=42 12/50 | AUROC 0.74176 AP 0.30004 | residual 0.093 | core grad 0.00437 | 4.3s
task_no_entanglement seed=42 13/50 | AUROC 0.74172 AP 0.29988 | residual 0.096 | core grad 0.00451 | 4.3s
task_no_entanglement seed=42 14/50 | AUROC 0.74175 AP 0.29984 | residual 0.100 | core grad 0.00493 | 4.3s
task_no_entanglement seed=42 15/50 | AUROC 0.74176 AP 0.29990 | residual 0.103 | core grad 0.00475 | 4.3s
linear_control seed=43 01/40 | AUROC 0.73243 AP 0.29498 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 02/40 | AUROC 0.73390 AP 0.29578 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 03/40 | AUROC 0.73514 AP 0.29619 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 04/40 | AUROC 0.73621 AP 0.29690 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 05/40 | AUROC 0.73748 AP 0.29852 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 06/40 | AUROC 0.73825 AP 0.29865 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 07/40 | AUROC 0.73874 AP 0.29890 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 08/40 | AUROC 0.73940 AP 0.29899 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 09/40 | AUROC 0.73988 AP 0.29911 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 10/40 | AUROC 0.74025 AP 0.29908 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 11/40 | AUROC 0.74088 AP 0.29845 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 12/40 | AUROC 0.74126 AP 0.29911 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 13/40 | AUROC 0.74152 AP 0.29946 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 14/40 | AUROC 0.74180 AP 0.29960 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 15/40 | AUROC 0.74219 AP 0.29979 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 16/40 | AUROC 0.74239 AP 0.29991 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 17/40 | AUROC 0.74272 AP 0.29987 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 18/40 | AUROC 0.74271 AP 0.29945 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 19/40 | AUROC 0.74298 AP 0.29963 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 20/40 | AUROC 0.74294 AP 0.30005 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 21/40 | AUROC 0.74293 AP 0.29992 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 22/40 | AUROC 0.74333 AP 0.30006 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 23/40 | AUROC 0.74332 AP 0.30038 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 24/40 | AUROC 0.74329 AP 0.30031 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 25/40 | AUROC 0.74321 AP 0.30050 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 26/40 | AUROC 0.74329 AP 0.30056 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 27/40 | AUROC 0.74317 AP 0.30045 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 28/40 | AUROC 0.74326 AP 0.30055 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 29/40 | AUROC 0.74323 AP 0.30060 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 30/40 | AUROC 0.74324 AP 0.30045 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 31/40 | AUROC 0.74320 AP 0.30054 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 32/40 | AUROC 0.74313 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 33/40 | AUROC 0.74305 AP 0.30078 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 34/40 | AUROC 0.74296 AP 0.30087 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 35/40 | AUROC 0.74289 AP 0.30084 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 36/40 | AUROC 0.74285 AP 0.30078 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 37/40 | AUROC 0.74291 AP 0.30090 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 38/40 | AUROC 0.74284 AP 0.30082 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 39/40 | AUROC 0.74289 AP 0.30099 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 40/40 | AUROC 0.74283 AP 0.30092 | residual 0.000 | core grad 0 | 0.1s
pca_classical seed=43 01/50 | AUROC 0.74227 AP 0.30041 | residual 0.016 | core grad 0 | 0.1s
pca_classical seed=43 02/50 | AUROC 0.74189 AP 0.29979 | residual 0.028 | core grad 0 | 0.1s
pca_classical seed=43 03/50 | AUROC 0.74138 AP 0.29948 | residual 0.040 | core grad 0 | 0.1s
pca_classical seed=43 04/50 | AUROC 0.74017 AP 0.29832 | residual 0.183 | core grad 0.0344 | 0.1s
pca_classical seed=43 05/50 | AUROC 0.73875 AP 0.29768 | residual 0.237 | core grad 0.039 | 0.1s
pca_classical seed=43 06/50 | AUROC 0.73594 AP 0.29647 | residual 0.221 | core grad 0.0608 | 0.1s
pca_classical seed=43 07/50 | AUROC 0.73563 AP 0.29546 | residual 0.321 | core grad 0.0656 | 0.1s
pca_classical seed=43 08/50 | AUROC 0.73529 AP 0.29500 | residual 0.285 | core grad 0.0908 | 0.1s
pca_classical seed=43 09/50 | AUROC 0.73493 AP 0.29482 | residual 0.312 | core grad 0.0782 | 0.1s
pca_classical seed=43 10/50 | AUROC 0.73491 AP 0.29559 | residual 0.321 | core grad 0.105 | 0.1s
pca_classical seed=43 11/50 | AUROC 0.73530 AP 0.29570 | residual 0.341 | core grad 0.0885 | 0.1s
pca_classical seed=43 12/50 | AUROC 0.73460 AP 0.29603 | residual 0.331 | core grad 0.0799 | 0.1s
pca_classical seed=43 13/50 | AUROC 0.73489 AP 0.29581 | residual 0.341 | core grad 0.0945 | 0.1s
pca_classical seed=43 14/50 | AUROC 0.73489 AP 0.29645 | residual 0.336 | core grad 0.0863 | 0.1s
pca_classical seed=43 15/50 | AUROC 0.73480 AP 0.29618 | residual 0.344 | core grad 0.0968 | 0.1s
pca_quantum seed=43 01/50 | AUROC 0.74274 AP 0.30073 | residual 0.009 | core grad 0 | 2.4s
pca_quantum seed=43 02/50 | AUROC 0.74249 AP 0.30070 | residual 0.016 | core grad 0 | 2.4s
pca_quantum seed=43 03/50 | AUROC 0.74237 AP 0.30064 | residual 0.022 | core grad 0 | 2.4s
pca_quantum seed=43 04/50 | AUROC 0.74214 AP 0.30033 | residual 0.029 | core grad 0.00143 | 5.2s
pca_quantum seed=43 05/50 | AUROC 0.74201 AP 0.30016 | residual 0.036 | core grad 0.00182 | 5.2s
pca_quantum seed=43 06/50 | AUROC 0.74183 AP 0.30000 | residual 0.042 | core grad 0.00232 | 5.2s
pca_quantum seed=43 07/50 | AUROC 0.74172 AP 0.29997 | residual 0.048 | core grad 0.00262 | 5.2s
pca_quantum seed=43 08/50 | AUROC 0.74181 AP 0.30012 | residual 0.053 | core grad 0.00402 | 5.2s
pca_quantum seed=43 09/50 | AUROC 0.74173 AP 0.30038 | residual 0.060 | core grad 0.00289 | 5.2s
pca_quantum seed=43 10/50 | AUROC 0.74156 AP 0.30030 | residual 0.065 | core grad 0.0035 | 5.2s
pca_quantum seed=43 11/50 | AUROC 0.74150 AP 0.30043 | residual 0.071 | core grad 0.00363 | 5.2s
pca_quantum seed=43 12/50 | AUROC 0.74132 AP 0.30023 | residual 0.077 | core grad 0.00408 | 5.2s
pca_quantum seed=43 13/50 | AUROC 0.74105 AP 0.30031 | residual 0.083 | core grad 0.00414 | 5.4s
pca_quantum seed=43 14/50 | AUROC 0.74098 AP 0.30020 | residual 0.089 | core grad 0.00475 | 5.2s
pca_quantum seed=43 15/50 | AUROC 0.74082 AP 0.30014 | residual 0.094 | core grad 0.00456 | 5.2s
task_classical seed=43 01/50 | AUROC 0.74239 AP 0.30065 | residual 0.015 | core grad 0 | 0.1s
task_classical seed=43 02/50 | AUROC 0.74166 AP 0.30062 | residual 0.028 | core grad 0 | 0.1s
task_classical seed=43 03/50 | AUROC 0.74141 AP 0.30052 | residual 0.038 | core grad 0 | 0.1s
task_classical seed=43 04/50 | AUROC 0.74017 AP 0.29934 | residual 0.199 | core grad 0.0316 | 0.1s
task_classical seed=43 05/50 | AUROC 0.73858 AP 0.29772 | residual 0.264 | core grad 0.0354 | 0.1s
task_classical seed=43 06/50 | AUROC 0.73618 AP 0.29667 | residual 0.251 | core grad 0.0615 | 0.1s
task_classical seed=43 07/50 | AUROC 0.73712 AP 0.29670 | residual 0.317 | core grad 0.0693 | 0.1s
task_classical seed=43 08/50 | AUROC 0.73822 AP 0.29805 | residual 0.322 | core grad 0.121 | 0.1s
task_classical seed=43 09/50 | AUROC 0.73756 AP 0.29777 | residual 0.323 | core grad 0.0783 | 0.1s
task_classical seed=43 10/50 | AUROC 0.73804 AP 0.29802 | residual 0.338 | core grad 0.129 | 0.1s
task_classical seed=43 11/50 | AUROC 0.73822 AP 0.29800 | residual 0.358 | core grad 0.112 | 0.1s
task_classical seed=43 12/50 | AUROC 0.73757 AP 0.29812 | residual 0.344 | core grad 0.097 | 0.1s
task_classical seed=43 13/50 | AUROC 0.73763 AP 0.29796 | residual 0.355 | core grad 0.0968 | 0.1s
task_classical seed=43 14/50 | AUROC 0.73765 AP 0.29811 | residual 0.354 | core grad 0.101 | 0.1s
task_classical seed=43 15/50 | AUROC 0.73762 AP 0.29827 | residual 0.358 | core grad 0.0891 | 0.1s
task_quantum seed=43 01/50 | AUROC 0.74263 AP 0.30065 | residual 0.011 | core grad 0 | 2.4s
task_quantum seed=43 02/50 | AUROC 0.74249 AP 0.30074 | residual 0.020 | core grad 0 | 2.4s
task_quantum seed=43 03/50 | AUROC 0.74222 AP 0.30085 | residual 0.029 | core grad 0 | 2.4s
task_quantum seed=43 04/50 | AUROC 0.74216 AP 0.30067 | residual 0.037 | core grad 0.00185 | 5.2s
task_quantum seed=43 05/50 | AUROC 0.74220 AP 0.30076 | residual 0.045 | core grad 0.00222 | 5.2s
task_quantum seed=43 06/50 | AUROC 0.74197 AP 0.30078 | residual 0.053 | core grad 0.0027 | 5.2s
task_quantum seed=43 07/50 | AUROC 0.74193 AP 0.30085 | residual 0.060 | core grad 0.00281 | 5.2s
task_quantum seed=43 08/50 | AUROC 0.74192 AP 0.30071 | residual 0.066 | core grad 0.00373 | 5.2s
task_quantum seed=43 09/50 | AUROC 0.74194 AP 0.30061 | residual 0.073 | core grad 0.00363 | 5.2s
task_quantum seed=43 10/50 | AUROC 0.74176 AP 0.30046 | residual 0.079 | core grad 0.00427 | 5.2s
task_quantum seed=43 11/50 | AUROC 0.74181 AP 0.30034 | residual 0.086 | core grad 0.00465 | 5.2s
task_quantum seed=43 12/50 | AUROC 0.74183 AP 0.30032 | residual 0.092 | core grad 0.00465 | 5.2s
task_quantum seed=43 13/50 | AUROC 0.74174 AP 0.30023 | residual 0.095 | core grad 0.00507 | 5.4s
task_quantum seed=43 14/50 | AUROC 0.74170 AP 0.30024 | residual 0.098 | core grad 0.00478 | 5.2s
task_quantum seed=43 15/50 | AUROC 0.74175 AP 0.30014 | residual 0.101 | core grad 0.00518 | 5.2s
task_frozen seed=43 01/50 | AUROC 0.74263 AP 0.30065 | residual 0.011 | core grad 0 | 2.4s
task_frozen seed=43 02/50 | AUROC 0.74249 AP 0.30074 | residual 0.020 | core grad 0 | 2.4s
task_frozen seed=43 03/50 | AUROC 0.74222 AP 0.30085 | residual 0.029 | core grad 0 | 2.4s
task_frozen seed=43 04/50 | AUROC 0.74215 AP 0.30068 | residual 0.037 | core grad 0 | 4.4s
task_frozen seed=43 05/50 | AUROC 0.74210 AP 0.30066 | residual 0.045 | core grad 0 | 4.4s
task_frozen seed=43 06/50 | AUROC 0.74189 AP 0.30089 | residual 0.053 | core grad 0 | 4.4s
task_frozen seed=43 07/50 | AUROC 0.74170 AP 0.30083 | residual 0.059 | core grad 0 | 4.4s
task_frozen seed=43 08/50 | AUROC 0.74154 AP 0.30086 | residual 0.065 | core grad 0 | 4.4s
task_frozen seed=43 09/50 | AUROC 0.74148 AP 0.30061 | residual 0.072 | core grad 0 | 4.4s
task_frozen seed=43 10/50 | AUROC 0.74121 AP 0.30058 | residual 0.078 | core grad 0 | 4.4s
task_frozen seed=43 11/50 | AUROC 0.74113 AP 0.30021 | residual 0.084 | core grad 0 | 4.4s
task_frozen seed=43 12/50 | AUROC 0.74108 AP 0.30017 | residual 0.086 | core grad 0 | 4.4s
task_frozen seed=43 13/50 | AUROC 0.74091 AP 0.30020 | residual 0.089 | core grad 0 | 4.6s
task_frozen seed=43 14/50 | AUROC 0.74088 AP 0.30025 | residual 0.091 | core grad 0 | 4.4s
task_frozen seed=43 15/50 | AUROC 0.74078 AP 0.30030 | residual 0.094 | core grad 0 | 4.4s
task_no_entanglement seed=43 01/50 | AUROC 0.74267 AP 0.30073 | residual 0.011 | core grad 0 | 2.1s
task_no_entanglement seed=43 02/50 | AUROC 0.74275 AP 0.30067 | residual 0.019 | core grad 0 | 2.1s
task_no_entanglement seed=43 03/50 | AUROC 0.74247 AP 0.30085 | residual 0.028 | core grad 0 | 2.1s
task_no_entanglement seed=43 04/50 | AUROC 0.74249 AP 0.30074 | residual 0.036 | core grad 0.0016 | 4.4s
task_no_entanglement seed=43 05/50 | AUROC 0.74244 AP 0.30060 | residual 0.044 | core grad 0.00186 | 4.4s
task_no_entanglement seed=43 06/50 | AUROC 0.74237 AP 0.30087 | residual 0.052 | core grad 0.00239 | 4.3s
task_no_entanglement seed=43 07/50 | AUROC 0.74242 AP 0.30088 | residual 0.059 | core grad 0.00263 | 4.4s
task_no_entanglement seed=43 08/50 | AUROC 0.74238 AP 0.30075 | residual 0.066 | core grad 0.00342 | 4.3s
task_no_entanglement seed=43 09/50 | AUROC 0.74233 AP 0.30060 | residual 0.073 | core grad 0.00331 | 4.3s
task_no_entanglement seed=43 10/50 | AUROC 0.74236 AP 0.30058 | residual 0.079 | core grad 0.00379 | 4.3s
task_no_entanglement seed=43 11/50 | AUROC 0.74238 AP 0.30055 | residual 0.086 | core grad 0.00411 | 4.3s
task_no_entanglement seed=43 12/50 | AUROC 0.74232 AP 0.30034 | residual 0.090 | core grad 0.00404 | 4.3s
task_no_entanglement seed=43 13/50 | AUROC 0.74232 AP 0.30033 | residual 0.093 | core grad 0.00471 | 4.3s
task_no_entanglement seed=43 14/50 | AUROC 0.74225 AP 0.30017 | residual 0.096 | core grad 0.00421 | 4.3s
task_no_entanglement seed=43 15/50 | AUROC 0.74220 AP 0.30016 | residual 0.100 | core grad 0.00434 | 4.3s
linear_control seed=44 01/40 | AUROC 0.73220 AP 0.29497 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 02/40 | AUROC 0.73370 AP 0.29591 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 03/40 | AUROC 0.73535 AP 0.29643 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 04/40 | AUROC 0.73600 AP 0.29809 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 05/40 | AUROC 0.73704 AP 0.29843 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 06/40 | AUROC 0.73828 AP 0.29854 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 07/40 | AUROC 0.73892 AP 0.29893 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 08/40 | AUROC 0.73936 AP 0.29888 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 09/40 | AUROC 0.73982 AP 0.29888 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 10/40 | AUROC 0.74043 AP 0.29887 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 11/40 | AUROC 0.74079 AP 0.29925 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 12/40 | AUROC 0.74113 AP 0.29909 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 13/40 | AUROC 0.74141 AP 0.29935 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 14/40 | AUROC 0.74202 AP 0.29952 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 15/40 | AUROC 0.74208 AP 0.29960 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 16/40 | AUROC 0.74232 AP 0.29974 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 17/40 | AUROC 0.74256 AP 0.29985 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 18/40 | AUROC 0.74263 AP 0.29994 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 19/40 | AUROC 0.74291 AP 0.29982 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 20/40 | AUROC 0.74290 AP 0.29984 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 21/40 | AUROC 0.74308 AP 0.29994 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 22/40 | AUROC 0.74335 AP 0.30008 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 23/40 | AUROC 0.74331 AP 0.30037 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 24/40 | AUROC 0.74344 AP 0.30032 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 25/40 | AUROC 0.74330 AP 0.30055 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 26/40 | AUROC 0.74322 AP 0.30046 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 27/40 | AUROC 0.74345 AP 0.30046 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 28/40 | AUROC 0.74337 AP 0.30064 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 29/40 | AUROC 0.74313 AP 0.30032 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 30/40 | AUROC 0.74314 AP 0.30062 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 31/40 | AUROC 0.74317 AP 0.30052 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 32/40 | AUROC 0.74300 AP 0.30047 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 33/40 | AUROC 0.74310 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 34/40 | AUROC 0.74301 AP 0.30079 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 35/40 | AUROC 0.74283 AP 0.30071 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 36/40 | AUROC 0.74294 AP 0.30071 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 37/40 | AUROC 0.74300 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 38/40 | AUROC 0.74295 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 39/40 | AUROC 0.74295 AP 0.30072 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 40/40 | AUROC 0.74289 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s
pca_classical seed=44 01/50 | AUROC 0.74284 AP 0.30082 | residual 0.015 | core grad 0 | 0.1s | best
pca_classical seed=44 02/50 | AUROC 0.74257 AP 0.30068 | residual 0.027 | core grad 0 | 0.1s
pca_classical seed=44 03/50 | AUROC 0.74252 AP 0.30090 | residual 0.037 | core grad 0 | 0.1s | best
pca_classical seed=44 04/50 | AUROC 0.74136 AP 0.30051 | residual 0.179 | core grad 0.0282 | 0.1s
pca_classical seed=44 05/50 | AUROC 0.74022 AP 0.29983 | residual 0.185 | core grad 0.0371 | 0.1s
pca_classical seed=44 06/50 | AUROC 0.73850 AP 0.29885 | residual 0.232 | core grad 0.0573 | 0.1s
pca_classical seed=44 07/50 | AUROC 0.73872 AP 0.29913 | residual 0.290 | core grad 0.0648 | 0.1s
pca_classical seed=44 08/50 | AUROC 0.73919 AP 0.29842 | residual 0.328 | core grad 0.0801 | 0.1s
pca_classical seed=44 09/50 | AUROC 0.73919 AP 0.29966 | residual 0.338 | core grad 0.093 | 0.1s
pca_classical seed=44 10/50 | AUROC 0.73993 AP 0.29988 | residual 0.345 | core grad 0.0881 | 0.1s
pca_classical seed=44 11/50 | AUROC 0.73961 AP 0.30013 | residual 0.360 | core grad 0.0942 | 0.1s
pca_classical seed=44 12/50 | AUROC 0.73951 AP 0.30035 | residual 0.373 | core grad 0.104 | 0.1s
pca_classical seed=44 13/50 | AUROC 0.73891 AP 0.30014 | residual 0.364 | core grad 0.0901 | 0.1s
pca_classical seed=44 14/50 | AUROC 0.73929 AP 0.30076 | residual 0.378 | core grad 0.0927 | 0.1s
pca_classical seed=44 15/50 | AUROC 0.73823 AP 0.30098 | residual 0.366 | core grad 0.0972 | 0.1s | best
pca_quantum seed=44 01/50 | AUROC 0.74289 AP 0.30077 | residual 0.010 | core grad 0 | 2.4s
pca_quantum seed=44 02/50 | AUROC 0.74269 AP 0.30078 | residual 0.016 | core grad 0 | 2.4s
pca_quantum seed=44 03/50 | AUROC 0.74262 AP 0.30084 | residual 0.023 | core grad 0 | 2.4s | best
pca_quantum seed=44 04/50 | AUROC 0.74256 AP 0.30089 | residual 0.030 | core grad 0.00165 | 5.1s | best
pca_quantum seed=44 05/50 | AUROC 0.74239 AP 0.30065 | residual 0.036 | core grad 0.00211 | 5.1s
pca_quantum seed=44 06/50 | AUROC 0.74213 AP 0.30042 | residual 0.042 | core grad 0.00266 | 5.1s
pca_quantum seed=44 07/50 | AUROC 0.74210 AP 0.30034 | residual 0.048 | core grad 0.00238 | 5.2s
pca_quantum seed=44 08/50 | AUROC 0.74199 AP 0.30023 | residual 0.054 | core grad 0.0025 | 5.1s
pca_quantum seed=44 09/50 | AUROC 0.74182 AP 0.30024 | residual 0.060 | core grad 0.00315 | 5.1s
pca_quantum seed=44 10/50 | AUROC 0.74181 AP 0.30014 | residual 0.062 | core grad 0.00345 | 5.1s
pca_quantum seed=44 11/50 | AUROC 0.74181 AP 0.30017 | residual 0.065 | core grad 0.00372 | 5.1s
pca_quantum seed=44 12/50 | AUROC 0.74181 AP 0.30027 | residual 0.068 | core grad 0.00355 | 5.1s
pca_quantum seed=44 13/50 | AUROC 0.74179 AP 0.30014 | residual 0.070 | core grad 0.00362 | 5.3s
pca_quantum seed=44 14/50 | AUROC 0.74183 AP 0.30026 | residual 0.073 | core grad 0.00433 | 5.2s
pca_quantum seed=44 15/50 | AUROC 0.74166 AP 0.30022 | residual 0.074 | core grad 0.0043 | 5.1s
task_classical seed=44 01/50 | AUROC 0.74263 AP 0.30066 | residual 0.015 | core grad 0 | 0.1s
task_classical seed=44 02/50 | AUROC 0.74203 AP 0.30051 | residual 0.026 | core grad 0 | 0.1s
task_classical seed=44 03/50 | AUROC 0.74173 AP 0.30042 | residual 0.037 | core grad 0 | 0.1s
task_classical seed=44 04/50 | AUROC 0.73937 AP 0.29946 | residual 0.195 | core grad 0.0324 | 0.1s
task_classical seed=44 05/50 | AUROC 0.73738 AP 0.29843 | residual 0.222 | core grad 0.0494 | 0.1s
task_classical seed=44 06/50 | AUROC 0.73746 AP 0.29716 | residual 0.280 | core grad 0.0605 | 0.1s
task_classical seed=44 07/50 | AUROC 0.73709 AP 0.29777 | residual 0.308 | core grad 0.0848 | 0.1s
task_classical seed=44 08/50 | AUROC 0.73765 AP 0.29744 | residual 0.332 | core grad 0.077 | 0.1s
task_classical seed=44 09/50 | AUROC 0.73772 AP 0.29824 | residual 0.358 | core grad 0.115 | 0.1s
task_classical seed=44 10/50 | AUROC 0.73834 AP 0.29785 | residual 0.358 | core grad 0.0879 | 0.1s
task_classical seed=44 11/50 | AUROC 0.73669 AP 0.29801 | residual 0.354 | core grad 0.0997 | 0.1s
task_classical seed=44 12/50 | AUROC 0.73762 AP 0.29817 | residual 0.374 | core grad 0.1 | 0.1s
task_classical seed=44 13/50 | AUROC 0.73736 AP 0.29830 | residual 0.359 | core grad 0.0886 | 0.1s
task_classical seed=44 14/50 | AUROC 0.73772 AP 0.29795 | residual 0.373 | core grad 0.106 | 0.1s
task_classical seed=44 15/50 | AUROC 0.73749 AP 0.29841 | residual 0.366 | core grad 0.091 | 0.1s
task_quantum seed=44 01/50 | AUROC 0.74279 AP 0.30072 | residual 0.011 | core grad 0 | 2.4s
task_quantum seed=44 02/50 | AUROC 0.74268 AP 0.30086 | residual 0.020 | core grad 0 | 2.4s | best
task_quantum seed=44 03/50 | AUROC 0.74242 AP 0.30071 | residual 0.029 | core grad 0 | 2.4s
task_quantum seed=44 04/50 | AUROC 0.74236 AP 0.30074 | residual 0.037 | core grad 0.00195 | 5.2s
task_quantum seed=44 05/50 | AUROC 0.74228 AP 0.30080 | residual 0.045 | core grad 0.00227 | 5.2s
task_quantum seed=44 06/50 | AUROC 0.74215 AP 0.30067 | residual 0.052 | core grad 0.00313 | 5.2s
task_quantum seed=44 07/50 | AUROC 0.74209 AP 0.30060 | residual 0.059 | core grad 0.00307 | 5.2s
task_quantum seed=44 08/50 | AUROC 0.74214 AP 0.30053 | residual 0.066 | core grad 0.0039 | 5.2s
task_quantum seed=44 09/50 | AUROC 0.74189 AP 0.30030 | residual 0.072 | core grad 0.00373 | 5.2s
task_quantum seed=44 10/50 | AUROC 0.74193 AP 0.30022 | residual 0.079 | core grad 0.00402 | 5.2s
task_quantum seed=44 11/50 | AUROC 0.74191 AP 0.30035 | residual 0.082 | core grad 0.0044 | 5.2s
task_quantum seed=44 12/50 | AUROC 0.74209 AP 0.30021 | residual 0.085 | core grad 0.00466 | 5.1s
task_quantum seed=44 13/50 | AUROC 0.74208 AP 0.29998 | residual 0.088 | core grad 0.00487 | 5.4s
task_quantum seed=44 14/50 | AUROC 0.74216 AP 0.29962 | residual 0.091 | core grad 0.00427 | 5.2s
task_quantum seed=44 15/50 | AUROC 0.74213 AP 0.29951 | residual 0.094 | core grad 0.00522 | 5.0s
task_frozen seed=44 01/50 | AUROC 0.74279 AP 0.30072 | residual 0.011 | core grad 0 | 2.3s
task_frozen seed=44 02/50 | AUROC 0.74268 AP 0.30086 | residual 0.020 | core grad 0 | 2.3s | best
task_frozen seed=44 03/50 | AUROC 0.74242 AP 0.30071 | residual 0.029 | core grad 0 | 2.3s
task_frozen seed=44 04/50 | AUROC 0.74231 AP 0.30075 | residual 0.037 | core grad 0 | 4.3s
task_frozen seed=44 05/50 | AUROC 0.74217 AP 0.30080 | residual 0.045 | core grad 0 | 4.2s
task_frozen seed=44 06/50 | AUROC 0.74193 AP 0.30079 | residual 0.052 | core grad 0 | 4.2s
task_frozen seed=44 07/50 | AUROC 0.74183 AP 0.30084 | residual 0.059 | core grad 0 | 4.2s
task_frozen seed=44 08/50 | AUROC 0.74165 AP 0.30059 | residual 0.065 | core grad 0 | 4.2s
task_frozen seed=44 09/50 | AUROC 0.74138 AP 0.30031 | residual 0.071 | core grad 0 | 4.2s
task_frozen seed=44 10/50 | AUROC 0.74123 AP 0.30045 | residual 0.078 | core grad 0 | 4.2s
task_frozen seed=44 11/50 | AUROC 0.74127 AP 0.30060 | residual 0.083 | core grad 0 | 4.2s
task_frozen seed=44 12/50 | AUROC 0.74134 AP 0.30055 | residual 0.089 | core grad 0 | 4.2s
task_frozen seed=44 13/50 | AUROC 0.74127 AP 0.30041 | residual 0.091 | core grad 0 | 4.5s
task_frozen seed=44 14/50 | AUROC 0.74112 AP 0.30031 | residual 0.094 | core grad 0 | 4.2s
task_frozen seed=44 15/50 | AUROC 0.74104 AP 0.30004 | residual 0.096 | core grad 0 | 4.2s
task_no_entanglement seed=44 01/50 | AUROC 0.74292 AP 0.30083 | residual 0.011 | core grad 0 | 2.0s | best
task_no_entanglement seed=44 02/50 | AUROC 0.74284 AP 0.30105 | residual 0.020 | core grad 0 | 2.0s | best
task_no_entanglement seed=44 03/50 | AUROC 0.74288 AP 0.30093 | residual 0.028 | core grad 0 | 2.0s
task_no_entanglement seed=44 04/50 | AUROC 0.74275 AP 0.30097 | residual 0.036 | core grad 0.00168 | 4.2s
task_no_entanglement seed=44 05/50 | AUROC 0.74262 AP 0.30106 | residual 0.044 | core grad 0.00213 | 4.2s | best
task_no_entanglement seed=44 06/50 | AUROC 0.74263 AP 0.30092 | residual 0.051 | core grad 0.00274 | 4.2s
task_no_entanglement seed=44 07/50 | AUROC 0.74261 AP 0.30085 | residual 0.058 | core grad 0.00279 | 4.2s
task_no_entanglement seed=44 08/50 | AUROC 0.74253 AP 0.30061 | residual 0.064 | core grad 0.00361 | 4.2s
task_no_entanglement seed=44 09/50 | AUROC 0.74261 AP 0.30051 | residual 0.071 | core grad 0.0033 | 4.2s
task_no_entanglement seed=44 10/50 | AUROC 0.74282 AP 0.30036 | residual 0.078 | core grad 0.00413 | 4.2s
task_no_entanglement seed=44 11/50 | AUROC 0.74278 AP 0.30028 | residual 0.081 | core grad 0.00456 | 4.2s
task_no_entanglement seed=44 12/50 | AUROC 0.74293 AP 0.30027 | residual 0.084 | core grad 0.00459 | 4.2s
task_no_entanglement seed=44 13/50 | AUROC 0.74297 AP 0.30032 | residual 0.088 | core grad 0.00429 | 4.2s
task_no_entanglement seed=44 14/50 | AUROC 0.74296 AP 0.30024 | residual 0.090 | core grad 0.00388 | 4.2s
task_no_entanglement seed=44 15/50 | AUROC 0.74301 AP 0.30010 | residual 0.093 | core grad 0.00462 | 4.2s

                    arm     seed  checkpoint_epoch  validation_macro_AUROC  validation_macro_AUPRC
experiment_2_reproduced   source               5.0                0.730784                0.294336
         linear_control       42              40.0                0.743003                0.301028
          pca_classical       42               2.0                0.742987                0.301622
            pca_quantum       42               2.0                0.742929                0.301321
         task_classical       42               0.0                0.743003                0.301028
           task_quantum       42               0.0                0.743003                0.301028
            task_frozen       42               0.0                0.743003                0.301028
   task_no_entanglement       42               0.0                0.743003                0.301028
         linear_control       43              39.0                0.742888                0.300988
          pca_classical       43               0.0                0.742888                0.300988
            pca_quantum       43               0.0                0.742888                0.300988
         task_classical       43               0.0                0.742888                0.300988
           task_quantum       43               0.0                0.742888                0.300988
            task_frozen       43               0.0                0.742888                0.300988
   task_no_entanglement       43               0.0                0.742888                0.300988
         linear_control       44              34.0                0.743007                0.300792
          pca_classical       44              15.0                0.738226                0.300976
            pca_quantum       44               4.0                0.742560                0.300890
         task_classical       44               0.0                0.743007                0.300792
           task_quantum       44               2.0                0.742678                0.300865
            task_frozen       44               2.0                0.742678                0.300865
   task_no_entanglement       44               5.0                0.742622                0.301057
         linear_control ensemble               NaN                0.742902                0.300829
          pca_classical ensemble               NaN                0.741813                0.300939
            pca_quantum ensemble               NaN                0.742614                0.301304
         task_classical ensemble               NaN                0.742902                0.300829
           task_quantum ensemble               NaN                0.742818                0.300961
            task_frozen ensemble               NaN                0.742818                0.300961
   task_no_entanglement ensemble               NaN                0.742779                0.301052

Paired macro AUPRC differences (positive favors left):
    seed                                comparison      metric  point_delta  ci_lower  ci_upper  confidence_level  bootstrap_samples_requested  bootstrap_samples_valid
      42           pca_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42          task_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42           task_frozen minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42  task_no_entanglement minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43           pca_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43          task_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43           task_frozen minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43  task_no_entanglement minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44           pca_quantum minus circuit_reset macro_AUPRC     0.000003 -0.000082  0.000075              0.95                          500                      493
      44          task_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44           task_frozen minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44  task_no_entanglement minus circuit_reset macro_AUPRC    -0.000044 -0.000284  0.000179              0.95                          500                      493
      42           pca_quantum minus pca_classical macro_AUPRC    -0.000301 -0.001276  0.001062              0.95                          500                      493
      42         task_quantum minus task_classical macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42            task_quantum minus pca_quantum macro_AUPRC    -0.000293 -0.000823  0.000518              0.95                          500                      493
      42            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42   task_quantum minus task_no_entanglement macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42        pca_classical minus linear_control macro_AUPRC     0.000594 -0.000566  0.001344              0.95                          500                      493
      42          pca_quantum minus linear_control macro_AUPRC     0.000293 -0.000518  0.000823              0.95                          500                      493
      42       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42         task_quantum minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42          task_frozen minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42 task_no_entanglement minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43           pca_quantum minus pca_classical macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43         task_quantum minus task_classical macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43            task_quantum minus pca_quantum macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43   task_quantum minus task_no_entanglement macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43        pca_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43          pca_quantum minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43         task_quantum minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43          task_frozen minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43 task_no_entanglement minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44           pca_quantum minus pca_classical macro_AUPRC    -0.000086 -0.010148  0.009225              0.95                          500                      493
      44         task_quantum minus task_classical macro_AUPRC     0.000073 -0.000731  0.001669              0.95                          500                      493
      44            task_quantum minus pca_quantum macro_AUPRC    -0.000025 -0.001049  0.001818              0.95                          500                      493
      44            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44   task_quantum minus task_no_entanglement macro_AUPRC    -0.000192 -0.002065  0.001202              0.95                          500                      493
      44        pca_classical minus linear_control macro_AUPRC     0.000184 -0.009434  0.010946              0.95                          500                      493
      44          pca_quantum minus linear_control macro_AUPRC     0.000098 -0.001311  0.001112              0.95                          500                      493
      44       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44         task_quantum minus linear_control macro_AUPRC     0.000073 -0.000731  0.001669              0.95                          500                      493
      44          task_frozen minus linear_control macro_AUPRC     0.000073 -0.000731  0.001669              0.95                          500                      493
      44 task_no_entanglement minus linear_control macro_AUPRC     0.000265 -0.001570  0.003165              0.95                          500                      493
ensemble           pca_quantum minus pca_classical macro_AUPRC     0.000366 -0.004877  0.004775              0.95                          500                      493
ensemble         task_quantum minus task_classical macro_AUPRC     0.000132 -0.000271  0.000691              0.95                          500                      493
ensemble            task_quantum minus pca_quantum macro_AUPRC    -0.000344 -0.001289  0.000372              0.95                          500                      493
ensemble            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
ensemble   task_quantum minus task_no_entanglement macro_AUPRC    -0.000091 -0.000692  0.000518              0.95                          500                      493
ensemble        pca_classical minus linear_control macro_AUPRC     0.000110 -0.004074  0.005510              0.95                          500                      493
ensemble          pca_quantum minus linear_control macro_AUPRC     0.000475 -0.000203  0.001496              0.95                          500                      493
ensemble       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
ensemble         task_quantum minus linear_control macro_AUPRC     0.000132 -0.000271  0.000691              0.95                          500                      493
ensemble          task_frozen minus linear_control macro_AUPRC     0.000132 -0.000271  0.000691              0.95                          500                      493
ensemble task_no_entanglement minus linear_control macro_AUPRC     0.000223 -0.000495  0.001020              0.95                          500                      493
Saved to: /workspace/unzippedarchive/unzippedarchive/xray_training_experiment_7_encoding/run_20260929T182317_419531Z
Test features, predictions and metrics were not computed.

"""Experiment 8: complementary encoder and controlled circuit training.

Vast.ai notebook (upload this file; E6's a.py is NOT required):
    %run c.py
Confirmation after a successful development run:
    %run c.py --stage confirm --protocol /path/to/develop_run/protocol.json
One final test evaluation after confirmation:
    %run c.py --stage final --confirmation-run /path/to/confirm_run
One-seed diagnostic run:
    %env E8_SEEDS=42
    %run c.py
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
SEEDS = [int(s) for s in os.environ.get('E8_SEEDS', '42,43,44').split(',')]
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
BOOTSTRAP_SAMPLES, BOOTSTRAP_ALPHA, BOOTSTRAP_SEED = 500, 0.05, 12345
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
    OUTPUT_DIR = ROOT / 'xray_training_experiment_8_complementary' / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
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
    if STAGE == 'develop':
        splitter = GroupShuffleSplit(n_splits=1, test_size=DEV_FRACTION,
                                     random_state=DEV_SPLIT_SEED)
        fit_ids, dev_ids = next(splitter.split(frames['train'],
                                               groups=frames['train'].subject_id))
        frames['fit'] = frames['train'].iloc[fit_ids].reset_index(drop=True)
        frames['dev'] = frames['train'].iloc[dev_ids].reset_index(drop=True)
        assert set(frames['fit'].subject_id).isdisjoint(set(frames['dev'].subject_id))
        assignments_inner = pd.concat([
            frames['fit'][['subject_id']].assign(inner_split='fit'),
            frames['dev'][['subject_id']].assign(inner_split='dev')
        ]).drop_duplicates().sort_values('subject_id')
        assignments_inner.to_csv(OUTPUT_DIR / 'inner_patient_splits.csv', index=False)
        print('Inner development split:', len(frames['fit']), 'fit images,',
              len(frames['dev']), 'development images')
    else:
        frames['fit'] = frames['train']
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


    # Keep the E7 cache signature, which includes train+val records, while only
    # loading original validation features during the explicit confirmation run.
    active_splits = ['train'] + (['val'] if STAGE == 'confirm' else
                                 ['test'] if STAGE == 'final' else [])
    source_hash = sha256_file(SOURCE_CHECKPOINT)
    assert source_hash == source_state_digest, 'Source checkpoint changed while reading.'
    cache_spec = {'revision': 'exp2-rgb320-pad-bilinear-imagenet-fp32-v1', 'checkpoint_sha256': source_hash,
                  'torch': str(torch.__version__), 'torchvision': torchvision.__version__, 'pillow': PIL.__version__,
                  'extraction_device': str(EXTRACT_DEVICE),
                  'gpu': torch.cuda.get_device_name(0) if EXTRACT_DEVICE.type == 'cuda' else None,
                  'batch_size': EXTRACT_BATCH_SIZE,
                  'records': [[str(row.dicom_id), row.split, str(row.image_path), Path(row.image_path).stat().st_size,
                               Path(row.image_path).stat().st_mtime_ns]
                              for row in df.loc[df.split.isin(
                                  ['train', 'val', 'test'] if STAGE == 'final' else ['train', 'val'])].itertuples(index=False)]}
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
    if STAGE == 'develop':
        feature_arrays['fit'] = feature_arrays['train'][fit_ids]
        feature_arrays['dev'] = feature_arrays['train'][dev_ids]
        active_splits += ['fit', 'dev']
    else:
        feature_arrays['fit'] = feature_arrays['train']
        active_splits += ['fit']
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


    eval_split = 'dev' if STAGE == 'develop' else 'test' if STAGE == 'final' else 'val'
    with torch.no_grad():
        z = torch.from_numpy(feature_arrays[eval_split])
        baseline_probabilities = torch.sigmoid(F.linear(z, original_weight, original_bias)).numpy()
    baseline_result = make_result(eval_split, baseline_probabilities)
    print('Saved E2 checkpoint metrics:', source_metrics)
    print(eval_split, 'source metrics:', {key: baseline_result['metrics'][key] for key in METRIC_NAMES})
    if STAGE == 'confirm':
        for key in ['macro_AUROC', 'macro_AUPRC']:
            if abs(baseline_result['metrics'][key] - source_metrics[key]) > .003:
                raise RuntimeError(f'Exp2 reproduction differs by >0.003 for {key}. Check preprocessing/checkpoint.')
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
    """Fixed 1024->48 complementary map with a common bounded angle transform."""
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
            self.encoder.requires_grad_(False)
            self.branch_dropout = nn.Dropout(BRANCH_DROPOUT)
            # Create shared objects before arm-specific random initializers.
            self.readout = nn.Linear(OBSERVABLE_DIM, len(LABELS))
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
            self.core = ClassicalCore() if arm == 'classical_residual' else quantum_core()
            self.base.requires_grad_(False)
            if arm == 'frozen_quantum':
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


def fit_complementary_encoder(data, base):
    """Fit PCA only on fit patients, orthogonal to the anchor's weight row space."""
    x = data.features['fit']
    _, singular, directions = torch.linalg.svd(base.weight.detach(), full_matrices=False)
    rank = int((singular > singular.max() * 1e-6).sum())
    basis = directions[:rank].T.contiguous()
    residual = x - (x @ basis) @ basis.T
    pca = PCA(n_components=ANGLE_DIM, svd_solver='randomized', random_state=42)
    pca.fit(residual.numpy())
    components = torch.from_numpy(pca.components_.astype(np.float32))
    center = torch.from_numpy(pca.mean_.astype(np.float32))
    scores = F.linear(residual - center, components)
    scale = scores.std(dim=0, unbiased=False).clamp_min(1e-4)
    weight = components / scale[:, None]
    bias = -weight @ center
    # The selected encoder directions must be orthogonal to anchor logits.
    max_overlap = float((weight @ basis).abs().max())
    if max_overlap > 1e-3:
        raise RuntimeError(f'Complementary encoder overlaps anchor directions: {max_overlap}')
    order = torch.arange(ANGLE_DIM).reshape(-1, Q_LAYERS).T.flatten()
    retained = float(pca.explained_variance_ratio_.sum())
    total_fraction = float(pca.explained_variance_.sum() / x.var(0, unbiased=True).sum().item())
    return (weight[order], bias[order]), pca, {
        'anchor_rank': rank, 'max_basis_overlap': max_overlap,
        'residual_variance_fraction_48': retained,
        'total_feature_variance_fraction_48': total_fraction}


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
    return {'experiment': 'Experiment 8 complementary encoding', 'arm': model.arm, 'seed': seed, 'epoch': epoch,
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'labels': LABELS, 'n_qubits': N_QUBITS, 'q_layers': Q_LAYERS,
            'feature_dim': FEATURE_DIM, 'image_size': IMAGE_SIZE, 'residual_limit': RESIDUAL_LIMIT,
            'feature_mean': torch.as_tensor(data.feature_mean), 'feature_scale': torch.as_tensor(data.feature_scale),
            'source_checkpoint_sha256': data.source_hash, 'source_epoch': data.source_epoch,
            'macro_AUROC': float(result['metrics']['macro_AUROC']),
            'macro_AUPRC': float(result['metrics']['macro_AUPRC']),
            'pennylane_version': str(qml.__version__),
            'note': 'Selected inference checkpoint; epoch 0 residual is inactive. Not an optimizer resume file.'}


def train_head(model, data, seed, directory, fixed_epochs=None):
    """Select on inner development patients or use an epoch locked by that run."""
    arm = model.arm
    is_base = arm == 'linear_control'
    monitor = data.eval_split if fixed_epochs is None else 'fit'
    if is_base:
        groups = [{'params': list(model.base.parameters()), 'lr': BASE_LR, 'name': 'base'}]
    else:
        model.core.requires_grad_(False)
        groups = [{'params': list(model.readout.parameters()), 'lr': READOUT_LR, 'name': 'readout'},
                  ]
        if arm != 'frozen_quantum':
            groups.append({'params': list(model.core.parameters()), 'lr': 0., 'name': 'core'})
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed + 1000)
    seed_everything(seed + 2000)
    criterion = make_loss(data)
    initial = evaluate_head(model, data, monitor)
    best_score = initial['metrics']['macro_AUPRC']
    if not math.isfinite(best_score):
        raise RuntimeError(f'No evaluable {monitor} macro AUPRC')
    best_epoch, stale, stopping_reference = 0, 0, best_score
    initial_state = copy.deepcopy(model.state_dict())
    atomic_torch_save(checkpoint_payload(model, data, seed, 0, initial), directory / 'best_model.pt')
    history = [{'epoch': 0, 'train_loss': np.nan, f'{monitor}_loss': initial['loss'],
                **{k: initial['metrics'][k] for k in METRIC_NAMES}, 'residual_logit_rms': initial['residual_logit_rms'],
                'core_parameter_change_l2': 0., 'core_gradient_max': 0.}]
    max_epochs = fixed_epochs if fixed_epochs is not None else (BASE_EPOCHS if is_base else MAX_EPOCHS)
    if max_epochs < 0:
        raise ValueError('Fixed epoch count cannot be negative')
    started_total = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        if not is_base and epoch == WARMUP_EPOCHS + 1:
            if arm != 'frozen_quantum':
                model.core.requires_grad_(True)
            for group in optimizer.param_groups:
                if group['name'] == 'core':
                    group['lr'] = CORE_LR
            if fixed_epochs is None:
                stale, stopping_reference = 0, best_score
        model.train()
        order = torch.randperm(len(data.features['fit']), generator=generator)
        loss_sum, valid_count, core_grad = 0., 0., 0.
        started = time.perf_counter()
        for selected in order.split(HEAD_BATCH_SIZE):
            mask = data.masks['fit'][selected]
            patient_weight = data.patient_weights[selected, None]
            weighted_mask = mask * patient_weight
            denominator = weighted_mask.sum()
            if denominator.item() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            base, residual = model.components(data.features['fit'][selected])
            masked_sum = (criterion(base + residual, data.targets['fit'][selected]) * weighted_mask).sum()
            loss = masked_sum / denominator + RESIDUAL_L2 * residual.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{arm}: non-finite loss')
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            if not is_base and arm != 'frozen_quantum':
                norm = math.sqrt(sum(p.grad.square().sum().item()
                                     for p in model.core.parameters() if p.grad is not None))
                core_grad = max(core_grad, norm)
            optimizer.step()
            loss_sum += masked_sum.detach().item()
            valid_count += denominator.item()
        result = evaluate_head(model, data, monitor)
        score = result['metrics']['macro_AUPRC']
        if not math.isfinite(score):
            raise FloatingPointError(f'Non-finite {monitor} score')
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            atomic_torch_save(checkpoint_payload(model, data, seed, epoch, result), directory / 'best_model.pt')
        if fixed_epochs is None and (is_base or epoch > WARMUP_EPOCHS):
            if score > stopping_reference + STOP_MIN_DELTA:
                stale, stopping_reference = 0, score
            else:
                stale += 1
        core_change = 0. if is_base else math.sqrt(sum(
            (p.detach() - initial_state[f'core.{key}']).square().sum().item()
            for key, p in model.core.named_parameters()))
        row = {'epoch': epoch, 'train_loss': loss_sum / max(valid_count, 1),
               f'{monitor}_loss': result['loss'], **{k: result['metrics'][k] for k in METRIC_NAMES},
               'residual_logit_rms': result['residual_logit_rms'],
               'core_parameter_change_l2': core_change, 'core_gradient_max': core_grad,
               'seconds': time.perf_counter() - started,
               **{f'lr_{g["name"]}': g['lr'] for g in optimizer.param_groups}}
        history.append(row)
        pd.DataFrame(history).to_csv(directory / 'training_history.csv', index=False)
        print(f'{arm} seed={seed} {epoch:02d}/{max_epochs} | {monitor} AUROC {result["metrics"]["macro_AUROC"]:.5f} '
              f'AP {score:.5f} | residual {row["residual_logit_rms"]:.3f} | '
              f'core grad {core_grad:.3g} change {core_change:.3g} | {row["seconds"]:.1f}s'
              + (' | best' if improved and fixed_epochs is None else ''), flush=True)
        if fixed_epochs is None and epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    atomic_torch_save(checkpoint_payload(model, data, seed, len(history)-1,
                                         result if max_epochs else initial),
                      directory / 'final_model.pt')
    if fixed_epochs is not None:
        # Confirmation uses the precommitted epoch, never a validation-selected one.
        atomic_torch_save(checkpoint_payload(model, data, seed, fixed_epochs,
                                             result if max_epochs else initial),
                          directory / 'best_model.pt')
        best_epoch = fixed_epochs
        best_score = (result if max_epochs else initial)['metrics']['macro_AUPRC']
    if not is_base and arm != 'frozen_quantum' and max_epochs > WARMUP_EPOCHS and max(r['core_gradient_max'] for r in history) <= 0:
        raise RuntimeError(f'{arm}: core did not receive gradients')
    best = torch.load(directory / 'best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'], strict=True)
    result = evaluate_head(model, data, monitor)
    assert abs(result['metrics']['macro_AUPRC'] - best_score) < 1e-7
    diagnostics = {}
    if not is_base:
        for name in ('base', 'encoder', 'core', 'readout'):
            diagnostics[f'{name}_parameter_change_l2'] = math.sqrt(sum(
                (p.detach() - initial_state[f'{name}.{key}']).square().sum().item()
                for key, p in getattr(model, name).named_parameters()))
        assert diagnostics['base_parameter_change_l2'] == 0, 'Frozen anchor changed'
        assert diagnostics['encoder_parameter_change_l2'] == 0, 'Fixed encoder changed'
        if arm == 'frozen_quantum':
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
        assert prefix == 'dev', 'Threshold fitting is limited to inner development patients.'
        fitted_thresholds, threshold_table = fit_validation_thresholds(result)
        threshold_table.to_csv(directory / 'dev_thresholds.csv', index=False)
        save_json(directory / 'thresholds.json', fitted_thresholds)
    reports = {'fixed_05': fixed}
    if fitted_thresholds is not None:
        fitted, fitted_per_class = threshold_metrics(result['targets'], result['probabilities'], result['masks'], fitted_thresholds)
        fitted.update(threshold_source='inner_development_fitted', threshold_fitting_on_evaluation_data=fit_thresholds,
                      subset_accuracy_definition='Exact match only on images with all selected labels observed.',
                      note='Labels below 30 development positives excluded. Development-fitted scores are in-sample and optimistic.')
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


def patient_sensitivity(result):
    """One patient = one row; mean observed probability and any observed positive."""
    subjects, inverse = np.unique(result['subject_ids'], return_inverse=True)
    patient_targets = np.zeros((len(subjects), len(LABELS)), dtype=np.float32)
    patient_masks = np.zeros_like(patient_targets)
    patient_probabilities = np.full_like(patient_targets, 0.5)
    for patient_index in range(len(subjects)):
        chosen = inverse == patient_index
        for label_index in range(len(LABELS)):
            observed = chosen & (result['masks'][:, label_index] == 1)
            if observed.any():
                patient_masks[patient_index, label_index] = 1
                patient_targets[patient_index, label_index] = result['targets'][observed, label_index].max()
                patient_probabilities[patient_index, label_index] = result['probabilities'][observed, label_index].mean()
    return calculate_metrics(patient_targets, patient_probabilities, patient_masks)


def protocol_settings():
    names = ('SEEDS', 'ARMS', 'LABELS', 'IMAGE_SIZE', 'FEATURE_DIM', 'N_QUBITS',
             'Q_LAYERS', 'ANGLE_DIM', 'OBSERVABLE_DIM', 'HEAD_BATCH_SIZE',
             'BASE_EPOCHS', 'MAX_EPOCHS', 'WARMUP_EPOCHS', 'MIN_EPOCHS',
             'PATIENCE', 'STOP_MIN_DELTA', 'BASE_LR', 'CORE_LR', 'READOUT_LR',
             'WEIGHT_DECAY', 'RESIDUAL_L2', 'DROPOUT', 'BRANCH_DROPOUT',
             'RESIDUAL_LIMIT', 'POS_WEIGHT_CAP', 'DEV_FRACTION', 'DEV_SPLIT_SEED',
             'MEANINGFUL_AP_GAIN', 'MAX_AUROC_DROP', 'BOOTSTRAP_SAMPLES',
             'BOOTSTRAP_ALPHA', 'BOOTSTRAP_SEED', 'MIN_POSITIVES_FOR_THRESHOLD')
    return {name: globals()[name] for name in names}


def protocol_fingerprint(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def run_experiment():
    settings = protocol_settings()
    fingerprint = protocol_fingerprint(settings)
    locked = None
    if STAGE == 'confirm':
        if PROTOCOL_PATH is None:
            raise ValueError('Confirmation requires --protocol from a development run')
        locked = json.loads(Path(PROTOCOL_PATH).read_text(encoding='utf-8'))
        if not locked.get('eligible_for_confirmation'):
            raise RuntimeError('Development protocol did not meet the predeclared confirmation gate')
        if locked.get('settings_fingerprint') != fingerprint:
            raise RuntimeError('Protocol settings changed since development')
    data = prepare_data()
    if locked and locked.get('source_hash') != data.source_hash:
        raise RuntimeError('Source checkpoint changed since development')
    save_json(data.output / 'configuration.json', {
        'experiment': 'Experiment 8 complementary encoder', 'stage': STAGE,
        'settings': settings, 'settings_fingerprint': fingerprint,
        'source_checkpoint_sha256': data.source_hash,
        'source_epoch': data.source_epoch, 'source_saved_metrics': data.source_metrics,
        'feature_cache': str(data.cache_dir), 'evaluation_split': data.eval_split,
        'selection': 'inner development patient macro AUPRC' if STAGE == 'develop'
                     else 'epoch fixed by the eligible development protocol',
        'labels': LABELS, 'label_policy': 'NaN/0 negative, 1 positive, -1 masked',
        'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
        'pennylane': str(qml.__version__), 'python': platform.python_version(),
        'development_protocol': str(PROTOCOL_PATH) if PROTOCOL_PATH else None})
    drawing_model = ResidualHead('quantum_residual', 42)
    drawing = qml.draw(drawing_model.core.qnode, decimals=2)(
        torch.zeros(ANGLE_DIM), drawing_model.core.rotations,
        drawing_model.core.couplings, drawing_model.core.final_y)
    (data.output / 'quantum_circuit.txt').write_text(drawing + '\n', encoding='utf-8')
    del drawing_model
    source_file = Path(globals().get('__file__', 'c.py'))
    if source_file.is_file():
        (data.output / 'c.py').write_bytes(source_file.read_bytes())
    atomic_torch_save({'features': data.backbone_state,
                       'source_sha256': data.source_hash}, data.output / 'frozen_backbone.pt')
    print('E8 stage:', STAGE, '| selection/report split:', data.eval_split, flush=True)
    if STAGE == 'develop':
        print('Caveat: the inherited E2 backbone trained on these development patients; '
              'the inner split is not an independent validation cohort.', flush=True)
    results, draws_by_run, rows, paired, ablation_rows, thresholds_by_run = {}, {}, [], [], [], {}

    def record(arm, seed, result, directory, info):
        key = f'{arm}:{seed}'
        fixed_thresholds = locked['thresholds'][key] if locked else None
        fixed, thresholds, draws = save_evaluation(
            result, directory, data.eval_split,
            fitted_thresholds=fixed_thresholds, fit_thresholds=STAGE == 'develop')
        patient = patient_sensitivity(result)
        save_json(directory / f'{data.eval_split}_patient_sensitivity.json', {
            'aggregation': 'mean probability over observed images; positive if any observed image is positive',
            'metrics': patient})
        results[(arm, seed)], draws_by_run[(arm, seed)] = result, draws
        thresholds_by_run[key] = thresholds
        rows.append({**info, 'arm': arm, 'seed': seed,
                     **{f'{data.eval_split}_{name}': result['metrics'][name] for name in METRIC_NAMES},
                     f'{data.eval_split}_patient_macro_AUPRC': patient['macro_AUPRC'],
                     f'{data.eval_split}_label_accuracy_05': fixed['label_accuracy']})
        pd.DataFrame(rows).to_csv(data.output / 'experiment_comparison.csv', index=False)
        return thresholds

    baseline_dir = data.output / 'experiment_2_source'
    baseline_dir.mkdir()
    record('experiment_2_source', 'source', data.baseline, baseline_dir,
           {'checkpoint_epoch': data.source_epoch})
    for seed in SEEDS:
        anchor = ResidualHead('linear_control', seed)
        with torch.no_grad():
            anchor.base.weight.copy_(data.original_weight * torch.as_tensor(data.feature_scale))
            anchor.base.bias.copy_(data.original_bias + data.original_weight @ torch.as_tensor(data.feature_mean))
        original_result = evaluate_head(anchor, data, data.eval_split)
        assert np.allclose(original_result['probabilities'], data.baseline['probabilities'],
                           atol=3e-5, rtol=3e-5)
        directory = data.output / f'linear_control_seed_{seed}'
        directory.mkdir()
        fixed_epochs = locked['epochs'][f'linear_control:{seed}'] if locked else None
        _, info, _ = train_head(anchor, data, seed, directory, fixed_epochs)
        anchor_result = evaluate_head(anchor, data, data.eval_split)
        if locked:
            atomic_torch_save(checkpoint_payload(anchor, data, seed, fixed_epochs, anchor_result),
                              directory / 'best_model.pt')
        record('linear_control', seed, anchor_result, directory, info)

        mapping, pca, encoder_info = fit_complementary_encoder(data, anchor.base)
        weight, bias = mapping
        with torch.no_grad():
            diagnostic_encoder = AngleEncoder()
            diagnostic_encoder.weight.copy_(weight)
            diagnostic_encoder.bias.copy_(bias)
            angles = diagnostic_encoder(data.features['fit'])
            encoder_info.update(angle_mean=float(angles.mean()),
                                angle_std=float(angles.std()),
                                fraction_abs_angle_over_2_5=float((angles.abs() > 2.5).float().mean()))
        np.savez(data.output / f'complementary_encoder_seed_{seed}.npz',
                 weight=weight.numpy(), bias=bias.numpy(),
                 pca_mean=pca.mean_, pca_components=pca.components_,
                 pca_explained_variance_ratio=pca.explained_variance_ratio_)
        save_json(data.output / f'complementary_encoder_seed_{seed}.json', encoder_info)
        print('Encoder seed', seed, encoder_info, flush=True)
        for arm in ARMS[1:]:
            model = ResidualHead(arm, seed)
            model.base.load_state_dict(anchor.base.state_dict())
            model.encoder.weight.copy_(weight)
            model.encoder.bias.copy_(bias)
            initialize_observation_scaling(model, data.features['fit'])
            initial = evaluate_head(model, data, data.eval_split)
            assert np.array_equal(initial['probabilities'], anchor_result['probabilities'])
            directory = data.output / f'{arm}_seed_{seed}'
            directory.mkdir()
            fixed_epochs = locked['epochs'][f'{arm}:{seed}'] if locked else None
            _, info, initial_state = train_head(model, data, seed, directory, fixed_epochs)
            result = evaluate_head(model, data, data.eval_split)
            if locked:
                atomic_torch_save(checkpoint_payload(model, data, seed, fixed_epochs, result),
                                  directory / 'best_model.pt')
            thresholds = record(arm, seed, result, directory, info)
            if arm in QUANTUM_ARMS:
                removed = evaluate_head(model, data, data.eval_split, base_only=True)
                assert np.array_equal(removed['probabilities'], anchor_result['probabilities'])
                saved = copy.deepcopy(model.core.state_dict())
                model.core.load_state_dict({k.removeprefix('core.'): v for k, v in initial_state.items()
                                            if k.startswith('core.')})
                reset = evaluate_head(model, data, data.eval_split)
                model.core.load_state_dict(saved)
                ablation_dir = directory / 'circuit_reset'
                ablation_dir.mkdir()
                _, _, reset_draws = save_evaluation(reset, ablation_dir, data.eval_split,
                                                     fitted_thresholds=thresholds)
                paired.extend(paired_rows_for(result, reset,
                              draws_by_run[(arm, seed)], reset_draws,
                              arm + ' minus circuit_reset', seed))
                ablation_rows.append({'arm': arm, 'seed': seed, 'ablation': 'circuit_reset',
                                      **{name: reset['metrics'][name] for name in METRIC_NAMES}})
                export_bundle(model, data, seed, directory, thresholds)
            del model
            gc.collect()
        del anchor

    comparisons = [('quantum_residual', 'classical_residual'),
                   ('quantum_residual', 'frozen_quantum'),
                   ('quantum_residual', 'linear_control'),
                   ('classical_residual', 'linear_control'),
                   ('frozen_quantum', 'linear_control')]
    for seed in SEEDS:
        for left, right in comparisons:
            paired.extend(paired_rows_for(results[(left, seed)], results[(right, seed)],
                           draws_by_run[(left, seed)], draws_by_run[(right, seed)],
                           left + ' minus ' + right, seed))
    if len(SEEDS) > 1:
        for arm in ARMS:
            directory = data.output / f'{arm}_ensemble'
            directory.mkdir()
            result = data.result(data.eval_split, np.mean(
                [results[(arm, seed)]['probabilities'] for seed in SEEDS], axis=0))
            record(arm, 'ensemble', result, directory, {'checkpoint_epoch': None})
            save_json(directory / 'ensemble_manifest.json', {
                'members': [str(data.output / f'{arm}_seed_{seed}' / 'best_model.pt')
                            for seed in SEEDS], 'weights': [1 / len(SEEDS)] * len(SEEDS)})
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
            values = selected[f'{data.eval_split}_{metric}'].to_numpy(float)
            seed_summary.append({'arm': arm, 'metric': metric, 'mean': values.mean(),
                                 'seed_sd': values.std(ddof=1) if len(values) > 1 else np.nan})
    pd.DataFrame(seed_summary).to_csv(data.output / 'seed_summary.csv', index=False)

    if STAGE == 'develop':
        gate_reasons = []
        if SEEDS != [42, 43, 44]:
            gate_reasons.append('The full prespecified seed set 42,43,44 is required')
        for seed in SEEDS:
            quantum = next(r for r in rows if r['arm'] == 'quantum_residual' and r['seed'] == seed)
            if quantum['checkpoint_epoch'] <= WARMUP_EPOCHS or quantum['core_parameter_change_l2'] <= 1e-6:
                gate_reasons.append(f'Seed {seed}: selected quantum circuit did not learn')
            reset_row = next(r for r in ablation_rows
                             if r['seed'] == seed and r['arm'] == 'quantum_residual')
            if (results[('quantum_residual', seed)]['metrics']['macro_AUPRC']
                    - reset_row['macro_AUPRC']) <= 0:
                gate_reasons.append(f'Seed {seed}: resetting the circuit did not reduce AUPRC')
            q_result = results[('quantum_residual', seed)]['metrics']
            anchor = results[('linear_control', seed)]['metrics']
            classical = results[('classical_residual', seed)]['metrics']
            frozen = results[('frozen_quantum', seed)]['metrics']
            if q_result['macro_AUPRC'] - anchor['macro_AUPRC'] < MEANINGFUL_AP_GAIN:
                gate_reasons.append(f'Seed {seed}: quantum gain over anchor below {MEANINGFUL_AP_GAIN}')
            if q_result['macro_AUPRC'] <= max(classical['macro_AUPRC'], frozen['macro_AUPRC']):
                gate_reasons.append(f'Seed {seed}: quantum did not exceed both controls')
            if q_result['macro_AUROC'] < anchor['macro_AUROC'] - MAX_AUROC_DROP:
                gate_reasons.append(f'Seed {seed}: AUROC dropped more than {MAX_AUROC_DROP}')
        if len(SEEDS) > 1:
            for right in ('linear_control', 'classical_residual', 'frozen_quantum'):
                name = 'quantum_residual minus ' + right
                selected = [r for r in paired if r['seed'] == 'ensemble'
                            and r['comparison'] == name and r['metric'] == 'macro_AUPRC']
                if not selected or not selected[0]['ci_lower'] > 0:
                    gate_reasons.append('Ensemble patient bootstrap interval includes zero versus ' + right)
        protocol = {'eligible_for_confirmation': not gate_reasons,
                    'gate_reasons': gate_reasons, 'settings_fingerprint': fingerprint,
                    'settings': settings, 'source_hash': data.source_hash,
                    'epochs': {f"{r['arm']}:{r['seed']}": int(r['checkpoint_epoch'])
                               for r in rows if r['arm'] in ARMS and isinstance(r['seed'], int)},
                    'thresholds': thresholds_by_run,
                    'inner_split_file': str(data.output / 'inner_patient_splits.csv'),
                    'rule': 'If eligible, confirmation retrains on original training patients for these fixed epochs.'}
        save_json(data.output / 'protocol.json', protocol)
        print('Eligible for confirmation:', protocol['eligible_for_confirmation'])
        for reason in gate_reasons:
            print('Gate:', reason)
    else:
        final_reasons = []
        for seed in SEEDS:
            quantum = next(r for r in rows if r['arm'] == 'quantum_residual' and r['seed'] == seed)
            if quantum['core_parameter_change_l2'] <= 1e-6:
                final_reasons.append(f'Seed {seed}: confirmed circuit change is zero')
            q = results[('quantum_residual', seed)]['metrics']
            reset_row = next(r for r in ablation_rows
                             if r['seed'] == seed and r['arm'] == 'quantum_residual')
            if q['macro_AUPRC'] - reset_row['macro_AUPRC'] <= 0:
                final_reasons.append(f'Seed {seed}: confirmed circuit reset has no effect')
            base = results[('linear_control', seed)]['metrics']
            classical = results[('classical_residual', seed)]['metrics']
            frozen = results[('frozen_quantum', seed)]['metrics']
            if q['macro_AUPRC'] - base['macro_AUPRC'] < MEANINGFUL_AP_GAIN:
                final_reasons.append(f'Seed {seed}: confirmed gain over anchor is too small')
            if q['macro_AUPRC'] <= max(classical['macro_AUPRC'], frozen['macro_AUPRC']):
                final_reasons.append(f'Seed {seed}: confirmed quantum model trails a control')
            if q['macro_AUROC'] < base['macro_AUROC'] - MAX_AUROC_DROP:
                final_reasons.append(f'Seed {seed}: confirmed AUROC loss is too large')
        if len(SEEDS) > 1:
            for right in ('linear_control', 'classical_residual', 'frozen_quantum'):
                name = 'quantum_residual minus ' + right
                selected = [r for r in paired if r['seed'] == 'ensemble'
                            and r['comparison'] == name and r['metric'] == 'macro_AUPRC']
                if not selected or not selected[0]['ci_lower'] > 0:
                    final_reasons.append('Confirmed ensemble interval includes zero versus ' + right)
        save_json(data.output / 'confirmation_manifest.json', {
            'development_protocol': str(PROTOCOL_PATH), 'source_hash': data.source_hash,
            'settings_fingerprint': fingerprint,
            'epochs': locked['epochs'], 'thresholds': locked['thresholds'],
            'eligible_for_final_test': not final_reasons, 'gate_reasons': final_reasons,
            'note': 'Original validation evaluated once per fixed trained arm/seed; no validation epoch selection.'})
        print('Eligible for final test:', not final_reasons)
        for reason in final_reasons:
            print('Gate:', reason)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, METRIC_NAMES[:2]):
        for i, arm in enumerate(ARMS):
            values = [results[(arm, seed)]['metrics'][metric] for seed in SEEDS]
            ax.scatter([i] * len(values), values)
            ax.plot([i - .2, i + .2], [np.mean(values)] * 2, color='black')
        ax.set_xticks(range(len(ARMS)), ARMS, rotation=20, ha='right')
        ax.set_ylabel(data.eval_split + ' ' + metric)
    fig.tight_layout()
    fig.savefig(data.output / 'model_comparison.png', dpi=160)
    plt.close(fig)
    notes = [
        'The default development run never computes original validation or test metrics.',
        'The inherited E2 backbone was trained on inner development patients; their scores are not independent validation.',
        'The E2 checkpoint was selected using original validation, so confirmation scores are also exploratory.',
        'Confirmation uses fixed epochs and thresholds from an eligible development protocol.',
        'Macro metrics are image-level; patient sensitivity aggregates observed images per patient.',
        'Patient bootstrap conditions on selected checkpoints and excludes retraining uncertainty.',
        'Circuit gradients alone do not demonstrate a useful trained quantum contribution.',
        'Analytic quantum simulation does not establish hardware or computational advantage.']
    save_json(data.output / 'summary.json', {
        'settings': settings, 'stage': STAGE, 'comparison': rows,
        'seed_summary': seed_summary, 'paired_comparisons': paired,
        'quantum_ablation_metrics': ablation_rows, 'interpretation': notes})
    lines = ['# Experiment 8 results', '', f'Stage: {STAGE}; evaluated split: {data.eval_split}', '',
             '| Arm | Seed | Epoch | AUROC | AUPRC |', '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['seed']} | {row.get('checkpoint_epoch')} | "
                     f"{row[f'{data.eval_split}_macro_AUROC']:.6f} | "
                     f"{row[f'{data.eval_split}_macro_AUPRC']:.6f} |")
    lines += ['', *['- ' + note for note in notes]]
    (data.output / 'c.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(comparison[['arm', 'seed', 'checkpoint_epoch',
                      f'{data.eval_split}_macro_AUROC', f'{data.eval_split}_macro_AUPRC']].to_string(index=False))
    print('Saved to:', data.output)
    return data.output

def run_final():
    """Evaluate fixed confirmation checkpoints on the untouched test patients once."""
    if CONFIRMATION_RUN is None:
        raise ValueError('Final test requires --confirmation-run')
    confirmed = Path(CONFIRMATION_RUN).expanduser().resolve()
    manifest_path = confirmed / 'confirmation_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Confirmation manifest missing: ' + str(manifest_path))
    consumed_path = confirmed / 'test_evaluation_complete.json'
    if consumed_path.exists():
        raise RuntimeError('This confirmation run already has a completed test evaluation')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('eligible_for_final_test'):
        raise RuntimeError('Confirmed models did not meet the predeclared final-test gate')
    fingerprint = protocol_fingerprint(protocol_settings())
    if manifest.get('settings_fingerprint') != fingerprint:
        raise RuntimeError('Final-stage settings differ from the confirmed protocol')
    for seed in SEEDS:
        for arm in ARMS:
            if not (confirmed / f'{arm}_seed_{seed}' / 'best_model.pt').is_file():
                raise FileNotFoundError(f'Missing confirmed checkpoint: {arm}, seed {seed}')
    data = prepare_data()
    if data.source_hash != manifest['source_hash']:
        raise RuntimeError('Source checkpoint differs from the confirmed run')
    save_json(data.output / 'configuration.json', {
        'experiment': 'Experiment 8 final test', 'confirmation_run': str(confirmed),
        'settings_fingerprint': fingerprint, 'source_hash': data.source_hash,
        'rule': 'Evaluate every predeclared confirmed model and ensemble; no test-based selection.'})
    results, draws_by_run, rows, paired = {}, {}, [], []

    def record(arm, seed, result, directory, thresholds, epoch):
        _, _, draws = save_evaluation(result, directory, 'test',
                                      fitted_thresholds=thresholds)
        patient = patient_sensitivity(result)
        save_json(directory / 'test_patient_sensitivity.json', {
            'aggregation': 'mean probability over observed images; positive if any observed image is positive',
            'metrics': patient})
        results[(arm, seed)], draws_by_run[(arm, seed)] = result, draws
        rows.append({'arm': arm, 'seed': seed, 'checkpoint_epoch': epoch,
                     **{f'test_{name}': result['metrics'][name] for name in METRIC_NAMES},
                     'test_patient_macro_AUPRC': patient['macro_AUPRC']})
        pd.DataFrame(rows).to_csv(data.output / 'experiment_comparison.csv', index=False)

    source_dir = data.output / 'experiment_2_source'
    source_dir.mkdir()
    record('experiment_2_source', 'source', data.baseline, source_dir, None,
           data.source_epoch)
    for seed in SEEDS:
        for arm in ARMS:
            checkpoint_path = confirmed / f'{arm}_seed_{seed}' / 'best_model.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if checkpoint['source_checkpoint_sha256'] != data.source_hash:
                raise RuntimeError('Confirmed head uses a different source checkpoint')
            if checkpoint['epoch'] != manifest['epochs'][f'{arm}:{seed}']:
                raise RuntimeError('Confirmed checkpoint epoch differs from locked protocol')
            if checkpoint['labels'] != LABELS:
                raise RuntimeError('Confirmed checkpoint label order differs')
            if not torch.allclose(checkpoint['feature_mean'], torch.from_numpy(data.feature_mean), atol=1e-6):
                raise RuntimeError('Feature mean differs from confirmed training')
            if not torch.allclose(checkpoint['feature_scale'], torch.from_numpy(data.feature_scale), atol=1e-6):
                raise RuntimeError('Feature scale differs from confirmed training')
            model = ResidualHead(arm, seed)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            result = evaluate_head(model, data, 'test')
            directory = data.output / f'{arm}_seed_{seed}'
            directory.mkdir()
            record(arm, seed, result, directory,
                   manifest['thresholds'][f'{arm}:{seed}'], checkpoint['epoch'])
            del model
            gc.collect()
    comparisons = [('quantum_residual', 'classical_residual'),
                   ('quantum_residual', 'frozen_quantum'),
                   ('quantum_residual', 'linear_control')]
    for seed in SEEDS:
        for left, right in comparisons:
            paired.extend(paired_rows_for(results[(left, seed)], results[(right, seed)],
                          draws_by_run[(left, seed)], draws_by_run[(right, seed)],
                          left + ' minus ' + right, seed))
    if len(SEEDS) > 1:
        for arm in ARMS:
            directory = data.output / f'{arm}_ensemble'
            directory.mkdir()
            result = data.result('test', np.mean(
                [results[(arm, seed)]['probabilities'] for seed in SEEDS], axis=0))
            record(arm, 'ensemble', result, directory,
                   manifest['thresholds'][f'{arm}:ensemble'], None)
        for left, right in comparisons:
            paired.extend(paired_rows_for(results[(left, 'ensemble')], results[(right, 'ensemble')],
                          draws_by_run[(left, 'ensemble')], draws_by_run[(right, 'ensemble')],
                          left + ' minus ' + right, 'ensemble'))
    pd.DataFrame(paired).to_csv(data.output / 'paired_patient_bootstrap_differences.csv', index=False)
    save_json(data.output / 'summary.json', {
        'stage': 'final', 'confirmation_run': str(confirmed), 'comparison': rows,
        'paired_comparisons': paired,
        'note': 'All predeclared arms reported; test used for final evaluation, not model selection.'})
    lines = ['# Experiment 8 final test', '',
             'All models come from the fixed confirmation run: ' + str(confirmed), '',
             '| Arm | Seed | Epoch | Test AUROC | Test AUPRC |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['checkpoint_epoch']} | "
                     f"{row['test_macro_AUROC']:.6f} | {row['test_macro_AUPRC']:.6f} |")
    (data.output / 'c.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save_json(consumed_path, {'final_output': str(data.output),
                              'completed_utc': datetime.now(timezone.utc).isoformat()})
    print('Final test results saved to:', data.output)
    return data.output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('develop', 'confirm', 'final'), default='develop')
    parser.add_argument('--protocol', type=Path, help='Eligible development protocol.json; required for confirmation')
    parser.add_argument('--confirmation-run', type=Path, help='Completed confirmation output; required for final test')
    parser.add_argument('-f', help=argparse.SUPPRESS)
    args = parser.parse_args()
    STAGE, PROTOCOL_PATH, CONFIRMATION_RUN = args.stage, args.protocol, args.confirmation_run
    if STAGE == 'confirm' and PROTOCOL_PATH is None:
        parser.error('--stage confirm requires --protocol')
    if STAGE == 'final' and CONFIRMATION_RUN is None:
        parser.error('--stage final requires --confirmation-run')
    if len(SEEDS) == 0 or len(SEEDS) != len(set(SEEDS)):
        parser.error('E8_SEEDS must contain unique seeds')
    assert ANGLE_DIM == 48 and N_QUBITS == 8 and Q_LAYERS == 3
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(min(CPU_HEAD_THREADS, os.cpu_count() or 1))
    ensure_quantum_dependency()
    seed_everything(42)
    print('EXPERIMENT 8 | stage:', STAGE, '| extraction:', EXTRACT_DEVICE,
          '| heads: CPU | seeds:', SEEDS)
    print('Torch:', torch.__version__, '| torchvision:', torchvision.__version__,
          '| PennyLane:', qml.__version__)
    OUTPUT_DIR = run_final() if STAGE == 'final' else run_experiment()
