from pathlib import Path

root = Path("/workspace/unzippedarchive/unzippedarchive/xray_images/x-ray images")

jpgs = [
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
]

print(f"Total JPG/JPEG files: {len(jpgs):,}")

from pathlib import Path
import pandas as pd

BASE = Path("/workspace/unzippedarchive/unzippedarchive")

IMG_DIR = BASE / "xray_images" / "x-ray images"
CSV_DIR = BASE / "csvs"

# ============================================================
# 1. LOAD FILES
# ============================================================

images = list(IMG_DIR.glob("*.jpg"))

print("=" * 70)
print("IMAGE DATASET")
print("=" * 70)
print(f"Images found: {len(images):,}")

# Filename format:
# 0006ffca-fee7bc9c-bb4e3942-4e61b867-7e77af78.jpg
#
# In MIMIC-CXR-JPG, this filename corresponds to dicom_id.

image_df = pd.DataFrame({
    "image_path": [str(p) for p in images],
    "dicom_id": [p.stem for p in images]
})

# ============================================================
# 2. LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    CSV_DIR / "mimic-cxr-2.0.0-metadata.csv"
)

print("\nMetadata rows:", len(metadata))

# ============================================================
# 3. LOAD CHEXPERT LABELS
# ============================================================

labels = pd.read_csv(
    CSV_DIR / "mimic-cxr-2.0.0-chexpert.csv"
)

print("Label rows:", len(labels))

# ============================================================
# 4. MATCH IMAGE -> METADATA
# ============================================================

merged = image_df.merge(
    metadata,
    on="dicom_id",
    how="left",
    indicator=True
)

matched_metadata = (merged["_merge"] == "both").sum()
unmatched_metadata = (merged["_merge"] == "left_only").sum()

print("\n" + "=" * 70)
print("IMAGE -> METADATA MATCH")
print("=" * 70)
print(f"Matched metadata  : {matched_metadata:,}")
print(f"Unmatched         : {unmatched_metadata:,}")

# Remove merge indicator
merged = merged.drop(columns="_merge")

# ============================================================
# 5. MATCH -> CHEXPERT LABELS
# ============================================================

merged = merged.merge(
    labels,
    on=["subject_id", "study_id"],
    how="left",
    indicator=True,
    suffixes=("", "_label")
)

matched_labels = (merged["_merge"] == "both").sum()
unmatched_labels = (merged["_merge"] == "left_only").sum()

print("\n" + "=" * 70)
print("IMAGE -> LABEL MATCH")
print("=" * 70)
print(f"Matched labels    : {matched_labels:,}")
print(f"Unmatched labels  : {unmatched_labels:,}")

merged = merged.drop(columns="_merge")

# ============================================================
# 6. LABEL COVERAGE
# ============================================================

label_columns = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices"
]

print("\n" + "=" * 70)
print("LABEL COVERAGE")
print("=" * 70)

for col in label_columns:
    if col in merged.columns:
        available = merged[col].notna().sum()
        print(f"{col:30} {available:,}")

# ============================================================
# 7. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

for col in label_columns:
    if col in merged.columns:
        print(f"\n--- {col} ---")
        print(merged[col].value_counts(dropna=False).to_string())

# ============================================================
# 8. SAVE MATCHED DATASET INDEX
# ============================================================

output = BASE / "xray_image_label_index.csv"

merged.to_csv(output, index=False)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)
print(output)

# ============================================================
# MIMIC-CXR MULTILABEL TRAINING — SINGLE NOTEBOOK CELL
# ============================================================
# Baseline:
# - AP + PA frontal X-rays only
# - 13 labels (No Finding excluded)
# - patient-level 70/15/15 split
# - NaN -> 0 (unmentioned treated as negative)
# - -1 -> ignored
# - DenseNet121, ImageNet pretrained
# - mixed precision
# - BCEWithLogitsLoss with masked uncertain labels
# - per-class AUROC / AUPRC
# - early stopping
# - threshold tuning on validation set
# - final untouched test evaluation
# - truncated JPEG handling
# - full image decode validation before split/training
# - saves checkpoint, splits, predictions, metrics, plots
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import gc
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFile

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 0.1 PIL JPEG SAFETY
# ============================================================

# Allows slightly truncated JPEG files to load.
# Example fixed error:
# OSError: image file is truncated (14 bytes not processed)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 1. CONFIG
# ============================================================

INDEX_CSV = "/workspace/unzippedarchive/unzippedarchive/xray_image_label_index.csv"

OUTPUT_DIR = Path(
    "/workspace/unzippedarchive/unzippedarchive/xray_training_output"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42

IMAGE_SIZE = 320

BATCH_SIZE = 24

NUM_WORKERS = 4

EPOCHS = 25

PATIENCE = 5

LR = 1e-4

WEIGHT_DECAY = 1e-4

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

SEARCH_SPLITS = 1000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


print("=" * 80)
print("CONFIG")
print("=" * 80)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "Torchvision:",
    torchvision.__version__,
)

print(
    "Image size:",
    IMAGE_SIZE,
)

print(
    "Batch size:",
    BATCH_SIZE,
)

print(
    "Workers:",
    NUM_WORKERS,
)

print()


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Faster convolution selection.
    torch.backends.cudnn.benchmark = True

    if hasattr(
        torch,
        "set_float32_matmul_precision",
    ):

        torch.set_float32_matmul_precision(
            "high"
        )


seed_everything(SEED)


# ============================================================
# 3. LOAD INDEX
# ============================================================

df = pd.read_csv(
    INDEX_CSV
)

print("=" * 80)
print("RAW INDEX")
print("=" * 80)

print(
    "Rows:",
    len(df),
)

print(
    "Patients:",
    df["subject_id"].nunique(),
)

print(
    "Studies:",
    df["study_id"].nunique(),
)

print(
    "DICOM IDs:",
    df["dicom_id"].nunique(),
)

print()


# ============================================================
# 4. BASIC INDEX VALIDATION
# ============================================================

required_cols = [
    "image_path",
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
] + LABELS

missing_cols = [
    c
    for c in required_cols
    if c not in df.columns
]

if missing_cols:

    raise RuntimeError(
        f"Missing required columns: {missing_cols}"
    )


# ----------------------------
# DICOM ID duplicate check
# ----------------------------

if df["dicom_id"].duplicated().any():

    dupes = df[
        df["dicom_id"].duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Duplicate dicom_id values exist "
        f"in index CSV: {len(dupes)} rows"
    )


# ============================================================
# 5. IMAGE VALIDATION
# ============================================================
#
# We do two checks:
#
# 1. File exists
# 2. PIL can fully decode it
#
# Slightly truncated JPEGs should load because
# LOAD_TRUNCATED_IMAGES=True.
#
# Images that still cannot decode are removed BEFORE
# patient splitting so all later counts remain consistent.
# ============================================================

print("=" * 80)
print("IMAGE VALIDATION")
print("=" * 80)

missing_files = []

bad_images = []

for p in tqdm(
    df["image_path"],
    desc="Validating JPEGs",
):

    # -------------------------------------
    # File existence
    # -------------------------------------

    if not os.path.exists(p):

        missing_files.append(
            p
        )

        continue

    # -------------------------------------
    # Full PIL decode
    # -------------------------------------

    try:

        with Image.open(p) as img:

            img.load()

    except Exception as e:

        bad_images.append(
            (
                p,
                repr(e),
            )
        )


print()

print(
    "Missing files:",
    len(missing_files),
)

print(
    "Unreadable files:",
    len(bad_images),
)


problem_paths = set(
    missing_files
)

problem_paths.update(
    p
    for p, _ in bad_images
)


if missing_files:

    print()
    print("Example missing files:")

    for p in missing_files[:10]:

        print(
            " -",
            p,
        )


if bad_images:

    print()
    print("Example unreadable files:")

    for p, error in bad_images[:20]:

        print(
            " -",
            p,
        )

        print(
            "   ",
            error,
        )


if problem_paths:

    original_count = len(df)

    df = df[
        ~df["image_path"].isin(
            problem_paths
        )
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    removed = (
        original_count
        - len(df)
    )

    print()
    print(
        f"Removed {removed} unusable images."
    )

else:

    print(
        "All indexed images exist and decode successfully."
    )


print()

print(
    "Remaining dataset rows:",
    len(df),
)

print()


# ============================================================
# 6. FILTER TO FRONTAL IMAGES
# ============================================================

df = df[
    df["ViewPosition"].isin(
        [
            "AP",
            "PA",
        ]
    )
].copy()

df.reset_index(
    drop=True,
    inplace=True,
)


print("=" * 80)
print("FRONTAL DATASET")
print("=" * 80)

print(
    "Images:",
    len(df),
)

print(
    "Patients:",
    df["subject_id"].nunique(),
)

print(
    "Studies:",
    df["study_id"].nunique(),
)

print()

print(
    df["ViewPosition"].value_counts(
        dropna=False
    )
)

print()


if len(df) == 0:

    raise RuntimeError(
        "No AP/PA frontal images remain after filtering."
    )


# ============================================================
# 7. LABEL SANITY CHECK
# ============================================================

print("=" * 80)
print("LABEL DISTRIBUTION — FRONTAL ONLY")
print("=" * 80)

for label in LABELS:

    s = df[label]

    pos = int(
        (s == 1).sum()
    )

    neg = int(
        (s == 0).sum()
    )

    unc = int(
        (s == -1).sum()
    )

    missing = int(
        s.isna().sum()
    )

    print(
        f"{label:30s} "
        f"pos={pos:4d} "
        f"explicit_neg={neg:4d} "
        f"uncertain={unc:4d} "
        f"unmentioned={missing:4d}"
    )


# ============================================================
# 8. PATIENT-LEVEL SPLIT
# ============================================================
#
# Searches many random patient-level splits and chooses
# one whose per-label prevalence best resembles the full
# frontal dataset.
#
# No image-level random splitting is performed.
# ============================================================

patients = df[
    "subject_id"
].unique()

n_patients = len(
    patients
)

n_train = round(
    n_patients
    * TRAIN_FRACTION
)

n_val = round(
    n_patients
    * VAL_FRACTION
)


target_prevalence = {}


for label in LABELS:

    vals = df[
        label
    ].copy()

    # NaN = unmentioned -> negative
    vals = vals.fillna(
        0
    )

    # -1 = uncertain -> excluded
    valid = (
        vals != -1
    )

    if valid.sum() > 0:

        target_prevalence[
            label
        ] = float(
            (
                vals[
                    valid
                ] == 1
            ).mean()
        )

    else:

        target_prevalence[
            label
        ] = 0.0


def score_split(
    train_ids,
    val_ids,
    test_ids,
):

    score = 0.0

    for split_ids in [
        train_ids,
        val_ids,
        test_ids,
    ]:

        part = df[
            df["subject_id"].isin(
                split_ids
            )
        ]

        for label in LABELS:

            vals = part[
                label
            ].fillna(
                0
            )

            valid = (
                vals != -1
            )

            if valid.sum() == 0:

                score += 100.0

                continue

            pos_count = int(
                (
                    vals[
                        valid
                    ] == 1
                ).sum()
            )

            neg_count = int(
                (
                    vals[
                        valid
                    ] == 0
                ).sum()
            )

            prevalence = float(
                (
                    vals[
                        valid
                    ] == 1
                ).mean()
            )

            target = target_prevalence[
                label
            ]

            score += abs(
                prevalence
                - target
            )

            # --------------------------------
            # Penalize complete loss of class
            # --------------------------------

            if pos_count == 0:

                score += 10.0

            if neg_count == 0:

                score += 10.0

    return score


best_score = float(
    "inf"
)

best_split = None

rng = np.random.default_rng(
    SEED
)


print()

print(
    "Searching for balanced patient-level split..."
)


for _ in tqdm(
    range(
        SEARCH_SPLITS
    )
):

    shuffled = rng.permutation(
        patients
    )

    train_ids = shuffled[
        :n_train
    ]

    val_ids = shuffled[
        n_train:
        n_train + n_val
    ]

    test_ids = shuffled[
        n_train + n_val:
    ]

    score = score_split(
        train_ids,
        val_ids,
        test_ids,
    )

    if score < best_score:

        best_score = score

        best_split = (
            set(
                train_ids
            ),
            set(
                val_ids
            ),
            set(
                test_ids
            ),
        )


if best_split is None:

    raise RuntimeError(
        "Could not construct patient-level split."
    )


train_ids, val_ids, test_ids = (
    best_split
)


train_df = df[
    df["subject_id"].isin(
        train_ids
    )
].copy()

val_df = df[
    df["subject_id"].isin(
        val_ids
    )
].copy()

test_df = df[
    df["subject_id"].isin(
        test_ids
    )
].copy()


# ============================================================
# 9. LEAKAGE ASSERTIONS
# ============================================================

assert train_ids.isdisjoint(
    val_ids
)

assert train_ids.isdisjoint(
    test_ids
)

assert val_ids.isdisjoint(
    test_ids
)


assert set(
    train_df["dicom_id"]
).isdisjoint(
    set(
        val_df["dicom_id"]
    )
)

assert set(
    train_df["dicom_id"]
).isdisjoint(
    set(
        test_df["dicom_id"]
    )
)

assert set(
    val_df["dicom_id"]
).isdisjoint(
    set(
        test_df["dicom_id"]
    )
)


print()

print("=" * 80)
print("PATIENT SPLIT")
print("=" * 80)


for name, part in [
    (
        "TRAIN",
        train_df,
    ),
    (
        "VAL",
        val_df,
    ),
    (
        "TEST",
        test_df,
    ),
]:

    print(
        f"{name:6s} "
        f"images={len(part):5d} "
        f"patients="
        f"{part['subject_id'].nunique():4d} "
        f"studies="
        f"{part['study_id'].nunique():4d}"
    )


print()

print(
    "Patient leakage check: PASSED"
)


# ============================================================
# 10. SAVE SPLITS
# ============================================================

train_df[
    "split"
] = "train"

val_df[
    "split"
] = "val"

test_df[
    "split"
] = "test"


split_df = pd.concat(
    [
        train_df,
        val_df,
        test_df,
    ],
    ignore_index=True,
)


split_csv_path = (
    OUTPUT_DIR
    / "patient_level_splits.csv"
)

split_df.to_csv(
    split_csv_path,
    index=False,
)


print(
    "Saved:",
    split_csv_path,
)


# ============================================================
# 11. PRINT SPLIT LABEL COUNTS
# ============================================================

print()

print("=" * 80)
print("LABEL COUNTS PER SPLIT")
print("=" * 80)


for label in LABELS:

    row = []

    for name, part in [
        (
            "train",
            train_df,
        ),
        (
            "val",
            val_df,
        ),
        (
            "test",
            test_df,
        ),
    ]:

        vals = part[
            label
        ].fillna(
            0
        )

        pos = int(
            (
                vals == 1
            ).sum()
        )

        neg = int(
            (
                vals == 0
            ).sum()
        )

        unc = int(
            (
                vals == -1
            ).sum()
        )

        row.append(
            f"{name}: "
            f"+{pos}/"
            f"-{neg}/"
            f"u{unc}"
        )

    print(
        f"{label:30s} "
        f"{' | '.join(row)}"
    )


# ============================================================
# 12. TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=(
                transforms
                .InterpolationMode
                .BILINEAR
            ),
        ),

        transforms.RandomAffine(
            degrees=5,
            translate=(
                0.02,
                0.02,
            ),
            scale=(
                0.95,
                1.05,
            ),
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


eval_transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=(
                transforms
                .InterpolationMode
                .BILINEAR
            ),
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 13. DATASET CLASS
# ============================================================

class XRayDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[
            idx
        ]

        path = row[
            "image_path"
        ]

        # -------------------------------------
        # Safe PIL load
        # -------------------------------------

        try:

            with Image.open(
                path
            ) as im:

                img = im.convert(
                    "RGB"
                )

        except Exception as e:

            raise RuntimeError(
                "Failed to load image:\n"
                f"{path}\n"
                f"Error: {repr(e)}"
            ) from e


        if self.transform is not None:

            img = self.transform(
                img
            )


        raw_labels = row[
            LABELS
        ].values.astype(
            np.float32
        )


        # -------------------------------------
        # LABEL POLICY
        #
        #  1   -> positive
        #  0   -> negative
        #  NaN -> negative / unmentioned
        # -1   -> ignored using mask
        # -------------------------------------

        uncertain_mask = (
            raw_labels
            == -1
        )


        labels = np.nan_to_num(
            raw_labels,
            nan=0.0,
        )


        labels[
            uncertain_mask
        ] = 0.0


        mask = (
            ~uncertain_mask
        ).astype(
            np.float32
        )


        return (
            img,
            torch.tensor(
                labels,
                dtype=torch.float32,
            ),
            torch.tensor(
                mask,
                dtype=torch.float32,
            ),
            row[
                "dicom_id"
            ],
        )


# ============================================================
# 14. DATA LOADERS
# ============================================================

train_dataset = XRayDataset(
    train_df,
    transform=train_transform,
)

val_dataset = XRayDataset(
    val_df,
    transform=eval_transform,
)

test_dataset = XRayDataset(
    test_df,
    transform=eval_transform,
)


loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(
        NUM_WORKERS
        > 0
    ),
)


train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    drop_last=False,
    **loader_kwargs,
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)


print()

print(
    "Train batches:",
    len(
        train_loader
    ),
)

print(
    "Val batches:",
    len(
        val_loader
    ),
)

print(
    "Test batches:",
    len(
        test_loader
    ),
)


# ============================================================
# 15. CLASS WEIGHTS
# ============================================================

pos_weights = []


for label in LABELS:

    vals = train_df[
        label
    ].fillna(
        0
    )

    vals = vals[
        vals != -1
    ]

    pos = int(
        (
            vals == 1
        ).sum()
    )

    neg = int(
        (
            vals == 0
        ).sum()
    )


    if pos == 0:

        weight = 1.0

    else:

        weight = (
            neg / pos
        )


    # Prevent extreme rare-class weighting
    weight = np.clip(
        weight,
        1.0,
        10.0,
    )


    pos_weights.append(
        weight
    )


    print(
        f"{label:30s} "
        f"train pos={pos:4d} "
        f"neg={neg:4d} "
        f"pos_weight="
        f"{weight:.3f}"
    )


pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# 16. MODEL
# ============================================================

weights = (
    DenseNet121_Weights
    .IMAGENET1K_V1
)


model = densenet121(
    weights=weights
)


in_features = (
    model
    .classifier
    .in_features
)


model.classifier = nn.Linear(
    in_features,
    len(
        LABELS
    ),
)


model = model.to(
    DEVICE
)


print()

print("=" * 80)
print("MODEL")
print("=" * 80)

print(
    model.__class__.__name__
)

print(
    "Output classes:",
    len(
        LABELS
    ),
)


# ============================================================
# 17. LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    reduction="none",
    pos_weight=pos_weights,
)


def masked_bce_loss(
    logits,
    targets,
    mask,
):

    raw_loss = criterion(
        logits,
        targets,
    )

    masked_loss = (
        raw_loss
        * mask
    )

    denom = mask.sum().clamp(
        min=1.0
    )

    return (
        masked_loss.sum()
        / denom
    )


# ============================================================
# 18. OPTIMIZER + SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        torch.cuda.is_available()
    ),
)


# ============================================================
# 19. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(
    targets,
    probs,
    masks,
):

    results = {}

    aurocs = []

    auprcs = []


    for i, label in enumerate(
        LABELS
    ):

        valid = (
            masks[
                :,
                i
            ]
            == 1
        )


        y_true = targets[
            valid,
            i
        ]


        y_prob = probs[
            valid,
            i
        ]


        if len(
            np.unique(
                y_true
            )
        ) < 2:

            auc = np.nan

            ap = np.nan

        else:

            auc = roc_auc_score(
                y_true,
                y_prob,
            )

            ap = average_precision_score(
                y_true,
                y_prob,
            )


        results[
            label
        ] = {
            "AUROC": auc,
            "AUPRC": ap,
        }


        if not np.isnan(
            auc
        ):

            aurocs.append(
                auc
            )


        if not np.isnan(
            ap
        ):

            auprcs.append(
                ap
            )


    results[
        "macro_AUROC"
    ] = (
        float(
            np.mean(
                aurocs
            )
        )
        if aurocs
        else np.nan
    )


    results[
        "macro_AUPRC"
    ] = (
        float(
            np.mean(
                auprcs
            )
        )
        if auprcs
        else np.nan
    )


    return results


# ============================================================
# 20. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
):

    model.train()

    running_loss = 0.0

    samples = 0


    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )


    for (
        images,
        targets,
        masks,
        _,
    ) in progress:


        images = images.to(
            DEVICE,
            non_blocking=True,
        )


        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )


        masks = masks.to(
            DEVICE,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type=(
                DEVICE.type
            ),
            dtype=torch.float16,
            enabled=(
                torch.cuda.is_available()
            ),
        ):

            logits = model(
                images
            )

            loss = masked_bce_loss(
                logits,
                targets,
                masks,
            )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        scaler.step(
            optimizer
        )

        scaler.update()


        bs = images.size(
            0
        )


        running_loss += (
            loss.item()
            * bs
        )

        samples += bs


        progress.set_postfix(
            loss=(
                f"{loss.item():.4f}"
            ),
            lr=(
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ),
        )


    return (
        running_loss
        / max(
            samples,
            1,
        )
    )


# ============================================================
# 21. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    model.eval()

    running_loss = 0.0

    samples = 0

    all_targets = []

    all_probs = []

    all_masks = []

    all_ids = []


    progress = tqdm(
        loader,
        desc="Evaluating",
        leave=False,
    )


    for (
        images,
        targets,
        masks,
        dicom_ids,
    ) in progress:


        images = images.to(
            DEVICE,
            non_blocking=True,
        )


        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )


        masks = masks.to(
            DEVICE,
            non_blocking=True,
        )


        with torch.autocast(
            device_type=(
                DEVICE.type
            ),
            dtype=torch.float16,
            enabled=(
                torch.cuda.is_available()
            ),
        ):

            logits = model(
                images
            )

            loss = masked_bce_loss(
                logits,
                targets,
                masks,
            )


        probs = torch.sigmoid(
            logits
        )


        bs = images.size(
            0
        )


        running_loss += (
            loss.item()
            * bs
        )

        samples += bs


        all_targets.append(
            targets
            .detach()
            .cpu()
            .numpy()
        )


        all_probs.append(
            probs
            .detach()
            .cpu()
            .numpy()
        )


        all_masks.append(
            masks
            .detach()
            .cpu()
            .numpy()
        )


        all_ids.extend(
            list(
                dicom_ids
            )
        )


    targets = np.concatenate(
        all_targets,
        axis=0,
    )


    probs = np.concatenate(
        all_probs,
        axis=0,
    )


    masks = np.concatenate(
        all_masks,
        axis=0,
    )


    metrics = calculate_metrics(
        targets,
        probs,
        masks,
    )


    return (
        running_loss
        / max(
            samples,
            1,
        ),
        metrics,
        targets,
        probs,
        masks,
        all_ids,
    )


# ============================================================
# 22. TRAINING LOOP
# ============================================================

history = []

best_metric = -np.inf

epochs_without_improvement = 0

best_checkpoint = (
    OUTPUT_DIR
    / "best_model.pt"
)


print()

print("=" * 80)
print("TRAINING")
print("=" * 80)


for epoch in range(
    1,
    EPOCHS + 1,
):


    print()

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        "-" * 80
    )


    train_loss = train_one_epoch(
        model,
        train_loader,
    )


    (
        val_loss,
        val_metrics,
        _,
        _,
        _,
        _,
    ) = evaluate(
        model,
        val_loader,
    )


    macro_auc = val_metrics[
        "macro_AUROC"
    ]

    macro_ap = val_metrics[
        "macro_AUPRC"
    ]


    scheduler.step(
        macro_auc
    )


    current_lr = optimizer.param_groups[
        0
    ][
        "lr"
    ]


    print(
        f"Train loss : "
        f"{train_loss:.5f}"
    )

    print(
        f"Val loss   : "
        f"{val_loss:.5f}"
    )

    print(
        f"Macro AUROC: "
        f"{macro_auc:.5f}"
    )

    print(
        f"Macro AUPRC: "
        f"{macro_ap:.5f}"
    )

    print(
        f"LR         : "
        f"{current_lr:.2e}"
    )


    history.append(
        {
            "epoch": epoch,
            "train_loss": (
                train_loss
            ),
            "val_loss": (
                val_loss
            ),
            "macro_AUROC": (
                macro_auc
            ),
            "macro_AUPRC": (
                macro_ap
            ),
            "lr": (
                current_lr
            ),
        }
    )


    # -----------------------------------------
    # Save best validation AUROC checkpoint
    # -----------------------------------------

    if macro_auc > best_metric:

        best_metric = (
            macro_auc
        )

        epochs_without_improvement = 0


        torch.save(
            {
                "epoch": (
                    epoch
                ),
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "macro_AUROC": (
                    macro_auc
                ),
                "macro_AUPRC": (
                    macro_ap
                ),
                "labels": (
                    LABELS
                ),
                "image_size": (
                    IMAGE_SIZE
                ),
            },
            best_checkpoint,
        )


        print(
            "✓ Saved new best model: "
            f"{macro_auc:.5f}"
        )


    else:

        epochs_without_improvement += 1


        print(
            "No improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break


# ============================================================
# 23. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_path = (
    OUTPUT_DIR
    / "training_history.csv"
)


history_df.to_csv(
    history_path,
    index=False,
)


# ============================================================
# 24. TRAINING CURVES
# ============================================================

plt.figure(
    figsize=(
        8,
        5,
    )
)

plt.plot(
    history_df[
        "epoch"
    ],
    history_df[
        "train_loss"
    ],
    label="Train",
)

plt.plot(
    history_df[
        "epoch"
    ],
    history_df[
        "val_loss"
    ],
    label="Validation",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training / Validation Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "loss_curve.png",
    dpi=150,
)


plt.show()


plt.figure(
    figsize=(
        8,
        5,
    )
)

plt.plot(
    history_df[
        "epoch"
    ],
    history_df[
        "macro_AUROC"
    ],
    label="Macro AUROC",
)

plt.plot(
    history_df[
        "epoch"
    ],
    history_df[
        "macro_AUPRC"
    ],
    label="Macro AUPRC",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Metric"
)

plt.title(
    "Validation Metrics"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "validation_metrics.png",
    dpi=150,
)


plt.show()


# ============================================================
# 25. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    best_checkpoint,
    map_location=DEVICE,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print()

print(
    "Loaded best checkpoint from epoch:",
    checkpoint[
        "epoch"
    ],
)


print(
    "Best validation macro AUROC:",
    checkpoint[
        "macro_AUROC"
    ],
)


# ============================================================
# 26. RUN VALIDATION AGAIN
# ============================================================

(
    val_loss,
    val_metrics,
    val_targets,
    val_probs,
    val_masks,
    val_ids,
) = evaluate(
    model,
    val_loader,
)


# ============================================================
# 27. OPTIMIZE THRESHOLDS ON VALIDATION SET
# ============================================================

thresholds = {}


print()

print("=" * 80)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 80)


for i, label in enumerate(
    LABELS
):


    valid = (
        val_masks[
            :,
            i
        ]
        == 1
    )


    y_true = val_targets[
        valid,
        i
    ]


    y_prob = val_probs[
        valid,
        i
    ]


    if len(
        np.unique(
            y_true
        )
    ) < 2:

        threshold = 0.5

        best_f1 = np.nan


    else:

        best_f1 = -1.0

        threshold = 0.5


        for t in np.linspace(
            0.05,
            0.95,
            181,
        ):


            pred = (
                y_prob
                >= t
            ).astype(
                int
            )


            score = f1_score(
                y_true,
                pred,
                zero_division=0,
            )


            if score > best_f1:

                best_f1 = (
                    score
                )

                threshold = float(
                    t
                )


    thresholds[
        label
    ] = threshold


    print(
        f"{label:30s} "
        f"threshold="
        f"{threshold:.3f} "
        f"val_F1="
        f"{best_f1:.3f}"
    )


with open(
    OUTPUT_DIR
    / "thresholds.json",
    "w",
) as f:

    json.dump(
        thresholds,
        f,
        indent=2,
    )


# ============================================================
# 28. FINAL TEST
# ============================================================

print()

print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)


(
    test_loss,
    test_metrics,
    test_targets,
    test_probs,
    test_masks,
    test_ids,
) = evaluate(
    model,
    test_loader,
)


print(
    "Test loss:",
    round(
        test_loss,
        5,
    ),
)


print(
    "Test macro AUROC:",
    round(
        test_metrics[
            "macro_AUROC"
        ],
        5,
    ),
)


print(
    "Test macro AUPRC:",
    round(
        test_metrics[
            "macro_AUPRC"
        ],
        5,
    ),
)


# ============================================================
# 29. PER-CLASS FINAL METRICS
# ============================================================

metric_rows = []


for i, label in enumerate(
    LABELS
):


    valid = (
        test_masks[
            :,
            i
        ]
        == 1
    )


    y_true = test_targets[
        valid,
        i
    ]


    y_prob = test_probs[
        valid,
        i
    ]


    threshold = thresholds[
        label
    ]


    y_pred = (
        y_prob
        >= threshold
    ).astype(
        int
    )


    if len(
        np.unique(
            y_true
        )
    ) >= 2:

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

        ap = average_precision_score(
            y_true,
            y_prob,
        )

    else:

        auc = np.nan

        ap = np.nan


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    specificity = (
        tn
        / (
            tn
            + fp
        )
        if (
            tn
            + fp
        ) > 0
        else np.nan
    )


    metric_rows.append(
        {
            "label": (
                label
            ),
            "AUROC": (
                auc
            ),
            "AUPRC": (
                ap
            ),
            "threshold": (
                threshold
            ),
            "precision": (
                precision
            ),
            "sensitivity_recall": (
                recall
            ),
            "specificity": (
                specificity
            ),
            "F1": (
                f1
            ),
            "TP": (
                int(
                    tp
                )
            ),
            "FP": (
                int(
                    fp
                )
            ),
            "TN": (
                int(
                    tn
                )
            ),
            "FN": (
                int(
                    fn
                )
            ),
            "n_valid": (
                len(
                    y_true
                )
            ),
        }
    )


metrics_df = pd.DataFrame(
    metric_rows
)


metrics_df.to_csv(
    OUTPUT_DIR
    / "test_metrics.csv",
    index=False,
)


print()

print(
    metrics_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 30. SAVE TEST PREDICTIONS
# ============================================================

pred_df = pd.DataFrame(
    {
        "dicom_id": (
            test_ids
        )
    }
)


for i, label in enumerate(
    LABELS
):


    pred_df[
        f"{label}_prob"
    ] = (
        test_probs[
            :,
            i
        ]
    )


    pred_df[
        f"{label}_target"
    ] = (
        test_targets[
            :,
            i
        ]
    )


    pred_df[
        f"{label}_mask"
    ] = (
        test_masks[
            :,
            i
        ]
    )


    pred_df[
        f"{label}_prediction"
    ] = (
        test_probs[
            :,
            i
        ]
        >= thresholds[
            label
        ]
    ).astype(
        int
    )


pred_df.to_csv(
    OUTPUT_DIR
    / "test_predictions.csv",
    index=False,
)


# ============================================================
# 31. SAVE SUMMARY
# ============================================================

summary = {

    "index_csv": (
        INDEX_CSV
    ),

    "image_size": (
        IMAGE_SIZE
    ),

    "batch_size": (
        BATCH_SIZE
    ),

    "num_workers": (
        NUM_WORKERS
    ),

    "epochs_requested": (
        EPOCHS
    ),

    "epochs_completed": (
        len(
            history_df
        )
    ),

    "best_epoch": (
        int(
            checkpoint[
                "epoch"
            ]
        )
    ),

    "best_validation_macro_AUROC": (
        float(
            checkpoint[
                "macro_AUROC"
            ]
        )
    ),

    "test_macro_AUROC": (
        float(
            test_metrics[
                "macro_AUROC"
            ]
        )
    ),

    "test_macro_AUPRC": (
        float(
            test_metrics[
                "macro_AUPRC"
            ]
        )
    ),

    "train_images": (
        len(
            train_df
        )
    ),

    "val_images": (
        len(
            val_df
        )
    ),

    "test_images": (
        len(
            test_df
        )
    ),

    "train_patients": (
        int(
            train_df[
                "subject_id"
            ].nunique()
        )
    ),

    "val_patients": (
        int(
            val_df[
                "subject_id"
            ].nunique()
        )
    ),

    "test_patients": (
        int(
            test_df[
                "subject_id"
            ].nunique()
        )
    ),

    "removed_missing_or_bad_images": (
        int(
            len(
                problem_paths
            )
        )
    ),

    "labels": (
        LABELS
    ),

}


with open(
    OUTPUT_DIR
    / "summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# 32. FINAL OUTPUT
# ============================================================

print()

print("=" * 80)
print("DONE")
print("=" * 80)


print(
    "Outputs saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Files:"
)


for p in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        " -",
        p.name,
    )


print()

print(
    "Best epoch:",
    checkpoint[
        "epoch"
    ],
)


print(
    "Validation macro AUROC:",
    round(
        checkpoint[
            "macro_AUROC"
        ],
        4,
    ),
)


print(
    "Test macro AUROC:",
    round(
        test_metrics[
            "macro_AUROC"
        ],
        4,
    ),
)


print(
    "Test macro AUPRC:",
    round(
        test_metrics[
            "macro_AUPRC"
        ],
        4,
    ),
)


print()

print(
    "Removed missing/unreadable images:",
    len(
        problem_paths
    ),
)


# ============================================================
# 33. CLEANUP
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# MIMIC-CXR MULTILABEL TRAINING — EXPERIMENT 2
# SINGLE NOTEBOOK CELL
# ============================================================
#
# EXPERIMENT 2 CHANGES
# ------------------------------------------------------------
# - SAME exact patient split as Experiment 1
# - AP + PA frontal X-rays only
# - 13 labels (No Finding excluded)
#
# Label policy:
#     1   -> positive
#     0   -> negative
#     NaN -> negative / unmentioned
#    -1   -> ignored
#
# Model:
# - DenseNet121
# - ImageNet pretrained
# - Dropout 0.30 before classifier
#
# Training:
# - image size 320
# - aspect-ratio-preserving resize + padding
# - LR 3e-5
# - weight decay 1e-3
# - pos_weight cap 5
# - mixed precision
# - early stopping patience 4
#
# Model selection:
# - BEST VALIDATION MACRO AUPRC
#
# Important:
# - RUN_TEST=False by default to avoid repeatedly inspecting
#   the same test set during model development.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import gc
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFile, ImageOps

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
)

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 0.1 PIL JPEG SAFETY
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 1. CONFIG
# ============================================================

INDEX_CSV = (
    "/workspace/unzippedarchive/unzippedarchive/"
    "xray_image_label_index.csv"
)

# ------------------------------------------------------------
# IMPORTANT:
# Experiment 1 split file
# ------------------------------------------------------------

EXPERIMENT_1_SPLIT_CSV = (
    "/workspace/unzippedarchive/unzippedarchive/"
    "xray_training_output/"
    "patient_level_splits.csv"
)

# ------------------------------------------------------------
# Separate output folder so Experiment 1 is untouched
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "workspace/unzippedarchive/unzippedarchive/"
    "xray_training_experiment_2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42

IMAGE_SIZE = 320

BATCH_SIZE = 24

NUM_WORKERS = 4

EPOCHS = 25

PATIENCE = 4

LR = 3e-5

WEIGHT_DECAY = 1e-3

DROPOUT = 0.30

POS_WEIGHT_CAP = 5.0


# ------------------------------------------------------------
# Leave False while comparing experiments.
#
# When you're satisfied with the final configuration,
# change to True and evaluate the held-out test split.
# ------------------------------------------------------------

RUN_TEST = False


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


print("=" * 80)
print("EXPERIMENT 2 CONFIG")
print("=" * 80)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )


print("PyTorch:", torch.__version__)

print(
    "Torchvision:",
    torchvision.__version__,
)

print("Image size:", IMAGE_SIZE)

print("Batch size:", BATCH_SIZE)

print("Workers:", NUM_WORKERS)

print("Learning rate:", LR)

print(
    "Weight decay:",
    WEIGHT_DECAY,
)

print("Dropout:", DROPOUT)

print(
    "Positive weight cap:",
    POS_WEIGHT_CAP,
)

print(
    "Early stopping patience:",
    PATIENCE,
)

print(
    "Checkpoint metric: Macro AUPRC"
)

print("RUN_TEST:", RUN_TEST)

print()


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed=42,
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    # Good speed for fixed-size tensors
    torch.backends.cudnn.benchmark = True

    if hasattr(
        torch,
        "set_float32_matmul_precision",
    ):

        torch.set_float32_matmul_precision(
            "high"
        )


seed_everything(SEED)


# ============================================================
# 3. LOAD INDEX
# ============================================================

df = pd.read_csv(
    INDEX_CSV
)


print("=" * 80)
print("RAW INDEX")
print("=" * 80)

print("Rows:", len(df))

print(
    "Patients:",
    df["subject_id"].nunique(),
)

print(
    "Studies:",
    df["study_id"].nunique(),
)

print(
    "DICOM IDs:",
    df["dicom_id"].nunique(),
)

print()


# ============================================================
# 4. VALIDATE REQUIRED COLUMNS
# ============================================================

required_cols = [
    "image_path",
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
] + LABELS


missing_cols = [
    c
    for c in required_cols
    if c not in df.columns
]


if missing_cols:

    raise RuntimeError(
        "Missing required columns: "
        f"{missing_cols}"
    )


if df[
    "dicom_id"
].duplicated().any():

    dupes = df[
        df[
            "dicom_id"
        ].duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Duplicate DICOM IDs found "
        f"in index CSV: {len(dupes)} rows"
    )


# ============================================================
# 5. IMAGE VALIDATION
# ============================================================

print("=" * 80)
print("IMAGE VALIDATION")
print("=" * 80)


missing_files = []

bad_images = []


for p in tqdm(
    df["image_path"],
    desc="Validating JPEGs",
):

    if not os.path.exists(
        p
    ):

        missing_files.append(
            p
        )

        continue


    try:

        with Image.open(
            p
        ) as img:

            img.load()

    except Exception as e:

        bad_images.append(
            (
                p,
                repr(e),
            )
        )


problem_paths = set(
    missing_files
)

problem_paths.update(
    p
    for p, _ in bad_images
)


print()

print(
    "Missing files:",
    len(missing_files),
)

print(
    "Unreadable files:",
    len(bad_images),
)


if bad_images:

    print()

    print(
        "Example unreadable images:"
    )

    for p, error in bad_images[:10]:

        print(p)

        print(
            "    ",
            error,
        )


if problem_paths:

    original_count = len(
        df
    )

    df = df[
        ~df[
            "image_path"
        ].isin(
            problem_paths
        )
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        "Removed unusable images:",
        original_count
        - len(df),
    )

else:

    print(
        "All indexed images decode successfully."
    )


print(
    "Remaining rows:",
    len(df),
)

print()


# ============================================================
# 6. FILTER FRONTAL AP + PA
# ============================================================

df = df[
    df[
        "ViewPosition"
    ].isin(
        [
            "AP",
            "PA",
        ]
    )
].copy()


df.reset_index(
    drop=True,
    inplace=True,
)


print("=" * 80)
print("FRONTAL DATASET")
print("=" * 80)

print(
    "Images:",
    len(df),
)

print(
    "Patients:",
    df[
        "subject_id"
    ].nunique(),
)

print(
    "Studies:",
    df[
        "study_id"
    ].nunique(),
)

print()

print(
    df[
        "ViewPosition"
    ].value_counts()
)

print()


# ============================================================
# 7. LABEL DISTRIBUTION
# ============================================================

print("=" * 80)
print("LABEL DISTRIBUTION — FRONTAL ONLY")
print("=" * 80)


for label in LABELS:

    s = df[
        label
    ]

    pos = int(
        (
            s == 1
        ).sum()
    )

    neg = int(
        (
            s == 0
        ).sum()
    )

    unc = int(
        (
            s == -1
        ).sum()
    )

    missing = int(
        s.isna().sum()
    )

    print(
        f"{label:30s} "
        f"pos={pos:4d} "
        f"explicit_neg={neg:4d} "
        f"uncertain={unc:4d} "
        f"unmentioned={missing:4d}"
    )


# ============================================================
# 8. LOAD EXACT EXPERIMENT 1 SPLIT
# ============================================================
#
# We DO NOT regenerate the split.
#
# This makes Experiment 1 vs Experiment 2 comparison fair.
# ============================================================

print()

print("=" * 80)
print("LOADING EXPERIMENT 1 PATIENT SPLIT")
print("=" * 80)


if not os.path.exists(
    EXPERIMENT_1_SPLIT_CSV
):

    raise RuntimeError(
        "Experiment 1 split file not found:\n"
        f"{EXPERIMENT_1_SPLIT_CSV}"
    )


old_split_df = pd.read_csv(
    EXPERIMENT_1_SPLIT_CSV
)


required_split_cols = [
    "subject_id",
    "split",
]


missing_split_cols = [
    c
    for c in required_split_cols
    if c not in old_split_df.columns
]


if missing_split_cols:

    raise RuntimeError(
        "Experiment 1 split CSV is "
        "missing required columns: "
        f"{missing_split_cols}"
    )


train_ids = set(
    old_split_df.loc[
        old_split_df[
            "split"
        ] == "train",
        "subject_id",
    ].unique()
)


val_ids = set(
    old_split_df.loc[
        old_split_df[
            "split"
        ] == "val",
        "subject_id",
    ].unique()
)


test_ids = set(
    old_split_df.loc[
        old_split_df[
            "split"
        ] == "test",
        "subject_id",
    ].unique()
)


if not train_ids:

    raise RuntimeError(
        "No training patients found "
        "in Experiment 1 split."
    )


if not val_ids:

    raise RuntimeError(
        "No validation patients found "
        "in Experiment 1 split."
    )


if not test_ids:

    raise RuntimeError(
        "No test patients found "
        "in Experiment 1 split."
    )


train_df = df[
    df[
        "subject_id"
    ].isin(
        train_ids
    )
].copy()


val_df = df[
    df[
        "subject_id"
    ].isin(
        val_ids
    )
].copy()


test_df = df[
    df[
        "subject_id"
    ].isin(
        test_ids
    )
].copy()


# ============================================================
# 9. LEAKAGE CHECK
# ============================================================

assert train_ids.isdisjoint(
    val_ids
)

assert train_ids.isdisjoint(
    test_ids
)

assert val_ids.isdisjoint(
    test_ids
)


assert set(
    train_df[
        "dicom_id"
    ]
).isdisjoint(
    set(
        val_df[
            "dicom_id"
        ]
    )
)


assert set(
    train_df[
        "dicom_id"
    ]
).isdisjoint(
    set(
        test_df[
            "dicom_id"
        ]
    )
)


assert set(
    val_df[
        "dicom_id"
    ]
).isdisjoint(
    set(
        test_df[
            "dicom_id"
        ]
    )
)


print()

for name, part in [
    (
        "TRAIN",
        train_df,
    ),
    (
        "VAL",
        val_df,
    ),
    (
        "TEST",
        test_df,
    ),
]:

    print(
        f"{name:6s} "
        f"images={len(part):5d} "
        f"patients="
        f"{part['subject_id'].nunique():4d} "
        f"studies="
        f"{part['study_id'].nunique():4d}"
    )


print()

print(
    "Patient leakage check: PASSED"
)

print(
    "Split source: Experiment 1"
)


# ============================================================
# 10. SAVE EXPERIMENT 2 SPLIT COPY
# ============================================================

train_df[
    "split"
] = "train"

val_df[
    "split"
] = "val"

test_df[
    "split"
] = "test"


split_df = pd.concat(
    [
        train_df,
        val_df,
        test_df,
    ],
    ignore_index=True,
)


split_df.to_csv(
    OUTPUT_DIR
    / "patient_level_splits.csv",
    index=False,
)


# ============================================================
# 11. LABEL COUNTS PER SPLIT
# ============================================================

print()

print("=" * 80)
print("LABEL COUNTS PER SPLIT")
print("=" * 80)


for label in LABELS:

    row_parts = []

    for name, part in [
        (
            "train",
            train_df,
        ),
        (
            "val",
            val_df,
        ),
        (
            "test",
            test_df,
        ),
    ]:

        vals = part[
            label
        ].fillna(
            0
        )

        pos = int(
            (
                vals == 1
            ).sum()
        )

        neg = int(
            (
                vals == 0
            ).sum()
        )

        unc = int(
            (
                vals == -1
            ).sum()
        )

        row_parts.append(
            f"{name}: "
            f"+{pos}/"
            f"-{neg}/"
            f"u{unc}"
        )

    print(
        f"{label:30s} "
        + " | ".join(
            row_parts
        )
    )


# ============================================================
# 12. ASPECT-RATIO PRESERVING RESIZE + PAD
# ============================================================

class ResizeAndPad:

    def __init__(
        self,
        size,
        fill=0,
    ):

        self.size = size

        self.fill = fill


    def __call__(
        self,
        img,
    ):

        width, height = (
            img.size
        )

        scale = min(
            self.size / width,
            self.size / height,
        )


        new_width = max(
            1,
            int(
                round(
                    width
                    * scale
                )
            ),
        )


        new_height = max(
            1,
            int(
                round(
                    height
                    * scale
                )
            ),
        )


        img = img.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.BILINEAR,
        )


        pad_width = (
            self.size
            - new_width
        )

        pad_height = (
            self.size
            - new_height
        )


        left = (
            pad_width
            // 2
        )

        right = (
            pad_width
            - left
        )

        top = (
            pad_height
            // 2
        )

        bottom = (
            pad_height
            - top
        )


        img = ImageOps.expand(
            img,
            border=(
                left,
                top,
                right,
                bottom,
            ),
            fill=self.fill,
        )


        return img


# ============================================================
# 13. TRANSFORMS
# ============================================================
#
# Important order:
#
# image
#   -> preserve aspect ratio + pad
#   -> mild affine augmentation
#   -> tensor
#   -> ImageNet normalization
#
# No horizontal flip.
# ============================================================

train_transform = transforms.Compose(
    [
        ResizeAndPad(
            IMAGE_SIZE
        ),

        transforms.RandomAffine(
            degrees=5,
            translate=(
                0.02,
                0.02,
            ),
            scale=(
                0.95,
                1.05,
            ),
            fill=0,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


eval_transform = transforms.Compose(
    [
        ResizeAndPad(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 14. DATASET
# ============================================================

class XRayDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.transform = (
            transform
        )


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[
            idx
        ]


        path = row[
            "image_path"
        ]


        try:

            with Image.open(
                path
            ) as im:

                img = im.convert(
                    "RGB"
                )

        except Exception as e:

            raise RuntimeError(
                "Failed to load image:\n"
                f"{path}\n"
                f"{repr(e)}"
            ) from e


        if (
            self.transform
            is not None
        ):

            img = (
                self.transform(
                    img
                )
            )


        raw_labels = row[
            LABELS
        ].values.astype(
            np.float32
        )


        # ----------------------------
        # Label policy
        # ----------------------------

        uncertain_mask = (
            raw_labels
            == -1
        )


        labels = np.nan_to_num(
            raw_labels,
            nan=0.0,
        )


        labels[
            uncertain_mask
        ] = 0.0


        mask = (
            ~uncertain_mask
        ).astype(
            np.float32
        )


        return (
            img,

            torch.tensor(
                labels,
                dtype=torch.float32,
            ),

            torch.tensor(
                mask,
                dtype=torch.float32,
            ),

            row[
                "dicom_id"
            ],
        )


# ============================================================
# 15. DATA LOADERS
# ============================================================

train_dataset = XRayDataset(
    train_df,
    transform=train_transform,
)


val_dataset = XRayDataset(
    val_df,
    transform=eval_transform,
)


test_dataset = XRayDataset(
    test_df,
    transform=eval_transform,
)


loader_kwargs = dict(
    batch_size=BATCH_SIZE,

    num_workers=NUM_WORKERS,

    pin_memory=(
        torch.cuda.is_available()
    ),

    persistent_workers=(
        NUM_WORKERS > 0
    ),
)


train_loader = DataLoader(
    train_dataset,

    shuffle=True,

    drop_last=False,

    **loader_kwargs,
)


val_loader = DataLoader(
    val_dataset,

    shuffle=False,

    drop_last=False,

    **loader_kwargs,
)


test_loader = DataLoader(
    test_dataset,

    shuffle=False,

    drop_last=False,

    **loader_kwargs,
)


print()

print(
    "Train batches:",
    len(train_loader),
)

print(
    "Val batches:",
    len(val_loader),
)

print(
    "Test batches:",
    len(test_loader),
)


# ============================================================
# 16. CLASS WEIGHTS — CAP 5
# ============================================================

print()

print("=" * 80)
print("CLASS WEIGHTS")
print("=" * 80)


pos_weights = []


for label in LABELS:

    vals = train_df[
        label
    ].fillna(
        0
    )


    vals = vals[
        vals != -1
    ]


    pos = int(
        (
            vals == 1
        ).sum()
    )


    neg = int(
        (
            vals == 0
        ).sum()
    )


    if pos == 0:

        weight = 1.0

    else:

        weight = (
            neg / pos
        )


    weight = float(
        np.clip(
            weight,
            1.0,
            POS_WEIGHT_CAP,
        )
    )


    pos_weights.append(
        weight
    )


    print(
        f"{label:30s} "
        f"pos={pos:4d} "
        f"neg={neg:4d} "
        f"weight={weight:.3f}"
    )


pos_weights = torch.tensor(
    pos_weights,

    dtype=torch.float32,

    device=DEVICE,
)


# ============================================================
# 17. MODEL — DenseNet121 + Dropout
# ============================================================

weights = (
    DenseNet121_Weights
    .IMAGENET1K_V1
)


model = densenet121(
    weights=weights
)


in_features = (
    model
    .classifier
    .in_features
)


model.classifier = nn.Sequential(

    nn.Dropout(
        p=DROPOUT
    ),

    nn.Linear(
        in_features,
        len(LABELS),
    ),
)


model = model.to(
    DEVICE
)


print()

print("=" * 80)
print("MODEL")
print("=" * 80)

print(
    model.__class__.__name__
)

print(
    "Output classes:",
    len(LABELS),
)

print(
    "Classifier dropout:",
    DROPOUT,
)


# ============================================================
# 18. LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    reduction="none",

    pos_weight=pos_weights,
)


def masked_bce_loss(
    logits,
    targets,
    mask,
):

    raw_loss = criterion(
        logits,
        targets,
    )


    masked_loss = (
        raw_loss
        * mask
    )


    denom = (
        mask
        .sum()
        .clamp(
            min=1.0
        )
    )


    return (
        masked_loss.sum()
        / denom
    )


# ============================================================
# 19. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),

    lr=LR,

    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 20. LR SCHEDULER
# ============================================================
#
# Experiment 2 selects models using validation AUPRC.
# Scheduler therefore also watches validation AUPRC.
# ============================================================

scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,

        mode="max",

        factor=0.5,

        patience=2,

        min_lr=1e-7,
    )
)


# ============================================================
# 21. MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",

    enabled=(
        torch.cuda.is_available()
    ),
)


# ============================================================
# 22. METRICS
# ============================================================

def calculate_metrics(
    targets,
    probs,
    masks,
):

    results = {}

    aurocs = []

    auprcs = []


    for i, label in enumerate(
        LABELS
    ):

        valid = (
            masks[
                :,
                i
            ]
            == 1
        )


        y_true = targets[
            valid,
            i
        ]


        y_prob = probs[
            valid,
            i
        ]


        if len(
            np.unique(
                y_true
            )
        ) < 2:

            auc = np.nan

            ap = np.nan


        else:

            auc = roc_auc_score(
                y_true,
                y_prob,
            )

            ap = average_precision_score(
                y_true,
                y_prob,
            )


        results[
            label
        ] = {
            "AUROC": auc,
            "AUPRC": ap,
        }


        if not np.isnan(
            auc
        ):

            aurocs.append(
                auc
            )


        if not np.isnan(
            ap
        ):

            auprcs.append(
                ap
            )


    results[
        "macro_AUROC"
    ] = (
        float(
            np.mean(
                aurocs
            )
        )
        if aurocs
        else np.nan
    )


    results[
        "macro_AUPRC"
    ] = (
        float(
            np.mean(
                auprcs
            )
        )
        if auprcs
        else np.nan
    )


    return results


# ============================================================
# 23. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
):

    model.train()


    running_loss = 0.0

    samples = 0


    progress = tqdm(
        loader,

        desc="Training",

        leave=False,
    )


    for (
        images,
        targets,
        masks,
        _,
    ) in progress:


        images = images.to(
            DEVICE,

            non_blocking=True,
        )


        targets = targets.to(
            DEVICE,

            non_blocking=True,
        )


        masks = masks.to(
            DEVICE,

            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type=DEVICE.type,

            dtype=torch.float16,

            enabled=(
                torch.cuda.is_available()
            ),
        ):

            logits = model(
                images
            )


            loss = masked_bce_loss(
                logits,
                targets,
                masks,
            )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),

            max_norm=5.0,
        )


        scaler.step(
            optimizer
        )


        scaler.update()


        bs = images.size(
            0
        )


        running_loss += (
            loss.item()
            * bs
        )


        samples += bs


        progress.set_postfix(
            loss=(
                f"{loss.item():.4f}"
            ),

            lr=(
                f"{optimizer.param_groups[0]['lr']:.2e}"
            ),
        )


    return (
        running_loss
        / max(
            samples,
            1,
        )
    )


# ============================================================
# 24. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    model.eval()


    running_loss = 0.0

    samples = 0


    all_targets = []

    all_probs = []

    all_masks = []

    all_ids = []


    progress = tqdm(
        loader,

        desc="Evaluating",

        leave=False,
    )


    for (
        images,
        targets,
        masks,
        dicom_ids,
    ) in progress:


        images = images.to(
            DEVICE,

            non_blocking=True,
        )


        targets = targets.to(
            DEVICE,

            non_blocking=True,
        )


        masks = masks.to(
            DEVICE,

            non_blocking=True,
        )


        with torch.autocast(
            device_type=DEVICE.type,

            dtype=torch.float16,

            enabled=(
                torch.cuda.is_available()
            ),
        ):

            logits = model(
                images
            )


            loss = masked_bce_loss(
                logits,
                targets,
                masks,
            )


        probs = torch.sigmoid(
            logits
        )


        bs = images.size(
            0
        )


        running_loss += (
            loss.item()
            * bs
        )


        samples += bs


        all_targets.append(
            targets
            .detach()
            .cpu()
            .numpy()
        )


        all_probs.append(
            probs
            .detach()
            .cpu()
            .numpy()
        )


        all_masks.append(
            masks
            .detach()
            .cpu()
            .numpy()
        )


        all_ids.extend(
            list(
                dicom_ids
            )
        )


    targets = np.concatenate(
        all_targets,

        axis=0,
    )


    probs = np.concatenate(
        all_probs,

        axis=0,
    )


    masks = np.concatenate(
        all_masks,

        axis=0,
    )


    metrics = calculate_metrics(
        targets,
        probs,
        masks,
    )


    return (
        running_loss
        / max(
            samples,
            1,
        ),

        metrics,

        targets,

        probs,

        masks,

        all_ids,
    )


# ============================================================
# 25. TRAINING LOOP
# ============================================================

history = []


best_metric = -np.inf


epochs_without_improvement = 0


best_checkpoint = (
    OUTPUT_DIR
    / "best_model.pt"
)


print()

print("=" * 80)
print("TRAINING — EXPERIMENT 2")
print("=" * 80)


for epoch in range(
    1,
    EPOCHS + 1,
):

    print()

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        "-" * 80
    )


    train_loss = train_one_epoch(
        model,
        train_loader,
    )


    (
        val_loss,
        val_metrics,
        _,
        _,
        _,
        _,
    ) = evaluate(
        model,
        val_loader,
    )


    macro_auc = val_metrics[
        "macro_AUROC"
    ]


    macro_ap = val_metrics[
        "macro_AUPRC"
    ]


    # --------------------------------------------------------
    # Scheduler watches AUPRC
    # --------------------------------------------------------

    scheduler.step(
        macro_ap
    )


    current_lr = (
        optimizer.param_groups[
            0
        ][
            "lr"
        ]
    )


    print(
        f"Train loss : "
        f"{train_loss:.5f}"
    )

    print(
        f"Val loss   : "
        f"{val_loss:.5f}"
    )

    print(
        f"Macro AUROC: "
        f"{macro_auc:.5f}"
    )

    print(
        f"Macro AUPRC: "
        f"{macro_ap:.5f}"
    )

    print(
        f"LR         : "
        f"{current_lr:.2e}"
    )


    history.append(
        {
            "epoch": epoch,

            "train_loss": (
                train_loss
            ),

            "val_loss": (
                val_loss
            ),

            "macro_AUROC": (
                macro_auc
            ),

            "macro_AUPRC": (
                macro_ap
            ),

            "lr": (
                current_lr
            ),
        }
    )


    # ========================================================
    # IMPORTANT:
    # Experiment 2 checkpoint selection uses MACRO AUPRC
    # ========================================================

    if macro_ap > best_metric:

        best_metric = (
            macro_ap
        )


        epochs_without_improvement = 0


        torch.save(
            {
                "epoch": (
                    epoch
                ),

                "model_state_dict": (
                    model.state_dict()
                ),

                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),

                "macro_AUROC": (
                    macro_auc
                ),

                "macro_AUPRC": (
                    macro_ap
                ),

                "labels": (
                    LABELS
                ),

                "image_size": (
                    IMAGE_SIZE
                ),

                "dropout": (
                    DROPOUT
                ),

                "learning_rate": (
                    LR
                ),

                "weight_decay": (
                    WEIGHT_DECAY
                ),

                "pos_weight_cap": (
                    POS_WEIGHT_CAP
                ),
            },

            best_checkpoint,
        )


        print(
            "✓ Saved new best model — "
            "Macro AUPRC: "
            f"{macro_ap:.5f}"
        )


    else:

        epochs_without_improvement += 1


        print(
            "No AUPRC improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break


# ============================================================
# 26. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    OUTPUT_DIR
    / "training_history.csv",

    index=False,
)


# ============================================================
# 27. TRAINING CURVES
# ============================================================

plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.plot(
    history_df[
        "epoch"
    ],

    history_df[
        "train_loss"
    ],

    label="Train",
)


plt.plot(
    history_df[
        "epoch"
    ],

    history_df[
        "val_loss"
    ],

    label="Validation",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Experiment 2 — Training / Validation Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "loss_curve.png",

    dpi=150,
)


plt.show()


plt.figure(
    figsize=(
        8,
        5,
    )
)


plt.plot(
    history_df[
        "epoch"
    ],

    history_df[
        "macro_AUROC"
    ],

    label="Macro AUROC",
)


plt.plot(
    history_df[
        "epoch"
    ],

    history_df[
        "macro_AUPRC"
    ],

    label="Macro AUPRC",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Metric"
)

plt.title(
    "Experiment 2 — Validation Metrics"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "validation_metrics.png",

    dpi=150,
)


plt.show()


# ============================================================
# 28. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    best_checkpoint,

    map_location=DEVICE,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print()

print("=" * 80)
print("BEST CHECKPOINT")
print("=" * 80)


print(
    "Best epoch:",
    checkpoint[
        "epoch"
    ],
)


print(
    "Best validation macro AUPRC:",
    checkpoint[
        "macro_AUPRC"
    ],
)


print(
    "Corresponding macro AUROC:",
    checkpoint[
        "macro_AUROC"
    ],
)


# ============================================================
# 29. FINAL VALIDATION EVALUATION
# ============================================================

(
    val_loss,
    val_metrics,
    val_targets,
    val_probs,
    val_masks,
    val_ids,
) = evaluate(
    model,
    val_loader,
)


print()

print("=" * 80)
print("BEST MODEL VALIDATION PERFORMANCE")
print("=" * 80)


print(
    "Validation loss:",
    round(
        val_loss,
        5,
    ),
)


print(
    "Validation macro AUROC:",
    round(
        val_metrics[
            "macro_AUROC"
        ],
        5,
    ),
)


print(
    "Validation macro AUPRC:",
    round(
        val_metrics[
            "macro_AUPRC"
        ],
        5,
    ),
)


# ============================================================
# 30. VALIDATION PER-CLASS METRICS
# ============================================================

validation_rows = []


for i, label in enumerate(
    LABELS
):

    valid = (
        val_masks[
            :,
            i
        ]
        == 1
    )


    y_true = val_targets[
        valid,
        i
    ]


    y_prob = val_probs[
        valid,
        i
    ]


    if len(
        np.unique(
            y_true
        )
    ) >= 2:

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

        ap = average_precision_score(
            y_true,
            y_prob,
        )

    else:

        auc = np.nan

        ap = np.nan


    validation_rows.append(
        {
            "label": label,

            "AUROC": auc,

            "AUPRC": ap,

            "positives": int(
                (
                    y_true == 1
                ).sum()
            ),

            "negatives": int(
                (
                    y_true == 0
                ).sum()
            ),

            "n_valid": int(
                len(
                    y_true
                )
            ),
        }
    )


validation_metrics_df = pd.DataFrame(
    validation_rows
)


validation_metrics_df.to_csv(
    OUTPUT_DIR
    / "validation_per_class_metrics.csv",

    index=False,
)


print()

print(
    validation_metrics_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 31. OPTIMIZE THRESHOLDS ON VALIDATION SET
# ============================================================

thresholds = {}


print()

print("=" * 80)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 80)


for i, label in enumerate(
    LABELS
):

    valid = (
        val_masks[
            :,
            i
        ]
        == 1
    )


    y_true = val_targets[
        valid,
        i
    ]


    y_prob = val_probs[
        valid,
        i
    ]


    if len(
        np.unique(
            y_true
        )
    ) < 2:

        threshold = 0.5

        best_f1 = np.nan


    else:

        threshold = 0.5

        best_f1 = -1.0


        for t in np.linspace(
            0.05,
            0.95,
            181,
        ):

            y_pred = (
                y_prob
                >= t
            ).astype(
                int
            )


            score = f1_score(
                y_true,
                y_pred,

                zero_division=0,
            )


            if score > best_f1:

                best_f1 = (
                    score
                )

                threshold = float(
                    t
                )


    thresholds[
        label
    ] = threshold


    print(
        f"{label:30s} "
        f"threshold="
        f"{threshold:.3f} "
        f"val_F1="
        f"{best_f1:.3f}"
    )


with open(
    OUTPUT_DIR
    / "thresholds.json",

    "w",
) as f:

    json.dump(
        thresholds,

        f,

        indent=2,
    )


# ============================================================
# 32. SAVE VALIDATION PREDICTIONS
# ============================================================

val_pred_df = pd.DataFrame(
    {
        "dicom_id": (
            val_ids
        )
    }
)


for i, label in enumerate(
    LABELS
):

    val_pred_df[
        f"{label}_prob"
    ] = (
        val_probs[
            :,
            i
        ]
    )


    val_pred_df[
        f"{label}_target"
    ] = (
        val_targets[
            :,
            i
        ]
    )


    val_pred_df[
        f"{label}_mask"
    ] = (
        val_masks[
            :,
            i
        ]
    )


val_pred_df.to_csv(
    OUTPUT_DIR
    / "validation_predictions.csv",

    index=False,
)


# ============================================================
# 33. OPTIONAL HELD-OUT TEST
# ============================================================

test_metrics = None

test_loss = None

test_targets = None

test_probs = None

test_masks = None

test_ids = None


if RUN_TEST:

    print()

    print("=" * 80)
    print("FINAL HELD-OUT TEST")
    print("=" * 80)


    (
        test_loss,
        test_metrics,
        test_targets,
        test_probs,
        test_masks,
        test_ids,
    ) = evaluate(
        model,
        test_loader,
    )


    print(
        "Test loss:",
        round(
            test_loss,
            5,
        ),
    )


    print(
        "Test macro AUROC:",
        round(
            test_metrics[
                "macro_AUROC"
            ],
            5,
        ),
    )


    print(
        "Test macro AUPRC:",
        round(
            test_metrics[
                "macro_AUPRC"
            ],
            5,
        ),
    )


    # ========================================================
    # TEST PER-CLASS METRICS
    # ========================================================

    metric_rows = []


    for i, label in enumerate(
        LABELS
    ):

        valid = (
            test_masks[
                :,
                i
            ]
            == 1
        )


        y_true = test_targets[
            valid,
            i
        ]


        y_prob = test_probs[
            valid,
            i
        ]


        threshold = thresholds[
            label
        ]


        y_pred = (
            y_prob
            >= threshold
        ).astype(
            int
        )


        if len(
            np.unique(
                y_true
            )
        ) >= 2:

            auc = roc_auc_score(
                y_true,
                y_prob,
            )

            ap = average_precision_score(
                y_true,
                y_prob,
            )

        else:

            auc = np.nan

            ap = np.nan


        precision = precision_score(
            y_true,
            y_pred,

            zero_division=0,
        )


        recall = recall_score(
            y_true,
            y_pred,

            zero_division=0,
        )


        f1 = f1_score(
            y_true,
            y_pred,

            zero_division=0,
        )


        cm = confusion_matrix(
            y_true,
            y_pred,

            labels=[
                0,
                1,
            ],
        )


        tn, fp, fn, tp = (
            cm.ravel()
        )


        specificity = (
            tn
            / (
                tn
                + fp
            )
            if (
                tn
                + fp
            ) > 0
            else np.nan
        )


        metric_rows.append(
            {
                "label": label,

                "AUROC": auc,

                "AUPRC": ap,

                "threshold": (
                    threshold
                ),

                "precision": (
                    precision
                ),

                "sensitivity_recall": (
                    recall
                ),

                "specificity": (
                    specificity
                ),

                "F1": f1,

                "TP": int(
                    tp
                ),

                "FP": int(
                    fp
                ),

                "TN": int(
                    tn
                ),

                "FN": int(
                    fn
                ),

                "n_valid": int(
                    len(
                        y_true
                    )
                ),
            }
        )


    test_metrics_df = pd.DataFrame(
        metric_rows
    )


    test_metrics_df.to_csv(
        OUTPUT_DIR
        / "test_metrics.csv",

        index=False,
    )


    print()

    print(
        test_metrics_df
        .round(
            4
        )
        .to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    pred_df = pd.DataFrame(
        {
            "dicom_id": (
                test_ids
            )
        }
    )


    for i, label in enumerate(
        LABELS
    ):

        pred_df[
            f"{label}_prob"
        ] = (
            test_probs[
                :,
                i
            ]
        )


        pred_df[
            f"{label}_target"
        ] = (
            test_targets[
                :,
                i
            ]
        )


        pred_df[
            f"{label}_mask"
        ] = (
            test_masks[
                :,
                i
            ]
        )


        pred_df[
            f"{label}_prediction"
        ] = (
            test_probs[
                :,
                i
            ]
            >= thresholds[
                label
            ]
        ).astype(
            int
        )


    pred_df.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",

        index=False,
    )


else:

    print()

    print("=" * 80)

    print(
        "TEST SET NOT EVALUATED"
    )

    print("=" * 80)

    print(
        "RUN_TEST=False"
    )

    print(
        "Use validation results to compare "
        "Experiment 2 against Experiment 1."
    )


# ============================================================
# 34. EXPERIMENT SUMMARY
# ============================================================

summary = {

    "experiment": (
        "experiment_2_regularized_densenet121"
    ),

    "index_csv": (
        INDEX_CSV
    ),

    "split_source": (
        EXPERIMENT_1_SPLIT_CSV
    ),

    "architecture": (
        "DenseNet121"
    ),

    "pretraining": (
        "ImageNet IMAGENET1K_V1"
    ),

    "views": [
        "AP",
        "PA",
    ],

    "image_size": (
        IMAGE_SIZE
    ),

    "resize_strategy": (
        "preserve_aspect_ratio_and_pad"
    ),

    "batch_size": (
        BATCH_SIZE
    ),

    "num_workers": (
        NUM_WORKERS
    ),

    "learning_rate": (
        LR
    ),

    "weight_decay": (
        WEIGHT_DECAY
    ),

    "dropout": (
        DROPOUT
    ),

    "positive_weight_cap": (
        POS_WEIGHT_CAP
    ),

    "early_stopping_patience": (
        PATIENCE
    ),

    "checkpoint_metric": (
        "macro_AUPRC"
    ),

    "epochs_requested": (
        EPOCHS
    ),

    "epochs_completed": int(
        len(
            history_df
        )
    ),

    "best_epoch": int(
        checkpoint[
            "epoch"
        ]
    ),

    "best_validation_macro_AUROC": float(
        checkpoint[
            "macro_AUROC"
        ]
    ),

    "best_validation_macro_AUPRC": float(
        checkpoint[
            "macro_AUPRC"
        ]
    ),

    "train_images": int(
        len(
            train_df
        )
    ),

    "val_images": int(
        len(
            val_df
        )
    ),

    "test_images": int(
        len(
            test_df
        )
    ),

    "train_patients": int(
        train_df[
            "subject_id"
        ].nunique()
    ),

    "val_patients": int(
        val_df[
            "subject_id"
        ].nunique()
    ),

    "test_patients": int(
        test_df[
            "subject_id"
        ].nunique()
    ),

    "removed_missing_or_bad_images": int(
        len(
            problem_paths
        )
    ),

    "test_evaluated": (
        RUN_TEST
    ),

    "labels": (
        LABELS
    ),
}


if (
    RUN_TEST
    and test_metrics
    is not None
):

    summary[
        "test_macro_AUROC"
    ] = float(
        test_metrics[
            "macro_AUROC"
        ]
    )

    summary[
        "test_macro_AUPRC"
    ] = float(
        test_metrics[
            "macro_AUPRC"
        ]
    )


with open(
    OUTPUT_DIR
    / "summary.json",

    "w",
) as f:

    json.dump(
        summary,

        f,

        indent=2,
    )


# ============================================================
# 35. EXPERIMENT 1 vs EXPERIMENT 2 QUICK COMPARISON
# ============================================================

EXP1_HISTORY = (
    "/workspace/private/unzippedarchive/"
    "xray_training_output/"
    "training_history.csv"
)


print()

print("=" * 80)
print("EXPERIMENT COMPARISON")
print("=" * 80)


if os.path.exists(
    EXP1_HISTORY
):

    exp1_history = pd.read_csv(
        EXP1_HISTORY
    )


    exp1_best_auroc = float(
        exp1_history[
            "macro_AUROC"
        ].max()
    )


    exp1_best_auprc = float(
        exp1_history[
            "macro_AUPRC"
        ].max()
    )


    exp2_best_auroc = float(
        history_df[
            "macro_AUROC"
        ].max()
    )


    exp2_best_auprc = float(
        history_df[
            "macro_AUPRC"
        ].max()
    )


    print(
        "Experiment 1"
    )

    print(
        "  Best val AUROC:",
        round(
            exp1_best_auroc,
            5,
        ),
    )

    print(
        "  Best val AUPRC:",
        round(
            exp1_best_auprc,
            5,
        ),
    )


    print()


    print(
        "Experiment 2"
    )

    print(
        "  Best val AUROC:",
        round(
            exp2_best_auroc,
            5,
        ),
    )

    print(
        "  Best val AUPRC:",
        round(
            exp2_best_auprc,
            5,
        ),
    )


    comparison = pd.DataFrame(
        [
            {
                "experiment": (
                    "experiment_1"
                ),

                "best_val_AUROC": (
                    exp1_best_auroc
                ),

                "best_val_AUPRC": (
                    exp1_best_auprc
                ),
            },

            {
                "experiment": (
                    "experiment_2"
                ),

                "best_val_AUROC": (
                    exp2_best_auroc
                ),

                "best_val_AUPRC": (
                    exp2_best_auprc
                ),
            },
        ]
    )


    comparison.to_csv(
        OUTPUT_DIR
        / "experiment_comparison.csv",

        index=False,
    )


else:

    print(
        "Experiment 1 training_history.csv "
        "not found."
    )


# ============================================================
# 36. FINAL OUTPUT
# ============================================================

print()

print("=" * 80)
print("EXPERIMENT 2 COMPLETE")
print("=" * 80)


print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()


print(
    "Best epoch:",
    checkpoint[
        "epoch"
    ],
)


print(
    "Best validation macro AUROC:",
    round(
        checkpoint[
            "macro_AUROC"
        ],
        5,
    ),
)


print(
    "Best validation macro AUPRC:",
    round(
        checkpoint[
            "macro_AUPRC"
        ],
        5,
    ),
)


print()

print(
    "Files:"
)


for p in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        " -",
        p.name
    )


print()

print(
    "Removed missing/unreadable images:",
    len(
        problem_paths
    ),
)


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


    