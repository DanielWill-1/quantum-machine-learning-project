# Experiment 7: quantum encoding and circuit comparison

Upload **b.py** to the Vast.ai notebook where E6 ran. It is standalone and uses the same E2 checkpoint, image index, patient splits, and frozen feature cache. It writes a new `xray_training_experiment_7_encoding/run_...` folder.

```python
%run b.py
```

If the dataset moved, set its location first:

```python
%env MIMIC_CXR_ROOT=/workspace/unzippedarchive/unzippedarchive
%run b.py
```

For an initial one-seed exploration, set `%env E7_SEEDS=42`. The default is `42,43,44`; use all three for the comparison. Run `%run b.py --self-check` to check circuits without loading the dataset. The normal run also performs these small checks before image work.

Use the working Torch/torchvision installation from E6. Required additional packages are NumPy, pandas, Pillow, scikit-learn, matplotlib, tqdm, and PennyLane. Missing PennyLane is installed as `pennylane==0.45.1`; the script does not replace Torch. Image feature extraction uses CUDA when available; the circuit and classical heads use CPU. Existing compatible E6 feature caches are reused.

## Why change E6?

In `a.md`, the E6 linear control reached validation macro AUPRC of 0.300378, 0.300553, and 0.300399. The learned quantum branch reached 0.294761, 0.294336, and 0.294336. Two seeds retained epoch zero. Its near equality to the frozen quantum control does not show a useful contribution from circuit training.

E7 first trains and selects the full 1024-feature linear classifier for each seed. Every residual arm then starts from that same classifier and keeps it frozen. This gives the quantum and classical branches a stronger common starting point. Smaller readout updates, a residual penalty of 0.05, and a smooth correction limit of 0.75 logits address the growing residuals observed in E6. Improvement remains a hypothesis to test.

## Two encoders, one main circuit

| Encoder | Inputs retained | Classical replacement |
|---|---|---|
| `pca` | 48 standardized principal components, fitted on training images | Same encoder and angle mapping, followed by a Fourier-feature MLP |
| `task` | 13 standardized logits from the frozen linear classifier plus 35 principal components | Same encoder and angle mapping, followed by the same MLP architecture |

Both encoders retain the full classifier as a separate prediction path. The 1024-to-48 maps remain fixed; only 48 bounded gains and 48 small shifts adapt after three readout warmup epochs. Each pair begins with identical encoder parameters, readout initialization, training order, and anchor weights. Encoder parameters can diverge during independent training.

The angle map is `2 * atan(0.5 * gain * feature + shift)`. It saturates more slowly than E6's `pi * tanh`. Training data supplies all PCA and normalization statistics. The output configuration records retained PCA variance so information loss is visible.

The quantum circuit has eight qubits and three layers. Each layer receives a different 16-feature block through RY and RZ gates, then applies trainable local rotations and neighboring IsingZZ interactions. A final RY layer mixes the measurement basis. Readout uses 32 measurements: local Z, local X, neighboring ZZ, and neighboring XX.

Layered encoding is motivated by [data re-uploading research](https://quantum-journal.org/papers/q-2020-02-06-226/) and the connection between [encoding and Fourier expressivity](https://pennylane.ai/demos/tutorial_expressivity_fourier_series). The implementation uses PennyLane's [IsingZZ gate](https://docs.pennylane.ai/en/stable/code/api/pennylane.IsingZZ.html). These references motivate the design; they do not establish that it improves this dataset. The script exports the actual circuit as `quantum_circuit.txt`.

## Predeclared comparisons

Seven arms run for every seed:

1. `linear_control`: full feature classifier, then frozen as the common anchor.
2. `pca_classical`: PCA encoder with classical core.
3. `pca_quantum`: PCA encoder with learned quantum core.
4. `task_classical`: task/PCA encoder with classical core.
5. `task_quantum`: task/PCA encoder with learned quantum core.
6. `task_frozen`: task/PCA encoder with a fixed initial quantum circuit; encoder/readout still train.
7. `task_no_entanglement`: task/PCA encoder and circuit with ZZ interactions removed.

The classical core receives identical angle inputs and produces the same number of outputs. Its 4,160 core parameters exceed the quantum core's 104; this is an input/output matched control, not a parameter matched comparison. Exact parameter counts are recorded. The no-entanglement circuit's unused coupling parameters remain frozen.

Every residual starts at exactly zero. Epoch zero stays eligible for checkpoint selection; a branch that fails to improve validation AUPRC can remain inactive. Additional checks compare the selected quantum model with its branch removed and its circuit reset. All declared seeds also form an equal probability ensemble. No best-seed selection or ensemble weight search is performed.

## Results to inspect

- `b.md`: compact result table.
- `experiment_comparison.csv` and `seed_summary.csv`: all arms and seed variation.
- `paired_patient_bootstrap_differences.csv`: paired comparisons using patient resampling.
- `quantum_ablation_metrics.csv`: circuit-reset diagnostics.
- `configuration.json`, `pca_train_only.npz`, `encoder_maps_seed_*.pt`: reproducible configuration and encoders.
- Per-arm `training_history.csv` and `training_summary.json`: gradients, residual size, parameter changes, and selected epoch.
- Per-arm `best_model.pt`: selected head and preprocessing statistics; saved during training.
- Quantum-arm `inference_bundle.pt`: frozen CNN, head, and thresholds together.
- Per-arm predictions, per-label metrics, AP/PA breakdowns, and threshold reports.

Inference checkpoints do not include optimizer state for resuming training. A rerun creates a new output folder and can reuse feature caches. To load a quantum inference bundle without training again:

```python
from b import load_exported_hybrid, ResizeAndPad, IMAGE_SIZE
from torchvision import transforms
from PIL import Image
import torch

model, thresholds = load_exported_hybrid('/path/to/task_quantum_seed_42/inference_bundle.pt')
preprocess = transforms.Compose([
    ResizeAndPad(IMAGE_SIZE), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
with Image.open('/path/to/image.jpg') as image:
    tensor = preprocess(image.convert('RGB')).unsqueeze(0)
with torch.inference_mode():
    probabilities = model(tensor).sigmoid()[0]
```

`RUN_TEST=False` by default. E7 preserves E6's labels, including its policy that blank labels count as negative and `-1` labels are masked. Checkpoints and thresholds use validation only. Since this validation split has already guided earlier experiments, its results are exploratory. Patient bootstrap intervals exclude training and model-selection uncertainty. A quantum contribution requires comparison against the common anchor and classical/frozen/no-entanglement controls; simulated circuit training does not establish quantum hardware advantage.
