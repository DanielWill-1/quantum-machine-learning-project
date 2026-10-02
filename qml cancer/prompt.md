# EXPERIMENT 9 — QUANTUM VALUE, ENCODING, CIRCUIT ARCHITECTURE, AND ABLATION STUDY

You are working on the next experiment in an existing MIMIC-CXR multilabel chest-X-ray research project.

This is **Experiment 9**, the second-to-last experimental stage.

**Experiment 10 will be the final locked experiment.**

Experiment 9 is NOT a reason to remove QML, replace QML with classical ML, or conclude that the project should become classical-only.

QML is a mandatory part of the research objective.

The scientific objective of Experiment 9 is:

> Determine whether a properly designed, trainable quantum component can extract useful complementary information from the frozen DenseNet representation and produce a reproducible improvement over classical alternatives, while demonstrating that any improvement actually depends on learned quantum circuit behavior.

We are investigating QML as the novelty of this project.

However, do not fake, force, cherry-pick, or manufacture a quantum advantage. The controls and ablations must be strong enough that a positive E9 result would be defensible.

Experiment 9 must preserve Experiment 8 and all earlier experiments unchanged.

Do NOT overwrite Experiment 8 files.

Do NOT modify Experiment 2's checkpoint.

Do NOT evaluate the final test set in Experiment 9.

Experiment 10 will be responsible for the final locked evaluation.

---

# 1. EXISTING EXPERIMENT 8 CONTEXT

Read and understand the supplied Experiment 8 code before implementing anything.

Preserve all useful E8 infrastructure, including:

- exact Experiment 2 cohort
- exact AP/PA image filtering
- exact 13 labels
- exact uncertainty/masking policy
- exact image preprocessing
- Experiment 2 `best_model.pt`
- frozen DenseNet121 feature extraction
- deterministic feature cache
- patient-level split integrity checks
- JPEG validation
- feature-cache verification
- deterministic execution
- multiple seeds
- patient-cluster bootstrap
- paired model comparisons
- fixed 0.5 threshold metrics
- development-fitted threshold metrics
- per-class metrics
- patient-level metrics
- source hashing
- configuration/protocol files
- model exportability
- circuit drawings
- parameter-change diagnostics
- gradient diagnostics

Do not regress any of these protections.

The relevant current dataset root is expected to be:

```text
/workspace/unzippedarchive/unzippedarchive
```

and Experiment 2 is expected at:

```text
/workspace/unzippedarchive/unzippedarchive/xray_training_experiment_2
```

Continue supporting:

```text
MIMIC_CXR_ROOT
```

for an explicit override.

Experiment 9 output must go into a new directory such as:

```text
xray_training_experiment_9_quantum_value/
    run_<UTC timestamp>/
```

Never overwrite an existing run.

---

# 2. WHAT EXPERIMENT 8 TAUGHT US

Experiment 8 technically succeeded.

The quantum circuit:

- received gradients,
- changed its parameters,
- remained numerically stable,
- produced predictions,
- and slightly improved AUPRC in some runs.

However, the key problem was that the trainable quantum model performed almost identically to the frozen-quantum control.

Therefore E9 must specifically attack:

```text
trainable quantum ≈ frozen quantum
```

The E9 design must make it possible to determine whether **learning inside the circuit** matters.

Another E8 observation was that performance generally peaked very early while the quantum residual continued increasing afterward.

Therefore E9 should use:

- gentler circuit optimization,
- explicit circuit-specific learning rates,
- a slightly longer readout warmup,
- tighter control of residual magnitude,
- early stopping,
- and detailed diagnostics of circuit behavior.

Do not simply increase circuit depth and call that Experiment 9.

---

# 3. DATA PROTOCOL — DO NOT TOUCH TEST

Use only the original Experiment 2 **training patient population** during E9 architecture development.

Create deterministic PATIENT-LEVEL subsets from the original training patients:

```text
fit
tune
report
```

Use approximately:

```text
70% fit
15% tune
15% report
```

or another deterministic nearby ratio if exact patient grouping makes that necessary.

Use a fixed seed such as:

```text
2027
```

Requirements:

- no patient overlap
- no study overlap
- no DICOM overlap
- no image-path overlap
- save exact assignments
- assert all disjointness programmatically

Roles:

```text
fit:
    train model parameters

tune:
    epoch selection
    architecture selection
    threshold fitting
    hyperparameter selection

report:
    E9 held-out head-development comparison
```

The `report` subset must not participate in fitting or architecture selection before its evaluation.

IMPORTANT CAVEAT:

The inherited E2 DenseNet was previously trained on these original training patients.

Therefore E9's `report` set is **not an independent end-to-end clinical validation cohort**.

Explicitly state this in outputs.

It is only a controlled head/architecture-development holdout.

Do not access the original test split.

Prefer not to extract test features at all.

Experiment 10 will perform the final untouched test evaluation.

---

# 4. PRIMARY E9 QUESTION

The primary scientific question is:

> Does a learned variational quantum circuit add predictive information beyond the frozen E2 classifier, a frozen/random quantum feature map, and matched classical nonlinear transformations?

The experiment must distinguish:

```text
benefit from adding parameters
benefit from nonlinear transformation
benefit from random quantum features
benefit from entanglement
benefit from correlation measurements
benefit from learned quantum parameters
```

These are not equivalent.

---

# 5. QUANTUM ARCHITECTURE SEARCH

Do a SMALL, PREDECLARED architecture comparison.

Do not perform a giant hyperparameter search.

Use 8 qubits unless there is a compelling implementation reason not to.

All quantum simulation remains exact analytic state-vector simulation using PennyLane.

No quantum hardware/API.

Do not claim quantum speedup or hardware advantage.

## Candidate Q0 — E8 reference

Reproduce the E8 quantum circuit as a baseline:

```text
8 qubits
3 layers
RY/RZ data encoding
trainable Rot gates
ring/alternating IsingZZ interactions
final mixer
Z/X/ZZ/XX observables
```

This ensures E9 can reproduce E8 behavior.

## Candidate Q1 — Data-reuploading quantum circuit

Implement a genuine data-reuploading architecture.

Suggested design:

```text
8 qubits
4 variational layers
16 encoded values
same encoded information reintroduced at each quantum layer

per layer:
    RY(data_i)
    RZ(data_j)
    trainable Rot(...)
    entanglement
```

Interleave data and trainable operations:

```text
encode
variational
entangle

encode
variational
entangle

...
```

Do NOT encode all data once at the beginning and then merely stack trainable gates.

The purpose of Q1 is to improve the accessible nonlinear/Fourier feature space.

## Candidate Q2 — Reuploading + noncommuting entanglement

Build on Q1 but alternate complementary entangling interactions.

For example:

```text
layer 1: IsingZZ ring
layer 2: IsingXX ring
layer 3: IsingZZ ring
layer 4: IsingXX ring
```

Use alternating ring pairings so qubits interact across the entire circuit rather than only fixed independent pairs.

The final measurement set should initially remain:

```text
<Z_i>
<X_i>
<Z_i Z_i+1>
<X_i X_i+1>
```

giving approximately 32 quantum observables.

Keeping the measurement dimensionality stable is important for fair control comparisons.

Save a text drawing of every candidate circuit.

---

# 6. INPUT ENCODING

E8 used a complementary PCA encoder orthogonal to the anchor classifier's row space.

Preserve this concept because it is scientifically useful.

The quantum branch should focus on DenseNet information not directly represented by the existing linear classifier.

For Q0 reproduce the E8 encoder.

For Q1/Q2, build an encoding appropriate for reuploading.

Prefer a compact complementary representation such as 16 or another justified small dimension.

Requirements:

- fit encoder ONLY using `fit`
- no `tune` or `report` leakage
- remain orthogonal/complementary to the anchor logit directions
- standardize using `fit` only
- bounded angle transformation
- record angle mean/std/min/max
- saturation statistics
- explained variance
- overlap with anchor space
- encoder rank
- output variance per dimension

Do not allow a huge trainable classical encoder to become the real model while the circuit does nothing.

The primary E9 QML architecture should use a **fixed or very tightly constrained encoder** so improvement can be attributed more cleanly to the circuit.

---

# 7. QUANTUM TRAINING CHANGES

E8 used roughly:

```text
core LR    = 1e-4
readout LR = 1e-4
```

and the quantum model often peaked within only a few epochs.

For E9, use more conservative circuit optimization.

A reasonable initial configuration is:

```text
readout LR = 1e-4
core LR    = 2e-5 to 5e-5
```

Choose ONE value before the final full E9 run and record it.

Use a readout warmup such as:

```text
2 epochs
```

during which:

```text
quantum core = frozen
readout      = trainable
```

Then unfreeze the quantum core.

Track separate learning rates for:

```text
readout
quantum core
```

Use:

- AdamW
- gradient clipping
- finite-value checks
- early stopping
- deterministic seeds
- masked BCEWithLogitsLoss
- existing positive-class weighting policy
- patient weighting

Do not unfreeze the DenseNet backbone.

The DenseNet remains frozen for E9.

---

# 8. RESIDUAL CONTROL

E8 showed residual magnitude continuing to grow after predictive performance peaked.

Make residual behavior an explicit part of E9.

Prefer a smaller primary bound than E8's 0.20, for example:

```text
RESIDUAL_LIMIT = 0.10
```

with the same smooth bounded formulation:

```python
residual = limit * tanh(raw / limit)
```

Do not hard clip.

Track every epoch:

```text
residual RMS
residual mean
residual std
residual max abs
base-logit RMS
residual/base RMS ratio
```

Add at least a small diagnostic ablation comparing:

```text
0.05
0.10
0.20
```

This diagnostic may use seed 42 only if running every combination across all seeds is prohibitively expensive.

The final selected residual limit must be chosen using `tune`, never `report`.

---

# 9. REQUIRED MODEL ARMS

The core E9 comparison must contain these arms.

## A. Experiment 2 source

Original frozen E2 classifier.

No retraining.

## B. Linear anchor/control

Exact E2 classifier expressed in the standardized frozen-feature space.

Must reproduce source probabilities before training.

## C. Parameter-matched classical nonlinear control

This is NEW and important.

Build a small classical nonlinear module receiving the EXACT SAME encoded input supplied to the quantum circuit and producing the EXACT SAME output dimensionality as the quantum observations.

Its trainable core parameter count should be as close as practical to the chosen quantum core.

Target:

```text
within approximately ±10%
```

If exact matching is awkward, report the exact counts and justify the difference.

This provides:

```text
quantum vs similarly sized classical nonlinear model
```

## D. Strong classical control

Keep a deliberately stronger classical nonlinear control similar in spirit to E8.

It may contain substantially more parameters than the quantum circuit.

It must receive the same encoded information and produce the same observation dimensionality.

This asks:

```text
does the quantum approach remain competitive with a more flexible ordinary ML head?
```

## E. Frozen quantum

Same circuit initialization.

Same encoder.

Same readout architecture.

Quantum circuit weights NEVER update.

Readout may train.

## F. Trainable quantum — PRIMARY MODEL

Selected E9 QML architecture.

Train:

```text
readout
+
quantum circuit
```

while keeping:

```text
DenseNet frozen
anchor classifier frozen
main encoder fixed/constrained
```

This is the main proposed model.

---

# 10. MANDATORY QUANTUM ABLATIONS

E9 must include formal ablations rather than only a circuit-reset check.

At minimum include:

### Circuit reset

After training, restore the quantum circuit parameters to their initial values while retaining the trained readout.

Performance should decline if learned circuit parameters matter.

### Frozen circuit

Already one core arm.

### No entanglement

Train the same circuit architecture with entangling gates disabled.

Everything else should remain matched.

This directly measures whether entanglement contributes.

### Correlation-observable removal

Evaluate/train a version using only local:

```text
Z
X
```

observables rather than:

```text
Z
X
ZZ
XX
```

This tests whether quantum correlation measurements contribute.

### No data reuploading

For a reuploading winner, compare against a corresponding single-encoding version.

### Random/complementary encoder diagnostic

Compare the chosen complementary encoder against a deterministic random orthogonal projection in at least a diagnostic arm.

This determines whether improvement comes mainly from the carefully chosen classical encoder.

Every ablation must be clearly labeled.

Never silently change multiple factors in one ablation.

---

# 11. CIRCUIT-DEPENDENCE DIAGNOSTICS

The most important E9 diagnostics are not merely gradient > 0.

For every quantum run, save:

```text
initial quantum parameters
selected/best quantum parameters
final quantum parameters
parameter change L2
parameter change relative L2
gradient L2/max per epoch
observable mean/std per dimension
observable covariance
observable variance
readout weight norm
residual RMS
```

Also measure prediction sensitivity to the trained circuit.

At the selected checkpoint evaluate:

```text
normal trained circuit
initial/reset circuit
frozen circuit
no-entanglement circuit where applicable
```

Compute differences in:

```text
macro AUPRC
macro AUROC
per-class AUPRC
per-class AUROC
prediction probability RMS difference
logit RMS difference
```

A learned circuit changing its parameters is NOT sufficient.

Its learned state must produce measurably useful predictive behavior.

---

# 12. ARCHITECTURE SELECTION PROTOCOL

Use all three main seeds:

```text
42
43
44
```

for the serious E9 comparison.

Candidate quantum architectures may be compared on `tune`.

Select the quantum architecture based primarily on:

```text
mean tune macro AUPRC across seeds
```

subject to:

```text
no major AUROC degradation
stable training
nonzero circuit learning
```

Record the entire candidate table.

Do not hide unsuccessful candidates.

After selecting the architecture, evaluate the chosen configuration and its matched controls/ablations on `report`.

Do not use `report` to go back and repeatedly redesign the architecture.

If a smoke-test mode is implemented, support something like:

```text
E9_SEEDS=42
```

but the official E9 run requires:

```text
42,43,44
```

---

# 13. METRICS — EXPAND THE REPORTING

Retain E8's AUROC/AUPRC metrics.

For every important arm report IMAGE-LEVEL and, where appropriate, PATIENT-LEVEL metrics.

## Ranking metrics

Per class and aggregate:

```text
AUROC
AUPRC / Average Precision
AUPRC lift over prevalence

macro AUROC
macro AUPRC

micro AUROC
micro AUPRC

macro AUROC for >=50-positive labels
macro AUPRC for >=50-positive labels
```

If weighted averages are implemented, clearly identify them as weighted rather than macro.

## Classification metrics

At BOTH:

```text
fixed threshold = 0.5
tune-fitted thresholds
```

report:

```text
Accuracy
Balanced Accuracy
Precision / PPV
Recall / Sensitivity / TPR
Specificity / TNR
NPV
F1
MCC
Jaccard / IoU
Hamming loss
exact-match accuracy where mathematically defined
subset accuracy
```

For multilabel accuracy, clearly distinguish:

```text
label-wise accuracy
exact-match/subset accuracy
```

Do not call one the other.

## Calibration

Add:

```text
Brier score
ECE / expected calibration error
```

and save calibration curves/reliability diagrams for the primary ensemble and major controls.

---

# 14. CONFUSION MATRICES — REQUIRED

Explicit confusion matrices are mandatory in E9.

For every label and important model arm calculate:

```text
TP
TN
FP
FN
```

at:

```text
threshold 0.5
tune-fitted threshold
```

Save machine-readable tables such as:

```text
report_confusion_matrices_05.csv
report_confusion_matrices_fitted.csv
```

Also create visual confusion matrices.

For the primary QML ensemble create a figure containing all 13 label confusion matrices.

Create both:

```text
raw counts
row-normalized values
```

If useful, use `sklearn.metrics.multilabel_confusion_matrix`.

Also save an aggregate micro-level binary confusion matrix over all observed label entries.

Do not include masked/uncertain entries in confusion-matrix counts.

---

# 15. CURVES AND FIGURES

Generate publication/research-friendly figures for at least the selected QML ensemble and major controls:

```text
ROC curves per class
PR curves per class
calibration curves
confusion matrices
training curves
AUPRC vs epoch
AUROC vs epoch
quantum gradient norm vs epoch
quantum parameter change vs epoch
residual RMS vs epoch
model comparison plot
per-class AUPRC comparison heatmap/bar chart
per-class AUROC comparison heatmap/bar chart
```

Do not use test data.

---

# 16. STATISTICS

Keep the patient-cluster bootstrap.

Increase to at least:

```text
1000 bootstrap samples
```

if runtime is reasonable.

Bootstrap by PATIENT, not image.

Report 95% confidence intervals.

Perform paired comparisons for the important contrasts:

```text
QML - E2 source
QML - linear control
QML - parameter-matched classical
QML - strong classical
QML - frozen quantum
QML - no-entanglement
QML - circuit-reset
```

At minimum compute paired intervals for:

```text
macro AUPRC
macro AUROC
stable-label macro AUPRC
stable-label macro AUROC
```

Also report point differences.

Do not describe a difference as established if its paired CI contains zero.

Seed variation and patient bootstrap variation are different quantities.

Report both separately.

---

# 17. E9 SUCCESS / E10 ADVANCEMENT GATE

Do not make Experiment 9 automatically succeed.

Predeclare the gate before viewing `report`.

The E9 configuration should be considered a strong candidate for Experiment 10 only if the selected QML model demonstrates evidence that the learned quantum component matters.

Use requirements along these lines:

```text
1. Quantum circuit parameters genuinely changed.

2. Circuit gradients are nonzero after unfreezing.

3. Resetting the trained circuit reduces report AUPRC.

4. Trainable QML exceeds frozen quantum.

5. Trainable QML exceeds the parameter-matched classical control.

6. QML improves meaningfully over the E2/linear anchor.
   Keep a meaningful target near the existing research standard,
   e.g. +0.003 macro AUPRC rather than moving the goalpost merely
   because E8 obtained a smaller result.

7. No unacceptable macro AUROC degradation.
   Preserve approximately the E8 tolerance, e.g. <=0.002 drop.

8. Ensemble patient-bootstrap comparison versus the key controls
   should favor QML, preferably with CI lower bound > 0 for the
   primary AUPRC comparisons.

9. No-entanglement / reset / frozen ablations should provide evidence
   that the learned circuit structure contributes rather than the
   readout alone.

10. Results should be reasonably consistent across seeds.
```

Do NOT weaken the gate after seeing results.

If E9 fails a criterion, record the failure honestly.

Experiment 10 can only make claims supported by E9.

---

# 18. IMPORTANT SCIENTIFIC INTERPRETATION

QML is mandatory for this project.

Do NOT add logic saying:

```text
if QML loses -> abandon QML
```

That is not the research plan.

Instead:

```text
Experiment 9 investigates which quantum design produces measurable value.
Experiment 10 will evaluate the final selected QML design.
```

Classical controls exist because a credible QML claim requires them.

They are not candidates for replacing QML as the project's novelty.

At the same time:

DO NOT write conclusions such as:

```text
quantum advantage proven
QML superior
quantum is better
```

unless the actual statistics support the precise claim being made.

Distinguish:

```text
predictive improvement
learned circuit contribution
parameter efficiency
statistical uncertainty
computational advantage
hardware quantum advantage
```

These are different claims.

This experiment can investigate the first three.

It does NOT establish computational or hardware quantum advantage.

---

# 19. OUTPUT FILES

Save enough information that Experiment 10 can be constructed without rerunning or guessing E9.

At minimum save:

```text
configuration.json
environment.json
dataset_manifest.csv
patient_level_splits.csv
e9_inner_patient_splits.csv

source_checkpoint_hash.json
feature_standardization.npz

architecture_candidates.csv
architecture_selection.json

quantum_circuit_Q0.txt
quantum_circuit_Q1.txt
quantum_circuit_Q2.txt

experiment_comparison.csv
seed_summary.csv
per_class_model_comparison.csv

paired_patient_bootstrap_differences.csv
bootstrap_ci.csv

quantum_ablation_metrics.csv
quantum_diagnostics.csv

thresholds.json
threshold_metrics.json

report_predictions.csv

report_per_class_metrics.csv
report_classification_per_class_05.csv
report_classification_per_class_fitted.csv

report_confusion_matrices_05.csv
report_confusion_matrices_fitted.csv

model_comparison.png
training_curves.png
quantum_training_diagnostics.png
confusion_matrices_counts.png
confusion_matrices_normalized.png
roc_curves.png
pr_curves.png
calibration_curves.png

protocol.json
summary.json
RESULTS.md

best_model.pt / inference bundle for each serious arm
```

Save the exact E9 source code into the run directory as well.

---

# 20. EXPERIMENT 10 HANDOFF

Create:

```text
e10_handoff.json
```

containing the exact locked recommendation produced by E9:

```text
selected quantum architecture
qubit count
layer count
entanglement design
encoding design
observable set
residual limit
core LR
readout LR
warmup epochs
selected epoch policy / epoch counts
seeds
thresholds
all preprocessing assumptions
source checkpoint SHA256
E9 run SHA256/config fingerprint
```

This is NOT Experiment 10.

Do not implement or run Experiment 10.

It is only a machine-readable handoff so E10 can later be built without silently changing the selected E9 design.

---

# 21. ROBUSTNESS / ENGINEERING REQUIREMENTS

Preserve E8's defensive engineering.

Specifically:

- deterministic seeds
- deterministic Torch behavior where supported
- no Torch reinstall
- PennyLane 0.45.1 compatibility
- CPU quantum heads acceptable
- CUDA frozen DenseNet extraction
- atomic checkpoint writes
- JSON without NaN/Infinity
- source hashes
- feature-cache hashes
- cache ordering validation
- finite tensor assertions
- strict checkpoint loading
- explicit label/order validation
- exact cohort validation against E2
- no silent missing-label handling changes
- no patient leakage
- no validation/test-based training
- no accidental test feature extraction
- no overwrite of earlier experiments

If the existing cached E2 frozen features are compatible and cryptographically/structurally verified, reuse them.

Do not unnecessarily repeat expensive DenseNet extraction.

---

# 22. FINAL CONSOLE SUMMARY

At completion print a concise but complete table showing:

```text
arm
seed
selected epoch
parameter count
quantum-core parameter count where applicable
report macro AUROC
report macro AUPRC
micro AUROC
micro AUPRC
accuracy
balanced accuracy
sensitivity
specificity
precision
F1
MCC
Brier
residual RMS
quantum parameter change
```

Then print ensemble comparisons:

```text
QML vs source
QML vs parameter-matched classical
QML vs strong classical
QML vs frozen quantum
QML vs circuit reset
QML vs no-entanglement
```

with:

```text
AUPRC delta
95% patient-bootstrap CI
AUROC delta
95% patient-bootstrap CI
```

Finally print:

```text
EXPERIMENT 9 E10 CANDIDATE: YES/NO
```

followed by every passed/failed gate individually.

A failed gate must not crash the completed experiment.

It should be recorded as a scientific result.

---

# 23. BEFORE CODING

First inspect the supplied Experiment 8 script thoroughly.

Then briefly report:

1. which E8 components will be reused unchanged;
2. which components will be refactored;
3. the exact E9 model arms;
4. the exact quantum candidate circuits;
5. the exact ablations;
6. the patient split protocol;
7. the metrics produced;
8. the E9 -> E10 gate.

Then implement Experiment 9 as one self-contained runnable Python script/notebook-compatible file.

Do not modify Experiment 8.

Do not implement Experiment 10.

The desired research progression is:

```text
E8:
Does a complementary quantum residual work at all?

↓

E9:
Can improved encoding/circuit design demonstrate that a LEARNED
quantum component adds useful information beyond rigorous
classical/random/frozen controls?

↓

E10:
Lock the best QML design and perform the final experiment.
```