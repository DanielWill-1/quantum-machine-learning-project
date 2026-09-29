# Experiment 8 (`c.py`)

E8 implements the rework in `b_rework.md`. Upload `c.py` to the same Vast.ai notebook and run it with the E2 dataset and checkpoint available. It creates a new `xray_training_experiment_8_complementary/run_...` directory. Set `MIMIC_CXR_ROOT` if automatic discovery does not find the dataset.

```python
%run c.py
```

This default **development** run uses a fixed patient split within the original training patients. It reports all four arms over seeds 42, 43, and 44: a full-feature linear anchor, a classical residual, a learned quantum residual, and a frozen-circuit residual. Every residual uses the same 48-feature encoder. The encoder projects DenseNet features away from directions used by the linear anchor, then fits PCA on the remaining features from the fit patients. E8 keeps the anchor and this encoder fixed while training each branch.

The readout trains for one epoch before the core can train. The learned core uses LR `1e-4`, and the correction is limited to 0.2 logits. Training weights images inversely to each patient's image count. It saves `best_model.pt` and `final_model.pt` separately. `training_history.csv` includes residual size, core gradient, and parameter change. `c.md`, per-label reports, a patient-level sensitivity report, and paired patient bootstrap differences show the result.

**Data limitation:** the inherited E2 backbone was trained on all original training patients, including the new inner development patients. This inner split helps develop the new residual branch but is **not an independent validation cohort**. E2 also used the original validation split for checkpoint selection, and E6/E7 examined it repeatedly. Treat both E8 development and confirmation scores as exploratory. The original test set remains untouched during development and confirmation.

For a cheaper diagnostic, set `%env E8_SEEDS=42` before running. Its `protocol.json` will be marked ineligible for confirmation because the declared three-seed comparison is incomplete. Reset with `%env E8_SEEDS=42,43,44` for a full run.

The full development run writes `protocol.json`. Its `eligible_for_confirmation` field is true only if the selected circuit actually changes, resetting it reduces AUPRC, the quantum arm gains at least 0.003 macro AUPRC over the anchor across seeds, it exceeds classical and frozen controls, AUROC does not materially fall, and the ensemble patient bootstrap intervals exclude zero. These are conservative gates, not proof of a quantum advantage.

If the development protocol is eligible, confirm it with fixed epochs and thresholds:

```python
%run c.py --stage confirm --protocol /path/to/development_run/protocol.json
```

Confirmation retrains on all original training patients and evaluates the original validation split once per fixed arm. It cannot use that split to choose a new epoch. It writes `confirmation_manifest.json` with an `eligible_for_final_test` field.

Only if that field is true, perform the single final test evaluation:

```python
%run c.py --stage final --confirmation-run /path/to/confirmation_run
```

This loads the confirmed checkpoints and reports every declared arm and ensemble on test patients. It records `test_evaluation_complete.json` in the confirmation folder after finishing, so the same confirmation run cannot be evaluated twice by the script. Do not choose a model or threshold from the test results.

The script reuses compatible E7 feature caches when possible, uses CUDA for frozen DenseNet extraction and CPU for the PennyLane circuit, and installs PennyLane 0.45.1 only if missing. It requires the same working Torch/torchvision stack and other Python packages listed at the top of `c.py`. Source E2 and E6/E7 files are not modified by this run.
