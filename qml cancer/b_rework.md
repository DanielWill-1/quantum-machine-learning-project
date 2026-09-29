# E7 review and proposed rework

Status: design only. This file does not change `b.py` or use the held-out test split.

## What `b.md` shows

The run finished normally: it reproduced the E2 checkpoint, reused the same frozen features, trained all 21 arm/seed runs, and saved comparisons. The failure is in the learned contribution of the residual branches, especially the quantum circuit.

| Validation macro AUPRC | E7 result | Interpretation |
|---|---:|---|
| Reproduced E2 | 0.294336 | Existing starting point |
| Linear anchor ensemble | 0.300829 | Most of the E7 improvement came from retraining the full feature classifier |
| PCA quantum ensemble | 0.301304 | Gain over anchor: **0.000475**; paired patient bootstrap 95% interval **[-0.000203, 0.001496]** |
| Task quantum ensemble | 0.300961 | Gain over anchor: **0.000132**; interval **[-0.000271, 0.000691]** |
| Task frozen-circuit ensemble | 0.300961 | Identical to task quantum ensemble |
| Task circuit without entanglement ensemble | 0.301052 | Slightly above the task quantum ensemble |

The selected task-quantum checkpoints are epoch **0, 0, and 2** for seeds 42, 43, and 44. In `b.py`, the circuit is frozen for epochs 1–3 and initially has a zero readout. Thus **none of the selected task models uses trained circuit parameters**. The matching frozen-circuit results and zero circuit-reset differences confirm this interpretation. PCA quantum selects epochs **2, 0, and 4**; only seed 44 retains a trained circuit, with a circuit-reset AUPRC difference of about **0.000003**.

Later training does happen: core gradients are nonzero in the log. It usually worsens validation ranking. For example, seed 42 PCA quantum falls from 0.301321 at epoch 2 to 0.30046 by epoch 15 while residual RMS rises from 0.016 to 0.076. PCA classical seed 42 falls from 0.301622 at epoch 2 to 0.29925 at epoch 15 as residual RMS rises from 0.023 to 0.350. That is evidence of a training/selection problem, although the log alone cannot establish the unique cause. It is **not** evidence that quantum circuits categorically cannot help this task.

## Likely causes and limitations

1. **Training changes too abruptly.** After three readout-only epochs, E7 unfreezes the encoder and core together, using core LR `1e-3`. The sharp residual growth and declining validation AUPRC after this point fit overfitting or an oversized update. The residual L2 coefficient is `0.05`, but the penalty is multiplied by the mean *squared logit*: at RMS 0.1, its contribution is only about 0.0005. This is an inference from the code and log, not a measured causal attribution.
2. **The task encoder repeats the anchor's information.** Its 13 task inputs are standardized logits of the same fixed linear classifier that already contributes the base prediction; the other 35 inputs are unsupervised principal components. The circuit can model nonlinear interactions, but the chosen inputs may provide little complementary signal. The 48 PCA components retain about 75% of training feature variance; explained variance does not measure retained pathology signal.
3. **The metric selects tiny differences.** Selection uses validation macro AUPRC alone. In seed 44, the selected PCA classical model gains about 0.000184 AUPRC over the anchor but loses about 0.0048 AUROC. The existing validation split has informed E2, E6, and E7, so tiny gains may reflect repeated model selection. The patient bootstrap intervals are conditional on the fitted checkpoints and omit retraining uncertainty.
4. **The circuit has not demonstrated a distinct effect.** Frozen and learned task models coincide at the selected checkpoints. The no-entanglement control is at least as good. Small nonzero gradients during training are insufficient evidence of a useful learned quantum component.

The available file is a console log. It does not contain the per-label prediction files, training losses, or saved parameter-change summaries. Those Vast.ai artifacts would help identify which labels and patients change, but they are not needed to conclude that E7 did not establish a quantum contribution.

## Proposed E8, once compute is available

### 1. Diagnose on existing E7 artifacts before another full run

Copy or inspect `experiment_comparison.csv`, `quantum_ablation_metrics.csv`, the per-arm `training_history.csv`, `training_summary.json`, and validation per-label metrics from the E7 output folder. Check the selected checkpoint's actual encoder/core parameter change, per-label AUPRC, train versus validation loss, and the patient distribution of score changes. This is read-only work and should determine whether one rare label caused the slight macro gain.

### 2. Make one encoder that targets information beyond the anchor

Keep the complete 1024-feature linear anchor. On **training patients only**, find the subspace spanned by its 13 weight vectors, subtract that component from the standardized DenseNet features, and fit a 48-component PCA to the remaining features. This supplies features from directions the linear anchor does not directly use. Use a fixed map and the same angle transformation for both quantum and classical arms. Record retained variance and angle distributions. This is a hypothesis about complementary information; it does not guarantee predictive signal.

Run only four arms per seed: the common linear anchor, a classical residual with the identical encoder, a learned quantum residual, and a frozen-circuit residual. Keep the classical core competitive and report its parameter count. Defer the no-entanglement arm unless the learned quantum circuit first shows a reproducible benefit. Do not sweep encoders, depths, or qubit counts over the same validation split.

### 3. Change training so circuit learning is actually evaluated

- Keep the encoder projection fixed for the first pilot. Give the readout one short warmup epoch, then train the readout and circuit together. Use a circuit LR around `1e-4`, one tenth of E7's starting value, and a readout LR around `1e-4`. Log train and validation masked loss, residual RMS, circuit parameter change, and AUPRC each epoch.
- Limit residual magnitude more tightly, provisionally to about **0.2 logits**, and monitor whether it stays small enough to preserve the anchor's ranking. Keep a frozen-circuit arm trained under the same schedule. Decide the cap and learning rates on training-patient development data before checking the original validation set.
- Try patient-balanced training weights so patients with many images do not dominate the branch update. Maintain the original image-level metrics for comparison and report a patient-level sensitivity analysis. Preserve label masks and the original cohort.
- Save both the best validation checkpoint and the final checkpoint; an early selected epoch must be visibly distinguished from later circuit learning. Do not infer contribution from the existence of gradients alone.

### 4. Spend validation and test information carefully

Make a fixed patient-level development split **inside the current training patients**. The inherited E2 backbone was trained on these patients, so this split guides residual development but is **not independent validation**. Choose the training schedule and single encoder there. Run the four fixed arms over seeds 42, 43, and 44. Compare macro AUPRC, macro AUROC, the 50+ positive-label subset, per-label scores, and paired patient bootstrap differences. Treat any improvement smaller than roughly **0.003 absolute macro AUPRC** as a weak exploratory signal rather than a win, especially if AUROC falls. A circuit claim also requires a nonzero selected circuit parameter change, an effect from resetting that circuit, and an advantage over both classical and frozen controls across seeds.

Once the procedure is fixed, evaluate it on the existing validation set as an exploratory check. The test set should be used **once**, only after the entire model and reporting rule are locked. Since previous experiments repeatedly used validation, even a positive E8 validation result is not an independent estimate. If the circuit still fails these checks, stop scaling it and put the next compute budget into improving the DenseNet representation and label handling, where the frozen-feature ceiling may be the larger constraint.

## Implementation order

1. Retrieve the E7 per-label and training-summary artifacts; confirm the diagnosis.
2. Build the training-only complementary encoder and identical classical control.
3. Implement the gradual training schedule and patient-balanced loss, with checkpoint and circuit-reset diagnostics.
4. Run one seed on the inner training-patient development split as a diagnostic, then all three prespecified seeds if learning is stable. Report every run.
5. Freeze the protocol before the exploratory validation check and any final test evaluation.

Evidence source: `b.md` from the Vast.ai E7 run and the local E7 implementation in `b.py`. No new cloud experiment was run for this review.
