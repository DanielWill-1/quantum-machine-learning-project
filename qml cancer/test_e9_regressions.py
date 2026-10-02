"""Synthetic regression tests only: no dataset access or E9 experiment execution.
Run: python "qml cancer/test_e9_regressions.py"
"""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('e9', Path(__file__).with_name('e9.py'))
e9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e9)


def fake_data():
    torch.manual_seed(42)
    features = {s: torch.randn(4, 1024) for s in ('fit', 'tune', 'report')}
    y = torch.tensor([[0.] * 13, [1.] * 13] * 2)
    data = SimpleNamespace(
        original_weight=torch.randn(13, 1024) * .001, original_bias=torch.zeros(13),
        feature_mean=np.zeros(1024, np.float32), feature_scale=np.ones(1024, np.float32),
        features=features, targets={s: y for s in features},
        masks={s: torch.ones_like(y) for s in features}, patient_weights=torch.ones(4),
        pos_weights=np.ones(13, np.float32), source_hash='synthetic', backbone_state={})
    def result(split, p):
        return dict(targets=y.numpy(), masks=np.ones_like(y.numpy()), probabilities=p,
                    subject_ids=np.array(['a', 'a', 'b', 'b']),
                    dicom_ids=np.array(['1', '2', '3', '4']),
                    metrics=e9.calculate_metrics(y.numpy(), p, np.ones_like(y.numpy())))
    data.result = result
    return data


class RegressionTests(unittest.TestCase):
    def test_every_candidate_and_arm(self):
        data = fake_data()
        for candidate in e9.CANDIDATES:
            dim = 48 if candidate == 'Q0' else 16
            mapping = (torch.randn(dim, 1024) * .01, torch.zeros(dim))
            for kind in ('quantum', 'frozen_quantum', 'no_entanglement', 'local_only',
                         'single_encode', 'random_encoder', 'matched_classical', 'strong_classical'):
                with self.subTest(candidate=candidate, kind=kind):
                    model = e9.Head(data, kind, 42, candidate, mapping,
                                    entangled=kind != 'no_entanglement',
                                    correlations=kind != 'local_only', reupload=kind != 'single_encode')
                    e9.init_scaling(model, data.features['fit'])
                    with torch.no_grad():
                        model.readout.weight.fill_(.01)
                    output = model(data.features['fit'])
                    self.assertEqual(tuple(output.shape), (4, 13))
                    self.assertTrue(torch.isfinite(output).all())
                    output.sum().backward()
                    for p in model.parameters():
                        if p.grad is not None:
                            self.assertTrue(torch.isfinite(p.grad).all())
                    model.eval()
                    cloned = copy.deepcopy(model)
                    self.assertTrue(torch.allclose(model(data.features['fit']), cloned(data.features['fit'])))
                    self.assertEqual(tuple(model(data.features['fit'][:1]).shape), (1, 13))
                    if kind == 'frozen_quantum':
                        self.assertTrue(all(p.grad is None for p in model.core.parameters()))

    def test_main_output_pipeline(self):
        # Exercise selection, Q0 random ablation, reporting and handoff without
        # real data or an architecture training run. Preserve real model forwards.
        data = fake_data()
        def encoder_stub(data, dim, random_projection=False):
            return (torch.randn(dim, 1024)*.01, torch.zeros(dim)), {'dim': dim}
        def train_stub(model, data, seed, directory):
            directory.mkdir(parents=True)
            e9.init_scaling(model, data.features['fit'])
            initial = e9.core_vector(model) if model.kind != 'linear_control' else None
            e9.save_checkpoint(model, data, seed, 0, directory)
            return dict(epoch=0, initial=initial, final=initial, history=[], gradient_nonzero=False)
        bootstrap = e9.patient_bootstrap
        with tempfile.TemporaryDirectory() as tmp:
            data.output = Path(tmp)
            data.source_path = Path(tmp)/'synthetic_source.pt'
            data.source_path.write_bytes(b'synthetic')
            data.source_epoch = 1
            data.cache_dir = Path(tmp)/'cache'
            data.cache_dir.mkdir()
            np.save(data.cache_dir/'train.npy', np.zeros((4,1024),np.float32))
            data.baseline = data.result('tune', np.full((4,13),.5,np.float32))
            with patch.object(e9,'prepare_data',return_value=data), \
                 patch.object(e9,'encoder_fit',side_effect=encoder_stub), \
                 patch.object(e9,'train',side_effect=train_stub), \
                 patch.object(e9,'SEEDS',[42]), \
                 patch.object(e9,'plot_results'), \
                 patch.object(e9,'patient_bootstrap',side_effect=lambda r,d: bootstrap(r,d,samples=2)), \
                 contextlib.redirect_stdout(io.StringIO()):
                e9.main()
            for filename in ('summary.json','e10_handoff.json','report_predictions.csv',
                             'report_confusion_matrices_fitted.csv','quantum_diagnostics.csv'):
                self.assertTrue((data.output/filename).is_file(), filename)

    def test_wrong_mapping_rejected_early(self):
        with self.assertRaisesRegex(ValueError, 'expected encoder'):
            e9.Head(fake_data(), 'random_encoder', 42, 'Q0',
                    (torch.zeros(16, 1024), torch.zeros(16)))

    def test_empty_fitted_thresholds(self):
        data = fake_data()
        result = data.result('report', np.full((4, 13), .5, np.float32))
        thresholds, _ = e9.fit_validation_thresholds(result)
        self.assertTrue(all(t is None for t in thresholds.values()))
        summary, rows = e9.extended(result, thresholds)
        self.assertEqual(summary['evaluated_labels'], 0)
        self.assertTrue({'label', 'TP', 'TN', 'FP', 'FN'}.issubset(rows.columns))

    def test_tiny_training_checkpoint(self):
        data = fake_data()
        mapping = (torch.randn(48, 1024) * .01, torch.zeros(48))
        model = e9.Head(data, 'random_encoder', 42, 'Q0', mapping)
        names = ('MAX_EPOCHS', 'MIN_EPOCHS', 'WARMUP_EPOCHS')
        saved = {k: getattr(e9, k) for k in names}
        try:
            e9.MAX_EPOCHS, e9.MIN_EPOCHS, e9.WARMUP_EPOCHS = 2, 1, 1
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / 'arm'
                record = e9.train(model, data, 42, directory)
                self.assertTrue(record['gradient_nonzero'])
                bundle = torch.load(directory / 'best_model.pt', weights_only=True)
                model.load_state_dict(bundle['state_dict'], strict=True)
                self.assertEqual(e9.evaluate(model, data, 'report')['probabilities'].shape, (4, 13))
        finally:
            for k, v in saved.items():
                setattr(e9, k, v)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
