"""Independent synthetic checks; no model or private dataset required."""
import copy
import unittest

import numpy as np
from scipy.stats import kendalltau

from thesis_exp.exp54_rar_sft.statistical_closure import (
    bootstrap_metrics, centered_p, cluster_codes, cluster_tensors,
    compare_historical, confusion_metrics, holm, interval, validate_predictions,
)


def matrix(gold, pred):
    c = np.zeros((5, 5))
    for g, p in zip(gold, pred, strict=True):
        c[g-1, p-1] += 1
    return c


class MetricTests(unittest.TestCase):
    def test_random_metrics_match_independent_row_computations(self):
        rng = np.random.default_rng(123)
        for n in (5, 20, 101, 664):
            for _ in range(10):
                g, p = rng.integers(1, 6, (2, n))
                m = confusion_metrics(matrix(g, p))
                self.assertAlmostEqual(float(m['MAE']), np.mean(abs(g-p)))
                self.assertAlmostEqual(float(m['Exact']), np.mean(g == p))
                self.assertAlmostEqual(float(m['Signed_Bias']), np.mean(p-g))
                self.assertAlmostEqual(float(m['Kendall_tau_b']), kendalltau(g, p).statistic)
                # Independent QWK via expected squared error of independent marginals.
                expected = np.mean(g*g) + np.mean(p*p) - 2*np.mean(g)*np.mean(p)
                self.assertAlmostEqual(float(m['QWK']), 1-np.mean((g-p)**2)/expected)

    def test_vectorized_matches_individual_matrices(self):
        rng = np.random.default_rng(42)
        c = rng.integers(0, 20, (3, 4, 5, 5))
        batched = confusion_metrics(c)
        for i in range(3):
            for j in range(4):
                for name, v in confusion_metrics(c[i, j]).items():
                    self.assertAlmostEqual(batched[name][i, j], v)

    def test_undefined_subgroup_and_rank_not_zero(self):
        m = confusion_metrics(matrix([5]*5, [5]*5))
        for k in ('L2H_rate', 'Label_2_Recall', 'Kendall_tau_b', 'QWK'):
            self.assertTrue(np.isnan(m[k]))
        self.assertEqual(m['MAE'], 0)

    def test_low_and_high_error_definitions(self):
        m = confusion_metrics(matrix([1, 2, 2, 3, 4, 5], [4, 3, 5, 1, 2, 5]))
        self.assertEqual(m['L2H_rate'], 2/3)
        self.assertEqual(m['H2L_rate'], 1/2)


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(record_id=f'r{i}', question_key=q, answer_key=a)
                     for i, (q, a) in enumerate([('a', 'x'), ('a', 'x'), ('a', 'y'), ('b', 'x')])]

    def test_question_and_qa_membership(self):
        self.assertEqual(cluster_codes(self.rows, 'question').tolist(), [0, 0, 0, 1])
        self.assertEqual(cluster_codes(self.rows, 'qa').tolist(), [0, 0, 1, 2])
        self.assertEqual(cluster_codes(self.rows, 'record').tolist(), [0, 1, 2, 3])

    def test_qa_tuple_has_no_delimiter_ambiguity(self):
        rows = [dict(record_id='x', question_key='a|b', answer_key='c'),
                dict(record_id='y', question_key='a', answer_key='b|c')]
        self.assertEqual(len(set(cluster_codes(rows, 'qa'))), 2)

    def test_missing_key_rejected(self):
        self.rows[0]['question_key'] = ''
        with self.assertRaises(ValueError):
            cluster_codes(self.rows, 'question')

    def test_unequal_cluster_sizes_keep_record_weighting(self):
        g = np.array([1, 2, 3, 4])
        p = np.array([[1, 2, 3, 5]])
        c, f = cluster_tensors(g, p, np.zeros_like(p), cluster_codes(self.rows, 'question'))
        # Sampling question A twice and B once expands to seven records.
        expanded = [0, 1, 2, 0, 1, 2, 3]
        combined = c[0]*2 + c[1]
        m = confusion_metrics(combined)
        e = confusion_metrics(matrix(g[expanded], p[0, expanded]))
        for k in m:
            np.testing.assert_allclose(m[k][0], e[k], equal_nan=True)
        self.assertAlmostEqual(float(m['MAE'][0]), 1/7)
        self.assertNotEqual(float(m['MAE'][0]), 1/3)  # not average cluster MAE

    def test_identical_models_stay_paired_and_deterministic(self):
        g = np.array([1, 2, 3, 4])
        pred = np.array([[2, 2, 3, 5]]*3)
        forced = np.array([[0, 0, 1, 0]]*3)
        c, f = cluster_tensors(g, pred, forced, cluster_codes(self.rows, 'question'))
        a, b = bootstrap_metrics(c, f, replicates=201), bootstrap_metrics(c, f, replicates=201)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
            np.testing.assert_array_equal(a[k][:, 0], a[k][:, 2])

    def test_full_sample_kendall_not_average_cluster_kendall(self):
        a, b = matrix([1, 2], [1, 2]), matrix([4, 5], [5, 4])
        full = confusion_metrics(a+b)['Kendall_tau_b']
        wrong = (confusion_metrics(a)['Kendall_tau_b']+confusion_metrics(b)['Kendall_tau_b'])/2
        self.assertNotAlmostEqual(float(full), float(wrong))

    def test_fixed_seed_metrics_not_ensemble_metric(self):
        c = np.array([matrix([1, 2], [1, 2]), matrix([1, 2], [3, 4]), matrix([1, 2], [1, 2])])
        self.assertAlmostEqual(float(confusion_metrics(c)['Exact'].mean()), 2/3)
        self.assertNotEqual(float(confusion_metrics(c)['Exact'].mean()), 0.0)


class InferenceTests(unittest.TestCase):
    def test_holm_known_example_and_missing_preserves_family_size(self):
        np.testing.assert_allclose(holm([.01, .04, .03, None]), [.04, .09, .09, 1])

    def test_undefined_bootstraps_counted(self):
        r = interval(np.array([0., 1., np.nan]), minimum=3)
        self.assertFalse(r['minimum_valid_met'])
        self.assertIsNone(r['ci95_low'])
        self.assertEqual(r['undefined_replicates'], 1)
        self.assertIsNone(centered_p(np.array([0, 1, np.nan]), 1, minimum=3))

    def test_centered_p_has_plus_one_and_null_behavior(self):
        self.assertEqual(centered_p(np.array([1., 1., 1.]), 1, minimum=3), .25)
        self.assertEqual(centered_p(np.array([-1., 0., 1.]), 0, minimum=3), 1)

    def test_historical_metric_tamper_rejected(self):
        metrics = {k: float(v) for k, v in confusion_metrics(matrix([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])).items()}
        self.assertGreater(compare_historical(metrics, metrics), 5)
        bad = dict(metrics, MAE=.1)
        with self.assertRaises(ValueError):
            compare_historical(metrics, bad)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = [dict(record_id=f'r{i}', label_5=i+1, metric_id='m', language='zh') for i in range(3)]
        self.pred = [dict(**r, row_position=i, parse_success=True, forced_completion=False,
                          prediction={'score': i+1}) for i, r in enumerate(self.source)]

    def test_valid_and_forced_close_not_excluded(self):
        self.pred[0]['forced_completion'] = True
        scores, flags = validate_predictions(self.pred, self.source)
        self.assertEqual(len(scores), 3)
        self.assertEqual(flags.sum(), 1)

    def test_identity_order_label_and_parse_tamper(self):
        for field, value in [('record_id', 'wrong'), ('row_position', -1), ('label_5', 2),
                             ('parse_success', False), ('forced_completion', 0), ('metric_id', 'other')]:
            with self.subTest(field=field):
                bad = copy.deepcopy(self.pred); bad[0][field] = value
                with self.assertRaises(ValueError):
                    validate_predictions(bad, self.source)
        with self.assertRaises(ValueError):
            validate_predictions(self.pred[::-1], self.source)

    def test_score_type_and_range(self):
        for value in (True, 1.0, '1', 0, 6):
            self.pred[0]['prediction']['score'] = value
            with self.assertRaises(ValueError):
                validate_predictions(self.pred, self.source)

    def test_duplicate_source_and_missing_prediction(self):
        with self.assertRaises(ValueError):
            validate_predictions(self.pred[:-1], self.source)
        self.source[1]['record_id'] = self.source[0]['record_id']
        with self.assertRaises(ValueError):
            validate_predictions(self.pred, self.source)


if __name__ == '__main__':
    unittest.main()
