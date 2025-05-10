import unittest

import numpy as np
import pandas as pd

from cross_validation import kfold_split, loo_split, cross_val_score_manual
from encoders import (
    label_encode,
    one_hot_encode,
    ordinal_encode,
)
from ensemble_manual import (
    BaggingRegressorManual,
    RandomForestRegressorManual,
    AdaBoostRegressorManual,
    GradientBoostingRegressorManual,
    StackingRegressorManual,
)
from gradient_descent import (
    batch_gradient_descent,
    stochastic_gradient_descent,
    momentum_gradient_descent,
)
from metrics import (
    mean_squared_error_manual,
    root_mean_squared_error_manual,
    mean_absolute_error_manual,
    median_absolute_error_manual,
    r2_score_manual,
    mean_absolute_percentage_error_manual,
    symmetric_mean_absolute_percentage_error_manual,
)
from neural_network import (
    linear_forward, relu_forward, identity_forward,
    linear_backward, relu_backward, compute_loss_mse, TwoLayerNet
)
from normalization import normalize_zscore, normalize_minmax
from regularization import lasso_regression, ridge_regression, elastic_net_regression, lp_regression


class DummyZero:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])


class DummyOne:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.ones(X.shape[0])


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    def test_mse(self):
        mse = mean_squared_error_manual(self.y_true, self.y_pred)
        expected = np.mean((self.y_true - self.y_pred) ** 2)
        self.assertAlmostEqual(mse, expected)

    def test_rmse(self):
        rmse = root_mean_squared_error_manual(self.y_true, self.y_pred)
        expected = np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
        self.assertAlmostEqual(rmse, expected)

    def test_mae(self):
        mae = mean_absolute_error_manual(self.y_true, self.y_pred)
        expected = np.mean(np.abs(self.y_true - self.y_pred))
        self.assertAlmostEqual(mae, expected)

    def test_medae(self):
        medae = median_absolute_error_manual(self.y_true, self.y_pred)
        expected = np.median(np.abs(self.y_true - self.y_pred))
        self.assertAlmostEqual(medae, expected)

    def test_r2(self):
        # perfect prediction
        r2 = r2_score_manual(self.y_true, self.y_true)
        self.assertAlmostEqual(r2, 1.0)
        # constant prediction → R2 <= 0
        r2_const = r2_score_manual(self.y_true, np.full_like(self.y_true, 2.5))
        self.assertLessEqual(r2_const, 0.0)

    def test_mape(self):
        mape = mean_absolute_percentage_error_manual(self.y_true, self.y_pred)
        ape = np.abs((self.y_true - self.y_pred) / self.y_true)
        expected = np.mean(ape) * 100
        self.assertAlmostEqual(mape, expected)

    def test_smape(self):
        smape = symmetric_mean_absolute_percentage_error_manual(self.y_true, self.y_pred)
        denom = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2
        ape = np.abs(self.y_true - self.y_pred) / denom
        expected = np.mean(ape) * 100
        self.assertAlmostEqual(smape, expected)


class TestCrossValidation(unittest.TestCase):
    def test_kfold_split_basic(self):
        X = list(range(10))
        splits = kfold_split(X, n_splits=5)
        self.assertEqual(len(splits), 5)
        all_test = []
        for tr, te in splits:
            # train+test covers all, no overlap
            self.assertTrue(set(tr).isdisjoint(te))
            all_test.extend(te)
        self.assertCountEqual(all_test, X)

    def test_loo_split(self):
        X = list(range(5))
        splits = loo_split(X)
        self.assertEqual(len(splits), 5)
        for tr, te in splits:
            self.assertEqual(len(te), 1)
            self.assertTrue(te[0] in X)

    def test_cross_val_score_manual(self):
        class DummyModel:
            def fit(self, X, y): self.mean_ = np.mean(y)

            def predict(self, X): return np.full(len(X), self.mean_)

        X = np.arange(10).reshape(-1, 1)
        y = np.linspace(0, 9, 10)
        scores = cross_val_score_manual(DummyModel(), X, y,
                                        cv=kfold_split(X, 5),
                                        scoring=mean_squared_error_manual)

        cv = kfold_split(X, 5)
        expected = []
        for train_idx, test_idx in cv:
            y_train = y[train_idx]
            y_test = y[test_idx]
            pred = np.full_like(y_test, np.mean(y_train))
            expected.append(mean_squared_error_manual(y_test, pred))

        self.assertEqual(len(scores), len(expected))
        for s, e in zip(scores, expected):
            self.assertAlmostEqual(s, e)


class TestGradientDescent(unittest.TestCase):
    def setUp(self):
        self.X = np.vstack([np.ones(20), np.linspace(0, 10, 20)]).T
        self.y = 3 + 2 * self.X[:, 1]


    def test_sgd(self):
        w_hist, loss_hist = stochastic_gradient_descent(self.X, self.y, alpha=0.01, n_iters=5000)
        w_final = w_hist[-1]
        self.assertAlmostEqual(w_final[0], 3, places=1)
        self.assertAlmostEqual(w_final[1], 2, places=1)


    def test_momentum_gd(self):
        w_hist, loss_hist = momentum_gradient_descent(self.X, self.y, alpha=0.05, beta=0.9, n_iters=500)
        w_final = w_hist[-1]
        self.assertAlmostEqual(w_final[0], 3, places=2)
        self.assertAlmostEqual(w_final[1], 2, places=2)


class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'col': ['a', 'b', 'a', 'c', 'b', 'a']
        })
        self.y = pd.Series([1, 2, 3, 4, 5, 6])

    def test_label_encode(self):
        le = label_encode(self.df['col'])
        self.assertTrue(set(le.unique()) <= set(range(len(le.unique()))))

    def test_one_hot_encode(self):
        oh = one_hot_encode(self.df['col'], drop_first=False)
        self.assertIn('a', oh.columns)
        self.assertIn('b', oh.columns)
        self.assertIn('c', oh.columns)

    def test_one_hot_encode_drop(self):
        oh = one_hot_encode(self.df['col'], drop_first=True)
        self.assertEqual(oh.shape[1], len(self.df['col'].unique()) - 1)

    def test_ordinal_encode(self):
        mapping = {'a': 1, 'b': 2, 'c': 3}
        oe = ordinal_encode(self.df['col'], mapping=mapping)
        self.assertTrue((oe == self.df['col'].map(mapping)).all())

class TestRegularization(unittest.TestCase):
    def setUp(self):
        self.X = np.vstack([np.ones(10), np.linspace(0, 9, 10)]).T
        self.y = 2 * self.X[:, 1] + 1

    def test_ridge_zero_alpha(self):
        w_ridge, _ = ridge_regression(self.X, self.y, alpha=0, alpha_lr=0.1, n_iters=500)
        w_gd, _ = batch_gradient_descent(self.X, self.y, alpha=0.1, n_iters=500)
        np.testing.assert_allclose(w_ridge[-1], w_gd[-1], atol=1e-2)

    def test_lasso_stability(self):
        w_lasso, _ = lasso_regression(self.X, self.y, alpha=1e3, alpha_lr=0.1, n_iters=100)
        self.assertTrue(np.allclose(w_lasso[-1], 0, atol=1e-3))

    def test_elastic_net(self):
        w_en, _ = elastic_net_regression(self.X, self.y, alpha1=0, alpha2=0, alpha_lr=0.1, n_iters=500)
        w_gd, _ = batch_gradient_descent(self.X, self.y, alpha=0.1, n_iters=500)
        np.testing.assert_allclose(w_en[-1], w_gd[-1], atol=1e-2)

    def test_lp_general(self):
        w_lp, _ = lp_regression(self.X, self.y, alpha=1.0, p=2, alpha_lr=0.1, n_iters=500)
        w_ridge, _ = ridge_regression(self.X, self.y, alpha=1.0, alpha_lr=0.1, n_iters=500)
        np.testing.assert_allclose(w_lp[-1], w_ridge[-1], atol=1e-2)

class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], name="s")
        self.df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [2, 4, 6, 8]
        })

    def test_zscore_series(self):
        S_norm, (mean, std) = normalize_zscore(self.series)
        np.testing.assert_allclose(mean, self.series.mean())
        np.testing.assert_allclose(std, self.series.std())
        self.assertAlmostEqual(S_norm.mean(), 0.0, places=6)
        self.assertAlmostEqual(S_norm.std(), 1.0, places=6)

    def test_zscore_dataframe(self):
        DF_norm, (mean_df, std_df) = normalize_zscore(self.df)
        expected_mean = self.df.mean()
        expected_std  = self.df.std()
        pd.testing.assert_series_equal(mean_df, expected_mean)
        pd.testing.assert_series_equal(std_df, expected_std)
        for col in DF_norm:
            self.assertAlmostEqual(DF_norm[col].mean(), 0.0, places=6)
            self.assertAlmostEqual(DF_norm[col].std(), 1.0, places=6)

    def test_zscore_with_ref_stats(self):
        ref_mean = 10.0
        ref_std = 2.0
        S_norm, stats = normalize_zscore(self.series, ref_stats=(ref_mean, ref_std))
        self.assertEqual(stats, (ref_mean, ref_std))
        expected = (self.series - ref_mean) / ref_std
        pd.testing.assert_series_equal(S_norm, expected)

    def test_zscore_zero_std(self):
        const = pd.Series([5, 5, 5])
        S_norm, (m, s) = normalize_zscore(const)
        self.assertTrue((s == 1).all() if isinstance(s, pd.Series) else s == 1)
        self.assertTrue((S_norm == 0).all())

    def test_minmax_series(self):
        S_norm, (mn, mx) = normalize_minmax(self.series)
        self.assertEqual(mn, self.series.min())
        self.assertEqual(mx, self.series.max())
        self.assertAlmostEqual(S_norm.min(), 0.0, places=6)
        self.assertAlmostEqual(S_norm.max(), 1.0, places=6)

    def test_minmax_dataframe(self):
        DF_norm, (min_df, max_df) = normalize_minmax(self.df)
        pd.testing.assert_series_equal(min_df, self.df.min())
        pd.testing.assert_series_equal(max_df, self.df.max())
        for col in DF_norm:
            self.assertAlmostEqual(DF_norm[col].min(), 0.0, places=6)
            self.assertAlmostEqual(DF_norm[col].max(), 1.0, places=6)

    def test_minmax_with_ref_stats(self):
        ref_min = 0.0
        ref_max = 10.0
        S_norm, stats = normalize_minmax(self.series, ref_stats=(ref_min, ref_max))
        self.assertEqual(stats, (ref_min, ref_max))
        expected = (self.series - ref_min) / (ref_max - ref_min)
        pd.testing.assert_series_equal(S_norm, expected)

    def test_minmax_zero_range(self):
        const = pd.Series([7, 7, 7])
        S_norm, (mn, mx) = normalize_minmax(const)
        self.assertTrue((mn == const.min()).all() if isinstance(mn, pd.Series) else mn == const.min())
        self.assertTrue((mx == const.max()).all() if isinstance(mx, pd.Series) else mx == const.max())
        self.assertTrue((S_norm == 0).all())

class TestEnsembleManual(unittest.TestCase):
        def setUp(self):
            self.X = np.arange(8).reshape(4, 2)
            self.y = np.array([0.0, 1.0, 2.0, 3.0])

        def test_bagging_constant(self):
            bag = BaggingRegressorManual(DummyOne(), n_estimators=5,
                                         max_samples=1.0, random_state=42)
            bag.fit(self.X, self.y)
            preds = bag.predict(self.X)
            self.assertTrue(np.all(preds == 1.0))

        def test_random_forest_constant(self):
            rf = RandomForestRegressorManual(DummyZero(), n_estimators=3,
                                             max_samples=1.0, max_features=0.5,
                                             random_state=0)
            rf.fit(self.X, self.y)
            preds = rf.predict(self.X)
            self.assertTrue(np.all(preds == 0.0))

        def test_adaboost_zero_base(self):
            ada = AdaBoostRegressorManual(DummyZero(), n_estimators=10,
                                          learning_rate=0.2)
            ada.fit(self.X, self.y)
            preds = ada.predict(self.X)
            self.assertTrue(np.allclose(preds, 0.0))

        def test_gradient_boosting_zero_base(self):
            gb = GradientBoostingRegressorManual(DummyZero(), n_estimators=5,
                                                 learning_rate=0.3)
            gb.fit(self.X, self.y)
            preds = gb.predict(self.X)
            self.assertTrue(np.allclose(preds, self.y.mean()))

        def test_stacking_simple(self):
            base0 = DummyZero()
            base1 = DummyOne()

            class Meta:
                def fit(self, X_meta, y):
                    self.coef_ = np.array([0.0, 1.0])
                    self.intercept_ = 0.0

                def predict(self, X_meta):
                    return X_meta.dot(self.coef_) + self.intercept_

            stack = StackingRegressorManual(
                base_estimators=[base0, base1],
                meta_estimator=Meta(),
                cv=2,
                shuffle=False,
                random_state=1
            )
            stack.fit(self.X, self.y)
            preds = stack.predict(self.X)
            self.assertTrue(np.allclose(preds, 1.0))

class TestNNUtils(unittest.TestCase):
    def test_linear_forward_backward(self):
        X = np.array([[1., 2.], [3., 4.]])
        W = np.eye(2)
        b = np.array([1., -1.])
        Z, cache = linear_forward(X, W, b)
        np.testing.assert_allclose(Z, X + b)
        dZ = np.ones_like(Z)
        dX, dW, db = linear_backward(dZ, cache)
        np.testing.assert_allclose(dW, X.T.dot(np.ones((2,2))) / 2)
        np.testing.assert_allclose(db, np.array([2., 2.]) / 2)

    def test_relu_forward_backward(self):
        Z = np.array([[-1., 2.], [0., -3.]])
        A, cache = relu_forward(Z)
        np.testing.assert_allclose(A, np.array([[0., 2.], [0., 0.]]))
        dA = np.ones_like(A)
        dZ = relu_backward(dA, cache)
        np.testing.assert_allclose(dZ, np.array([[0., 1.], [0., 0.]]))

    def test_identity_forward(self):
        Z = np.random.randn(3,2)
        A, _ = identity_forward(Z)
        np.testing.assert_allclose(A, Z)

    def test_compute_loss_mse(self):
        y = np.array([1., 3., 5.])
        pred = np.array([2., 2., 6.])
        loss, dA = compute_loss_mse(pred, y)
        self.assertAlmostEqual(loss, 0.5)
        np.testing.assert_allclose(dA, np.array([1/3, -1/3, 1/3]))

    def test_two_layer_net(self):
        X = np.array([[1.],[2.]])
        y = np.array([1., 2.])
        net = TwoLayerNet(n_input=1, n_hidden=2, n_output=1)
        # One forward + backward + update step
        A2 = net.forward(X)
        loss_before, grads = net.backward(A2, y)
        net.update_parameters(grads, lr=0.1)
        A2_after = net.forward(X)
        loss_after, _ = compute_loss_mse(A2_after, y)
        self.assertLessEqual(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main()
