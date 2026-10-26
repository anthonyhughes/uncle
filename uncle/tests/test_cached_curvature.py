"""Unit tests for CachedCurvature."""

import os
import tempfile
import unittest

import numpy as np
from scipy.linalg import cho_solve

from uncle.simplified.cached_curvature import CachedCurvature
from tests.certifier_fixtures import retrain_exact


def _gaussian_pac(n, d, lam, rng):
    theta_star = rng.standard_normal(d)
    theta_star /= np.linalg.norm(theta_star)
    X = rng.standard_normal((n, d))
    p = 1.0 / (1.0 + np.exp(-(X @ theta_star)))
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return X, y, theta_star


class TestCachedCurvatureBuild(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n, self.d, self.lam = 200, 8, 0.1
        self.X, self.y, _ = _gaussian_pac(self.n, self.d, self.lam, self.rng)
        self.theta_hat = retrain_exact(self.X, self.y, self.lam, self.d)

    def test_build_shapes(self):
        c = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        self.assertEqual(c.theta_0.shape, (self.d,))
        self.assertEqual(c.H_chol_L.shape, (self.d, self.d))
        self.assertEqual(c.G.shape, (self.n, self.n))
        self.assertEqual(c.w_full.shape, (self.n,))
        self.assertEqual(c.G_diag.shape, (self.n,))

    def test_G_correctness(self):
        """G should equal X @ H^-1 @ X.T to high precision."""
        c = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        # Recompute G via Cholesky-solved form
        H_inv_Xt = cho_solve((c.H_chol_L, True), self.X.T, check_finite=False)
        G_ref = self.X @ H_inv_Xt
        G_ref = 0.5 * (G_ref + G_ref.T)
        np.testing.assert_allclose(c.G, G_ref, rtol=1e-10, atol=1e-12)

    def test_G_is_symmetric_psd(self):
        c = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        # Symmetry
        np.testing.assert_allclose(c.G, c.G.T, atol=1e-12)
        # PSD-ish: eigenvalues should be ≥ 0 (allowing tiny negative round-off)
        eigvals = np.linalg.eigvalsh(c.G)
        self.assertGreater(eigvals.min(), -1e-9)

    def test_constants_match_recomputation(self):
        c = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        np.testing.assert_allclose(c.G_diag, np.diag(c.G), atol=1e-12)
        absG = np.abs(c.G)
        np.fill_diagonal(absG, 0.0)
        self.assertAlmostEqual(c.C_O, float(np.max(absG)), places=10)
        op = float(np.max(np.abs(np.linalg.eigvalsh(c.G))))
        self.assertAlmostEqual(c.C_op, op, places=8)

    def test_logistic_weights(self):
        c = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        p_ref = 1.0 / (1.0 + np.exp(-(self.X @ self.theta_hat)))
        np.testing.assert_allclose(c.w_full, p_ref * (1.0 - p_ref), atol=1e-12)


class TestCachedCurvatureSerialisation(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.n, self.d, self.lam = 100, 5, 0.5
        self.X, self.y, _ = _gaussian_pac(self.n, self.d, self.lam, self.rng)
        self.theta_hat = retrain_exact(self.X, self.y, self.lam, self.d)

    def test_save_load_roundtrip(self):
        c = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.npz")
            c.save(path)
            self.assertTrue(os.path.isfile(path))
            c2 = CachedCurvature.load(path)
            with np.load(path, allow_pickle=False) as saved:
                for name in (
                    "H", "p_full", "gamma_full", "C_D",
                    "gamma_anchor", "C_D_anchor", "C_O_anchor", "C_op_anchor",
                ):
                    self.assertNotIn(name, saved.files)
        np.testing.assert_array_equal(c.G, c2.G)
        np.testing.assert_array_equal(c.H_chol_L, c2.H_chol_L)
        np.testing.assert_array_equal(c.theta_0, c2.theta_0)
        np.testing.assert_array_equal(c.w_full, c2.w_full)
        np.testing.assert_array_equal(c.G_diag, c2.G_diag)
        self.assertEqual(c.C_O, c2.C_O)
        self.assertEqual(c.C_op, c2.C_op)
        self.assertEqual(c.n, c2.n)
        self.assertEqual(c.d, c2.d)
        self.assertEqual(c.lam, c2.lam)
        self.assertEqual(c.loss_fn_id, c2.loss_fn_id)

    def test_load_rejects_version_mismatch(self):
        c = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.npz")
            c.save(path)
            # Mutate version on disk
            data = dict(np.load(path, allow_pickle=False))
            data["version"] = np.array(99)
            np.savez_compressed(path, **data)
            with self.assertRaises(ValueError):
                CachedCurvature.load(path)


class TestCachedCurvatureUnsupported(unittest.TestCase):
    def test_unsupported_loss_raises(self):
        rng = np.random.default_rng(2)
        n, d, lam = 50, 3, 0.1
        X, y, _ = _gaussian_pac(n, d, lam, rng)
        theta = np.zeros(d)
        with self.assertRaises(NotImplementedError):
            CachedCurvature.build(X, y, theta, lam, loss_fn_id="hinge")


class TestCachedCurvatureLegacyFiles(unittest.TestCase):
    """Older cache files still load. Extra historical arrays are ignored."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n, self.d, self.lam = 80, 4, 0.1
        self.X, self.y, _ = _gaussian_pac(self.n, self.d, self.lam, self.rng)
        self.theta_hat = retrain_exact(self.X, self.y, self.lam, self.d)

    def _legacy_payload(self, cache, *, version, include_diag):
        payload = dict(
            theta_0=cache.theta_0,
            H=np.eye(cache.d),
            H_chol_L=cache.H_chol_L,
            G=cache.G,
            p_full=np.zeros(cache.n),
            w_full=cache.w_full,
            gamma_full=np.zeros(cache.n),
            C_D=np.array(0.0),
            C_O=np.array(cache.C_O),
            C_op=np.array(cache.C_op),
            n=np.array(cache.n),
            d=np.array(cache.d),
            lam=np.array(cache.lam),
            loss_fn_id=np.array(cache.loss_fn_id),
            version=np.array(version),
            gamma_anchor=np.zeros(cache.n),
            C_D_anchor=np.array(1.0),
            C_O_anchor=np.array(1.0),
            C_op_anchor=np.array(1.0),
        )
        if include_diag:
            payload["G_diag"] = cache.G_diag
        return payload

    def test_v1_file_reconstructs_diag_and_ignores_extras(self):
        cache = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache_v1.npz")
            np.savez_compressed(
                path, **self._legacy_payload(cache, version=1, include_diag=False)
            )
            loaded = CachedCurvature.load(path)
        np.testing.assert_allclose(loaded.G_diag, np.diag(cache.G))
        np.testing.assert_array_equal(loaded.G, cache.G)
        np.testing.assert_array_equal(loaded.w_full, cache.w_full)
        self.assertEqual(loaded.C_O, cache.C_O)
        self.assertEqual(loaded.C_op, cache.C_op)
        self.assertEqual(loaded.version, 3)

    def test_v2_file_keeps_stored_diag(self):
        cache = CachedCurvature.build(self.X, self.y, self.theta_hat, self.lam)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache_v2.npz")
            np.savez_compressed(
                path, **self._legacy_payload(cache, version=2, include_diag=True)
            )
            loaded = CachedCurvature.load(path)
        np.testing.assert_array_equal(loaded.G_diag, cache.G_diag)
        self.assertEqual(loaded.C_O, cache.C_O)
        self.assertEqual(loaded.version, 3)


if __name__ == "__main__":
    unittest.main()
