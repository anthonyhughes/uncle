"""Tests for the naïve simple_certifier reference implementation."""

import unittest

import numpy as np
from scipy.special import expit

from tests.certifier_fixtures import (
    default_R2,
    logistic_gradient,
    newton_step_hypothesis,
    remove_samples,
    retrain_exact,
)
from uncle.simplified.cached_curvature import CachedCurvature
from uncle.simplified.main_certifier import main_certifier
from uncle.simplified.model_certify import certify
from uncle.simplified.simple_certifier import (
    newton_residual_at_candidate,
    retained_hessian_inverse_at_theta0,
    simple_certifier,
)


def _gaussian_pac(n, d, rng):
    theta_star = rng.standard_normal(d)
    theta_star /= np.linalg.norm(theta_star)
    X = rng.standard_normal((n, d))
    p = expit(X @ theta_star)
    y = (rng.uniform(size=n) < p).astype(np.float64)
    return X, y


class TestSimpleCertifierAgreement(unittest.TestCase):
    """simple_certifier must match main_certifier on small problems."""

    def _run_trial(self, seed, n=120, d=8, k=5, lam=0.1, sf=1.0):
        rng = np.random.default_rng(seed)
        X, y = _gaussian_pac(n, d, rng)
        theta_0 = retrain_exact(X, y, lam, d)
        T_idx = np.array(rng.choice(n, k, replace=False), dtype=np.intp)
        T = set(T_idx.tolist())
        X_ret, y_ret = remove_samples(X, y, T)
        theta_1 = newton_step_hypothesis(theta_0, X, y, T, lam)
        R_2 = default_R2(k, d, n, safety_factor=sf)

        cache = CachedCurvature.build(X, y, theta_0, lam)
        ref = main_certifier(
            X_ret,
            y_ret,
            T_idx,
            theta_1,
            R_2,
            lam,
            cache,
            X_full=X,
        )
        naive = simple_certifier(X_ret, y_ret, theta_0, theta_1, R_2, lam, n)
        return ref, naive

    def test_agrees_with_main_certifier_multiple_seeds(self):
        # Since the Lemma A.7 change, the main certifier feeds the predicate
        # fast VALID BOUNDS for C_op / C_O (Theorem A.9's query path) rather
        # than exact constants; only the diagonal C_D stays exact. The
        # contract with the simple reference is therefore: identical A/B
        # quantities, identical exact C_D, and dominance (bound >= exact)
        # for C_op / C_O.
        for seed in (0, 1, 7, 42):
            with self.subTest(seed=seed):
                ref, naive = self._run_trial(seed)
                self.assertEqual(ref["accepted"], naive["accepted"])
                self.assertAlmostEqual(ref["B_2"], naive["B_2"], places=8)
                self.assertAlmostEqual(ref["B_inf"], naive["B_inf"], places=8)
                # Exact diagonal agrees across certifiers
                # (main: mean-form G then /n; simple: sum-form G).
                self.assertAlmostEqual(
                    ref["quantities"]["C_D_eff"],
                    naive["quantities"]["C_D_eff"],
                    places=8,
                )
                # Lemma A.7 bounds dominate the exact constants.
                for key in ("C_O_eff", "C_op_eff"):
                    self.assertGreaterEqual(
                        ref["quantities"][key] + 1e-12,
                        naive["quantities"][key],
                        msg=key,
                    )
                # Main's raw C_* = n · its scaled C_*_eff (internal consistency).
                n = 120
                for raw, eff in (("C_D", "C_D_eff"), ("C_O", "C_O_eff"), ("C_op", "C_op_eff")):
                    self.assertAlmostEqual(
                        ref["quantities"][raw],
                        n * ref["quantities"][eff],
                        places=8,
                        msg=f"main raw {raw}",
                    )

    def test_agrees_across_removal_sizes(self):
        for k in (1, 3, 10):
            with self.subTest(k=k):
                ref, naive = self._run_trial(seed=99, k=k)
                self.assertEqual(ref["accepted"], naive["accepted"])
                self.assertAlmostEqual(ref["B_2"], naive["B_2"], places=8)

    def test_variant_label(self):
        _, naive = self._run_trial(seed=3)
        self.assertEqual(naive["variant"], "simple")


class TestSimpleCertifierEdgeCases(unittest.TestCase):
    def test_k_equals_one_runs(self):
        rng = np.random.default_rng(0)
        n, d, lam = 30, 2, 0.5
        X, y = _gaussian_pac(n, d, rng)
        theta_0 = retrain_exact(X, y, lam, d)
        T_idx = np.array([5], dtype=np.intp)
        X_ret, y_ret = remove_samples(X, y, {5})
        theta_1 = newton_step_hypothesis(theta_0, X, y, {5}, lam)
        R_2 = default_R2(1, d, n)
        result = simple_certifier(X_ret, y_ret, theta_0, theta_1, R_2, lam, n)
        self.assertIn("accepted", result)
        self.assertTrue(np.isfinite(result["B_2"]))


class TestSimpleCertifierPredicateWiring(unittest.TestCase):
    def test_accepted_implies_four_conditions(self):
        rng = np.random.default_rng(11)
        n, d, lam = 80, 6, 0.2
        X, y = _gaussian_pac(n, d, rng)
        theta_0 = retrain_exact(X, y, lam, d)
        T_idx = np.array(rng.choice(n, 4, replace=False), dtype=np.intp)
        X_ret, y_ret = remove_samples(X, y, set(T_idx.tolist()))
        theta_1 = newton_step_hypothesis(theta_0, X, y, set(T_idx.tolist()), lam)
        R_2 = default_R2(4, d, n, safety_factor=50.0)

        result = simple_certifier(X_ret, y_ret, theta_0, theta_1, R_2, lam, n)
        if not result["accepted"]:
            self.skipTest("certificate did not accept on this draw")

        q = result["quantities"]
        A_2, A_inf = q["A_2"], q["A_inf"]
        B_2, B_inf = result["B_2"], result["B_inf"]
        G_op, G_D, G_O = q["C_op_eff"], q["C_D_eff"], q["C_O_eff"]
        # R_∞ is existentially quantified: the certifier accepts at the
        # per-query witness it searched for, so check the four conditions at
        # the R_∞ it actually used, not at the legacy closed form.
        R_inf = result["R_inf"]
        S_2 = A_2 + R_2
        S_inf = A_inf + R_inf

        self.assertLess(G_op * S_inf, 1.0)
        self.assertLess(G_D * S_inf, 1.0)
        self.assertLess(B_2 / (1.0 - G_op * S_inf), R_2)
        Lambda_off = G_O * S_2
        c_4 = (B_inf + Lambda_off * B_2 / (1.0 - G_op * S_inf)) / (1.0 - G_D * S_inf)
        self.assertLess(c_4, R_inf)
        self.assertTrue(
            certify(
                G_op=G_op,
                G_D=G_D,
                A_inf=A_inf,
                B_2=B_2,
                B_inf=B_inf,
                R_2=R_2,
                R_inf=R_inf,
                Lambda_off=Lambda_off,
            )
        )


class TestFixedNvsNretRegularization(unittest.TestCase):
    """Empirical comparison: paper fixed-ridge (original n) vs n_ret rescaled ridge.

    Passing ``n_ret`` into ``simple_certifier`` simulates the common mistake of
    re-averaging the regularizer over the retained set only. These tests show
    that the two conventions are genuinely different after deletion, and that
    only the original-``n`` choice matches ``main_certifier`` / ``retrain_exact``.
    """

    def _deletion_trial(self, seed=42, n=120, d=8, k=15, lam=0.1):
        rng = np.random.default_rng(seed)
        X, y = _gaussian_pac(n, d, rng)
        theta_0 = retrain_exact(X, y, lam, d)
        T_idx = np.array(rng.choice(n, k, replace=False), dtype=np.intp)
        T = set(T_idx.tolist())
        X_ret, y_ret = remove_samples(X, y, T)
        theta_1 = newton_step_hypothesis(theta_0, X, y, T, lam)
        R_2 = default_R2(k, d, n)
        n_ret = X_ret.shape[0]
        cache = CachedCurvature.build(X, y, theta_0, lam)
        main = main_certifier(
            X_ret, y_ret, T_idx, theta_1, R_2, lam, cache, X_full=X,
        )
        cert_fixed_n = simple_certifier(
            X_ret, y_ret, theta_0, theta_1, R_2, lam, n,
        )
        cert_n_ret = simple_certifier(
            X_ret, y_ret, theta_0, theta_1, R_2, lam, n_ret,
        )
        return {
            "n": n,
            "k": k,
            "n_ret": n_ret,
            "lam": lam,
            "X_ret": X_ret,
            "y_ret": y_ret,
            "theta_0": theta_0,
            "theta_1": theta_1,
            "R_2": R_2,
            "main": main,
            "cert_fixed_n": cert_fixed_n,
            "cert_n_ret": cert_n_ret,
        }

    def test_no_deletion_n_equals_n_ret_is_identical(self):
        """With k=0, n and n_ret coincide so both ridge conventions agree."""
        rng = np.random.default_rng(0)
        n, d, lam = 120, 8, 0.1
        X, y = _gaussian_pac(n, d, rng)
        theta_0 = retrain_exact(X, y, lam, d)
        theta_1 = theta_0.copy()
        R_2 = default_R2(0, d, n)
        cert_fixed_n = simple_certifier(X, y, theta_0, theta_1, R_2, lam, n)
        cert_n_ret = simple_certifier(X, y, theta_0, theta_1, R_2, lam, n)
        for key in ("B_2", "B_inf"):
            self.assertAlmostEqual(cert_fixed_n[key], cert_n_ret[key], places=12)
        for key in ("C_D_eff", "C_O_eff", "C_op_eff"):
            self.assertAlmostEqual(
                cert_fixed_n["quantities"][key],
                cert_n_ret["quantities"][key],
                places=12,
            )

    def test_hessian_and_residual_differ_after_deletion(self):
        """Ridge strength nλ vs n_retλ changes H⁻¹ and the Newton residual."""
        trial = self._deletion_trial(seed=42, k=15)
        n, n_ret = trial["n"], trial["n_ret"]
        self.assertLess(n_ret, n)

        H_inv_n = retained_hessian_inverse_at_theta0(
            trial["theta_0"], trial["X_ret"], trial["lam"], n,
        )
        H_inv_n_ret = retained_hessian_inverse_at_theta0(
            trial["theta_0"], trial["X_ret"], trial["lam"], n_ret,
        )
        self.assertGreater(
            np.linalg.norm(H_inv_n - H_inv_n_ret, "fro"),
            1e-6,
            "Hessian inverses should differ when n ≠ n_ret",
        )

        res_n = newton_residual_at_candidate(
            trial["theta_1"],
            trial["y_ret"],
            trial["X_ret"],
            H_inv_n,
            trial["lam"],
            n,
        )
        res_n_ret = newton_residual_at_candidate(
            trial["theta_1"],
            trial["y_ret"],
            trial["X_ret"],
            H_inv_n_ret,
            trial["lam"],
            n_ret,
        )
        self.assertGreater(
            abs(res_n["B_2"] - res_n_ret["B_2"]),
            1e-8,
            "B_2 should differ under the two ridge conventions",
        )

    def test_fixed_n_matches_main_certifier_not_n_ret(self):
        """Only original-n ridge matches the production certifier."""
        trial = self._deletion_trial(seed=7, k=10)
        fixed, wrong, main = (
            trial["cert_fixed_n"],
            trial["cert_n_ret"],
            trial["main"],
        )
        self.assertAlmostEqual(fixed["B_2"], main["B_2"], places=8)
        self.assertAlmostEqual(fixed["B_inf"], main["B_inf"], places=8)
        self.assertEqual(fixed["accepted"], main["accepted"])
        self.assertNotAlmostEqual(wrong["B_2"], main["B_2"], places=4)

    def test_retrained_optima_differ(self):
        """Paper retained ERM (1/n sum + λ) ≠ sklearn-style (1/n_ret sum + λ)."""
        trial = self._deletion_trial(seed=99, k=20)
        d = trial["X_ret"].shape[1]
        theta_paper = retrain_exact(
            trial["X_ret"], trial["y_ret"], trial["lam"], d, n=trial["n"],
        )
        theta_sklearn = retrain_exact(
            trial["X_ret"], trial["y_ret"], trial["lam"], d, n=None,
        )
        dist = np.linalg.norm(theta_paper - theta_sklearn)
        self.assertGreater(
            dist,
            1e-4,
            "retained optima should differ when k > 0",
        )

    def test_each_optimum_zeros_its_own_gradient_not_the_other(self):
        """Each objective's minimizer annihilates its own gradient, not the rival's."""
        trial = self._deletion_trial(seed=11, k=12)
        n, n_ret, lam = trial["n"], trial["n_ret"], trial["lam"]
        X_ret, y_ret = trial["X_ret"], trial["y_ret"]
        d = X_ret.shape[1]

        theta_paper = retrain_exact(X_ret, y_ret, lam, d, n=n)
        theta_sklearn = retrain_exact(X_ret, y_ret, lam, d, n=None)

        # Paper mean-form: (1/n) Xᵀ(p−y) + λθ
        grad_paper_at_paper = logistic_gradient(theta_paper, X_ret, y_ret, lam * n / n_ret)
        grad_paper_at_sklearn = logistic_gradient(theta_sklearn, X_ret, y_ret, lam * n / n_ret)

        # Sklearn mean-form: (1/n_ret) Xᵀ(p−y) + λθ  (lam_eff = lam when n=None)
        grad_sklearn_at_sklearn = logistic_gradient(theta_sklearn, X_ret, y_ret, lam)
        grad_sklearn_at_paper = logistic_gradient(theta_paper, X_ret, y_ret, lam)

        self.assertLess(np.linalg.norm(grad_paper_at_paper), 1e-6)
        self.assertLess(np.linalg.norm(grad_sklearn_at_sklearn), 1e-6)
        self.assertGreater(np.linalg.norm(grad_paper_at_sklearn), 1e-4)
        self.assertGreater(np.linalg.norm(grad_sklearn_at_paper), 1e-4)

    def test_ridge_ratio_scales_with_deletion_fraction(self):
        """Larger deletions widen the gap between n and n_ret ridge strengths."""
        gaps = []
        for k in (5, 25, 50):
            trial = self._deletion_trial(seed=3, k=k)
            b_fixed = trial["cert_fixed_n"]["B_2"]
            b_n_ret = trial["cert_n_ret"]["B_2"]
            gaps.append(abs(b_fixed - b_n_ret))
        self.assertLess(gaps[0], gaps[1])
        self.assertLess(gaps[1], gaps[2])


if __name__ == "__main__":
    unittest.main()
