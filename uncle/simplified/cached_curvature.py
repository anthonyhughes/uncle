"""Cached full-data curvature for the deletion-aware verifier.

The cache holds the expensive objects associated with a fixed reference point
θ_0 — typically the full-data optimum:

    H_chol  = Cholesky factor of ∇²L_all(θ_0)   (for cheap solves)
    G       = X H⁻¹ X^T                          (n × n) — the Gram matrix
    G_diag  = diagonal of G
    C_O, C_op = scalar Gram constants the certifier reads
    w_full  = per-sample logistic weights at θ_0

Per-query Woodbury rank-k updates against this cache let the verifier
operate at O(nk + k³) cost instead of forming the retained Gram.

Caveats:
- The cache is loss-specific only via the saved w_full.
  G and the Cholesky factor are loss-agnostic.
- All floating-point storage is float64 to avoid precision drift inside the
  certificate. The Gram matrix dominates disk and memory; for very large n
  this is the constraint that determines what hardware is required.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.linalg import cho_factor, eigvalsh, solve as la_solve
from scipy.sparse.linalg import ArpackError, eigsh
from scipy.special import expit

_CACHE_VERSION = 3
_DENSE_EIG_THRESHOLD = 500
_GRAM_BLOCK_ROWS = 1024


def _largest_psd_eigenvalue(G: np.ndarray) -> float:
    """Largest eigenvalue of a symmetric PSD Gram matrix, deterministically."""
    n = G.shape[0]
    if n == 0:
        return 0.0
    if n < _DENSE_EIG_THRESHOLD:
        return float(eigvalsh(G, subset_by_index=[n - 1, n - 1])[0])
    v0 = np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
    try:
        return float(eigsh(
            G, k=1, which="LA", v0=v0, return_eigenvectors=False,
            tol=1e-10,
        )[0])
    except (ArpackError, np.linalg.LinAlgError):
        # Deterministic power iteration avoids an O(n^3) dense fallback on
        # the real embedding datasets.
        v = v0
        value = 0.0
        for _ in range(500):
            u = G @ v
            next_value = float(np.linalg.norm(u))
            if next_value < 1e-30:
                return 0.0
            v = u / next_value
            if abs(next_value - value) <= 1e-10 * max(next_value, 1.0):
                break
            value = next_value
        return float(v @ (G @ v))


def _off_diagonal_abs_max(G: np.ndarray) -> float:
    """Exact max off-diagonal ``|G_ij|`` in blocks."""
    n = G.shape[0]
    best = 0.0
    for start in range(0, n, _GRAM_BLOCK_ROWS):
        stop = min(start + _GRAM_BLOCK_ROWS, n)
        block = np.abs(G[start:stop])
        rows = np.arange(stop - start)
        block[rows, start + rows] = 0.0
        best = max(best, float(np.max(block, initial=0.0)))
    return best


@dataclass
class CachedCurvature:
    """Cached full-data curvature at the reference point θ_0.

    Build once per (dataset, model, λ) triple via ``CachedCurvature.build``,
    optionally serialise via ``save``/``load``, and reuse across many
    deletion-aware verifier queries.

    Memory footprint is dominated by ``G`` which is n × n float64
    (8 · n² bytes). Caller is responsible for ensuring the host has enough
    RAM; on HPC nodes with 256 GB+ RAM, n up to ~150 000 is comfortable.
    """

    theta_0: np.ndarray
    H_chol_L: np.ndarray
    w_full: np.ndarray
    n: int
    d: int
    lam: float
    C_O: float
    C_op: float
    G: np.ndarray
    loss_fn_id: str = "logistic"
    version: int = _CACHE_VERSION
    G_diag: Optional[np.ndarray] = None

    @classmethod
    def build(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        theta_0: np.ndarray,
        lam: float,
        *,
        loss_fn_id: str = "logistic",
    ) -> "CachedCurvature":
        """Construct the full curvature cache from data and a reference point.

        Cost: O(nd² + d³ + n²d). The n²d term (forming G) dominates for
        large n; for sst2-scale (n ≈ 50 000) this is several minutes on a
        modern multicore CPU. The cost is paid once per dataset.
        """
        if loss_fn_id != "logistic":
            raise NotImplementedError(
                f"CachedCurvature.build only supports logistic loss; got {loss_fn_id}"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        theta_0 = np.asarray(theta_0, dtype=np.float64)

        n, d = X.shape

        # Per-sample logistic weights at θ_0.
        p = expit(X @ theta_0)
        w = p * (1.0 - p)

        # Hessian: (1/n) X^T diag(w) X + λ I
        H = (X.T * w) @ X / n + lam * np.eye(d)

        # decomp the hessian
        H_chol_L, _ = cho_factor(H, lower=True, check_finite=False)

        # H⁻¹ X^T then G = X (H⁻¹ X^T)
        H_inv_Xt = la_solve(H, X.T, assume_a="pos", check_finite=False)
        G = X @ H_inv_Xt

        # Force exact symmetry against numerical asymmetry from the BLAS
        # multiply. G is theoretically symmetric (X H⁻¹ X^T with H SPD).
        G = 0.5 * (G + G.T)

        diag_G = np.diag(G)
        return cls(
            theta_0=theta_0.copy(),
            H_chol_L=H_chol_L,
            w_full=w,
            n=n,
            d=d,
            lam=float(lam),
            loss_fn_id=loss_fn_id,
            C_O=_off_diagonal_abs_max(G),
            C_op=_largest_psd_eigenvalue(G),
            G=G,
            G_diag=np.ascontiguousarray(diag_G),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the cache to a single .npz file (compressed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        g_diag = self.G_diag if self.G_diag is not None else np.diag(self.G)
        np.savez_compressed(
            path,
            theta_0=self.theta_0,
            H_chol_L=self.H_chol_L,
            G=self.G,
            w_full=self.w_full,
            C_O=np.array(self.C_O),
            C_op=np.array(self.C_op),
            n=np.array(self.n),
            d=np.array(self.d),
            lam=np.array(self.lam),
            loss_fn_id=np.array(self.loss_fn_id),
            version=np.array(self.version),
            G_diag=np.ascontiguousarray(g_diag),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CachedCurvature":
        """Load a previously-saved cache.

        v1 files omit ``G_diag``; it is reconstructed from ``G``. Files may
        also contain historical arrays the certifier does not read (the
        Hessian, per-sample probabilities and curvature weights, the scalar
        ``C_D``, and the v3 anchor summaries). Those arrays are ignored.
        """
        data = np.load(path, allow_pickle=False)
        version = int(data["version"]) if "version" in data.files else 0
        if version not in (1, 2, _CACHE_VERSION):
            raise ValueError(
                f"CachedCurvature version mismatch at {path}: file is v{version}, "
                f"current code is v{_CACHE_VERSION}. Rebuild via CachedCurvature.build."
            )

        if "G_diag" in data.files:
            g_diag = np.ascontiguousarray(data["G_diag"])
        else:
            g_diag = np.ascontiguousarray(np.diag(data["G"]))

        return cls(
            theta_0=data["theta_0"],
            H_chol_L=data["H_chol_L"],
            G=data["G"],
            w_full=data["w_full"],
            C_O=float(data["C_O"]),
            C_op=float(data["C_op"]),
            n=int(data["n"]),
            d=int(data["d"]),
            lam=float(data["lam"]),
            loss_fn_id=str(data["loss_fn_id"]),
            version=_CACHE_VERSION,
            G_diag=g_diag,
        )

    def memory_bytes(self) -> int:
        """Approximate in-memory footprint of the large arrays (bytes)."""
        return (
            self.H_chol_L.nbytes
            + self.G.nbytes
            + self.w_full.nbytes
            + self.theta_0.nbytes
        )

    def __repr__(self) -> str:
        gb = self.memory_bytes() / 1e9
        return (
            f"CachedCurvature(n={self.n}, d={self.d}, lam={self.lam:.3e}, "
            f"loss={self.loss_fn_id}, ~{gb:.2f} GB)"
        )
