"""One explicit geometry for BOTH UNCLE certification and Gaussian noise.

Q_R = X_R.T X_R + tau I. Tau is a direct precision coefficient, NOT training
ridge and NOT (n-k)*ridge. There is no jitter, pseudoinverse, or other fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

from .logistic import fit
from .privacy import Privacy
if TYPE_CHECKING:
    from .uncle import UncleRequest, Certificate

Array = NDArray[np.float64]


@dataclass(frozen=True)
class AugmentedGeometry:
    """Full-data setup, shared by every deletion request at the SAME tau."""
    X: Array
    tau: float
    cholesky: Array

    @classmethod
    def build(cls, X: Array, tau: float) -> AugmentedGeometry:
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("geometry ridge tau must be strictly positive")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError("finite nonempty design required")
        precision = X.T @ X + tau * np.eye(X.shape[1])
        factor = np.linalg.cholesky(precision)
        if not np.isfinite(factor).all() or np.any(np.diag(factor) <= 0):
            raise ValueError("augmented geometry factor must be finite positive definite")
        X.setflags(write=False)
        factor.setflags(write=False)
        return cls(X, float(tau), factor)

    def prepare(self, indices: NDArray[np.int64]) -> RetainedGeometry:
        """Rank-k downdate; never refactor a retained d-by-d precision."""
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) >= len(self.X)
                or np.any(indices < 0) or np.any(indices >= len(self.X))
                or np.unique(indices).size != indices.size):
            raise ValueError("deleted rows must be unique, in range, and leave data")
        indices = indices.copy()
        indices.setflags(write=False)
        return RetainedGeometry(self, indices)


@dataclass(frozen=True)
class RetainedGeometry:
    full: AugmentedGeometry
    indices: NDArray[np.int64]

    @property
    def X(self) -> Array:
        return self.full.X

    @property
    def tau(self) -> float:
        return self.full.tau

    @cached_property
    def downdate(self) -> tuple[Array, Array]:
        """Compute V and Cholesky(K) once, only if noise is actually needed.

        Certification-only requests need tau and row identity but no Gaussian
        draw. This cache is algebraic reuse, not a different certificate.
        """
        deleted = self.X[self.indices]
        lower = self.full.cholesky
        V = solve_triangular(lower.T, solve_triangular(lower, deleted.T, lower=True,
                             check_finite=False), lower=False, check_finite=False)
        small = np.eye(len(self.indices)) - deleted @ V
        return V, np.linalg.cholesky((small + small.T) * 0.5)

    def draw(self, rng: np.random.Generator) -> Array:
        """u=L^-T xi + V K^-T eta; Cov(u)=(X_R.T X_R+tau I)^-1.

        xi and eta are INDEPENDENT d- and k-dimensional standard Gaussians.
        Reusing u across radii changes only noise scale, not its marginal law.
        """
        base = solve_triangular(self.full.cholesky.T, rng.standard_normal(self.X.shape[1]),
                                lower=False, check_finite=False)
        if not len(self.indices):
            return base
        V, update_cholesky = self.downdate
        correction = solve_triangular(update_cholesky.T, rng.standard_normal(len(self.indices)),
                                      lower=False, check_finite=False)
        return base + V @ correction


@dataclass(frozen=True)
class UncleOutput:
    theta: Array
    certificate: Certificate
    noise_std: float


@dataclass(frozen=True)
class PreparedRelease:
    """Readable accepted/retrain/mask path; cache only radius-independent work."""
    request: UncleRequest
    geometry: RetainedGeometry
    direction: Array
    _fallback: Array | None = field(default=None, repr=False)

    @classmethod
    def build(cls, request: UncleRequest, geometry: RetainedGeometry,
              rng: np.random.Generator, fallback_theta: Array | None = None) -> PreparedRelease:
        """Optional fallback is the same retained ERM, shared across tau scans."""
        if geometry.X is not request.model.X or not np.array_equal(geometry.indices, request.indices):
            raise ValueError("certificate and mask must use the same full design and deletion")
        if fallback_theta is not None:
            fallback_theta = np.array(fallback_theta, dtype=float, copy=True)
            if fallback_theta.shape != request.candidate.shape or not np.isfinite(fallback_theta).all():
                raise ValueError("fallback must be a finite retained-model parameter vector")
            fallback_theta.setflags(write=False)
        direction = geometry.draw(rng)
        direction.setflags(write=False)
        return cls(request, geometry, direction, fallback_theta)

    @cached_property
    def retrained(self) -> Array:
        """Lazy retained ERM; caching cannot rebind the noise geometry."""
        if self._fallback is not None:
            return self._fallback
        model = self.request.model
        keep = np.ones(len(model.X), dtype=bool)
        keep[self.request.indices] = False
        return fit(model.X[keep], model.y[keep], model.training_ridge,
                   normalization_n=len(model.X))

    def release(self, radius: float, privacy: Privacy) -> UncleOutput:
        # The SAME geometry supplies tau to C3 and Q_R^-1 to the noise sampler.
        certificate = self.request.certify(radius, self.geometry)
        if certificate.accepted:
            center = self.request.candidate
        else:
            center = self.retrained
        sigma = privacy.gaussian_std(radius)
        return UncleOutput(center + sigma * self.direction, certificate, sigma)
