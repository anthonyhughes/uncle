"""Explicit independent streams; compatible with the current frozen campaigns."""
from __future__ import annotations
import hashlib
import numpy as np
from numpy.typing import NDArray


def trial_seeds(master_seed: int, dataset: str, count: int) -> NDArray[np.int64]:
    """Use the campaign's canonical dataset name (e.g. cifar10_catdog_resnet18)."""
    key = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:4], "big")
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, key, 0x53454544]))
    return rng.integers(1, 2**31 - 1, size=count, dtype=np.int64)


def deletion_requests(seed: int, n: int, max_k: int, count: int) -> NDArray[np.int64]:
    """Nested prefixes across k; independent request stream within each seed."""
    rows = []
    for index in range(count):
        entropy = [seed, 0x44524F50] + ([index] if index else [])
        rows.append(np.random.default_rng(np.random.SeedSequence(entropy)).choice(n, max_k, replace=False))
    return np.asarray(rows, dtype=np.int64)


def selected_request(seed: int, k: int, count: int) -> int:
    """Predetermined utility request; selection does not depend on certification."""
    if count == 1:
        return 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4556414C, k]))
    return int(rng.integers(count))


def mask_rng(seed: int, k: int, request_index: int) -> np.random.Generator:
    key = f"UNERRING_R2_SHARED_MASK_V2|{seed}|{k}|{request_index}".encode("ascii")
    words = np.frombuffer(hashlib.sha256(key).digest()[:16], dtype=np.uint32).astype(np.uint64)
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


def sekhari_rng(master: int, dataset: str, seed: int, k: int) -> np.random.Generator:
    key = f"sekhari-isotropic-v1:{master}:{dataset}:{seed}:{k}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
