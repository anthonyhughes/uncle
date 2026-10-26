"""Small static checks for the standalone distribution and its two UNCLE paths."""
import ast
from pathlib import Path


def test_package_never_imports_historical_codebase():
    root = Path(__file__).resolve().parents[1]
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(name.name.split('.')[0] not in {"unerring", "scripts"} for name in node.names)
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or '').split('.')[0] not in {"unerring", "scripts"}


def test_algorithm_modules_do_not_reference_the_other_certificate():
    root = Path(__file__).resolve().parents[1]
    ab2 = (root / "ab2.py").read_text()
    retained = (root / "retained.py").read_text()
    for name in ("retained_spectra", "hull_coefficients"):
        assert name not in ab2
    for name in ("fixed_coefficients", "full_weighted_norm"):
        assert name not in retained
    from uncle import ab2 as ab2_module
    from uncle import retained as retained_module
    assert not hasattr(ab2_module, "DEFAULT_R_INF")
    assert ab2_module.DEFAULT_S_INF == (10., 1., .1, .01, .001)
    assert ab2_module.CERTIFICATE_VARIANT == "theta0-A-B2-augmented-v1"
    assert retained_module.DEFAULT_R_INF == (1., .1, .01, .001, .0001)
