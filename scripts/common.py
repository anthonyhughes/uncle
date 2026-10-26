"""Output safety shared by the frozen submission plotting entry points."""
from pathlib import Path


FROZEN_PACKAGES = (
    "uncle-nine-mean-retention-refresh-20260924",
    "ab2-nine-raised-ridge-final-20260921",
    "uncle-nine-test-selected-refresh-20260922",
    "experiment-design-paretos-20260920",
    "experiment-design-appendix-20260920",
    "retention-nine-dense30-20260922a",
)


def prepare_output(artifact_root: Path, out_dir: Path) -> Path:
    """Allow new output only below the external artifact root, never in inputs."""
    root, out = Path(artifact_root).resolve(), Path(out_dir).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not root.is_dir():
        raise ValueError(f"Artifact root must already exist: {root}")
    if out == root or not out.is_relative_to(root):
        raise ValueError("Output must be a subdirectory of the external artifact root")
    if out.is_relative_to(repo) or any(p.lower().startswith("onedrive") for p in out.parts):
        raise ValueError("Generated figures must remain outside OneDrive and the repository")
    frozen_names = {name.casefold() for name in FROZEN_PACKAGES} | {"figure 2"}
    if any(part.casefold() in frozen_names for part in out.parts):
        raise ValueError("Output must not be inside a frozen figure input package")
    for name in FROZEN_PACKAGES:
        frozen = (root / "diagnostics" / name).resolve()
        if out == frozen or out.is_relative_to(frozen) or frozen.is_relative_to(out):
            raise ValueError(f"Do not overwrite or write into the frozen input package: {frozen}")
    out.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in out.iterdir()):
        raise ValueError("Output directory must not contain file or directory symlinks")
    return out
