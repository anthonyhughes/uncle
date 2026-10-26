# Reproduce the submission figures

This package regenerates **all seven figures** used in `9306_Certified_Unlearning_with.pdf`, from frozen numeric plotting ledgers and the supplied Figure 2 mathematical drawing recipe. It does not train models, run unlearning, refit calibration draws, or substitute newer experiment results.

The renderers are independent of the evolving `scripts/` and `unerring/` implementations. All input ledgers, original figure references, and expected output hashes are pinned in [manifest.json](manifest.json). Generated files and plotting environments stay outside this OneDrive repository.

## Figure map

| Figure | Renderer | Frozen content |
| --- | --- | --- |
| 1 | [figure_1.py](figure_1.py) | Native nine-panel CE increase, at most 50% retraining, arithmetic-mean half-data-loss allowance, automatic v1 clipping at 3 times the allowance. The submitted old k grid is preserved. |
| 2 | [figure_2.py](figure_2.py) | Three negative-gradient field panels: all space, an ellipsoid, and an ellipsoid/polyhedron intersection. Each panel is drawn from mathematical coordinates, not copied from a PDF. |
| 3 | [figure_3.py](figure_3.py) | Raised-regularization accuracy at k=100, including native 1x and the factor-specific no-unlearning references. |
| 4 | [figure_4.py](figure_4.py) | The September 22 test-selected baseline regularization searches actually embedded in the submission. |
| 5 | [figure_5.py](figure_5.py) | The illustrative Pareto schematic, with its original fixed illustrative coordinates. |
| 6 | [figure_6.py](figure_6.py) | Guo coarse/refinement hyperparameter exploration and the original selected-point membership. |
| 7 | [figure_7.py](figure_7.py) | Arithmetic-mean CE increase across the 30 random-retention draws, with the original logarithmic y-axis. |

Original script paths and exact external input paths are recorded per figure in the manifest. These scripts reproduce the submitted graphics, not the full upstream experiments. For Figure 7, unused descriptive percentile columns in the historical ledger are not used as CE allowances.

## Required external files

The artifact root is:

```text
C:\Temp.NoBackup\project_files\unerring
```

Keep the manifest-listed files under their existing relative `diagnostics/...` paths. Do not relocate them into this repository. The submission PDF used for verification is:

```text
C:\Users\ittai\Downloads\9306_Certified_Unlearning_with.pdf
```

The runner checks its SHA-256, so a different revision of the paper is rejected rather than silently compared with the wrong figures. Copying this scripts directory alone to another machine is insufficient: copy the manifest-listed input/reference files into a separate artifact root as well.

## Plotting environments

Exact rendering uses Linux/WSL and three environments: CPython 3.10.12 with Matplotlib 3.7.3 for Figures 1, 3, 4, 5, and 6; CPython 3.13.5 with Matplotlib 3.10.8 for Figure 2; and CPython 3.13.7 with Matplotlib 3.11.1 for Figure 7. Do not upgrade Matplotlib or substitute a native Windows renderer when checking byte-for-byte output. Font rendering and PDF output depend on the environment.

The dependency pins are in `requirements-legacy.txt`, `requirements-figure2.txt`, and `requirements-modern.txt`. They describe isolated plotting environments, not changes to the project's scientific environment. For a new machine, an example using `uv` from a WSL shell at the repository root is:

```bash
plot_env_root=/mnt/c/Temp.NoBackup/project_files/unerring/diagnostics/submission-figure-environments
export UV_CACHE_DIR="$plot_env_root/uv-cache"
export UV_PYTHON_INSTALL_DIR="$plot_env_root/python"
uv venv --python 3.10.12 "$plot_env_root/legacy"
uv pip install --python "$plot_env_root/legacy/bin/python" -r scripts/requirements-legacy.txt
uv venv --python 3.13.5 "$plot_env_root/figure2"
uv pip install --python "$plot_env_root/figure2/bin/python" -r scripts/requirements-figure2.txt
uv venv --python 3.13.7 "$plot_env_root/modern"
uv pip install --python "$plot_env_root/modern/bin/python" -r scripts/requirements-modern.txt
```

Use those three environment interpreters as the arguments below on that machine. The output checks, not successful dependency installation alone, establish exact reproduction.

## Regenerate and verify

From **PowerShell at the repository root**, this command uses the tested environments on this machine. Choose a new or empty output directory each time:

```powershell
wsl.exe -e python3 -B scripts/reproduce.py `
  --artifact-root /mnt/c/Temp.NoBackup/project_files/unerring `
  --submission-pdf /mnt/c/Users/ittai/Downloads/9306_Certified_Unlearning_with.pdf `
  --legacy-python /usr/bin/python3 `
  --figure2-python /mnt/c/Temp.NoBackup/project_files/unerring/diagnostics/submission-figure-reproduction-20261020/env-figure2/bin/python `
  --modern-python /mnt/c/Temp.NoBackup/project_files/unerring/diagnostics/submission-figure-reproduction-20261020/env-modern/bin/python `
  --out-dir /mnt/c/Temp.NoBackup/project_files/unerring/diagnostics/submission-figure-reproduction-20261020/rerun-01
```

Each figure produces `figure_N.pdf` and `figure_N.png`; Figure 2 produces three pairs named `figure_2a`, `figure_2b`, and `figure_2c`. [figure_2_layout.tex](figure_2_layout.tex) preserves the **supplied** LaTeX assembly and captions, with only image filenames changed to these output names. That caption is a different draft from the submission and still contains `TODO-global-analyses`; it is not covered by the exact-graphics guarantee. It is an inclusion snippet, not a standalone document; the enclosing paper must provide its existing macros, bibliography, and packages. No captions have been rewritten here.

`verification.json` records the current run time, environment, script/input hashes, and comparison results. A mismatch is an error, not a successful reproduction. `--figures 1 3` can be used for a partial check, which is explicitly recorded as partial coverage.

Individual `figure_N.py` entry points accept `--artifact-root` and `--out-dir`, but use the combined runner for the full provenance and exactness checks. No renderer copies the original PDF or PNG to its output.

## What exact reproduction means

The combined runner verifies every regenerated **PDF byte-for-byte** against its original SHA-256. This includes the original PDF creation-date metadata, deliberately retained for deterministic output; the actual regeneration date is in `verification.json`.

PNGs for Figures 1, 3, 4, 5, and 6 are also checked byte-for-byte. Figure 7's PNG is checked for identical mode, dimensions, and every decoded pixel; its file encoding differs while its PDF is byte-identical. Figure 2's PNGs are new previews: no original PNGs were supplied, and their receipt status explicitly says `no_original`. None of these PNG distinctions weaken the exact PDF checks against the submitted graphics.

It also verifies that the generated PDF matches the corresponding figure embedded in the submitted paper: decoded drawing commands and rendering resources, including fonts, glyphs, and marker paths. PDF object IDs and compression are ignored for this embedded comparison. The only numeric accommodations are pdfTeX's import rounding of bounding boxes: six decimal places for resource `/BBox` coordinates and at most 0.0001 point for the outer figure box. Drawing commands and font programs have no tolerance. The stricter whole-file output hash comparison has no tolerance at all.

Historical data, reference selection, midpoint conventions, missing/failed-fit records, colors, labels, layout, clipping, and sampled grids are preserved. This package does not revise any captions or scientific results to reconcile them with later presentation preferences.

## Fresh MNIST panel

`write_run_ledger.py` turns a fresh Guo versus UNCLE comparison CSV into a ledger with columns `dataset`, `method`, `k`, `ce_increase`, `success_rate`, and `status`. Method names are `uncle-augmented` and `Guo-v4+v5`. `status` is `attained` when the CE increase is finite. The ledger has no 50% retraining threshold and no CE allowance.

`plot_run_panel.py` draws one panel from that ledger. `experiments/compare_guo_ab2_mnist.sh` writes the ledger to `experiments/guo_ab2_mnist/ce_increase_ledger.csv` and the panel to `experiments/guo_ab2_mnist/plots/mnist_ce_increase.pdf`. The y-axis is test CE increase at the script's fixed operating point. This plot is outside the byte-for-byte check.
