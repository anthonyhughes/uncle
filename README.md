# UNCLE - Unlearning Certifiably via Local Estimates

Certified removal for logistic regression.

- `uncle.ab2` the UNCLE algorithm

## Structure

```
data/            Loaders for MNIST, synthetic data, and the KATL19 sets
experiments/     Comparison scripts
uncle/           All removal algorithms and the experiment runners
  ab2.py         A+B2. This is the algorithm used in the paper.
  guo.py         Guo et al. removal baseline
  sekhari.py     Sekhari et al. removal baseline
  simplified/    Our alternative unlearning algorithms
  tests/         All unit tests
```

## Install

```bash
uv sync
```

The certifiers install with the uncle package:

```bash
python -m pip install ./uncle
```

## Tests

```bash
uv run pytest \
  uncle/tests/test_simple_certifier.py \
  uncle/tests/test_main_certifier_prepared.py \
  uncle/tests/test_r_inf_search.py \
  uncle/tests/test_cached_curvature.py
```

## Experiments

`experiments/compare_guo_ab2_mnist.sh` compares Guo and UNCLE on MNIST digits 3 and 8. Removal sizes run from 1 through 10.

```bash
./experiments/compare_guo_ab2_mnist.sh
```

The script writes `experiments/guo_ab2_mnist.csv`, the fresh-run ledger `experiments/guo_ab2_mnist/ce_increase_ledger.csv`, and the figures under `experiments/guo_ab2_mnist/plots`. `mnist_ce_increase.pdf` in that directory is the test CE increase against samples removed, in the style of the paper's MNIST panel. The ledger uses the figure method names and has no 50% retraining threshold.

`scripts/` redraws figures the paper given experiment outputs. See `scripts/README.md`.

`experiments/compare_guo_ab2_molhiv_transformer.sh` and `experiments/compare_guo_ab2_fingerprints.sh` run the same comparison. Their matrices are referenced at `data/molhiv_transformer/molhiv_transformer.npz` and `data/fingerprints/fingerprints.npz`. Those files are not stored in the repository.
