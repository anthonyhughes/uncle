# UNCLE: retained-spectrum and A+B2 experiments

This standalone package contains both augmented UNCLE certificates, Guo's
removal baseline, and Sekhari Algorithm 1. Retained-spectrum UNCLE is
`retained.py`. A+B2 is `ab2.py`. There is no package-level
`UncleModel` and no silent default: a config names `uncle_variant` explicitly.

The paper certifiers moved here from `unerring.core` live in
[`simplified/`](simplified/). `uncle.simplified.main_certifier` evaluates the
certificate from a cached full-data curvature and a rank-k Woodbury downdate.
`uncle.simplified.simple_certifier` is the reference form: it inverts the retained
Hessian and builds the Gram matrix directly. `uncle.simplified.model_certify.certify`
is the shared predicate. This folder is separate from the retained-spectrum
and A+B2 experiment code. It is also separate from the earlier standalone
package that the validation record calls "formerly simplified/".
Its production modules import only this package, NumPy, SciPy, pandas, and
Matplotlib. It consumes frozen binary-classification arrays and plain JSON/CSV
experiment specifications. Dataset preparation, downloads, cloud scheduling,
historical variants, and presentation infrastructure are outside its scope.

Start with [ALGORITHMS.md](ALGORITHMS.md) for the equations and numerical caveats.
Then read the code in this order:

1. [logistic.py](logistic.py): objective normalization, numerical ERM, test metrics.
2. [retained.py](retained.py) or [ab2.py](ab2.py): full-model setup, one deletion request, four scalar checks.
3. [curvature.py](curvature.py): both local logistic envelopes and both spectral quantities.
4. [masking.py](masking.py) and [privacy.py](privacy.py): the same augmented geometry for certification and noise.
5. [guo.py](guo.py) and [sekhari.py](sekhari.py): the two distinct baseline mechanisms.
6. [experiment.py](experiment.py) and [randomness.py](randomness.py): explicit loops and reproducible requests/noise.
7. [design.py](design.py) and [reporting.py](reporting.py): parameter selection, seed summaries, and small plots.

## Retained-spectrum single-request path

With the data and operating-point parameters already supplied, this is the
public API. The experiment loop repeats this path and reuses its prepared state:

```python
import numpy as np
from uncle.logistic import fit
from uncle.retained import UncleModel
from uncle.masking import AugmentedGeometry, PreparedRelease
from uncle.privacy import Privacy

theta0 = fit(X, y, training_ridge, polish=True)
model = UncleModel.build(X, y, theta0, training_ridge)
geometry = AugmentedGeometry.build(X, tau).prepare(deleted_indices)
request = model.prepare_request(deleted_indices, r_inf_values)
release = PreparedRelease.build(request, geometry, np.random.default_rng(seed))
output = release.release(R2, Privacy(epsilon=1, delta=1e-4))
# output.theta is already masked, including after a rejected request's retrain.
# output.certificate.branches exposes the LHS/RHS of all four checks.
```

The exported witness grid is `[1, .1, .01, .001, .0001]`. Config field
`uncle_variant` is `theta0-non-A-retained-augmented-v1`. That config uses
`r_inf_values` and must not set `s_inf_values`.

## A+B2 single-request path

```python
import numpy as np
from uncle.logistic import fit
from uncle.ab2 import UncleModel
from uncle.masking import AugmentedGeometry, PreparedRelease
from uncle.privacy import Privacy

theta0 = fit(X, y, training_ridge, polish=True)
model = UncleModel.build(X, y, theta0, training_ridge,
                         s_inf_values=[10, 1, .1, .01, .001])
geometry = AugmentedGeometry.build(X, tau).prepare(deleted_indices)
request = model.prepare_request(deleted_indices)
release = PreparedRelease.build(request, geometry, np.random.default_rng(seed))
output = release.release(R2, Privacy(epsilon=1, delta=1e-4))
```

`UncleModel.build` fixes the local intervals and their full-data spectra
before any deletion. `prepare_request` computes the small Woodbury solve and
B2 inflation. A query derives `Rinf = Sinf - Ainf` and rejects nonpositive
values. Config field `uncle_variant` is `theta0-A-B2-augmented-v1`. That
config uses `s_inf_values` and must not set `r_inf_values`.

`training_ridge`, `tau`, and `R2` are deliberately separate named quantities.
No function substitutes one for another or selects a different masking metric.

## Inputs and execution

Install with `python -m pip install ./uncle`. Use the user's MIT GCP worker
environment for fitting, spectral setup, and tests; the package itself does not
submit cloud jobs. Local syntax checks and plotting are permitted. Run the
standalone tests there with `python -m pytest uncle/tests`.
[Validation](VALIDATION.md) records the two historical gates. Those counts are
not a cloud gate for this unified tree.

The input NPZ contains `X_train`, `y_train`, `X_test`, and `y_test`, with samples
as rows and labels in `{0,1}`. Feature columns and row order are used as supplied:
there is no normalization, projection, intercept insertion, or train/test split.
Every coordinate, including an existing bias column, is regularized.

`export-config` reads a completed matched-Guo native JSON configuration and
copies its recorded data path, lambda grids, sigma grid, seeds, deletion sizes,
and separate CE/accuracy references. `--variant retained` or `--variant ab2`
states that certificate's geometry multipliers and witness grid. The shared
experiment budget is `epsilon=1, delta=1e-4`. Exporting configuration does not
copy or modify old results. Review the exported JSON and planned point table
before running.

The exporter creates a new paired experiment with a common master seed. The
completed Guo and augmented UNCLE campaigns used different master seeds
(`2026091601` and `337396328` respectively), so one exported config does not
byte-replay both historical runs. Mathematical migration parity is evaluated
at identical given inputs, including the random streams.

All generated outputs must be outside this source workspace. The examples use
a cloud results directory; local plots belong under
`C:\Temp.NoBackup\project_files\unerring`. Run directories are new independent
shards; completed directories are not overwritten or silently resumed.

## Experiment workflow

The following commands show one dataset and the retained-spectrum certificate.
`/results/mnist` is an explicit external output directory; substitute actual
frozen input/result paths. Use `--variant ab2` to export the A+B2 witness grid
instead.

```sh
python -m uncle export-config /frozen/matched-guo/discovery/mnist/config.json --variant retained --out /results/mnist/config.json
python -m uncle plan /results/mnist/config.json --methods uncle guo sekhari --out /results/mnist/coarse.csv
python -m uncle run /results/mnist/config.json /results/mnist/coarse.csv --out /results/mnist/coarse-run
python -m uncle summarize /results/mnist/coarse-run --out /results/mnist/coarse-summary
python -m uncle refine /results/mnist/config.json /results/mnist/coarse-summary/summary.csv --method guo --out /results/mnist/guo-refined.csv
python -m uncle run /results/mnist/config.json /results/mnist/guo-refined.csv --out /results/mnist/guo-refined-run
```

Repeat refinement/run with `--method sekhari` for Sekhari's second stage.
Guo keeps the existing k=1 CE-frontier Gaussian tube in log(lambda, sigma),
including its seeded arc-length sampling and rejection at frozen bounds; it
does not use Sobol points. Sekhari keeps exactly 32 coarse plus 64 new refined
lambdas. Its refinement uses the union of adjacent-coarse brackets around each
k's CE and accuracy winners. Boundary winners are recorded without expanding
the range. UNCLE uses the explicit lambda/geometry/radius grid. A+B2 also uses
the fixed-S grid `[10, 1, .1, .01, .001]`.

For parallel workers, `run --seed-start START --seed-stop STOP` selects a
half-open seed-index range. Give every shard a distinct output directory, keep
the same config and point table within a stage, and summarize all disjoint
shards. The number of requests and the full configured k grid remain fixed so
request prefixes and the utility-request selection do not change.

After all stages, combine their run directories and freeze the union of both
utility frontiers:

```sh
python -m uncle summarize /results/mnist/coarse-run /results/mnist/guo-refined-run /results/mnist/sekhari-refined-run --out /results/mnist/discovery
python -m uncle select /results/mnist/discovery/summary.csv --out /results/mnist/frozen-frontier.csv
python -m uncle plot /results/mnist/discovery/summary.csv --dataset mnist --out /results/mnist/plots
```

For fresh inference, make a separate config with a new `master_seed`, preserve
the data, references, privacy budget, request convention, variant, and parameter
choices, and run that config against `frozen-frontier.csv`. Do not refine or
retune on the fresh outcomes. Retain the complete frozen-target summary; any
newly nondominated subset is a separate descriptive result. A summary from one
`uncle_variant` does not validate against the other.

One raw row is one operating point and algorithm seed: utility comes from the
predetermined selected request. UNCLE and Guo estimate their no-retraining
fractions from the ten requests in that seed's cluster. Means and SEM aggregate
independent seeds, not individual requests. Sekhari evaluates only the selected
utility request; its ten-request convention fixes that request's identity. Its
rate of one is by construction for valid outputs, not an empirical certificate
pass frequency. Failed or absent outputs do not become successful observations.

Plots write PNG/PDF plus explanatory metadata. The two summary panels show
minimum accuracy drop at mean no-retraining rate at least 90%, and minimum
retraining rate at mean CE increase at most 0.01. Infeasible choices remain gaps.
Available error bars are pointwise plus/minus one SEM, not confidence bands for
the selected frontier.
