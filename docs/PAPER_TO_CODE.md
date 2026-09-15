# Paper-to-code map

This document connects each scientific component of the paper to its executable command, configuration, input, and output artifact.

## Operational system model

SentrySem uses five pre-refresh transmitted frames: proposal signs, candidate-lock record, audit seed, audit signs, and receiver decision. The nearest-candidate search is performed locally at the receiver between the first and second frames. A refresh decision adds one feature frame carrying the current quantized feature.

The protocol implementation is installed from `src/sentrysem/` and exercised through `main.py`. The traffic constants and the selected proposal-audit allocation are declared in `configs/paper.json`.

## Source modules

| Module | Responsibility |
|---|---|
| `src/sentrysem/config.py` | Typed configuration loading and scientific-parameter validation |
| `src/sentrysem/data.py` | CIFAR-10 partitioning, deterministic view generation, manifests, and frozen-feature loading |
| `src/sentrysem/encoder.py` | Frozen ResNet-18 construction, preprocessing, feature extraction, and encoder provenance |
| `src/sentrysem/sketch.py` | Random-hyperplane signs, binary symmetric channel, Hamming search, and stable candidate ranking |
| `src/sentrysem/statistics.py` | Observed mismatch model, exact binomial boundaries, Wilson intervals, sequential audit lookup, and alpha spending |
| `src/sentrysem/traffic.py` | Five-frame serialization, candidate-lock binding, int8 refresh, and completion-bit accounting |
| `src/sentrysem/protocol.py` | Threshold calibration, fixed independent audit, same-evidence diagnostic, and sequential cache protocol |
| `src/sentrysem/results.py` | Stable CSV/JSON serialization, seed aggregation, identity-cluster bootstrap, and artifact manifests |
| `src/sentrysem/experiments/allocation.py` | CAL-TUNE proposal-audit allocation study and frozen selection |
| `src/sentrysem/experiments/comparators.py` | Common-seed method comparison, aggregation, and identity-cluster bootstrap |
| `src/sentrysem/experiments/stress.py` | Partial-cache, missing-class, strong-view, and novel-source evaluation |
| `src/sentrysem/figures/paper.py` | Quantitative manuscript figure construction |
| `src/sentrysem/figures/qualitative.py` | Representative cached, repeated, strong-shift, and novel-source image panel |

## Frozen semantic representation

| Paper component | Repository implementation |
|---|---|
| CIFAR-10 deterministic partitions | `python main.py extract-embeddings --config configs/paper.json --output artifacts/generated_inputs` |
| Standard repeated-view transformation | Raw-image extraction pipeline configured by `configs/paper.json` |
| Strong-view transformation | Raw-image extraction pipeline configured by `configs/paper.json` |
| Frozen ResNet-18 feature encoder | Raw-image extraction pipeline in `src/sentrysem/` |
| Published embeddings and identities | `artifacts/frozen_inputs/frozen_embeddings.npz` |
| Exact raw-image split indices | `artifacts/frozen_inputs/split_indices.json` |
| Qualitative figure source images | `artifacts/frozen_inputs/qualitative_examples.npz` |
| Dataset split and provenance metadata | `artifacts/frozen_inputs/data_manifest.json` |
| Encoder provenance | `artifacts/frozen_inputs/verifier_manifest.json` |

The feature archive contains the development cache, CAL-TUNE queries, CAL-VALIDATE queries, evaluation cache, standard repeated views, strong views, and novel-source queries required by every downstream experiment.

## Calibration and certification

| Paper component | Configuration or output |
|---|---|
| Cosine-verifier threshold selection | `src/sentrysem/protocol.py` and `artifacts/frozen_inputs/calibration.json` |
| Candidate threshold grid | `configs/paper.json` |
| Random-hyperplane proposal | SentrySem protocol implementation in `src/sentrysem/` |
| Candidate lock and fresh audit | SentrySem protocol implementation in `src/sentrysem/` |
| Exact binomial audit boundary | SentrySem statistical utilities in `src/sentrysem/` |
| Verifier target `alpha = 0.01` | `configs/paper.json` |
| Selected threshold `0.7717004776000975` | `artifacts/frozen_inputs/calibration.json` |

`python main.py verify` checks the calibration split, selected threshold, exact boundary masses, and independence-critical protocol ordering.

## Experimental questions

### Post-selection false acceptance

The boundary experiment searches 1, 8, or 64 candidates with a 64-sign sketch and compares reuse of the winning search evidence with an independent audit. Its frozen numerical records are stored with the comparator reference artifacts under `results/paper/comparators/`. The corresponding manuscript plot is under `figures/paper/`.

### Proposal-audit allocation

The allocation study evaluates seven 1,024-sign splits from 128/896 through 896/128 on CAL-TUNE, using five predeclared seeds at BSC crossover probabilities 0 and 0.05. The selected 256/768 split is then frozen before evaluation.

- Reference outputs: `results/paper/allocation/`
- Fresh outputs: `results/generated/allocation/`
- Configuration: `configs/paper.json`
- Command: `python main.py reproduce --config configs/paper.json --overwrite`

### Method comparison

The comparison stage evaluates the selected SentrySem method together with the always-refresh reference, Bonferroni controls, fixed-split ablations, unverified nearest-neighbor reuse, same-evidence screening, and sequential audit variants.

- Reference outputs: `results/paper/comparators/`
- Fresh outputs: `results/generated/comparators/`
- Figures: `figures/paper/` and `figures/generated/`

### Selected-method stress evaluation

The stress stage evaluates the frozen 256/768 method under the full cache, a stratified 50% cache, a cache missing three classes, standard repeated views, strong views, and novel sources.

- Reference outputs: `results/paper/stress/`
- Fresh outputs: `results/generated/stress/`
- Principal aggregate: `sentrysem_fixed_aggregate.csv`
- Identity-cluster intervals: `sentrysem_fixed_cluster_bootstrap.csv`
- Cluster bootstrap: 2,000 resamples of independent source-identity clusters while retaining all five protocol seeds per identity

## Figures

Run:

```bash
python main.py figures
```

The figure builder reads `results/paper/` by default and writes a fresh set to `figures/generated/`. The manuscript versions remain under `figures/paper/`. Numerical labels, thresholds, and confidence intervals are read from the result records; panel layout and visual styling are fixed by the figure builder.

## End-to-end execution

```bash
python main.py verify
python main.py reproduce --config configs/paper.json --overwrite
python -m pytest
```

This sequence validates the release, recreates the experiment tables and figures from the frozen semantic representations, and tests the package-level numerical invariants.
