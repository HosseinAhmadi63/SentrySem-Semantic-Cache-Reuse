# Result schema

SentrySem writes comma-separated tables for analysis and JSON records for configuration, selected aggregates, provenance, and programmatic verification. Rates are stored as fractions in `[0, 1]`; plot labels convert them to percentages. Bit counts include logical framing exactly as declared in `configs/paper.json`.

## Directory layout

```text
results/paper/
├── comparators/             Frozen method-comparison records
├── allocation/              Frozen CAL-TUNE allocation selection
└── stress/                  Frozen selected-method stress records

results/generated/
├── comparators/             Fresh method-comparison records
├── allocation/              Fresh allocation-selection records
├── stress/                  Fresh stress records
├── calibration/             Fresh threshold-selection records
├── validation/              Fresh post-selection and null checks
└── reproduction_manifest.json
```

`results/paper/` is immutable. Reproduction commands write to `results/generated/`.

The frozen `selection.json` and experiment-configuration records preserve the provenance fields written by the original experiment bundle. In those records, the relative input label `inputs` corresponds to `artifacts/frozen_inputs/` in this repository, and each `script_sha256` identifies the original single-file experiment runner. The publication release reorganizes those runners into `src/sentrysem/experiments/`; `python main.py reproduce --config configs/paper.json --overwrite` executes the maintained modular equivalents and reproduces the numerical records.

## Comparator session records

Each row represents one query, one protocol seed, one BSC condition, and one method.

The comparator directory contains:

```text
experiment_configuration.json
protocol_validation.json
publication_session_results.csv
publication_seed_summary.csv
publication_aggregate_summary.csv
publication_aggregate_summary.json
identity_cluster_bootstrap.csv
identity_cluster_bootstrap.json
```

The reference tables retain stable machine identifiers so generated rows can be compared directly with the archived records. The normalized public names used by `configs/paper.json` map to those identifiers as follows:

| Table identifier | Public method name | Method |
|---|---|---|
| `proposal64_unverified` | `nearest-neighbor-64` | 64-sign nearest-neighbor reuse without certification |
| `screened512_heuristic` | `same-evidence-512` | 512 signs reused for search and threshold screening |
| `bonferroni512` | `bonferroni-512` | 512-sign multiplicity-adjusted certificate |
| `split256_256` | `fixed-256-256` | Independent 256-sign proposal and 256-sign audit |
| `screened1024_heuristic` | `same-evidence-1024` | 1,024 signs reused for search and threshold screening |
| `bonferroni1024` | `bonferroni-1024` | 1,024-sign multiplicity-adjusted certificate |
| `split512_512` | `fixed-512-512` | Independent 512-sign proposal and 512-sign audit |
| `sentrysem1` | `sequential-1` | Sequential fresh audit of one locked candidate |
| `sentrysem3` | `sequential-3` | Sequential fresh audits of at most three locked candidates |
| `int8_fallback` | `always-refresh` | Quantized 512-feature refresh in every session |

| Field group | Fields | Meaning |
|---|---|---|
| Run key | `protocol_seed`, `bsc`, `query_index`, `target_id`, `method` | Unique experimental condition and query identity |
| Decision | `decision`, `certified`, `direct_output`, `heuristic_reuse`, `cache_reuse`, `fallback_used` | Protocol path and final reuse/refresh action |
| Candidate | `candidate_bank_index`, `candidate_original_id`, `candidate_cosine`, `candidate_verifier_good`, `candidate_identity_match` | Locked cache candidate and its semantic status |
| Error | `false_certification`, `bad_reuse` | Certification of a verifier-bad candidate and reuse of a verifier-bad representation |
| Completion | `completion_bank_index`, `completion_original_id`, `completion_cosine`, `completion_verifier_good`, `completion_identity_match` | Representation available after reuse or refresh |
| Traffic | `protocol_bits`, `protocol_forward_bits`, `protocol_reverse_bits`, `fallback_bits`, `completion_forward_bits`, `completion_bits` | Logical traffic ledger |
| Sequential use | `audit_bits`, `attempts`, `logical_round_trips` | Audit resources used by fixed or sequential methods |

## Seed-level comparator summaries

Each row aggregates all queries for one `protocol_seed`, `bsc`, and `method`. The table reports:

- session count;
- certification, direct-output, reuse, fallback, false-certification, and bad-reuse rates;
- identity precision among certified, direct-output, and reused candidates;
- verifier-good completion rate and service completion rate;
- mean protocol, forward, reverse, fallback, and total completion bits;
- traffic saving relative to always refresh; and
- mean audit bits, attempts, and logical round trips.

## Aggregate comparator summaries

Each row aggregates the five protocol-seed summaries for one `bsc` and `method`. A metric named `x` appears as:

- `x_mean`: mean across protocol seeds;
- `x_seed_sd`: sample standard deviation across protocol seeds;
- `x_seed_min`: minimum seed value; and
- `x_seed_max`: maximum seed value.

The source-identity cluster-bootstrap table gives 95% bounds for reuse, bad reuse, identity precision, completion traffic, and saving. A bootstrap resample selects source identities and retains every protocol seed associated with each selected identity.

## Allocation records

The CAL-TUNE seed table is keyed by `phase`, `protocol_seed`, `bsc`, `search_bits`, and `audit_bits`. It reports the exact mismatch boundary, its null probability, reuse, bad reuse, identity precision, protocol traffic, completion traffic, and saving.

The allocation directory contains:

```text
cal_tune_ratio_seed_results.csv
cal_tune_ratio_aggregate.csv
cal_tune_ratio_aggregate.json
selected_ratio_final_seed_results.csv
selected_ratio_final_aggregate.csv
selected_ratio_final_aggregate.json
selection.json
```

The allocation aggregate table summarizes each BSC and proposal-audit split across five CAL-TUNE seeds. `selection.json` records:

- all seven candidate allocations;
- the traffic-minimizing selection rule;
- the selected 256/768 allocation;
- the development and evaluation seed sets;
- the 1% verifier target;
- the 4,272-bit refresh reference; and
- the hashes of the frozen input files.

The selected-ratio evaluation tables have the same core fields and use the disjoint evaluation seed set.

## Stress session records

The stress directory contains:

```text
sentrysem_fixed_session_results.csv
sentrysem_fixed_seed_summary.csv
sentrysem_fixed_aggregate.csv
sentrysem_fixed_aggregate.json
sentrysem_fixed_cluster_bootstrap.csv
sentrysem_fixed_cluster_bootstrap.json
```

Each row in `sentrysem_fixed_session_results.csv` contains:

| Field | Meaning |
|---|---|
| `scenario` | Cache construction: full, stratified half, or missing three classes |
| `condition` | Standard repeated view, strong view, or novel source |
| `protocol_seed`, `bsc`, `query_index`, `target_id` | Run and query key |
| `target_present` | Whether the exact source identity exists in the active cache |
| `candidate_original_id`, `candidate_cosine` | Selected candidate identity and similarity |
| `cache_reuse`, `verifier_good`, `bad_reuse`, `identity_match` | Decision-quality fields |
| `audit_mismatches`, `audit_boundary` | Observed audit statistic and acceptance boundary |
| `protocol_bits`, `fallback_bits`, `completion_bits` | Traffic ledger |

Stress seed summaries add reuse rates conditioned on target presence, identity recovery when present, reuse when absent, and saving relative to refresh. Aggregate tables apply the `_mean`, `_seed_sd`, `_seed_min`, and `_seed_max` convention. The stress bootstrap table reports 95% identity-cluster intervals for the same principal outcomes.

## Missing values

Undefined conditional quantities are encoded as empty CSV fields and JSON `null`. Examples include identity precision when a method produces no reuses and identity recovery when a condition has no target-present queries. They are excluded from arithmetic summaries rather than interpreted as zero.

## Reproduction manifest

`results/generated/reproduction_manifest.json` records the configuration path and digest, input-artifact digests, package version, interpreter and library versions, run stages, seeds, output files, and file hashes. It provides the machine-readable link between a generated result set and the inputs that produced it.
