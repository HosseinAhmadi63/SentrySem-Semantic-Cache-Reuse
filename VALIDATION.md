# Validation record

The checked-in paper artifacts form a read-only reference set. Run the complete validation suite from the repository root with:

```bash
python main.py verify
python -m pytest
```

`python main.py verify` reads the frozen inputs, reference results, configuration, and figure files without modifying them. A successful run ends with `PASS`.

## Validated experimental contract

| Item | Reference value |
|---|---:|
| Frozen embedding archive SHA-256 | `77173ff70b1fbe318219692771120ade7aa430fc2a0683b82f25a486631d88a0` |
| Calibration file SHA-256 | `94aae0e166c7181c87dd3911ae9c44378820e39f70ecc188581ee1ef19720cdb` |
| Base seed | `20260907` |
| Calibration threshold | `0.7717004776000975` |
| Verifier error target | `0.01` |
| Selected proposal bits | `256` |
| Selected audit bits | `768` |
| Audit boundary at BSC 0 | `141` mismatches |
| Audit boundary at BSC 0.05 | `162` mismatches |
| Exact null mass at BSC 0 | `0.00831` |
| Exact null mass at BSC 0.05 | `0.00972` |
| Feature-refresh traffic | `4272` bits |
| Selected protocol traffic before fallback | `1944` bits |

## Headline numerical checks

The verifier checks the following paper values against the immutable result tables:

| Quantity | BSC 0 | BSC 0.05 |
|---|---:|---:|
| Reuse rate | 86.80% | 83.04% |
| Source-instance precision among reuses | 99.723% | 99.710% |
| Mean fallback-inclusive completion traffic | 2507.9 bits | 2668.5 bits |
| Traffic reduction from refresh | 41.29% | 37.53% |
| Bad-reuse rate | 0 | 0 |

The post-selection boundary checks are:

| Searched candidates | Reused search evidence | Fresh independent audit |
|---:|---:|---:|
| 1 | 0.47% | 0.57% |
| 8 | 4.14% | 0.50% |
| 64 | 25.15% | 0.45% |

The finite sequential validation uses 50,000 null sessions per channel condition and produces familywise false-acceptance estimates of 0.088% at BSC 0 and 0.104% at BSC 0.05.

## Structural checks

The validation command confirms:

- all required input arrays and metadata fields are present;
- the development, calibration-selection, calibration-validation, and evaluation partitions have their declared sizes;
- CAL-TUNE and CAL-VALIDATE identities are disjoint;
- development and evaluation protocol-seed sets are disjoint;
- the selected allocation is 256/768, its recorded tuning and evaluation seed sets match the configuration, and those seed sets are disjoint;
- the exact binomial boundaries satisfy the 1% null-probability requirement;
- principal aggregate statistics recompute from the corresponding seed-level tables;
- all 50,000 comparator session rows and 22,000 stress session rows are present;
- each primary query identity has all five protocol-seed realizations required by the cluster bootstrap;
- the traffic ledger closes for reuse and refresh outcomes; and
- every manuscript figure file is present and nonempty; and
- the curated Figure 1 exports and editable PowerPoint source form a valid publication bundle.

## Fresh-run comparison

To generate a separate full result set and compare it with the reference:

```bash
python main.py reproduce --config configs/paper.json --overwrite
python main.py verify --results results/generated
```

Deterministic fields are compared exactly. Floating-point summaries are compared with the tolerances declared by the verifier so that supported operating systems and processor architectures produce the same scientific conclusions.
