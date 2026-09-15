# Reproducibility contract

## Two supported routes

### Frozen-input reproduction

This is the reference route for the paper's protocol experiments. It begins with the checked-in semantic features and requires no dataset or model download:

```bash
python main.py verify
python main.py reproduce --config configs/paper.json --overwrite
```

The command leaves `artifacts/frozen_inputs/`, `results/paper/`, and `figures/paper/` unchanged. New outputs are written to `results/generated/` and `figures/generated/`. The reviewed conceptual Figure 1 exports are copied unchanged into the generated figure set, while Figures 2--7 are rebuilt from frozen records.

### Raw-image reproduction

This route reconstructs the semantic features before rerunning the protocol experiments:

```bash
python -m pip install -e ".[raw]"
python main.py extract-embeddings --config configs/paper.json --output artifacts/generated_inputs
python main.py reproduce --config configs/paper.json --inputs artifacts/generated_inputs --overwrite
```

The extraction command downloads CIFAR-10 and `ResNet18_Weights.IMAGENET1K_V1` through TorchVision when the files are absent from the local cache.

## Software environment

The recorded paper environment is:

- Python 3.12.11;
- PyTorch 2.6.0;
- TorchVision 0.21.0;
- NumPy 1.26.4; and
- CPU execution.

`requirements.txt` pins the frozen-input runtime packages, while `requirements-raw.txt` pins PyTorch and TorchVision for raw-image extraction. `environment.yml` supplies the complete Conda environment. The protocol experiments operate on NumPy arrays and are CPU-oriented; the raw-image feature extractor uses PyTorch and also runs on CPU.

## Data partitions

All partitions are deterministic functions of base seed `20260907`.

| Partition | Size | Role |
|---|---:|---|
| Development cache | 1,000 | Candidate bank for calibration and allocation selection |
| CAL-TUNE | 500 | Threshold and proposal-audit allocation selection |
| CAL-VALIDATE | 500 | Independent threshold validation |
| Evaluation cache | 1,000 | Candidate bank for final protocol evaluation |
| Standard repeated views | 500 | Primary evaluation queries |
| Strong views | 500 | Distribution-shift stress queries |
| Novel sources | 200 | Source identities absent from the cache |

CAL-TUNE and CAL-VALIDATE use disjoint source identities. The evaluation pool is excluded from threshold and allocation selection.

## Image transformations

Every image is converted to RGB, resized to 128 × 128 by bilinear interpolation with antialiasing, and normalized by ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

The standard repeated-view transformation uses reflect-padded translations of at most 3 pixels, horizontal flip probability 0.5, brightness and contrast factors in `[0.9, 1.1]`, and Gaussian pixel noise with standard deviation `2/255`.

The strong-view transformation uses reflect-padded translations of at most 5 pixels, rotation uniformly distributed from −12° to +12°, brightness and contrast factors in `[0.75, 1.25]`, and Gaussian pixel noise with standard deviation `6/255`.

The frozen ResNet-18 classification layer is replaced by identity. The resulting 512-dimensional global-average-pooled feature is L2-normalized before storage or comparison.

## Calibration

Threshold selection searches 91 empirical similarity quantiles from the 5th through the 95th percentile on CAL-TUNE. A candidate threshold must retain at least 200 pairs and achieve at least 95% empirical source-identity precision. Among feasible thresholds, the procedure maximizes the retained-pair count. The selected threshold is then accepted only when its CAL-VALIDATE source-identity precision has a Wilson 95% lower bound of at least 90%.

The frozen result is:

```text
cosine threshold = 0.7717004776000975
CAL-TUNE selected pairs = 305 / 500
CAL-TUNE identity precision = 0.9639344262
CAL-VALIDATE selected pairs = 288 / 500
CAL-VALIDATE identity precision = 0.9479166667
CAL-VALIDATE Wilson lower 95% bound = 0.9158556008
```

## Random-hyperplane channel model

For two unit-normalized features with cosine similarity `s`, the clean sign-mismatch probability is `arccos(s)/pi`. Under a binary symmetric channel with crossover probability `p`, the observed mismatch probability is:

```text
q(s, p) = p + (1 - 2p) arccos(s) / pi
```

Proposal signs select the minimum-Hamming cache entry, with ties resolved by the lowest cache index. The candidate index is fixed before the independent audit seed and audit signs are generated. The fixed audit accepts only when the mismatch count is at or below the largest exact binomial lower-tail boundary whose null probability does not exceed `alpha = 0.01`.

## Allocation selection

The total sign budget is 1,024. CAL-TUNE evaluates:

```text
128/896, 256/768, 384/640, 512/512,
640/384, 768/256, 896/128
```

Each pair is `proposal bits / audit bits`. Five development protocol seeds are evaluated at BSC crossover probabilities 0 and 0.05. The selection minimizes mean fallback-inclusive logical bits after averaging the two channel conditions equally; ties favor the allocation closest to balanced and then the one with fewer proposal bits. This selects 256/768.

The development protocol seeds are:

```text
4108815543219190477
7915450270055424108
14611438597939087046
17838361416203552926
15742982603362453269
```

The disjoint evaluation protocol seeds are:

```text
13181815969110816250
14673129866313320424
13231421562932395684
5635005200607147554
17148364466832244007
```

## Traffic accounting

Every logical frame has an 18-byte overhead. For the selected 256/768 protocol:

| Component | Bits |
|---|---:|
| Proposal frame | 400 |
| Candidate-lock frame | 272 |
| Audit-seed frame | 208 |
| Audit-sign frame | 912 |
| Decision frame | 152 |
| Total before fallback | 1,944 |

The feature refresh contains 512 signed 8-bit coefficients, a 32-bit quantization scale, and a 144-bit frame, totaling 4,272 bits. If `C` denotes cache-reuse success, fallback-inclusive completion traffic is:

```text
B_completion = B_protocol + (1 - C) × 4272
```

## Replication and uncertainty

The primary experiment evaluates 500 independent source identities under five disjoint evaluation protocol seeds at both BSC conditions. Confidence intervals are produced by 2,000 source-identity cluster-bootstrap resamples. Each resampled identity retains all five associated protocol seeds, preserving the experiment's dependence structure.

The sequential null validation uses 50,000 sessions per channel condition. The post-selection boundary experiment uses 10,000 trials for each candidate-pool size.

## Artifact integrity

Run:

```bash
python main.py verify
```

The verifier checks the published hashes, schema invariants, sample counts, selected allocation, exact audit masses, headline metrics, and figure presence. Fresh runs record their own hashes and environment information in `results/generated/reproduction_manifest.json`.
