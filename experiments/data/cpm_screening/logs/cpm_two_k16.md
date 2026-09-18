# CPM screening -- blocksize 50, 92160 corpus rows, 10240 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 51, 10-90% 7-173; 30.8% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0912 | 0.2446 | **2.68** | 6.16 (scalars_only) | 2.8% |
| 8 | 1 | 4 | 0.0679 | 0.2655 | **3.91** | 17.16 (scalars_only) | 5.8% |
| 16 | 1 | 1 | 0.0636 | 0.2179 | **3.42** | 6.16 (scalars_only) | 3.3% |
| 16 | 1 | 4 | 0.0657 | 0.2168 | **3.30** | 17.16 (scalars_only) | 5.7% |
| 32 | 1 | 1 | 0.0823 | 0.1686 | **2.05** | 6.16 (scalars_only) | 3.0% |
| 32 | 1 | 4 | 0.0700 | 0.1955 | **2.79** | 17.31 (snr_pruned) | 6.2% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 3.91

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0282 | 0.9225 | 32.66 | 0.619 |
| log_r95 | 1 | 0.0301 | 0.8927 | 29.61 | 0.706 |
| pair_correlation_gofr | 6 | 1.2237 | 1.5569 | 1.27 | 20.711 |
| dbscan_gaslike_fraction | 1 | 0.5398 | 0.5853 | 1.08 | 1.724 |
| radial_density_profile_equal_volume | 6 | 1.8190 | 1.2685 | 0.70 | 37.414 |
| radial_s2_equal_volume | 6 | 2.0676 | 1.2253 | 0.59 | 14.547 |
| radial_fa_equal_volume | 6 | 2.2097 | 0.8205 | 0.37 | 310.106 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0679 | 0.2655 | **3.91** |
| drop_dead | 7 | 0.0679 | 0.2655 | **3.91** |
| scalars_only | 2 | 0.0480 | 0.8228 | **17.16** |
| snr_pruned | 4 | 0.0783 | 0.4439 | **5.67** |

## Parameter screen under that protocol

Summary space: 27 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **13.76** | 56 | log_n 35%, log_r95 34% | [0.00118, 0.01669] | 50% |
| motility | [100, 4000] log | **0.85** | 3 | pair_correlation_gofr 97%, log_n 1% | [141, 3564] | 88% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | motility |
|---|---|---|
| **division_rate** | 1.00 | 0.11 |
| **motility** | 0.11 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_two_k16/screening_report.json
