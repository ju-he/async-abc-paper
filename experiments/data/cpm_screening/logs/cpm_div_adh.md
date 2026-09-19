# CPM screening -- blocksize 50, 41040 corpus rows, 4560 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 35, 10-90% 7-91; 33.8% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0834 | 0.1892 | **2.27** | 4.28 (scalars_only) | 3.3% |
| 8 | 1 | 4 | 0.0562 | 0.1622 | **2.89** | 8.02 (scalars_only) | 0.3% |
| 16 | 1 | 1 | 0.0793 | 0.1477 | **1.86** | 4.28 (scalars_only) | 3.2% |
| 16 | 1 | 4 | 0.0653 | 0.1370 | **2.10** | 8.02 (scalars_only) | 0.3% |
| 32 | 1 | 1 | 0.0798 | 0.1607 | **2.01** | 4.28 (scalars_only) | 3.2% |
| 32 | 1 | 4 | 0.0603 | 0.1282 | **2.13** | 8.02 (scalars_only) | 0.3% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 2.89

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0441 | 0.8861 | 20.07 | 1.040 |
| log_r95 | 1 | 0.0604 | 0.8700 | 14.41 | 1.182 |
| pair_correlation_gofr | 6 | 1.3198 | 2.0000 | 1.52 | 76.782 |
| dbscan_gaslike_fraction | 1 | 0.6734 | 0.7400 | 1.10 | 2.020 |
| radial_density_profile_equal_volume | 6 | 1.6743 | 1.7620 | 1.05 | 34.684 |
| radial_s2_equal_volume | 7 | 1.9278 | 1.7298 | 0.90 | 17.144 |
| radial_fa_equal_volume | 6 | 1.7663 | 1.0753 | 0.61 | 147.924 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0562 | 0.1622 | **2.89** |
| drop_dead | 7 | 0.0562 | 0.1622 | **2.89** |
| scalars_only | 2 | 0.0609 | 0.4882 | **8.02** |
| snr_pruned | 5 | 0.0381 | 0.2274 | **5.97** |

## Parameter screen under that protocol

Summary space: 28 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **12.11** | 40 | log_n 41%, radial_fa_equal_volume 29% | [0.00118, 0.01669] | 50% |
| adhesion_cl | [20, 200] lin | **1.73** | 5 | radial_fa_equal_volume 86%, radial_density_profile_equal_volume 4% | [149, 183] | 19% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | adhesion_cl |
|---|---|---|
| **division_rate** | 1.00 | 0.05 |
| **adhesion_cl** | 0.05 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_div_adh/screening_report.json
