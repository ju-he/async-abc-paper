# CPM screening -- blocksize 50, 69360 corpus rows, 4624 evaluations
snapshots [300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 47, 10-90% 7-214; 30.3% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.1192 | 0.2672 | **2.24** | 5.14 (scalars_only) | 3.2% |
| 8 | 1 | 4 | 0.1051 | 0.2367 | **2.25** | 12.85 (scalars_only) | 0.2% |
| 16 | 1 | 1 | 0.1092 | 0.2316 | **2.12** | 5.14 (scalars_only) | 3.3% |
| 16 | 1 | 4 | 0.1447 | 0.1550 | **1.07** | 12.85 (scalars_only) | 0.1% |
| 32 | 1 | 1 | 0.1194 | 0.1717 | **1.44** | 5.14 (scalars_only) | 3.3% |
| 32 | 1 | 4 | 0.1219 | 0.1442 | **1.18** | 12.85 (scalars_only) | 0.1% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 2.25

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0323 | 0.9737 | 30.12 | 1.217 |
| log_r95 | 1 | 0.0420 | 0.8684 | 20.67 | 1.328 |
| msd | 2 | 0.3245 | 1.3904 | 4.28 | 2.547 |
| pair_correlation_gofr | 6 | 1.1468 | 2.0873 | 1.82 | 37.877 |
| non_gaussian_parameter | 3 | 0.8289 | 1.4699 | 1.77 | 10.957 |
| radial_density_profile_equal_volume | 6 | 1.2817 | 2.0173 | 1.57 | 50.373 |
| radial_s2_equal_volume | 6 | 1.5993 | 1.7701 | 1.11 | 14.711 |
| radial_fa_equal_volume | 6 | 1.6610 | 1.4691 | 0.88 | 260.337 |
| dbscan_gaslike_fraction | 1 | 1.1090 | 0.0000 | 0.00 | 2.156 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 9 | 0.1051 | 0.2367 | **2.25** |
| drop_dead | 9 | 0.1051 | 0.2367 | **2.25** |
| scalars_only | 2 | 0.0319 | 0.4095 | **12.85** |
| snr_pruned | 7 | 0.1051 | 0.2638 | **2.51** |

## Parameter screen under that protocol

Summary space: 32 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **17.53** | 54 | log_n 40%, log_r95 26% | [0.001208, 0.01709] | 50% |
| cell_volume | [200, 1200] log | **15.46** | 45 | radial_density_profile_equal_volume 53%, pair_correlation_gofr 24% | [242, 767] | 64% |
| motility | [100, 4000] log | **2.44** | 16 | log_n 38%, log_r95 22% | [722, 3506] | 43% |
| temperature | [15, 150] log | **2.61** | 10 | radial_fa_equal_volume 72%, radial_density_profile_equal_volume 10% | [16, 117] | 86% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | cell_volume | motility | temperature |
|---|---|---|---|---|
| **division_rate** | 1.00 | 0.17 | 0.82 | 0.27 |
| **cell_volume** | 0.17 | 1.00 | 0.15 | 0.08 |
| **motility** | 0.82 | 0.15 | 1.00 | 0.32 |
| **temperature** | 0.27 | 0.08 | 0.32 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_new4/screening_report.json
