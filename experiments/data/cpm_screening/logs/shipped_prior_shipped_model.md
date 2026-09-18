# CPM screening -- blocksize 50, 37008 corpus rows, 2056 evaluations
snapshots [250, 300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 58, 10-90% 29-159; 3.7% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | skipped: the shipped feature-space model is fitted for 32 bins | | | | |
| 8 | 1 | 4 | skipped: the shipped feature-space model is fitted for 32 bins | | | | |
| 16 | 1 | 1 | skipped: the shipped feature-space model is fitted for 32 bins | | | | |
| 16 | 1 | 4 | skipped: the shipped feature-space model is fitted for 32 bins | | | | |
| 32 | 1 | 1 | 0.1409 | 0.5328 | **3.78** | 5.00 (scalars_only) | 0.8% |
| 32 | 1 | 4 | 0.0565 | 0.4441 | **7.86** | 13.49 (scalars_only) | 0.4% |

Best protocol on the shipped (equal-weight) metric: **b32_s1_k4** -- 32 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 7.86

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0781 | 0.9719 | 12.45 | 2.104 |
| log_r95 | 1 | 0.1019 | 0.9203 | 9.03 | 2.136 |
| radial_fa_equal_volume | 10 | 1.0900 | 1.3529 | 1.24 | 32.407 |
| pair_correlation_gofr | 10 | 1.2937 | 1.2639 | 0.98 | 44.743 |
| radial_s2_equal_volume | 10 | 1.1426 | 0.9423 | 0.82 | 21.205 |
| radial_density_profile_equal_volume | 10 | 1.2968 | 1.0054 | 0.78 | 28.503 |
| dbscan_gaslike_fraction  (dead) | 1 | 0.0000 | 0.0375 | n/a | 1.000 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0565 | 0.4441 | **7.86** |
| drop_dead | 6 | 0.0659 | 0.5181 | **7.86** |
| scalars_only | 2 | 0.1102 | 1.4869 | **13.49** |
| snr_pruned | 3 | 0.0973 | 1.0265 | **10.55** |

## Parameter screen under that protocol

Summary space: 42 live coordinates (1 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| motility | [0, 10000] lin | **13.36** | 35 | log_n 48%, log_r95 25% | [357, 6786] | 64% |
| division_rate | [6e-05, 0.6] lin | **0.40** | 14 | log_n 46%, log_r95 30% | [0.02149, 0.1929] | 29% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | motility | division_rate |
|---|---|---|
| **motility** | 1.00 | 0.89 |
| **division_rate** | 0.89 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/shipped_prior/screening_report_shipped_model.json
