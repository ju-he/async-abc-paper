# CPM screening -- blocksize 50, 39888 corpus rows, 4432 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 27, 10-90% 19-34; 10.1% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.1771 | 0.0596 | **0.34** | 0.62 (scalars_only) | 1.0% |
| 8 | 1 | 4 | 0.1210 | 0.1100 | **0.91** | 1.79 (scalars_only) | 0.1% |
| 16 | 1 | 1 | 0.1194 | 0.0507 | **0.42** | 0.62 (scalars_only) | 1.1% |
| 16 | 1 | 4 | 0.0820 | 0.1212 | **1.48** | 1.79 (scalars_only) | 0.3% |
| 32 | 1 | 1 | 0.1189 | 0.0523 | **0.44** | 0.62 (scalars_only) | 1.1% |
| 32 | 1 | 4 | 0.1202 | 0.0960 | **0.80** | 1.79 (scalars_only) | 0.3% |

Best protocol on the shipped (equal-weight) metric: **b16_s1_k4** -- 16 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 1.48

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_r95 | 1 | 0.6344 | 0.7729 | 1.22 | 2.180 |
| radial_s2_equal_volume | 10 | 2.3169 | 2.1469 | 0.93 | 20.375 |
| radial_fa_equal_volume | 10 | 2.3639 | 2.1180 | 0.90 | 20.391 |
| radial_density_profile_equal_volume | 10 | 2.5341 | 1.9092 | 0.75 | 20.265 |
| pair_correlation_gofr | 10 | 2.6701 | 1.6698 | 0.63 | 44.746 |
| log_n | 1 | 0.8979 | 0.4285 | 0.48 | 2.048 |
| dbscan_gaslike_fraction  (dead) | 1 | 0.0000 | 0.0000 | n/a | 1.000 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0820 | 0.1212 | **1.48** |
| drop_dead | 6 | 0.0956 | 0.1414 | **1.48** |
| scalars_only | 2 | 0.1480 | 0.2646 | **1.79** |
| snr_pruned | 1 | 0.1794 | 0.2453 | **1.37** |

## Parameter screen under that protocol

Summary space: 42 live coordinates (1 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| adhesion_cl | [25, 110] lin | **0.00** | 3 | pair_correlation_gofr 40%, radial_fa_equal_volume 29% | [28, 44] | 19% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | adhesion_cl |
|---|---|
| **adhesion_cl** | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_adh_transition/screening_report.json
