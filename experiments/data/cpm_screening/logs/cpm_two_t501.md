# CPM screening -- blocksize 50, 48240 corpus rows, 5360 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 52, 10-90% 7-176; 30.5% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0550 | 0.2268 | **4.13** | 5.96 (scalars_only) | 3.9% |
| 8 | 1 | 4 | 0.0327 | 0.1817 | **5.55** | 15.61 (scalars_only) | 0.2% |
| 16 | 1 | 1 | 0.0612 | 0.2027 | **3.31** | 5.96 (scalars_only) | 3.3% |
| 16 | 1 | 4 | 0.0411 | 0.1637 | **3.98** | 15.61 (scalars_only) | 0.1% |
| 32 | 1 | 1 | 0.0659 | 0.1481 | **2.25** | 5.96 (scalars_only) | 3.4% |
| 32 | 1 | 4 | 0.0401 | 0.1277 | **3.19** | 15.61 (scalars_only) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 5.55

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0270 | 0.9057 | 33.52 | 1.011 |
| log_r95 | 1 | 0.0334 | 0.8814 | 26.39 | 1.119 |
| dbscan_gaslike_fraction | 1 | 0.4451 | 0.8959 | 2.01 | 1.914 |
| radial_density_profile_equal_volume | 6 | 1.5201 | 1.8391 | 1.21 | 37.956 |
| radial_s2_equal_volume | 7 | 1.8521 | 1.8124 | 0.98 | 18.315 |
| pair_correlation_gofr | 6 | 1.7942 | 1.2634 | 0.70 | 361.308 |
| radial_fa_equal_volume | 6 | 1.9435 | 1.0815 | 0.56 | 324.284 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0327 | 0.1817 | **5.55** |
| drop_dead | 7 | 0.0327 | 0.1817 | **5.55** |
| scalars_only | 2 | 0.0320 | 0.4997 | **15.61** |
| snr_pruned | 4 | 0.0204 | 0.2699 | **13.24** |

## Parameter screen under that protocol

Summary space: 28 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **17.83** | 62 | log_n 47%, log_r95 33% | [0.00118, 0.02324] | 56% |
| motility | [100, 4000] log | **4.15** | 17 | log_n 49%, log_r95 25% | [564, 2831] | 44% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | motility |
|---|---|---|
| **division_rate** | 1.00 | 0.86 |
| **motility** | 0.86 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_two_t501/screening_report.json
