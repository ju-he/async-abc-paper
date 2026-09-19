# CPM screening -- blocksize 50, 41040 corpus rows, 4560 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 35, 10-90% 7-91; 33.4% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0773 | 0.1834 | **2.37** | 4.41 (scalars_only) | 3.0% |
| 8 | 1 | 4 | 0.0787 | 0.1309 | **1.66** | 11.74 (scalars_only) | 0.3% |
| 16 | 1 | 1 | 0.0893 | 0.1455 | **1.63** | 4.41 (scalars_only) | 2.8% |
| 16 | 1 | 4 | 0.0628 | 0.1291 | **2.06** | 11.74 (scalars_only) | 0.3% |
| 32 | 1 | 1 | 0.0855 | 0.1508 | **1.76** | 4.41 (scalars_only) | 3.2% |
| 32 | 1 | 4 | 0.0985 | 0.1073 | **1.09** | 11.80 (snr_pruned) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k1** -- 8 bins, 1 snapshot(s) at [500], 1 seed(s) per evaluation; SNR 2.37

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0905 | 0.8600 | 9.51 | 0.676 |
| log_r95 | 1 | 0.1514 | 0.8590 | 5.67 | 0.852 |
| pair_correlation_gofr | 7 | 1.7251 | 1.6038 | 0.93 | 232.937 |
| dbscan_gaslike_fraction | 1 | 0.6899 | 0.3705 | 0.54 | 1.755 |
| radial_s2_equal_volume | 7 | 2.4386 | 0.8945 | 0.37 | 17.376 |
| radial_density_profile_equal_volume | 7 | 2.5090 | 0.7267 | 0.29 | 38.775 |
| radial_fa_equal_volume | 7 | 2.5142 | 0.7002 | 0.28 | 415.870 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0773 | 0.1834 | **2.37** |
| drop_dead | 7 | 0.0773 | 0.1834 | **2.37** |
| scalars_only | 2 | 0.1529 | 0.6747 | **4.41** |
| snr_pruned | 2 | 0.1529 | 0.6747 | **4.41** |

## Parameter screen under that protocol

Summary space: 31 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **2.55** | 15 | radial_fa_equal_volume 49%, log_n 32% | [0.002288, 0.01198] | 31% |
| adhesion_cl | [110, 230] lin | **0.00** | 5 | radial_fa_equal_volume 93%, radial_s2_equal_volume 4% | [114, 121] | 6% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | adhesion_cl |
|---|---|---|
| **division_rate** | 1.00 | 0.60 |
| **adhesion_cl** | 0.60 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_div_adh_narrow/screening_report.json
