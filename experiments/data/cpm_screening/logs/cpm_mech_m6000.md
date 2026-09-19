# CPM screening -- blocksize 50, 35928 corpus rows, 3992 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 13, 10-90% 5-30; 74.2% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.1459 | 0.2382 | **1.63** | 2.85 (scalars_only) | 3.9% |
| 8 | 1 | 4 | 0.1127 | 0.2334 | **2.07** | 7.01 (scalars_only) | 0.0% |
| 16 | 1 | 1 | 0.1193 | 0.2097 | **1.76** | 2.85 (scalars_only) | 4.7% |
| 16 | 1 | 4 | 0.1341 | 0.1825 | **1.36** | 7.01 (scalars_only) | 0.1% |
| 32 | 1 | 1 | 0.1390 | 0.2156 | **1.55** | 2.85 (scalars_only) | 4.2% |
| 32 | 1 | 4 | 0.0785 | 0.2287 | **2.91** | 7.01 (scalars_only) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b32_s1_k4** -- 32 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 2.91

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0764 | 0.9928 | 12.99 | 1.475 |
| log_r95 | 1 | 0.1391 | 0.9818 | 7.06 | 1.472 |
| dbscan_gaslike_fraction | 1 | 0.5684 | 0.8235 | 1.45 | 1.886 |
| pair_correlation_gofr | 10 | 1.9841 | 2.0058 | 1.01 | 139.449 |
| radial_fa_equal_volume | 10 | 2.2991 | 2.1839 | 0.95 | 20.195 |
| radial_s2_equal_volume | 10 | 2.3144 | 2.1317 | 0.92 | 25.687 |
| radial_density_profile_equal_volume | 10 | 2.5752 | 1.8294 | 0.71 | 36.185 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0785 | 0.2287 | **2.91** |
| drop_dead | 7 | 0.0785 | 0.2287 | **2.91** |
| scalars_only | 2 | 0.0723 | 0.5065 | **7.01** |
| snr_pruned | 4 | 0.0631 | 0.3262 | **5.17** |

## Parameter screen under that protocol

Summary space: 43 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **10.97** | 28 | log_n 45%, radial_fa_equal_volume 35% | [0.001208, 0.05318] | 71% |
| surface_lambda | [0.25, 8] log | **3.26** | 13 | pair_correlation_gofr 68%, log_n 14% | [0.3624, 2.626] | 57% |
| recalc_time | [2, 60] log | **2.78** | 11 | pair_correlation_gofr 41%, log_n 20% | [8, 42] | 49% |
| persistence | [0.01, 0.99] lin | **0.00** | 5 | pair_correlation_gofr 83%, radial_s2_equal_volume 6% | [0.115, 0.885] | 79% |
| adhesion_cl | [20, 200] lin | **1.18** | 3 | pair_correlation_gofr 88%, radial_fa_equal_volume 5% | [116, 181] | 36% |
| adhesion_cc | [40, 400] lin | **0.73** | 1 | pair_correlation_gofr 75%, radial_fa_equal_volume 11% | [156, 181] | 7% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | surface_lambda | recalc_time | persistence | adhesion_cl | adhesion_cc |
|---|---|---|---|---|---|---|
| **division_rate** | 1.00 | 0.55 | 0.78 | 0.32 | 0.00 | 0.40 |
| **surface_lambda** | 0.55 | 1.00 | 0.68 | 0.16 | 0.10 | 0.22 |
| **recalc_time** | 0.78 | 0.68 | 1.00 | 0.26 | 0.05 | 0.27 |
| **persistence** | 0.32 | 0.16 | 0.26 | 1.00 | 0.23 | 0.84 |
| **adhesion_cl** | 0.00 | 0.10 | 0.05 | 0.23 | 1.00 | 0.25 |
| **adhesion_cc** | 0.40 | 0.22 | 0.27 | 0.84 | 0.25 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_mech_m6000/screening_report.json
