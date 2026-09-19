# CPM screening -- blocksize 50, 35928 corpus rows, 3992 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 49, 10-90% 7-150; 27.3% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0566 | 0.2325 | **4.11** | 6.37 (scalars_only) | 5.3% |
| 8 | 1 | 4 | 0.0420 | 0.2018 | **4.81** | 11.53 (scalars_only) | 0.3% |
| 16 | 1 | 1 | 0.0731 | 0.2120 | **2.90** | 6.37 (scalars_only) | 3.3% |
| 16 | 1 | 4 | 0.0477 | 0.1787 | **3.74** | 11.53 (scalars_only) | 0.2% |
| 32 | 1 | 1 | 0.0909 | 0.1529 | **1.68** | 6.37 (scalars_only) | 3.2% |
| 32 | 1 | 4 | 0.0543 | 0.1477 | **2.72** | 11.53 (scalars_only) | 0.3% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 4.81

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0309 | 0.9009 | 29.12 | 1.163 |
| log_r95 | 1 | 0.0551 | 0.9103 | 16.52 | 1.317 |
| pair_correlation_gofr | 6 | 1.2047 | 2.0644 | 1.71 | 60.293 |
| dbscan_gaslike_fraction | 1 | 0.5245 | 0.8522 | 1.62 | 1.882 |
| radial_s2_equal_volume | 6 | 1.4644 | 1.9018 | 1.30 | 14.601 |
| radial_fa_equal_volume | 6 | 1.8576 | 1.3079 | 0.70 | 344.477 |
| radial_density_profile_equal_volume | 6 | 1.9978 | 1.3571 | 0.68 | 37.573 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0420 | 0.2018 | **4.81** |
| drop_dead | 7 | 0.0420 | 0.2018 | **4.81** |
| scalars_only | 2 | 0.0357 | 0.4114 | **11.53** |
| snr_pruned | 5 | 0.0440 | 0.2298 | **5.22** |

## Parameter screen under that protocol

Summary space: 27 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **14.87** | 52 | log_n 59%, log_r95 20% | [0.001208, 0.01709] | 50% |
| surface_lambda | [0.25, 8] log | **3.17** | 8 | pair_correlation_gofr 47%, log_n 36% | [0.3624, 2.05] | 50% |
| recalc_time | [2, 60] log | **0.00** | 7 | pair_correlation_gofr 53%, radial_fa_equal_volume 16% | [2, 33] | 82% |
| persistence | [0.01, 0.99] lin | **0.00** | 2 | pair_correlation_gofr 59%, radial_s2_equal_volume 17% | [0.045, 0.955] | 93% |
| adhesion_cl | [20, 200] lin | **0.00** | 2 | pair_correlation_gofr 68%, log_n 15% | [168, 194] | 14% |
| adhesion_cc | [40, 400] lin | **0.00** | 1 | pair_correlation_gofr 100%, radial_s2_equal_volume 0% | [79, 284] | 57% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | surface_lambda | recalc_time | persistence | adhesion_cl | adhesion_cc |
|---|---|---|---|---|---|---|
| **division_rate** | 1.00 | 0.71 | 0.45 | 0.39 | 0.41 | 0.04 |
| **surface_lambda** | 0.71 | 1.00 | 0.66 | 0.70 | 0.09 | 0.41 |
| **recalc_time** | 0.45 | 0.66 | 1.00 | 0.52 | 0.28 | 0.34 |
| **persistence** | 0.39 | 0.70 | 0.52 | 1.00 | 0.40 | 0.51 |
| **adhesion_cl** | 0.41 | 0.09 | 0.28 | 0.40 | 1.00 | 0.55 |
| **adhesion_cc** | 0.04 | 0.41 | 0.34 | 0.51 | 0.55 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_mech_m500/screening_report.json
