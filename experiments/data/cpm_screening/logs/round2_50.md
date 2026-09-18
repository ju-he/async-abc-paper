# CPM screening -- blocksize 50, 44568 corpus rows, 2476 evaluations
snapshots [250, 300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 40, 10-90% 10-90; 25.9% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0770 | 0.2838 | **3.69** | 5.30 (scalars_only) | 3.9% |
| 8 | 1 | 2 | 0.0631 | 0.3113 | **4.93** | 7.95 (scalars_only) | 4.8% |
| 8 | 1 | 4 | 0.0605 | 0.2701 | **4.47** | 13.41 (scalars_only) | 0.4% |
| 8 | 3 | 1 | 0.1114 | 0.2601 | **2.33** | 5.40 (scalars_only) | 2.3% |
| 8 | 3 | 2 | 0.0782 | 0.2886 | **3.69** | 8.28 (snr_pruned) | 3.8% |
| 8 | 3 | 4 | 0.0783 | 0.2575 | **3.29** | 13.40 (scalars_only) | 0.7% |
| 8 | 6 | 1 | 0.1132 | 0.2494 | **2.20** | 4.31 (scalars_only) | 3.6% |
| 8 | 6 | 2 | 0.0867 | 0.3038 | **3.50** | 7.55 (scalars_only) | 4.6% |
| 8 | 6 | 4 | 0.1180 | 0.2539 | **2.15** | 10.36 (scalars_only) | 0.9% |
| 16 | 1 | 1 | 0.0829 | 0.2691 | **3.25** | 5.30 (scalars_only) | 2.8% |
| 16 | 1 | 2 | 0.0588 | 0.2937 | **5.00** | 7.95 (scalars_only) | 4.0% |
| 16 | 1 | 4 | 0.0666 | 0.2509 | **3.77** | 13.41 (scalars_only) | 0.4% |
| 16 | 3 | 1 | 0.0952 | 0.2277 | **2.39** | 5.40 (scalars_only) | 2.8% |
| 16 | 3 | 2 | 0.0655 | 0.2782 | **4.24** | 6.79 (scalars_only) | 5.4% |
| 16 | 3 | 4 | 0.0831 | 0.2313 | **2.78** | 13.40 (scalars_only) | 0.7% |
| 16 | 6 | 1 | 0.1118 | 0.2204 | **1.97** | 4.31 (scalars_only) | 3.9% |
| 16 | 6 | 2 | 0.0920 | 0.2589 | **2.81** | 7.55 (scalars_only) | 5.0% |
| 16 | 6 | 4 | 0.1782 | 0.1949 | **1.09** | 10.36 (scalars_only) | 0.2% |
| 32 | 1 | 1 | 0.0821 | 0.2091 | **2.55** | 5.30 (scalars_only) | 3.3% |
| 32 | 1 | 2 | 0.0911 | 0.2455 | **2.70** | 7.95 (scalars_only) | 2.8% |
| 32 | 1 | 4 | 0.0698 | 0.2300 | **3.30** | 13.41 (scalars_only) | 0.7% |
| 32 | 3 | 1 | 0.1054 | 0.2077 | **1.97** | 5.40 (scalars_only) | 2.1% |
| 32 | 3 | 2 | 0.0780 | 0.2195 | **2.82** | 6.79 (scalars_only) | 4.5% |
| 32 | 3 | 4 | 0.0699 | 0.2150 | **3.08** | 13.40 (scalars_only) | 0.7% |
| 32 | 6 | 1 | 0.1205 | 0.2149 | **1.78** | 4.31 (scalars_only) | 3.6% |
| 32 | 6 | 2 | 0.1051 | 0.2487 | **2.37** | 7.55 (scalars_only) | 4.3% |
| 32 | 6 | 4 | 0.0949 | 0.2301 | **2.42** | 10.36 (scalars_only) | 0.7% |

Best protocol on the shipped (equal-weight) metric: **b16_s1_k2** -- 16 bins, 1 snapshot(s) at [500], 2 seed(s) per evaluation; SNR 5.00

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0982 | 0.9752 | 9.94 | 1.015 |
| log_r95 | 1 | 0.1266 | 0.9911 | 7.83 | 1.172 |
| radial_fa_equal_volume | 10 | 1.9137 | 1.8525 | 0.97 | 35.090 |
| pair_correlation_gofr | 10 | 2.3019 | 2.1761 | 0.95 | 231.929 |
| radial_s2_equal_volume | 10 | 2.5933 | 1.4118 | 0.54 | 21.682 |
| radial_density_profile_equal_volume | 10 | 2.7045 | 1.2647 | 0.47 | 41.395 |
| dbscan_gaslike_fraction | 1 | 0.9175 | 0.2744 | 0.30 | 1.717 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0588 | 0.2937 | **5.00** |
| drop_dead | 7 | 0.0588 | 0.2937 | **5.00** |
| scalars_only | 2 | 0.0772 | 0.6131 | **7.95** |
| snr_pruned | 2 | 0.0772 | 0.6131 | **7.95** |

## Parameter screen under that protocol

Summary space: 43 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.002, 0.06] log | **7.18** | 18 | log_n 37%, radial_fa_equal_volume 29% | [0.003671, 0.02011] | 50% |
| motility | [300, 3000] log | **4.18** | 11 | log_n 55%, log_r95 29% | [629, 2763] | 64% |
| surface_lambda | [0.25, 8] log | **3.36** | 10 | log_n 35%, log_r95 27% | [0.5946, 7.069] | 71% |
| adhesion_cc | [40, 400] lin | **0.65** | 3 | pair_correlation_gofr 78%, radial_fa_equal_volume 10% | [79, 387] | 86% |
| adhesion_cl | [20, 200] lin | **0.00** | 2 | pair_correlation_gofr 91%, radial_fa_equal_volume 4% | [52, 194] | 79% |
| persistence | [0.002, 0.3] log | **1.54** | 2 | radial_fa_equal_volume 76%, radial_s2_equal_volume 9% | [0.08572, 0.1754] | 14% |
| recalc_time | [2, 60] log | **0.00** | 2 | pair_correlation_gofr 68%, radial_s2_equal_volume 13% | [3, 4] | 8% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | motility | surface_lambda | adhesion_cc | adhesion_cl | persistence | recalc_time |
|---|---|---|---|---|---|---|---|
| **division_rate** | 1.00 | 0.77 | 0.83 | 0.09 | 0.02 | 0.07 | 0.23 |
| **motility** | 0.77 | 1.00 | 0.82 | 0.12 | 0.08 | 0.10 | 0.13 |
| **surface_lambda** | 0.83 | 0.82 | 1.00 | 0.03 | 0.11 | 0.06 | 0.20 |
| **adhesion_cc** | 0.09 | 0.12 | 0.03 | 1.00 | 0.09 | 0.08 | 0.33 |
| **adhesion_cl** | 0.02 | 0.08 | 0.11 | 0.09 | 1.00 | 0.11 | 0.44 |
| **persistence** | 0.07 | 0.10 | 0.06 | 0.08 | 0.11 | 1.00 | 0.02 |
| **recalc_time** | 0.23 | 0.13 | 0.20 | 0.33 | 0.44 | 0.02 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/round2_50/screening_report.json
