# CPM screening -- blocksize 80, 43056 corpus rows, 2392 evaluations
snapshots [250, 300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 60, 10-90% 12-162; 19.6% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0950 | 0.2593 | **2.73** | 5.94 (scalars_only) | 2.1% |
| 8 | 1 | 2 | 0.0748 | 0.2584 | **3.45** | 7.97 (scalars_only) | 1.6% |
| 8 | 1 | 4 | 0.0704 | 0.2256 | **3.21** | 15.62 (scalars_only) | 0.7% |
| 8 | 3 | 1 | 0.0852 | 0.2606 | **3.06** | 5.39 (scalars_only) | 3.4% |
| 8 | 3 | 2 | 0.0802 | 0.2903 | **3.62** | 7.02 (scalars_only) | 3.4% |
| 8 | 3 | 4 | 0.0799 | 0.2532 | **3.17** | 9.93 (scalars_only) | 1.1% |
| 8 | 6 | 1 | 0.1086 | 0.2761 | **2.54** | 4.73 (scalars_only) | 3.5% |
| 8 | 6 | 2 | 0.1026 | 0.2946 | **2.87** | 7.42 (scalars_only) | 3.8% |
| 8 | 6 | 4 | 0.0928 | 0.2751 | **2.97** | 11.51 (scalars_only) | 0.7% |
| 16 | 1 | 1 | 0.0774 | 0.2826 | **3.65** | 5.94 (scalars_only) | 3.0% |
| 16 | 1 | 2 | 0.0625 | 0.2785 | **4.45** | 7.97 (scalars_only) | 2.3% |
| 16 | 1 | 4 | 0.0443 | 0.2165 | **4.88** | 15.62 (scalars_only) | 0.4% |
| 16 | 3 | 1 | 0.0803 | 0.2524 | **3.14** | 5.39 (scalars_only) | 2.9% |
| 16 | 3 | 2 | 0.0517 | 0.2609 | **5.04** | 7.02 (scalars_only) | 4.9% |
| 16 | 3 | 4 | 0.0594 | 0.2225 | **3.75** | 9.93 (scalars_only) | 0.4% |
| 16 | 6 | 1 | 0.0975 | 0.2338 | **2.40** | 4.73 (scalars_only) | 3.8% |
| 16 | 6 | 2 | 0.0872 | 0.2791 | **3.20** | 7.42 (scalars_only) | 4.0% |
| 16 | 6 | 4 | 0.0805 | 0.2555 | **3.17** | 11.51 (scalars_only) | 0.9% |
| 32 | 1 | 1 | 0.0902 | 0.2405 | **2.67** | 5.94 (scalars_only) | 1.9% |
| 32 | 1 | 2 | 0.0741 | 0.2415 | **3.26** | 7.97 (scalars_only) | 1.3% |
| 32 | 1 | 4 | 0.0455 | 0.2018 | **4.44** | 15.62 (scalars_only) | 0.7% |
| 32 | 3 | 1 | 0.0783 | 0.2104 | **2.69** | 5.39 (scalars_only) | 2.6% |
| 32 | 3 | 2 | 0.0798 | 0.2261 | **2.83** | 7.02 (scalars_only) | 3.6% |
| 32 | 3 | 4 | 0.0771 | 0.1934 | **2.51** | 9.93 (scalars_only) | 0.4% |
| 32 | 6 | 1 | 0.1043 | 0.2157 | **2.07** | 4.73 (scalars_only) | 3.6% |
| 32 | 6 | 2 | 0.0879 | 0.2399 | **2.73** | 7.42 (scalars_only) | 4.0% |
| 32 | 6 | 4 | 0.1043 | 0.2366 | **2.27** | 11.51 (scalars_only) | 0.4% |

Best protocol on the shipped (equal-weight) metric: **b16_s3_k2** -- 16 bins, 3 snapshot(s) at [400, 450, 500], 2 seed(s) per evaluation; SNR 5.04

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0694 | 1.0067 | 14.51 | 0.943 |
| log_r95 | 1 | 0.1063 | 1.0086 | 9.49 | 1.014 |
| radial_fa_equal_volume | 10 | 1.9417 | 1.8390 | 0.95 | 48.101 |
| pair_correlation_gofr | 10 | 2.3164 | 2.0738 | 0.90 | 135.650 |
| radial_density_profile_equal_volume | 10 | 2.3117 | 1.7766 | 0.77 | 40.130 |
| radial_s2_equal_volume | 10 | 2.7812 | 1.4419 | 0.52 | 21.368 |
| dbscan_gaslike_fraction | 1 | 1.1471 | 0.0000 | 0.00 | 2.207 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0517 | 0.2609 | **5.04** |
| drop_dead | 7 | 0.0517 | 0.2609 | **5.04** |
| scalars_only | 2 | 0.0931 | 0.6535 | **7.02** |
| snr_pruned | 2 | 0.0931 | 0.6535 | **7.02** |

## Parameter screen under that protocol

Summary space: 43 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.002, 0.2] log | **7.50** | 28 | radial_fa_equal_volume 43%, log_n 34% | [0.003276, 0.01697] | 36% |
| motility | [100, 4000] log | **1.76** | 11 | log_n 53%, radial_fa_equal_volume 19% | [939, 3506] | 36% |
| surface_lambda | [0.25, 8] log | **4.44** | 11 | log_n 41%, log_r95 20% | [0.3624, 4.308] | 71% |
| recalc_time | [2, 60] log | **3.51** | 7 | radial_fa_equal_volume 25%, radial_density_profile_equal_volume 17% | [6, 33] | 50% |
| persistence | [0, 0.99] lin | **1.31** | 6 | radial_s2_equal_volume 28%, pair_correlation_gofr 20% | [0.03536, 0.6718] | 64% |
| adhesion_cc | [40, 200] lin | **0.46** | 3 | radial_fa_equal_volume 42%, pair_correlation_gofr 41% | [46, 183] | 86% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | motility | surface_lambda | recalc_time | persistence | adhesion_cc |
|---|---|---|---|---|---|---|
| **division_rate** | 1.00 | 0.76 | 0.83 | 0.61 | 0.57 | 0.10 |
| **motility** | 0.76 | 1.00 | 0.82 | 0.56 | 0.59 | 0.20 |
| **surface_lambda** | 0.83 | 0.82 | 1.00 | 0.73 | 0.62 | 0.23 |
| **recalc_time** | 0.61 | 0.56 | 0.73 | 1.00 | 0.49 | 0.14 |
| **persistence** | 0.57 | 0.59 | 0.62 | 0.49 | 1.00 | 0.07 |
| **adhesion_cc** | 0.10 | 0.20 | 0.23 | 0.14 | 0.07 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/blocksize80/screening_report.json
