# CPM screening -- blocksize 50, 48240 corpus rows, 5360 evaluations
snapshots [600, 800, 1000]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=1000: median 269, 10-90% 16-296; 12.7% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0461 | 0.0910 | **1.97** | 3.76 (scalars_only) | 8.5% |
| 8 | 1 | 4 | 0.0245 | 0.1118 | **4.55** | 7.91 (scalars_only) | 0.5% |
| 16 | 1 | 1 | 0.0287 | 0.0682 | **2.38** | 3.76 (scalars_only) | 10.2% |
| 16 | 1 | 4 | 0.0242 | 0.0691 | **2.86** | 7.91 (scalars_only) | 0.7% |
| 32 | 1 | 1 | 0.0359 | 0.0764 | **2.13** | 3.76 (scalars_only) | 6.8% |
| 32 | 1 | 4 | 0.0310 | 0.0994 | **3.21** | 7.91 (scalars_only) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [1000], 4 seed(s) per evaluation; SNR 4.55

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0044 | 0.1146 | 26.24 | 1.928 |
| log_r95 | 1 | 0.0253 | 0.2884 | 11.42 | 1.557 |
| pair_correlation_gofr | 4 | 0.9031 | 1.6032 | 1.78 | 9.152 |
| radial_s2_equal_volume | 6 | 1.7931 | 1.5737 | 0.88 | 28.381 |
| radial_fa_equal_volume | 7 | 1.8571 | 1.5244 | 0.82 | 316.645 |
| radial_density_profile_equal_volume  (dead) | 2 | 0.0000 | 0.0000 | n/a | 1.000 |
| dbscan_gaslike_fraction  (dead) | 1 | 0.0000 | 1.0004 | n/a | 2.673 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0245 | 0.1118 | **4.55** |
| drop_dead | 5 | 0.0344 | 0.1565 | **4.55** |
| scalars_only | 2 | 0.0149 | 0.1178 | **7.91** |
| snr_pruned | 3 | 0.0528 | 0.2214 | **4.20** |

## Parameter screen under that protocol

Summary space: 19 live coordinates (3 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **34.30** | 567 | log_n 98%, log_r95 2% | [0.00118, 0.008606] | 38% |
| motility | [100, 4000] log | **31.93** | 121 | log_n 98%, log_r95 2% | [355, 2831] | 56% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | motility |
|---|---|---|
| **division_rate** | 1.00 | 1.00 |
| **motility** | 1.00 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_two_t1001/screening_report.json
