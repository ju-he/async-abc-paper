# CPM screening -- blocksize 50, 47088 corpus rows, 5232 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 53, 10-90% 7-132; 30.8% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0607 | 0.1849 | **3.04** | 5.24 (scalars_only) | 4.8% |
| 8 | 1 | 4 | 0.0457 | 0.1742 | **3.81** | 14.75 (scalars_only) | 0.3% |
| 16 | 1 | 1 | 0.0701 | 0.1633 | **2.33** | 5.24 (scalars_only) | 3.5% |
| 16 | 1 | 4 | 0.0388 | 0.1420 | **3.66** | 14.75 (scalars_only) | 0.2% |
| 32 | 1 | 1 | 0.0768 | 0.1089 | **1.42** | 5.24 (scalars_only) | 3.4% |
| 32 | 1 | 4 | 0.0576 | 0.1177 | **2.04** | 14.75 (scalars_only) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 3.81

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0256 | 0.8209 | 32.12 | 1.000 |
| log_r95 | 1 | 0.0418 | 0.8162 | 19.55 | 1.099 |
| pair_correlation_gofr | 6 | 1.2433 | 1.8307 | 1.47 | 35.936 |
| dbscan_gaslike_fraction | 1 | 0.6096 | 0.7932 | 1.30 | 1.974 |
| radial_fa_equal_volume | 6 | 1.7677 | 1.3661 | 0.77 | 318.886 |
| radial_s2_equal_volume | 7 | 2.0622 | 1.5598 | 0.76 | 17.715 |
| radial_density_profile_equal_volume | 5 | 1.9412 | 1.0433 | 0.54 | 36.380 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0457 | 0.1742 | **3.81** |
| drop_dead | 7 | 0.0457 | 0.1742 | **3.81** |
| scalars_only | 2 | 0.0272 | 0.4017 | **14.75** |
| snr_pruned | 4 | 0.0425 | 0.2521 | **5.93** |

## Parameter screen under that protocol

Summary space: 27 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **13.97** | 54 | log_n 50%, radial_fa_equal_volume 26% | [0.00118, 0.01669] | 50% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate |
|---|---|
| **division_rate** | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_one_t501/screening_report.json
