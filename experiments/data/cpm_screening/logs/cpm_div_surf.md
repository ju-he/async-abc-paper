# CPM screening -- blocksize 50, 41040 corpus rows, 4560 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 42, 10-90% 7-131; 31.5% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0867 | 0.2350 | **2.71** | 6.09 (scalars_only) | 4.0% |
| 8 | 1 | 4 | 0.0535 | 0.2146 | **4.01** | 13.16 (scalars_only) | 0.4% |
| 16 | 1 | 1 | 0.0783 | 0.2056 | **2.63** | 6.09 (scalars_only) | 4.2% |
| 16 | 1 | 4 | 0.0519 | 0.1688 | **3.25** | 13.16 (scalars_only) | 0.4% |
| 32 | 1 | 1 | 0.0803 | 0.1481 | **1.84** | 6.09 (scalars_only) | 4.4% |
| 32 | 1 | 4 | 0.0628 | 0.1431 | **2.28** | 13.16 (scalars_only) | 0.3% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 4.01

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0327 | 0.8983 | 27.49 | 1.168 |
| log_r95 | 1 | 0.0450 | 0.9212 | 20.49 | 1.312 |
| dbscan_gaslike_fraction | 1 | 0.5524 | 0.8342 | 1.51 | 2.085 |
| pair_correlation_gofr | 6 | 1.6977 | 1.6764 | 0.99 | 26.391 |
| radial_s2_equal_volume | 7 | 1.9733 | 1.7309 | 0.88 | 17.257 |
| radial_fa_equal_volume | 6 | 1.8523 | 1.2063 | 0.65 | 292.665 |
| radial_density_profile_equal_volume | 5 | 1.8320 | 1.1647 | 0.64 | 30.208 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0535 | 0.2146 | **4.01** |
| drop_dead | 7 | 0.0535 | 0.2146 | **4.01** |
| scalars_only | 2 | 0.0335 | 0.4410 | **13.16** |
| snr_pruned | 3 | 0.0223 | 0.2940 | **13.16** |

## Parameter screen under that protocol

Summary space: 27 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **16.90** | 55 | log_n 50%, log_r95 30% | [0.00118, 0.02324] | 56% |
| surface_lambda | [0.25, 8] log | **1.30** | 5 | pair_correlation_gofr 96%, log_n 1% | [0.6626, 5.781] | 62% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | surface_lambda |
|---|---|---|
| **division_rate** | 1.00 | 0.17 |
| **surface_lambda** | 0.17 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_div_surf/screening_report.json
