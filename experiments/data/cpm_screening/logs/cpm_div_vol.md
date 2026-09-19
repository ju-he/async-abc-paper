# CPM screening -- blocksize 50, 80400 corpus rows, 5360 evaluations
snapshots [300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 47, 10-90% 7-159; 30.4% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.1282 | 0.2239 | **1.75** | 7.32 (scalars_only) | 3.4% |
| 8 | 1 | 4 | 0.0916 | 0.2238 | **2.44** | 17.28 (scalars_only) | 0.1% |
| 16 | 1 | 1 | 0.1197 | 0.1929 | **1.61** | 7.32 (scalars_only) | 3.6% |
| 16 | 1 | 4 | 0.0953 | 0.1966 | **2.06** | 17.28 (scalars_only) | 0.1% |
| 32 | 1 | 1 | 0.1161 | 0.1730 | **1.49** | 7.32 (scalars_only) | 3.8% |
| 32 | 1 | 4 | 0.0617 | 0.1870 | **3.03** | 17.28 (scalars_only) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b32_s1_k4** -- 32 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 3.03

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0355 | 0.8938 | 25.16 | 1.164 |
| log_r95 | 1 | 0.0432 | 0.8245 | 19.10 | 1.190 |
| msd | 2 | 0.2884 | 1.3394 | 4.64 | 2.578 |
| non_gaussian_parameter | 3 | 0.8241 | 1.5244 | 1.85 | 8.972 |
| dbscan_gaslike_fraction | 1 | 0.5960 | 0.8035 | 1.35 | 2.247 |
| radial_density_profile_equal_volume | 10 | 2.0833 | 2.3255 | 1.12 | 99.947 |
| radial_fa_equal_volume | 10 | 2.1866 | 2.1236 | 0.97 | 25.775 |
| pair_correlation_gofr | 10 | 2.3198 | 2.1238 | 0.92 | 87.876 |
| radial_s2_equal_volume | 10 | 2.3454 | 2.1333 | 0.91 | 23.219 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 9 | 0.0617 | 0.1870 | **3.03** |
| drop_dead | 9 | 0.0617 | 0.1870 | **3.03** |
| scalars_only | 2 | 0.0244 | 0.4210 | **17.28** |
| snr_pruned | 6 | 0.0571 | 0.3216 | **5.63** |

## Parameter screen under that protocol

Summary space: 48 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| cell_volume | [200, 1200] log | **23.49** | 65 | radial_density_profile_equal_volume 85%, pair_correlation_gofr 5% | [212, 648] | 62% |
| division_rate | [0.001, 0.2] log | **25.54** | 62 | log_n 38%, log_r95 27% | [0.002288, 0.03236] | 50% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | cell_volume | division_rate |
|---|---|---|
| **cell_volume** | 1.00 | 0.06 |
| **division_rate** | 0.06 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_div_vol/screening_report.json
