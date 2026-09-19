# CPM screening -- blocksize 50, 71880 corpus rows, 4792 evaluations
snapshots [300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 35, 10-90% 7-132; 32.6% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.1080 | 0.3163 | **2.93** | 7.22 (scalars_only) | 4.6% |
| 8 | 1 | 4 | 0.0736 | 0.3264 | **4.43** | 21.94 (scalars_only) | 0.1% |
| 16 | 1 | 1 | 0.0972 | 0.2976 | **3.06** | 7.22 (scalars_only) | 5.7% |
| 16 | 1 | 4 | 0.0902 | 0.2974 | **3.30** | 21.94 (scalars_only) | 0.2% |
| 32 | 1 | 1 | 0.1009 | 0.2678 | **2.66** | 7.22 (scalars_only) | 5.2% |
| 32 | 1 | 4 | 0.1074 | 0.2498 | **2.33** | 21.94 (scalars_only) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 4.43

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0332 | 0.9680 | 29.15 | 1.150 |
| log_r95 | 1 | 0.0519 | 0.9491 | 18.29 | 1.325 |
| cell_shape_index | 2 | 0.1887 | 1.3021 | 6.90 | 3.980 |
| msd | 2 | 0.2791 | 1.3838 | 4.96 | 5.061 |
| aggregation_omega | 1 | 0.3091 | 0.7925 | 2.56 | 2.936 |
| outer_fraction | 1 | 0.2868 | 0.6397 | 2.23 | 3.757 |
| cell_volume_dispersion | 1 | 0.4807 | 0.6421 | 1.34 | 2.569 |
| pair_correlation_gofr | 6 | 1.4588 | 1.8825 | 1.29 | 39.956 |
| dbscan_gaslike_fraction | 1 | 0.6247 | 0.6967 | 1.12 | 2.298 |
| radial_variance_fraction | 1 | 0.8550 | 0.9201 | 1.08 | 3.194 |
| radial_density_profile_equal_volume | 6 | 1.6304 | 1.7372 | 1.07 | 35.054 |
| radial_s2_equal_volume | 6 | 1.7030 | 1.7348 | 1.02 | 14.611 |
| motility_order | 2 | 1.0293 | 0.7681 | 0.75 | 6.040 |
| radial_fa_equal_volume | 6 | 2.6770 | 0.0000 | 0.00 | 135.209 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 14 | 0.0736 | 0.3264 | **4.43** |
| drop_dead | 14 | 0.0736 | 0.3264 | **4.43** |
| scalars_only | 2 | 0.0201 | 0.4418 | **21.94** |
| snr_pruned | 12 | 0.0762 | 0.3139 | **4.12** |

## Parameter screen under that protocol

Summary space: 37 live coordinates (0 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.001, 0.2] log | **21.91** | 51 | log_n 40%, msd 31% | [0.001208, 0.02495] | 57% |
| cell_volume | [200, 1200] log | **14.04** | 44 | radial_density_profile_equal_volume 73%, cell_shape_index 11% | [213, 594] | 57% |
| surface_lambda | [0.25, 8] log | **6.14** | 18 | msd 41%, cell_shape_index 27% | [0.2829, 3.364] | 71% |
| recalc_time | [2, 60] log | **1.83** | 12 | msd 52%, radial_density_profile_equal_volume 15% | [12, 42] | 37% |
| persistence | [0.01, 0.99] lin | **2.04** | 7 | outer_fraction 38%, aggregation_omega 21% | [0.535, 0.955] | 43% |
| temperature | [15, 150] log | **0.00** | 4 | log_n 28%, msd 26% | [16, 61] | 58% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | cell_volume | surface_lambda | recalc_time | persistence | temperature |
|---|---|---|---|---|---|---|
| **division_rate** | 1.00 | 0.10 | 0.81 | 0.68 | 0.00 | 0.85 |
| **cell_volume** | 0.10 | 1.00 | 0.17 | 0.24 | 0.04 | 0.14 |
| **surface_lambda** | 0.81 | 0.17 | 1.00 | 0.65 | 0.04 | 0.64 |
| **recalc_time** | 0.68 | 0.24 | 0.65 | 1.00 | 0.29 | 0.65 |
| **persistence** | 0.00 | 0.04 | 0.04 | 0.29 | 1.00 | 0.30 |
| **temperature** | 0.85 | 0.14 | 0.64 | 0.65 | 0.30 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_custom/screening_report.json
