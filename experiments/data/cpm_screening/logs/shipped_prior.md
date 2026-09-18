# CPM screening -- blocksize 50, 37008 corpus rows, 2056 evaluations
snapshots [250, 300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 58, 10-90% 29-159; 3.7% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.1048 | 0.2307 | **2.20** | 3.47 (scalars_only) | 2.6% |
| 8 | 1 | 2 | 0.0929 | 0.2169 | **2.33** | 6.09 (scalars_only) | 1.5% |
| 8 | 1 | 4 | 0.0632 | 0.1768 | **2.80** | 11.32 (scalars_only) | 1.3% |
| 8 | 3 | 1 | 0.1143 | 0.2695 | **2.36** | 3.71 (scalars_only) | 1.8% |
| 8 | 3 | 2 | 0.0927 | 0.2130 | **2.30** | 4.90 (scalars_only) | 1.2% |
| 8 | 3 | 4 | 0.0834 | 0.1527 | **1.83** | 8.76 (snr_pruned) | 0.7% |
| 8 | 6 | 1 | 0.1201 | 0.2196 | **1.83** | 3.14 (scalars_only) | 1.6% |
| 8 | 6 | 2 | 0.0763 | 0.2185 | **2.87** | 3.92 (scalars_only) | 1.8% |
| 8 | 6 | 4 | 0.0647 | 0.1834 | **2.84** | 7.49 (scalars_only) | 0.9% |
| 16 | 1 | 1 | 0.1041 | 0.2445 | **2.35** | 3.47 (scalars_only) | 1.6% |
| 16 | 1 | 2 | 0.0922 | 0.2298 | **2.49** | 6.09 (scalars_only) | 1.6% |
| 16 | 1 | 4 | 0.0692 | 0.1790 | **2.59** | 11.32 (scalars_only) | 0.7% |
| 16 | 3 | 1 | 0.0980 | 0.2356 | **2.40** | 3.71 (scalars_only) | 1.6% |
| 16 | 3 | 2 | 0.1032 | 0.2188 | **2.12** | 4.90 (scalars_only) | 1.3% |
| 16 | 3 | 4 | 0.0798 | 0.1589 | **1.99** | 8.76 (scalars_only) | 0.2% |
| 16 | 6 | 1 | 0.1183 | 0.1859 | **1.57** | 3.14 (scalars_only) | 1.5% |
| 16 | 6 | 2 | 0.0984 | 0.2054 | **2.09** | 3.92 (scalars_only) | 1.5% |
| 16 | 6 | 4 | 0.0719 | 0.1507 | **2.10** | 7.49 (scalars_only) | 0.2% |
| 32 | 1 | 1 | 0.0821 | 0.2184 | **2.66** | 3.47 (scalars_only) | 1.8% |
| 32 | 1 | 2 | 0.0728 | 0.2132 | **2.93** | 6.09 (scalars_only) | 1.1% |
| 32 | 1 | 4 | 0.0688 | 0.1838 | **2.67** | 11.32 (scalars_only) | 0.4% |
| 32 | 3 | 1 | 0.0873 | 0.1879 | **2.15** | 3.71 (scalars_only) | 1.8% |
| 32 | 3 | 2 | 0.0760 | 0.1868 | **2.46** | 4.90 (scalars_only) | 2.2% |
| 32 | 3 | 4 | 0.0897 | 0.1690 | **1.88** | 8.76 (scalars_only) | 0.4% |
| 32 | 6 | 1 | 0.0979 | 0.1737 | **1.77** | 3.14 (scalars_only) | 1.9% |
| 32 | 6 | 2 | 0.0965 | 0.1848 | **1.91** | 3.92 (scalars_only) | 2.0% |
| 32 | 6 | 4 | 0.0855 | 0.1616 | **1.89** | 7.49 (scalars_only) | 0.4% |

Best protocol on the shipped (equal-weight) metric: **b32_s1_k2** -- 32 bins, 1 snapshot(s) at [500], 2 seed(s) per evaluation; SNR 2.93

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0896 | 0.9438 | 10.54 | 1.431 |
| log_r95 | 1 | 0.1633 | 0.8933 | 5.47 | 1.477 |
| pair_correlation_gofr | 10 | 2.4880 | 1.6247 | 0.65 | 35.863 |
| radial_s2_equal_volume | 10 | 2.4789 | 1.4661 | 0.59 | 20.631 |
| radial_fa_equal_volume | 10 | 2.7650 | 1.4004 | 0.51 | 31.889 |
| radial_density_profile_equal_volume | 10 | 3.1304 | 0.9038 | 0.29 | 35.514 |
| dbscan_gaslike_fraction  (dead) | 1 | 0.0000 | 0.9498 | n/a | 1.491 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.0728 | 0.2132 | **2.93** |
| drop_dead | 6 | 0.0850 | 0.2487 | **2.93** |
| scalars_only | 2 | 0.0700 | 0.4261 | **6.09** |
| snr_pruned | 2 | 0.0700 | 0.4261 | **6.09** |

## Parameter screen under that protocol

Summary space: 42 live coordinates (1 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| motility | [0, 10000] lin | **10.40** | 29 | log_n 60%, log_r95 17% | [357, 6786] | 64% |
| division_rate | [6e-05, 0.6] lin | **0.54** | 12 | log_n 62%, log_r95 24% | [0.02149, 0.15] | 21% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | motility | division_rate |
|---|---|---|
| **motility** | 1.00 | 0.92 |
| **division_rate** | 0.92 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/shipped_prior/screening_report.json
