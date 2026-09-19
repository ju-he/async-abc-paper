# CPM screening -- blocksize 50, 41040 corpus rows, 4560 evaluations
snapshots [300, 400, 500]   bins [8, 16, 32]   replicate seeds up to 64
cells at t=500: median 32, 10-90% 18-52; 13.3% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.1659 | 0.1180 | **0.71** | 0.85 (scalars_only) | 0.8% |
| 8 | 1 | 4 | 0.1063 | 0.1186 | **1.12** | 1.43 (scalars_only) | 0.0% |
| 16 | 1 | 1 | 0.1629 | 0.0897 | **0.55** | 0.85 (scalars_only) | 0.5% |
| 16 | 1 | 4 | 0.1260 | 0.0841 | **0.67** | 1.43 (scalars_only) | 0.0% |
| 32 | 1 | 1 | 0.1656 | 0.0757 | **0.46** | 0.85 (scalars_only) | 0.4% |
| 32 | 1 | 4 | 0.1064 | 0.1172 | **1.10** | 1.43 (scalars_only) | 0.0% |

Best protocol on the shipped (equal-weight) metric: **b8_s1_k4** -- 8 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 1.12

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.2229 | 0.9217 | 4.14 | 1.607 |
| log_r95 | 1 | 0.2280 | 0.8920 | 3.91 | 1.573 |
| pair_correlation_gofr | 7 | 1.8831 | 1.8132 | 0.96 | 21.991 |
| radial_s2_equal_volume | 7 | 1.9411 | 1.8033 | 0.93 | 14.182 |
| radial_fa_equal_volume | 7 | 2.0344 | 1.3166 | 0.65 | 61.986 |
| radial_density_profile_equal_volume  (dead) | 2 | 0.0000 | 0.0000 | n/a | 1.000 |
| dbscan_gaslike_fraction  (dead) | 1 | 0.0000 | 0.0000 | n/a | 1.000 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 7 | 0.1063 | 0.1186 | **1.12** |
| drop_dead | 5 | 0.1488 | 0.1660 | **1.12** |
| scalars_only | 2 | 0.2195 | 0.3141 | **1.43** |
| snr_pruned | 2 | 0.2195 | 0.3141 | **1.43** |

## Parameter screen under that protocol

Summary space: 23 live coordinates (3 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| motility | [300, 6000] log | **3.41** | 11 | log_n 38%, log_r95 34% | [1222, 5464] | 50% |
| adhesion_cl | [20, 200] lin | **1.32** | 3 | radial_fa_equal_volume 68%, pair_correlation_gofr 18% | [37, 71] | 19% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | motility | adhesion_cl |
|---|---|---|
| **motility** | 1.00 | 0.10 |
| **adhesion_cl** | 0.10 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_mot_adh/screening_report.json
