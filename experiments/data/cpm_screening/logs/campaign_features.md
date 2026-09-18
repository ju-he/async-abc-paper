# CPM screening -- blocksize 50, 43056 corpus rows, 2392 evaluations
snapshots [250, 300, 350, 400, 450, 500]   bins [8, 16, 32]   replicate seeds up to 48
cells at t=500: median 61, 10-90% 12-161; 19.4% of evaluations below 20 cells, where the curve summaries stop meaning anything

## Protocol screen -- signal-to-noise of the reported discrepancy

| bins | snapshots | seeds/eval | within-theta | between-theta | SNR (equal weights) | SNR (best weighting) | outliers |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 0.0923 | 0.2155 | **2.33** | 5.59 (scalars_only) | 3.5% |
| 8 | 1 | 4 | 0.0586 | 0.2207 | **3.77** | 14.56 (scalars_only) | 0.2% |
| 16 | 1 | 1 | 0.0866 | 0.2217 | **2.56** | 5.59 (scalars_only) | 2.8% |
| 16 | 1 | 4 | 0.0482 | 0.2237 | **4.64** | 14.56 (scalars_only) | 0.7% |
| 32 | 1 | 1 | 0.0851 | 0.1921 | **2.26** | 5.59 (scalars_only) | 3.0% |
| 32 | 1 | 4 | 0.0670 | 0.1849 | **2.76** | 14.56 (scalars_only) | 0.2% |

Best protocol on the shipped (equal-weight) metric: **b16_s1_k4** -- 16 bins, 1 snapshot(s) at [500], 4 seed(s) per evaluation; SNR 4.64

## Per-block behaviour under that protocol

| block | dims | within-theta | between-theta | SNR | block norm |
|---|---|---|---|---|---|
| log_n | 1 | 0.0291 | 0.9761 | 33.54 | 1.267 |
| log_r95 | 1 | 0.0644 | 1.0020 | 15.57 | 1.334 |
| invasion_ratio | 1 | 0.3604 | 0.8404 | 2.33 | 3.124 |
| surface_roughness | 1 | 0.5125 | 0.8833 | 1.72 | 6.456 |
| radial_fa_equal_volume | 10 | 1.5549 | 2.3837 | 1.53 | 77.238 |
| radial_s2_equal_volume | 10 | 2.1745 | 2.3042 | 1.06 | 22.990 |
| radial_density_profile_equal_volume | 10 | 2.1420 | 2.2445 | 1.05 | 40.668 |
| pair_correlation_gofr | 10 | 2.2272 | 2.1625 | 0.97 | 56.325 |
| dbscan_gaslike_fraction  (dead) | 1 | 0.0000 | 1.0013 | n/a | 1.827 |

## Block weighting

| weighting | blocks kept | within-theta | between-theta | SNR |
|---|---|---|---|---|
| equal | 9 | 0.0482 | 0.2237 | **4.64** |
| drop_dead | 8 | 0.0542 | 0.2517 | **4.64** |
| scalars_only | 2 | 0.0302 | 0.4398 | **14.56** |
| snr_pruned | 7 | 0.0406 | 0.2289 | **5.64** |

## Parameter screen under that protocol

Summary space: 44 live coordinates (1 dead). Distances are in units of the Monte Carlo noise along each parameter's own response direction.

| parameter | prior | identifiability | sigmas moved | carried by | responsive window | share of prior |
|---|---|---|---|---|---|---|
| division_rate | [0.002, 0.2] log | **16.23** | 58 | log_n 55%, radial_fa_equal_volume 28% | [0.002358, 0.02358] | 50% |
| motility | [100, 4000] log | **2.06** | 21 | log_n 66%, log_r95 9% | [939, 3506] | 36% |
| surface_lambda | [0.25, 8] log | **7.57** | 17 | log_n 65%, pair_correlation_gofr 14% | [0.3624, 4.308] | 71% |
| persistence | [0, 0.99] lin | **0.00** | 6 | log_n 32%, radial_fa_equal_volume 31% | [0.1768, 0.3182] | 14% |
| recalc_time | [2, 60] log | **1.20** | 6 | pair_correlation_gofr 89%, radial_fa_equal_volume 6% | [8, 53] | 56% |
| adhesion_cc | [40, 200] lin | **0.00** | 4 | pair_correlation_gofr 88%, radial_s2_equal_volume 5% | [137, 194] | 36% |

## Confounding -- |cos| between response directions

A pair near 1 moves the summary the same way and cannot be separated, however identifiable each is on its own.

| | division_rate | motility | surface_lambda | persistence | recalc_time | adhesion_cc |
|---|---|---|---|---|---|---|
| **division_rate** | 1.00 | 0.75 | 0.85 | 0.70 | 0.15 | 0.14 |
| **motility** | 0.75 | 1.00 | 0.87 | 0.42 | 0.21 | 0.11 |
| **surface_lambda** | 0.85 | 0.87 | 1.00 | 0.56 | 0.18 | 0.25 |
| **persistence** | 0.70 | 0.42 | 0.56 | 1.00 | 0.04 | 0.14 |
| **recalc_time** | 0.15 | 0.21 | 0.18 | 0.04 | 1.00 | 0.45 |
| **adhesion_cc** | 0.14 | 0.11 | 0.25 | 0.14 | 0.45 | 1.00 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/campaign_features/screening_report_campaign.json
