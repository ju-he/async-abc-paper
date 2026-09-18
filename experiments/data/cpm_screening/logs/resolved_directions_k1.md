# Resolved directions -- campaign_features, 400 thetas, 1600 evaluations at 1 seed(s) each, 9 blocks, 6 parameters

| block weighting | r=0 | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | resolved |
|---|---|---|---|---|---|---|---|---|
| equal (9 blocks, shipped) | -0.002 | 0.114 | 0.117 | 0.116 | 0.115 | 0.114 | 0.114 | **1** |
| size scalars only | -0.008 | 0.663 | 0.685 | 0.684 | 0.682 | 0.683 | 0.685 | **2** |
| scalars + campaign features at 25% | -0.006 | 0.658 | 0.677 | 0.673 | 0.677 | 0.674 | 0.674 | **2** |
| scalars + campaign features at 50% | -0.004 | 0.544 | 0.555 | 0.554 | 0.551 | 0.550 | 0.549 | **2** |
| size blocks removed | -0.002 | 0.027 | 0.035 | 0.033 | 0.032 | 0.034 | 0.034 | **2** |

Best: **size scalars only** -- 2 direction(s), held-out R2 0.685
Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/campaign_features/resolved_directions_k1.json
