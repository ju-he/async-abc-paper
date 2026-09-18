# Resolved directions -- campaign_features, 400 thetas, 400 evaluations at 4 seed(s) each, 9 blocks, 6 parameters

| block weighting | r=0 | r=1 | r=2 | r=3 | r=4 | r=5 | r=6 | resolved |
|---|---|---|---|---|---|---|---|---|
| equal (9 blocks, shipped) | -0.008 | 0.204 | 0.205 | 0.199 | 0.196 | 0.193 | 0.191 | **1** |
| size scalars only | -0.009 | 0.799 | 0.825 | 0.822 | 0.822 | 0.823 | 0.826 | **2** |
| scalars + campaign features at 25% | -0.006 | 0.758 | 0.773 | 0.771 | 0.768 | 0.767 | 0.764 | **2** |
| scalars + campaign features at 50% | -0.004 | 0.608 | 0.608 | 0.609 | 0.603 | 0.599 | 0.596 | **1** |
| size blocks removed | -0.010 | 0.068 | 0.080 | 0.073 | 0.073 | 0.069 | 0.066 | **2** |

Best: **size scalars only** -- 2 direction(s), held-out R2 0.826
Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/campaign_features/resolved_directions_k4.json
