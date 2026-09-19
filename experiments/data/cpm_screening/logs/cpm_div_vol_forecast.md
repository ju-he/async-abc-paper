# Posterior forecast -- cpm_div_vol
weighting 'scalars' over 2 of 7 blocks, 4 seed(s) per evaluation, snapshot t=500, 1200 prior draws, reference = single (x16 realisations)

truth (prior-normalised): division_rate 0.415 = 0.009, cell_volume 0.511 = 500

| accept | ESS | tolerance | division_rate bias | division_rate contraction | division_rate 90% | cell_volume bias | cell_volume contraction | cell_volume 90% | corr |
|---|---|---|---|---|---|---|---|---|---|
| 20.0% | 240 | 0.1024 | +0.019 ±0.023 | 77% | 100% | +0.030 ±0.013 | 4% | 100% | +0.22 |
| 10.0% | 120 | 0.0380 | +0.013 ±0.023 | 85% | 94% | +0.027 ±0.017 | 16% | 100% | +0.13 |
| 5.0% | 60 | 0.0173 | +0.014 ±0.023 | 88% | 94% | +0.034 ±0.031 | 38% | 100% | +0.09 |
| 2.0% | 24 | 0.0060 | +0.016 ±0.023 | 91% | 94% | +0.027 ±0.041 | 64% | 100% | +0.04 |
| 1.0% | 12 | 0.0028 | +0.019 ±0.024 | 91% | 94% | +0.025 ±0.040 | 78% | 94% | -0.26 |

Wrote /home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/cpm_screening/cpm_div_vol/posterior_forecast_scalars_k4_single.json
