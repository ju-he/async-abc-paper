# SBC extra diagnostics (concern 6): multidim + multimodal

## sbc_gandk

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline     A           0.500               0.540  0.509  0.571          False
   abc_smc_baseline     A           0.800               0.834  0.811  0.857          False
   abc_smc_baseline     A           0.900               0.917  0.900  0.934           True
   abc_smc_baseline     A           0.950               0.949  0.935  0.963           True
   abc_smc_baseline     B           0.500               0.447  0.416  0.478          False
   abc_smc_baseline     B           0.800               0.741  0.714  0.768          False
   abc_smc_baseline     B           0.900               0.850  0.828  0.872          False
   abc_smc_baseline     B           0.950               0.905  0.887  0.923          False
   abc_smc_baseline     g           0.500               0.508  0.477  0.539           True
   abc_smc_baseline     g           0.800               0.806  0.781  0.831           True
   abc_smc_baseline     g           0.900               0.901  0.882  0.920           True
   abc_smc_baseline     g           0.950               0.952  0.939  0.965           True
   abc_smc_baseline     k           0.500               0.461  0.430  0.492          False
   abc_smc_baseline     k           0.800               0.751  0.724  0.778          False
   abc_smc_baseline     k           0.900               0.858  0.836  0.880          False
   abc_smc_baseline     k           0.950               0.915  0.898  0.932          False
async_propulate_abc     A           0.500               0.498  0.467  0.529           True
async_propulate_abc     A           0.800               0.768  0.742  0.794          False
async_propulate_abc     A           0.900               0.847  0.825  0.869          False
async_propulate_abc     A           0.950               0.890  0.871  0.909          False
async_propulate_abc     B           0.500               0.456  0.425  0.487          False
async_propulate_abc     B           0.800               0.718  0.690  0.746          False
async_propulate_abc     B           0.900               0.811  0.787  0.835          False
async_propulate_abc     B           0.950               0.862  0.841  0.883          False
async_propulate_abc     g           0.500               0.512  0.481  0.543           True
async_propulate_abc     g           0.800               0.763  0.737  0.789          False
async_propulate_abc     g           0.900               0.862  0.841  0.883          False
async_propulate_abc     g           0.950               0.899  0.880  0.918          False
async_propulate_abc     k           0.500               0.465  0.434  0.496          False
async_propulate_abc     k           0.800               0.746  0.719  0.773          False
async_propulate_abc     k           0.900               0.831  0.808  0.854          False
async_propulate_abc     k           0.950               0.877  0.857  0.897          False

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials       chi2  uniformity_p  uniform_ok
   abc_smc_baseline     A      1000    38.1600        0.0057       False
   abc_smc_baseline     B      1000    39.4800        0.0038       False
   abc_smc_baseline     g      1000    29.9200        0.0528        True
   abc_smc_baseline     k      1000    30.1200        0.0503        True
async_propulate_abc     A      1000   750.4400        0.0000       False
async_propulate_abc     B      1000   867.2800        0.0000       False
async_propulate_abc     g      1000   935.2400        0.0000       False
async_propulate_abc     k      1000  1039.7600        0.0000       False

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
    A      1000           0.8126        0.6841        0.0327     0.1555               0.0412
    B      1000           0.8126        0.6841        0.0327     0.1555               0.0412
    g      1000           0.8126        0.6841        0.0327     0.1555               0.0412
    k      1000           0.8126        0.6841        0.0327     0.1555               0.0412

## sbc_bimodal

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline theta           0.500               0.451  0.420  0.482          False
   abc_smc_baseline theta           0.800               0.812  0.788  0.836           True
   abc_smc_baseline theta           0.900               0.896  0.877  0.915           True
   abc_smc_baseline theta           0.950               0.949  0.935  0.963           True
async_propulate_abc theta           0.500               0.478  0.447  0.509           True
async_propulate_abc theta           0.800               0.791  0.766  0.816           True
async_propulate_abc theta           0.900               0.879  0.859  0.899          False
async_propulate_abc theta           0.950               0.928  0.912  0.944          False

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials     chi2  uniformity_p  uniform_ok
   abc_smc_baseline theta      1000  51.2400        0.0001       False
async_propulate_abc theta      1000  33.4000        0.0216       False

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
theta      1000           0.9934        0.9408        0.0114     0.0217               0.0114

### Bimodal mode coverage (async top-k archive)

trials=1000  both-modes-kept=0.857  true-value-mode-kept=0.867
