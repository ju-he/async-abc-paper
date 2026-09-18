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

## sbc_gandk_2x

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline     A           0.500               0.508  0.464  0.552           True
   abc_smc_baseline     A           0.800               0.808  0.773  0.843           True
   abc_smc_baseline     A           0.900               0.884  0.856  0.912           True
   abc_smc_baseline     A           0.950               0.942  0.922  0.962           True
   abc_smc_baseline     B           0.500               0.428  0.385  0.471          False
   abc_smc_baseline     B           0.800               0.722  0.683  0.761          False
   abc_smc_baseline     B           0.900               0.830  0.797  0.863          False
   abc_smc_baseline     B           0.950               0.892  0.865  0.919          False
   abc_smc_baseline     g           0.500               0.482  0.438  0.526           True
   abc_smc_baseline     g           0.800               0.768  0.731  0.805           True
   abc_smc_baseline     g           0.900               0.898  0.871  0.925           True
   abc_smc_baseline     g           0.950               0.946  0.926  0.966           True
   abc_smc_baseline     k           0.500               0.492  0.448  0.536           True
   abc_smc_baseline     k           0.800               0.748  0.710  0.786          False
   abc_smc_baseline     k           0.900               0.850  0.819  0.881          False
   abc_smc_baseline     k           0.950               0.902  0.876  0.928          False
async_propulate_abc     A           0.500               0.492  0.448  0.536           True
async_propulate_abc     A           0.800               0.774  0.737  0.811           True
async_propulate_abc     A           0.900               0.848  0.817  0.879          False
async_propulate_abc     A           0.950               0.890  0.863  0.917          False
async_propulate_abc     B           0.500               0.438  0.395  0.481          False
async_propulate_abc     B           0.800               0.712  0.672  0.752          False
async_propulate_abc     B           0.900               0.796  0.761  0.831          False
async_propulate_abc     B           0.950               0.834  0.801  0.867          False
async_propulate_abc     g           0.500               0.468  0.424  0.512           True
async_propulate_abc     g           0.800               0.752  0.714  0.790          False
async_propulate_abc     g           0.900               0.848  0.817  0.879          False
async_propulate_abc     g           0.950               0.886  0.858  0.914          False
async_propulate_abc     k           0.500               0.488  0.444  0.532           True
async_propulate_abc     k           0.800               0.738  0.699  0.777          False
async_propulate_abc     k           0.900               0.812  0.778  0.846          False
async_propulate_abc     k           0.950               0.860  0.830  0.890          False

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials      chi2  uniformity_p  uniform_ok
   abc_smc_baseline     A       500   29.4400        0.0594        True
   abc_smc_baseline     B       500   37.6000        0.0067       False
   abc_smc_baseline     g       500   12.5600        0.8603        True
   abc_smc_baseline     k       500   33.1200        0.0233       False
async_propulate_abc     A       500  281.4400        0.0000       False
async_propulate_abc     B       500  283.7600        0.0000       False
async_propulate_abc     g       500  357.5200        0.0000       False
async_propulate_abc     k       500  375.6000        0.0000       False

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
    A       500           0.8118        0.6667        0.0372     0.1697               0.0425
    B       500           0.8118        0.6667        0.0372     0.1697               0.0425
    g       500           0.8118        0.6667        0.0372     0.1697               0.0425
    k       500           0.8118        0.6667        0.0372     0.1697               0.0425

## sbc_gandk_fullhist

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline     A           0.500               0.542  0.498  0.586           True
   abc_smc_baseline     A           0.800               0.836  0.804  0.868          False
   abc_smc_baseline     A           0.900               0.926  0.903  0.949          False
   abc_smc_baseline     A           0.950               0.956  0.938  0.974           True
   abc_smc_baseline     B           0.500               0.452  0.408  0.496          False
   abc_smc_baseline     B           0.800               0.746  0.708  0.784          False
   abc_smc_baseline     B           0.900               0.826  0.793  0.859          False
   abc_smc_baseline     B           0.950               0.894  0.867  0.921          False
   abc_smc_baseline     g           0.500               0.492  0.448  0.536           True
   abc_smc_baseline     g           0.800               0.814  0.780  0.848           True
   abc_smc_baseline     g           0.900               0.904  0.878  0.930           True
   abc_smc_baseline     g           0.950               0.948  0.929  0.967           True
   abc_smc_baseline     k           0.500               0.526  0.482  0.570           True
   abc_smc_baseline     k           0.800               0.746  0.708  0.784          False
   abc_smc_baseline     k           0.900               0.858  0.827  0.889          False
   abc_smc_baseline     k           0.950               0.912  0.887  0.937          False
async_propulate_abc     A           0.500               0.654  0.612  0.696          False
async_propulate_abc     A           0.800               0.926  0.903  0.949          False
async_propulate_abc     A           0.900               0.968  0.953  0.983          False
async_propulate_abc     A           0.950               0.986  0.976  0.996          False
async_propulate_abc     B           0.500               0.532  0.488  0.576           True
async_propulate_abc     B           0.800               0.842  0.810  0.874          False
async_propulate_abc     B           0.900               0.926  0.903  0.949          False
async_propulate_abc     B           0.950               0.964  0.948  0.980           True
async_propulate_abc     g           0.500               0.558  0.514  0.602          False
async_propulate_abc     g           0.800               0.884  0.856  0.912          False
async_propulate_abc     g           0.900               0.956  0.938  0.974          False
async_propulate_abc     g           0.950               0.982  0.970  0.994          False
async_propulate_abc     k           0.500               0.578  0.535  0.621          False
async_propulate_abc     k           0.800               0.834  0.801  0.867          False
async_propulate_abc     k           0.900               0.914  0.889  0.939           True
async_propulate_abc     k           0.950               0.962  0.945  0.979           True

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials     chi2  uniformity_p  uniform_ok
   abc_smc_baseline     A       500  24.0000        0.1962        True
   abc_smc_baseline     B       500  50.7200        0.0001       False
   abc_smc_baseline     g       500  21.7600        0.2964        True
   abc_smc_baseline     k       500  29.0400        0.0654        True
async_propulate_abc     A       500  96.1600        0.0000       False
async_propulate_abc     B       500  27.7600        0.0882        True
async_propulate_abc     g       500  44.3200        0.0009       False
async_propulate_abc     k       500  29.1200        0.0641        True

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
    A       500           1.0000        1.0000        0.0005     0.0005               0.0100
    B       500           1.0000        1.0000        0.0005     0.0005               0.0100
    g       500           1.0000        1.0000        0.0005     0.0005               0.0100
    k       500           1.0000        1.0000        0.0005     0.0005               0.0100

## sbc_gandk_topm500

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline     A           0.500               0.566  0.523  0.609          False
   abc_smc_baseline     A           0.800               0.832  0.799  0.865           True
   abc_smc_baseline     A           0.900               0.922  0.898  0.946           True
   abc_smc_baseline     A           0.950               0.954  0.936  0.972           True
   abc_smc_baseline     B           0.500               0.472  0.428  0.516           True
   abc_smc_baseline     B           0.800               0.754  0.716  0.792          False
   abc_smc_baseline     B           0.900               0.852  0.821  0.883          False
   abc_smc_baseline     B           0.950               0.900  0.874  0.926          False
   abc_smc_baseline     g           0.500               0.484  0.440  0.528           True
   abc_smc_baseline     g           0.800               0.800  0.765  0.835           True
   abc_smc_baseline     g           0.900               0.900  0.874  0.926           True
   abc_smc_baseline     g           0.950               0.950  0.931  0.969           True
   abc_smc_baseline     k           0.500               0.498  0.454  0.542           True
   abc_smc_baseline     k           0.800               0.760  0.723  0.797          False
   abc_smc_baseline     k           0.900               0.860  0.830  0.890          False
   abc_smc_baseline     k           0.950               0.910  0.885  0.935          False
async_propulate_abc     A           0.500               0.574  0.531  0.617          False
async_propulate_abc     A           0.800               0.860  0.830  0.890          False
async_propulate_abc     A           0.900               0.932  0.910  0.954          False
async_propulate_abc     A           0.950               0.962  0.945  0.979           True
async_propulate_abc     B           0.500               0.506  0.462  0.550           True
async_propulate_abc     B           0.800               0.796  0.761  0.831           True
async_propulate_abc     B           0.900               0.894  0.867  0.921           True
async_propulate_abc     B           0.950               0.938  0.917  0.959           True
async_propulate_abc     g           0.500               0.512  0.468  0.556           True
async_propulate_abc     g           0.800               0.828  0.795  0.861           True
async_propulate_abc     g           0.900               0.934  0.912  0.956          False
async_propulate_abc     g           0.950               0.960  0.943  0.977           True
async_propulate_abc     k           0.500               0.548  0.504  0.592          False
async_propulate_abc     k           0.800               0.798  0.763  0.833           True
async_propulate_abc     k           0.900               0.884  0.856  0.912           True
async_propulate_abc     k           0.950               0.940  0.919  0.961           True

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials     chi2  uniformity_p  uniform_ok
   abc_smc_baseline     A       500  33.8400        0.0192       False
   abc_smc_baseline     B       500  46.2400        0.0005       False
   abc_smc_baseline     g       500  26.5600        0.1153        True
   abc_smc_baseline     k       500  25.6000        0.1417        True
async_propulate_abc     A       500  48.1600        0.0002       False
async_propulate_abc     B       500  17.9200        0.5278        True
async_propulate_abc     g       500  17.6000        0.5493        True
async_propulate_abc     k       500  32.8800        0.0248       False

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
    A       500           1.0000        1.0000        0.0005     0.0005               0.0100
    B       500           1.0000        1.0000        0.0005     0.0005               0.0100
    g       500           1.0000        1.0000        0.0005     0.0005               0.0100
    k       500           1.0000        1.0000        0.0005     0.0005               0.0100

## sbc_gandk_topm1500

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline     A           0.500               0.548  0.504  0.592          False
   abc_smc_baseline     A           0.800               0.834  0.801  0.867          False
   abc_smc_baseline     A           0.900               0.922  0.898  0.946           True
   abc_smc_baseline     A           0.950               0.954  0.936  0.972           True
   abc_smc_baseline     B           0.500               0.454  0.410  0.498          False
   abc_smc_baseline     B           0.800               0.744  0.706  0.782          False
   abc_smc_baseline     B           0.900               0.842  0.810  0.874          False
   abc_smc_baseline     B           0.950               0.896  0.869  0.923          False
   abc_smc_baseline     g           0.500               0.504  0.460  0.548           True
   abc_smc_baseline     g           0.800               0.804  0.769  0.839           True
   abc_smc_baseline     g           0.900               0.908  0.883  0.933           True
   abc_smc_baseline     g           0.950               0.948  0.929  0.967           True
   abc_smc_baseline     k           0.500               0.510  0.466  0.554           True
   abc_smc_baseline     k           0.800               0.748  0.710  0.786          False
   abc_smc_baseline     k           0.900               0.860  0.830  0.890          False
   abc_smc_baseline     k           0.950               0.912  0.887  0.937          False
async_propulate_abc     A           0.500               0.592  0.549  0.635          False
async_propulate_abc     A           0.800               0.890  0.863  0.917          False
async_propulate_abc     A           0.900               0.948  0.929  0.967          False
async_propulate_abc     A           0.950               0.976  0.963  0.989          False
async_propulate_abc     B           0.500               0.512  0.468  0.556           True
async_propulate_abc     B           0.800               0.818  0.784  0.852           True
async_propulate_abc     B           0.900               0.912  0.887  0.937           True
async_propulate_abc     B           0.950               0.954  0.936  0.972           True
async_propulate_abc     g           0.500               0.540  0.496  0.584           True
async_propulate_abc     g           0.800               0.858  0.827  0.889          False
async_propulate_abc     g           0.900               0.944  0.924  0.964          False
async_propulate_abc     g           0.950               0.966  0.950  0.982          False
async_propulate_abc     k           0.500               0.556  0.512  0.600          False
async_propulate_abc     k           0.800               0.806  0.771  0.841           True
async_propulate_abc     k           0.900               0.908  0.883  0.933           True
async_propulate_abc     k           0.950               0.952  0.933  0.971           True

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials     chi2  uniformity_p  uniform_ok
   abc_smc_baseline     A       500  34.6400        0.0154       False
   abc_smc_baseline     B       500  39.1200        0.0043       False
   abc_smc_baseline     g       500  15.8400        0.6679        True
   abc_smc_baseline     k       500  28.0800        0.0819        True
async_propulate_abc     A       500  52.6400        0.0001       False
async_propulate_abc     B       500  16.4800        0.6251        True
async_propulate_abc     g       500  30.2400        0.0488       False
async_propulate_abc     k       500  26.4800        0.1174        True

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
    A       500           1.0000        1.0000        0.0005     0.0005               0.0100
    B       500           1.0000        1.0000        0.0005     0.0005               0.0100
    g       500           1.0000        1.0000        0.0005     0.0005               0.0100
    k       500           1.0000        1.0000        0.0005     0.0005               0.0100

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

## sbc_bimodal_fullhist

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline theta           0.500               0.446  0.402  0.490          False
   abc_smc_baseline theta           0.800               0.808  0.773  0.843           True
   abc_smc_baseline theta           0.900               0.902  0.876  0.928           True
   abc_smc_baseline theta           0.950               0.942  0.922  0.962           True
async_propulate_abc theta           0.500               0.474  0.430  0.518           True
async_propulate_abc theta           0.800               0.826  0.793  0.859           True
async_propulate_abc theta           0.900               0.912  0.887  0.937           True
async_propulate_abc theta           0.950               0.960  0.943  0.977           True

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials     chi2  uniformity_p  uniform_ok
   abc_smc_baseline theta       500  39.0400        0.0044       False
async_propulate_abc theta       500  25.6000        0.1417        True

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
theta       500           1.0000        1.0000        0.0005     0.0005               0.0100

### Bimodal mode coverage (async top-k archive)

trials=500  both-modes-kept=0.858  true-value-mode-kept=0.860

## sbc_gaussian_archive

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline    mu           0.500               0.398  0.355  0.441          False
   abc_smc_baseline    mu           0.800               0.638  0.596  0.680          False
   abc_smc_baseline    mu           0.900               0.750  0.712  0.788          False
   abc_smc_baseline    mu           0.950               0.828  0.795  0.861          False
async_propulate_abc    mu           0.500               0.528  0.484  0.572           True
async_propulate_abc    mu           0.800               0.794  0.759  0.829           True
async_propulate_abc    mu           0.900               0.894  0.867  0.921           True
async_propulate_abc    mu           0.950               0.924  0.901  0.947          False

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials      chi2  uniformity_p  uniform_ok
   abc_smc_baseline    mu       500  124.7200        0.0000       False
async_propulate_abc    mu       500   13.2000        0.8282        True

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
   mu       500           0.8985        0.7659        0.0291     0.0514               0.0291

## sbc_gaussian_fullhist

### Coverage with binomial 95% CI (nominal_in_ci = calibrated at that level)

             method param  coverage_level  empirical_coverage  ci_lo  ci_hi  nominal_in_ci
   abc_smc_baseline    mu           0.500               0.398  0.355  0.441          False
   abc_smc_baseline    mu           0.800               0.696  0.656  0.736          False
   abc_smc_baseline    mu           0.900               0.786  0.750  0.822          False
   abc_smc_baseline    mu           0.950               0.860  0.830  0.890          False
async_propulate_abc    mu           0.500               0.522  0.478  0.566           True
async_propulate_abc    mu           0.800               0.814  0.780  0.848           True
async_propulate_abc    mu           0.900               0.908  0.883  0.933           True
async_propulate_abc    mu           0.950               0.950  0.931  0.969           True

### Rank uniformity (chi-square, 20 bins)

             method param  n_trials     chi2  uniformity_p  uniform_ok
   abc_smc_baseline    mu       500  68.4000        0.0000       False
async_propulate_abc    mu       500  11.8400        0.8923        True

### Async importance-weight health (per-trial, aggregated)

param  n_trials  ess_frac_median  ess_frac_p05  max_w_median  max_w_p95  top1pct_mass_median
   mu       500           1.0000        1.0000        0.0005     0.0005               0.0100
