# Concern-5 (T2.1) analysis

root: /home/juhe/async-abc-rerun-staging/concern5_20260718

## (A) AMIS vs no-AMIS under coupling — weighted posterior (async arm)

### MILD (base 0.02s, cap 0.5s)

AMIS (m=20) weighted:
  sigma  w_mean  w_mean_err   w_var  ci90_width  ess_frac   max_w  cover90  n
 0.0000 -0.0504      0.0091  0.0085      0.2918    0.9035  0.0292   1.0000  5
 0.5000 -0.0492      0.0102  0.0084      0.2990    0.8906  0.0313   1.0000  5
 1.0000 -0.0672      0.0245  0.0097      0.3197    0.8959  0.0283   1.0000  5
 2.0000 -0.0433      0.0086  0.0088      0.3145    0.8788  0.0322   1.0000  5

no-AMIS (m=0) weighted:
  sigma  w_mean  w_mean_err   w_var  ci90_width  ess_frac   max_w  cover90  n
 0.0000 -0.0532      0.0106  0.0058      0.2551    0.9955  0.0108   1.0000  5
 0.5000 -0.0608      0.0105  0.0067      0.2628    0.9958  0.0108   1.0000  5
 1.0000 -0.0505      0.0043  0.0068      0.2701    0.9973  0.0107   1.0000  5
 2.0000 -0.0433      0.0080  0.0075      0.2785    0.9948  0.0108   1.0000  5

AMIS: weighted vs unweighted mean-abs-error (does reweighting change the metric?):
  sigma  unw_mean_err  w_mean_err  weighted_minus_unweighted_err
 0.0000        0.0090      0.0091                         0.0001
 0.5000        0.0055      0.0102                         0.0047
 1.0000        0.0148      0.0245                         0.0097
 2.0000        0.0048      0.0086                         0.0038

### STEEP (base 0.5s, cap 8s)

AMIS (m=20) weighted:
  sigma  w_mean  w_mean_err   w_var  ci90_width  ess_frac   max_w  cover90  n
 0.0000 -0.0510      0.0117  0.0080      0.2906    0.9144  0.0267   1.0000  5
 0.5000 -0.0474      0.0070  0.0087      0.3038    0.9137  0.0268   1.0000  5
 1.0000 -0.0407      0.0137  0.0118      0.3640    0.8516  0.0409   1.0000  5
 2.0000 -0.0477      0.0078  0.0104      0.3479    0.8833  0.0315   1.0000  5

no-AMIS (m=0) weighted:
  sigma  w_mean  w_mean_err   w_var  ci90_width  ess_frac   max_w  cover90  n
 0.0000 -0.0463      0.0065  0.0067      0.2622    1.0000  0.0101   1.0000  5
 0.5000 -0.0469      0.0038  0.0066      0.2798    0.9999  0.0102   1.0000  5
 1.0000 -0.0530      0.0061  0.0056      0.2429    1.0000  0.0102   1.0000  5
 2.0000 -0.0544      0.0117  0.0080      0.3013    0.9999  0.0102   1.0000  5

AMIS: weighted vs unweighted mean-abs-error (does reweighting change the metric?):
  sigma  unw_mean_err  w_mean_err  weighted_minus_unweighted_err
 0.0000        0.0077      0.0117                         0.0040
 0.5000        0.0063      0.0070                         0.0007
 1.0000        0.0090      0.0137                         0.0046
 2.0000        0.0026      0.0078                         0.0052

## (B) Censored vs drained — deadline-censoring effect (async arm, AMIS)

### MILD: parameter_bias (censored) vs parameter_bias_drain (drained)

  sigma  w_mean_cens  w_mean_drain  mean_shift  w_var_cens  w_var_drain  var_ratio
 0.0000      -0.0504       -0.0562     -0.0058      0.0085       0.0083     0.9828
 0.5000      -0.0492       -0.0426      0.0066      0.0084       0.0085     1.0076
 1.0000      -0.0672       -0.0547      0.0125      0.0097       0.0089     0.9173
 2.0000      -0.0433       -0.0309      0.0124      0.0088       0.0091     1.0380

### STEEP: parameter_bias_strong (censored) vs parameter_bias_strong_drain (drained)

  sigma  w_mean_cens  w_mean_drain  mean_shift  w_var_cens  w_var_drain  var_ratio
 0.0000      -0.0510       -0.0494      0.0016      0.0080       0.0085     1.0731
 0.5000      -0.0474       -0.0514     -0.0040      0.0087       0.0092     1.0607
 1.0000      -0.0407       -0.0518     -0.0111      0.0118       0.0091     0.7658
 2.0000      -0.0477       -0.0564     -0.0087      0.0104       0.0094     0.9077
