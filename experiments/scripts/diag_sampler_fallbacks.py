"""How often do the two sampler deviations actually fire?

Both leave signatures in a stored history:
  reject-resample fallback : tolerance is not None AND weight == 1.0 exactly
                             (a prior draw stamped with the active bandwidth)
  underflow-retry exhaustion: weight == 0.0
Same seed as posthoc_r.py, so the history is identical.
"""
import sys, numpy as np, random
sys.path.insert(0, "propulate")
from propulate.propagators.abcpmc import ABCPMC

LO, HI = -3.0, 3.0
LIMITS = {"x": (LO, HI)}
SIG, NOBS, MU_TRUE, SEED = 1.0, 25, 0.4, 20260730
N_SIMS, K = 12000, 100

rng_np = np.random.default_rng(SEED)
ybar = float(rng_np.normal(MU_TRUE, SIG / np.sqrt(NOBS)))
prop = ABCPMC(LIMITS, k=K, kernel="gaussian", scheduler_type="quantile",
              amis_snapshots=20, rng=random.Random(SEED))
hist = []
for i in range(N_SIMS):
    c = prop(hist)
    c.generation = i
    c.loss = abs(float(rng_np.normal(c.position[0], SIG / np.sqrt(NOBS))) - ybar)
    hist.append(c)

n = len(hist)
boot = sum(1 for i in hist if i.tolerance is None)
fallback = sum(1 for i in hist if i.tolerance is not None and i.weight == 1.0)
zero_w = sum(1 for i in hist if i.weight == 0.0)
print(f"n = {n}")
print(f"  bootstrap prior draws (tolerance is None)      : {boot:6d}  {boot/n:.3%}")
print(f"  reject-resample fallbacks (stamped, weight==1) : {fallback:6d}  {fallback/n:.3%}")
print(f"  underflow-retry exhaustions (weight==0)        : {zero_w:6d}  {zero_w/n:.3%}")
