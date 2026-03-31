Root Cause: Implementation bug — O(N²) overhead in ABCPMC.__call__                                                                                                                                                                          
                                                                                                                                                                                                                                              
  This is not a measurement/analysis issue. It is a real performance defect in the async ABC implementation that masquerades as algorithm behaviour in the scaling plots.                                                                     
                                                                                                                                                                                                                                              
  ---                                                                                                                                                                                                                                         
  What the bug is                                                                                                                                                                                                                             
                                                                                                                                                                                                                                              
  Every call to ABCPMC.__call__ in propulate/propagators/abcpmc.py receives the full growing history of all evaluated individuals (inds) and iterates over it at least 3–4 times:                                                             
                                                                                                                                                                                                                                              
  # All O(N) in len(inds):                                                                                                                                                                                                                    
  tol_from_history = min(ind.tolerance for ind in inds ...)   # O(N)                                                                                                                                                                          
  preliminary_archive = self.select_archive(inds, tol)         # O(N) + O(N log N) sort                                                                                                                                                       
  proposed_tol = self.tolerance_scheduler.compute(inds, tol)   # O(N) — all three schedulers                                                                                                                                                  
  # possibly a second select_archive call:                                                                                                                                                                                                    
  self.select_archive(inds, candidate_tol)                     # O(N)                                                                                                                                                                         
                                                                                                                                                                                                                                              
  Since Propulate calls this propagator once per evaluation, and the history grows by 1 each time, total cost is:                                                                                                                             
                                                                                                                                                                                                                                              
  $$T_\text{overhead} = \sum_{n=1}^{N} O(n) = O(N^2)$$                                                                                                                                                                                        
                                                                                                                                                                                                                                              
  This is confirmed directly from the data: fitting a linear model to gap-vs-history-size gives a = 7 µs per evaluation per history element, and the predicted curve matches the observed 1327 s for k=1000, w=1 almost exactly (predicted    
  1406 s).                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                              
  ---                                                                                                                                                                                                                                         
  Why it explains all the anomalous results                                                                                                                                                                                                   
                                                                                                                                                                                                                                              
  1. The "super-linear efficiency" for large k at many workers                                                                                                                                                                                
                                                                                                                                                                                                                                              
  The efficiency metric is throughput(w) / (w × throughput(1)). For k=1000 w=1, throughput(1) is devastated by O(N²) overhead: 20 000 evals each worker, 1400 s of overhead, leaving only ~14 sim/s. When going to 16 workers, each worker    
  only processes ceil(20000/16) = 1250 evals, reducing overhead from 1400 s to 5.5 s per worker. The per-worker overhead drops by 256× while n_workers only grows 16×, so the efficiency figure (computed relative to the degraded 1-worker   
  baseline) exceeds 1.0. This is an artifact — not a real algorithmic gain.                                                                                                                                                                   
                                                                                                                                                                                                                                              
  2. The "low 1-worker throughput" at large k                                                                                                                                                                                                 
   
  Directly caused by the O(N²) overhead. At k=1000, 97% of wall-clock time is spent inside ABCPMC.__call__ iterating history, not running simulations. The actual Lotka-Volterra simulation takes ~0.002–0.003 s/call; propulate overhead     
  takes ~0.066 s/call on average over the run.
                                                                                                                                                                                                                                              
  3. The archive never filling for k=1000, w=1

  At k=1000, max_sims = 20 000, the ABCPMC archive needs 1 000 individuals with loss < 500 000. From the data, only 492 out of 40 000 evals pass this threshold (1.2% acceptance rate in prior phase). The archive never fills, so the        
  tolerance scheduler never fires and tol stays at the initial value of 500 000 throughout the entire run. This is a separate algorithmic problem (the prior acceptance rate at the given tol_init is too low for the Lotka-Volterra model),
  but it's compounded by the O(N²) overhead that makes the run take 1327 s without convergence instead of ~40 s.                                                                                                                              

  4. Why it doesn't affect the baseline                                                                                                                                                                                                       
   
  abc_smc_baseline (pyABC) maintains only the current generation of k particles in memory at any time. Its per-step cost is O(k), not O(N_total). For k=1000, w=1, baseline processes ~163 000 simulations in 267 s (wall-time stopped at 300 
  s), efficiently.

---
  Where to fix it
                                                                                                                                                                                                                                              
  In ABCPMC.__call__ (and the schedulers), the full inds list is scanned repeatedly on every call. The fix is to maintain incremental state rather than scanning history from scratch:
                                                                                                                                                                                                                                              
  - tol_from_history: keep a running minimum updated each call rather than min() over all inds.                                                                                                                                               
  - select_archive: maintain a sorted structure (e.g. a heap or sorted list capped at k) updated incrementally, instead of filtering and sorting all inds on every call.                                                                      
  - Schedulers (especially AcceptanceRateScheduler): operate on a sliding window of recent inds; this window should be pre-extracted, not scanned by iterating sorted(inds, ...)[-window_size:].                                              
                                                                                                                                                                                                                                              
  Each of these reduces the per-call cost from O(N) to O(log k) or O(1), converting the total cost from O(N²) to O(N log k).                                                                                                                  
                                                                                                                                                                                                                                              
  ---                                                                                                                                                                                                                                         
  Summary table
                                                                                                                                                                                                                                              
  ┌─────────────────────────────────────────┬──────────────────────┬───────────────────────────────────────────────────────────────┐
  │                  Issue                  │       Category       │                          Root cause                           │                                                                                                          
  ├─────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────┤                                                                                                          
  │ Low async throughput at k≥200, w=1      │ Implementation bug   │ O(N²) ABCPMC.__call__ cost                                    │                                                                                                          
  ├─────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────┤                                                                                                          
  │ "Super-linear" efficiency at large k    │ Measurement artifact │ Efficiency normalised to degraded 1-worker baseline           │                                                                                                          
  ├─────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────┤                                                                                                          
  │ Archive never fills at k=1000, w=1      │ Algorithm + config   │ Too-low prior acceptance rate at tol_init=500000 for LV model │                                                                                                          
  ├─────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────┤                                                                                                          
  │ All observations disappear above T=300s │ Correct behaviour    │ Propulate runs to simulation cap, not wall-time cap           │
  └─────────────────────────────────────────┴──────────────────────┴───────────────────────────────────────────────────────────────┘                                                                                                          
   
