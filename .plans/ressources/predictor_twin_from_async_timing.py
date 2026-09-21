import pandas as pd, numpy as np
S='/home/juhe/remotes/scratch/herold2/async-abc'
V='/home/juhe/bwSyncShare/Code/async-abc-paper/experiments/data/paper_figures'
SP='/tmp/claude-1000/-home-juhe-bwSyncShare-Code-async-abc-paper/d94b1997-2664-437f-a4b6-e53c38b01f5b/scratchpad'
rng=np.random.default_rng(0); rows=[]
# ---- straggler: per-worker mean recorded duration (busy/n) + the fast workers' per-eval overhead
dbg=pd.read_csv(f'{S}/twin2_20260729/straggler_async_wall/data/runtime_debug_summary.csv')
dbg['slow']=dbg.method.str.extract(r'slowdown([0-9.]+)x').astype(float)
tw=pd.read_csv(f'{V}/tab_twin/twin_straggler_raw.csv')
for (s,rep),g in dbg.groupby(['slow','replicate']):
    W=g.worker_id.nunique(); busy=(g.total_busy_s/g.n_attempts).to_numpy()
    ovh=np.median(((g.active_span_s-g.total_busy_s)/g.n_attempts).to_numpy()[busy<busy.max()]) if s>0 else np.median((g.active_span_s-g.total_busy_s)/g.n_attempts)
    d=busy+ovh; T_async=g.n_attempts.sum()/g.elapsed_wall_s.max(); T_sync_pred=W/d.max()
    for _,t in tw[(tw.slowdown_factor==s)&(tw.replicate==rep)&(tw.arm=='twin_fine')].iterrows():
        rows.append(dict(workload='straggler',level=s,W=W,rep=rep,T_async=T_async,T_sync_pred=T_sync_pred,T_sync_meas=t.throughput_sims_per_s,
                         ratio_pred=T_async/T_sync_pred,ratio_meas=T_async/t.throughput_sims_per_s,note=f'straggler busy {busy.max():.3f}s ovh {ovh*1e3:.2f}ms'))
# ---- heterogeneity: injected law LN(0,sigma) x base, uncensored, parametric E[max_48]
hd=pd.read_csv(f'{S}/rerun_20260707/runtime_heterogeneity/data/runtime_debug_summary.csv')
hd=hd[hd.method.str.startswith('async')]; hd['sigma']=hd.method.str.extract(r'sigma([0-9.]+)').astype(float)
th=pd.read_csv(f'{V}/tab_twin_hetero/twin_hetero_raw.csv')
base0=hd[hd.sigma==0].pipe(lambda g:(g.total_busy_s.sum()/g.n_attempts.sum()))
print(f'hetero base duration at sigma=0 (async busy/attempt): {base0:.4f}s')
for (sg,rep),g in hd.groupby(['sigma','replicate']):
    W=48; ovh=((g.active_span_s-g.total_busy_s)/g.n_attempts).median(); T_async=g.n_attempts.sum()/g.elapsed_wall_s.max()
    mult=np.exp(sg*rng.standard_normal((40000,W))); Emax=(base0*mult+ovh).max(axis=1).mean()
    T_sync_pred=W/Emax
    # empirical-censored variant for the record
    for _,t in th[(th.sigma==sg)&(th.replicate==rep)&(th.arm=='twin_fine')].iterrows():
        rows.append(dict(workload='hetero',level=sg,W=W,rep=rep,T_async=T_async,T_sync_pred=T_sync_pred,T_sync_meas=t.throughput_sims_per_s,
                         ratio_pred=T_async/T_sync_pred,ratio_meas=T_async/t.throughput_sims_per_s,note=f'E[max48]={Emax:.2f}s ovh {ovh*1e3:.1f}ms'))
# ---- CPM 50^3: reuse first pass rows (utilisation-based measured ratio)
p1=pd.read_csv(f'{SP}/predictor_rows.csv'); c=p1[p1.workload=='cpm50'].copy(); c['note']='async in-run durations + overhead; measured = utilisation ratio'
c['ratio_meas']=c['ratio_meas_util']; rows+=c[['workload','level','W','rep','T_async','T_sync_pred','T_sync_meas','ratio_pred','ratio_meas','note']].to_dict('records')
rows.append(dict(workload='cpm80',level=48,W=48,rep=-1,T_async=np.nan,T_sync_pred=np.nan,T_sync_meas=np.nan,ratio_pred=1.90,ratio_meas=2.01,note='from study log, job 14262841 (in-run CV 0.23)'))
df=pd.DataFrame(rows); df.to_csv(f'{SP}/predictor_rows2.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.float_format',lambda x:f'{x:.3g}')
summ=df.groupby(['workload','level']).agg(W=('W','first'),T_async=('T_async','median'),T_sync_pred=('T_sync_pred','median'),T_sync_meas=('T_sync_meas','median'),
     ratio_pred=('ratio_pred','median'),ratio_meas=('ratio_meas','median'),pred_over_meas=('ratio_pred',lambda x: np.nan),note=('note','first'))
summ['pred_over_meas']=summ.ratio_pred/summ.ratio_meas
print(summ.to_string())
