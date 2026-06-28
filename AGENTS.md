# Exception Handling
Never hide failures, prefer to crash loudly, only handle errors with clear recovery path, otherwise pass upwards. No silent handling. 

# Testing
Use the venv at nastjapy_copy/.venv

# Bug fixing
Keep track of previous bug fixes by updating .plans/bug-fixes/previous-fixes.md. Consult it when needed.

# Ressources
Plans are meant to be stored in .plans. Further ressources on the intentions of the experiments can be found in .plans/ressources if needed.

# JSC cluster runs (jsc-mpc MCP)
When using the jsc-mpc tools (submit_job, sync_code, start_session, run_on_login, remaining_budget, estimate_cost, ...), ALWAYS pass `project="async-abc-paper"` and `cluster="juwels-cluster"`. That selects this repo's experimentation budget — soft 24 node-hours / 24h rolling, hard 96 / session, per-job 48 (NODE-hours; 1 JUWELS node = 48 cores). Omitting the project falls back to a conservative default and runs will be refused early.
Cluster ops need a live SSH ControlMaster (else "Connection closed"); if so, call `login_master_status` for the exact pre-auth command to run in a terminal. Before `sync_code`, commit/stash local changes (it refuses a dirty tree).