# Realistic simulator-workload assets

These files configure the **realistic simulator workload** benchmark — the
expensive, heterogeneous-runtime external simulator used to evaluate the
asynchronous method at scale (RQ3 in the paper).

The workload is driven through a pluggable simulation backend
(`experiments/async_abc/benchmarks/realistic_workload.py`), which resolves a
`ParameterSpace`, a `SimulationManager`, and a `DistanceMetric` from the active
environment. Two of the files here are **placeholders**: the concrete,
backend-specific configuration template (`sim_config.json`) and its
config-builder parameters (`config_builder_params.json`) are provided by the
simulation backend and are intentionally omitted from this branch.

| File | Role |
|------|------|
| `parameter_space.json` | The two inferred parameters, θ₁ (well-identified) and θ₂ (weakly-identified), with their public `[0,1]` and physical ranges. |
| `distance_metric_params.json` | Summary-statistic / feature configuration for the ABC distance metric (spatial point-pattern features). |
| `sim_config.json` | Placeholder for the backend's native simulator template. |
| `config_builder_params.json` | Placeholder for the backend's config-builder parameters. |
| `reference_data/` | Reference simulation output the distance metric compares against (columns de-identified). |

To run the benchmark against a concrete backend, point
`SIM_BACKEND_PARAMSPACE_MODULE` (and, if needed, `SIM_BACKEND_VENV`) at your
simulation backend and supply the backend-specific `sim_config.json` /
`config_builder_params.json`.
