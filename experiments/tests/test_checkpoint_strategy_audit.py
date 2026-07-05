"""W2.5 audit: paper-facing wall-time comparison plots must use ``time_uniform``.

Paper §6 comparisons across methods (async vs sync) are only meaningful when
both methods are sampled on a shared wall-clock checkpoint grid. ``time_uniform``
LOCF-resamples records onto that grid; the legacy ``quantile`` strategy uses
each method's native per-event granularity, which produces unequal checkpoint
density and biases curve comparisons.

This audit pins the paper-facing reporters that *must* use ``time_uniform``:
- ``plot_quality_vs_wall_time``
- ``plot_quality_by_sigma``
- ``plot_ablation_amis_isolation``

Diagnostic reporters (``*_diagnostic``) and non-wall-time-axis reporters
(``plot_quality_vs_posterior_samples``, ``plot_quality_vs_attempt_budget``)
are intentionally excluded — ``time_uniform`` only applies to a wall-clock
axis.
"""
import re
from pathlib import Path


REPORTERS_PY = (
    Path(__file__).parent.parent
    / "async_abc"
    / "plotting"
    / "reporters.py"
).resolve()


PAPER_WALL_TIME_REPORTERS = (
    "plot_quality_vs_wall_time",
    "plot_quality_by_sigma",
    "plot_ablation_amis_isolation",
)


def _function_bodies(source: str) -> dict[str, str]:
    """Map ``def name(...)`` → its body text (until next top-level ``def`` or EOF)."""
    bodies: dict[str, str] = {}
    pattern = re.compile(r"^def (\w+)\(", re.MULTILINE)
    matches = list(pattern.finditer(source))
    for idx, m in enumerate(matches):
        name = m.group(1)
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
        bodies[name] = source[start:end]
    return bodies


def test_paper_wall_time_reporters_use_time_uniform():
    source = REPORTERS_PY.read_text()
    bodies = _function_bodies(source)
    for fname in PAPER_WALL_TIME_REPORTERS:
        assert fname in bodies, (
            f"Paper-facing reporter `{fname}` not found in reporters.py; "
            "did the function move or get renamed?"
        )
        body = bodies[fname]
        # Verify the function calls posterior_quality_curve AND that call
        # passes checkpoint_strategy="time_uniform".
        if "posterior_quality_curve(" not in body:
            # Some reporters wrap subfunctions; allow that.
            continue
        assert 'checkpoint_strategy="time_uniform"' in body, (
            f"Paper-facing wall-time reporter `{fname}` must call "
            f"posterior_quality_curve(checkpoint_strategy=\"time_uniform\"). "
            f"Current body uses a different strategy — see W2.5 audit."
        )


def test_no_unflagged_default_checkpoint_strategy_in_paper_plots():
    """No call to posterior_quality_curve(... axis_kind="wall_time" ...) in a
    paper-facing reporter may omit checkpoint_strategy (which would default to
    'all'). This catches an accidental regression where a new reporter is
    added without specifying the strategy."""
    source = REPORTERS_PY.read_text()
    # Find all posterior_quality_curve(...) call blocks
    pattern = re.compile(
        r"posterior_quality_curve\((.*?)\)",
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        block = match.group(1)
        # Locate the enclosing function name by scanning backwards.
        prefix = source[: match.start()]
        func_match = list(re.finditer(r"^def (\w+)\(", prefix, re.MULTILINE))
        if not func_match:
            continue
        enclosing = func_match[-1].group(1)
        # Skip diagnostic and non-paper reporters.
        if enclosing.endswith("_diagnostic"):
            continue
        if 'axis_kind="wall_time"' not in block:
            continue
        # Paper-facing + wall-time axis: must have an explicit strategy.
        assert "checkpoint_strategy=" in block, (
            f"Paper-facing wall-time `posterior_quality_curve` call in "
            f"`{enclosing}` omits checkpoint_strategy — defaults to 'all'. "
            f"Set checkpoint_strategy=\"time_uniform\" to enable cross-method "
            f"comparability (W2.5 audit)."
        )
