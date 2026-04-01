"""The junction — 3-arm router connecting all bands."""
from __future__ import annotations
from arm.void.formats import ExperimentRecord
from arm.identity.pipeline import run_experiment

def measure(
    source, source_type: str = "auto", mode: str = "topology",
    experiment_id: str = "ARM-MEASURE", series: str = "ad-hoc",
    results_dir: str = "arm/results", hypothesis: str = "",
    eps_max: float = 5.0, n_steps: int = 100,
) -> ExperimentRecord:
    if source_type == "auto":
        source_type = _detect_source_type(source)
    return run_experiment(
        experiment_id=experiment_id, series=series, source=source,
        source_type=source_type, mode=mode, results_dir=results_dir,
        hypothesis=hypothesis, eps_max=eps_max, n_steps=n_steps,
    )

def _detect_source_type(source) -> str:
    if isinstance(source, str):
        lines = source.strip().split("\n")
        if len(lines) > 1 and all("," in line for line in lines[:3]):
            try:
                float(lines[0].split(",")[0])
                return "csv"
            except ValueError:
                pass
        return "text"
    return "generic"
