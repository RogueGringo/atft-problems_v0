"""The scaffolding — connects stages, manages experiment records."""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from arm.void.formats import ExperimentRecord, PointCloud
from arm.void.transducers import TextTransducer, GenericTransducer, VeilbreakTransducer
from arm.identity.persistence import compute_h0
from arm.prime.invariants import gini, onset_scale, effective_rank, crystal_from_persistence

TRANSDUCERS = {
    "text": TextTransducer,
    "csv": GenericTransducer,
    "generic": GenericTransducer,
    "veilbreak": VeilbreakTransducer,
}

def _get_next_run(experiment_id: str, results_dir: str) -> int:
    if not os.path.exists(results_dir):
        return 1
    existing = [f for f in os.listdir(results_dir)
                if f.startswith(experiment_id) and f.endswith(".json")]
    return len(existing) + 1

def _save_record(record: ExperimentRecord, results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    ts = record.timestamp.replace(":", "-").replace(".", "-")
    filename = f"{record.id}_run{record.run}_{ts}.json"
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        f.write(record.to_json())
    return path

def run_experiment(
    experiment_id: str, series: str, source, source_type: str,
    mode: str = "topology", results_dir: str = "arm/results",
    hypothesis: str = "", eps_max: float = 5.0, n_steps: int = 100,
) -> ExperimentRecord:
    t0 = time.time()
    annotations = []
    run = _get_next_run(experiment_id, results_dir)
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. TRANSDUCE
    transducer_cls = TRANSDUCERS.get(source_type, GenericTransducer)
    transducer = transducer_cls()
    cloud = transducer.transduce(source)
    annotations.append({
        "stage": "transduce",
        "transducer": transducer.describe(),
        "points": cloud.data.shape[0],
        "dims": cloud.data.shape[1] if cloud.data.ndim > 1 else 1,
        "hash": cloud.hash,
    })

    result = {}
    verdict = "PARTIAL"

    if mode in ("topology", "full"):
        diagram = compute_h0(cloud, eps_max=eps_max, n_steps=n_steps)
        annotations.append({
            "stage": "persistence",
            "h0_bars": diagram.h0.shape[0],
            "h1_bars": diagram.h1.shape[0],
            "filtration_range": list(diagram.filtration_range),
        })

        finite_bars = diagram.h0[:, 1] - diagram.h0[:, 0]
        finite_bars = finite_bars[np.isfinite(finite_bars)]

        g = gini(finite_bars)
        eps_star = onset_scale(diagram)
        crystal = crystal_from_persistence(diagram)
        crystal.validate()

        result["gini"] = g
        result["onset_scale"] = eps_star
        result["h0_bar_count"] = int(diagram.h0.shape[0])
        result["crystal"] = {
            "void": crystal.void_ratio,
            "identity": crystal.identity_ratio,
            "prime": crystal.prime_ratio,
        }
        result["eff_rank"] = crystal.eff_rank
        annotations.append({
            "stage": "invariants",
            "gini": g, "onset_scale": eps_star,
            "crystal_void": crystal.void_ratio,
            "crystal_identity": crystal.identity_ratio,
            "crystal_prime": crystal.prime_ratio,
        })
        verdict = "PASS"

    if mode in ("crystal", "full"):
        annotations.append({
            "stage": "crystal_forward",
            "status": "NOT_IMPLEMENTED",
            "note": "Requires pre-trained weights from desktop via export_for_arm()",
        })
        if mode == "crystal":
            verdict = "BLOCKED"

    elapsed = time.time() - t0
    annotations.append({"stage": "timing", "elapsed_seconds": round(elapsed, 3)})

    record = ExperimentRecord(
        id=experiment_id, run=run, series=series, timestamp=timestamp,
        hypothesis=hypothesis,
        protocol=f"mode={mode}, source_type={source_type}, eps_max={eps_max}, n_steps={n_steps}",
        input_hash=cloud.hash, annotations=annotations,
        result=result, comparison={}, verdict=verdict, notes="",
    )
    _save_record(record, results_dir)
    return record
