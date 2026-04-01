"""Human boundary — command-line interface for the measurement arm."""
from __future__ import annotations
import argparse
import json
import os
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(prog="arm", description="ARM: Universal Topological Measurement Arm")
    sub = parser.add_subparsers(dest="command")

    p_measure = sub.add_parser("measure", help="Measure a source")
    p_measure.add_argument("source_type", choices=["text", "csv", "veilbreak", "auto"])
    p_measure.add_argument("source", nargs="?", default=None)
    p_measure.add_argument("--mode", default="topology", choices=["topology", "crystal", "full"])
    p_measure.add_argument("--eps-max", type=float, default=5.0)
    p_measure.add_argument("--id", default=None)
    p_measure.add_argument("--results-dir", default="arm/results")

    p_compare = sub.add_parser("compare", help="Compare crystal ratios between experiments")
    p_compare.add_argument("source1")
    p_compare.add_argument("source2")
    p_compare.add_argument("--results-dir", default="arm/results")

    p_validate = sub.add_parser("validate", help="Validate crystal weights match desktop")
    p_validate.add_argument("weights")

    p_results = sub.add_parser("results", help="Show experiment records")
    p_results.add_argument("--results-dir", default="arm/results")

    p_series = sub.add_parser("series", help="Run full experiment series")
    p_series.add_argument("--results-dir", default="arm/results")

    args = parser.parse_args(argv)
    if args.command == "measure":
        _cmd_measure(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "results":
        _cmd_results(args)
    elif args.command == "series":
        _cmd_series(args)
    else:
        parser.print_help()

def _cmd_measure(args):
    from arm.measure import measure
    source = args.source
    if args.source_type != "veilbreak" and source and os.path.isfile(source):
        with open(source) as f:
            source = f.read()
    elif args.source_type == "veilbreak":
        source = None
    exp_id = args.id or f"ARM-{args.source_type.upper()}"
    record = measure(source=source, source_type=args.source_type, mode=args.mode,
                     experiment_id=exp_id, results_dir=args.results_dir, eps_max=args.eps_max)
    _print_record(record)

def _cmd_compare(args):
    from arm.prime.compare import crystal_distance
    from arm.void.formats import Crystal
    def _load_crystal(ref, results_dir):
        path = ref if os.path.isfile(ref) else None
        if not path:
            candidates = [f for f in os.listdir(results_dir) if f.startswith(ref) and f.endswith(".json")]
            if candidates:
                path = os.path.join(results_dir, sorted(candidates)[-1])
        if not path:
            print(f"Cannot find experiment or file: {ref}")
            return None
        with open(path) as f:
            rec = json.load(f)
        c = rec.get("result", {}).get("crystal")
        if not c:
            print(f"No crystal in {ref}")
            return None
        return Crystal(c["void"], c["identity"], c["prime"], 0.0, rec["id"])
    c1 = _load_crystal(args.source1, args.results_dir)
    c2 = _load_crystal(args.source2, args.results_dir)
    if c1 and c2:
        d = crystal_distance(c1, c2)
        print(f"Crystal distance ({c1.source} vs {c2.source}): {d:.4f}")
        print(f"  {c1.source}: void={c1.void_ratio:.3f} identity={c1.identity_ratio:.3f} prime={c1.prime_ratio:.3f}")
        print(f"  {c2.source}: void={c2.void_ratio:.3f} identity={c2.identity_ratio:.3f} prime={c2.prime_ratio:.3f}")

def _cmd_validate(args):
    from arm.identity.weights import load_weights, unpack_ternary
    import numpy as np
    layers, config = load_weights(args.weights)
    print(f"Loaded weights: {len(layers)} layers, config: {config}")
    all_values = []
    for name, packed in layers.items():
        count = packed.shape[0] * 4
        values = unpack_ternary(packed, count)
        all_values.append(values)
    if all_values:
        combined = np.concatenate(all_values)
        n = len(combined)
        from arm.void.formats import Crystal
        c = Crystal(float(np.sum(combined == 0))/n, float(np.sum(combined == 1))/n,
                     float(np.sum(combined == 3))/n, 0.0, "weights")
        c.validate()
        print(f"Crystal: void={c.void_ratio:.3f} identity={c.identity_ratio:.3f} prime={c.prime_ratio:.3f}")

def _cmd_results(args):
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    if not files:
        print("No experiment records found.")
        return
    for f in files:
        with open(os.path.join(results_dir, f)) as fh:
            rec = json.load(fh)
        print(f"{rec['id']} run{rec['run']} [{rec['verdict']}] {rec['timestamp']}")
        if "crystal" in rec.get("result", {}):
            c = rec["result"]["crystal"]
            print(f"  crystal: void={c['void']:.3f} identity={c['identity']:.3f} prime={c['prime']:.3f}")

def _cmd_series(args):
    from arm.measure import measure
    print("=== ARM Experiment Series: Phase A ===\n")
    sample = _get_sample_text()

    print("--- ARM-001: Text topology on ARM ---")
    r = measure(sample, "text", "topology", "ARM-001", "phase-a", args.results_dir,
                hypothesis="H0 persistence produces valid barcodes on ARM hardware")
    _print_record(r)

    print("\n--- ARM-002: Character harmonic crystal ---")
    r = measure(sample, "text", "topology", "ARM-002", "phase-a", args.results_dir,
                hypothesis="Character harmonics produce crystal ratios near 22/42/36")
    _print_record(r)

    print("\n--- ARM-003: Veilbreak observation topology ---")
    r = measure(None, "veilbreak", "topology", "ARM-003", "phase-a", args.results_dir,
                hypothesis="Veilbreak data produces measurable topological structure")
    _print_record(r)

    print("\n--- ARM-005: Cross-domain comparison ---")
    _run_comparison(args.results_dir)
    print("\n=== Series complete ===")

def _get_sample_text() -> str:
    candidates = ["data/wikitext-103-raw/wiki.test.raw", "data/wikitext/test.txt"]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return f.read()[:10000]
    return ("The topology doesn't care what the structure is. It cares how the structure "
            "changes across scales. Every mathematical structure that describes reality has "
            "a topological signature. The adaptive operator is the instrument that reads "
            "this signature. Primes are the zero-dimensional framework of computational reality. ") * 10

def _run_comparison(results_dir: str):
    from arm.prime.compare import universality_test
    from arm.void.formats import Crystal
    import glob
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    crystals = []
    for f in files:
        with open(f) as fh:
            rec = json.load(fh)
        if "crystal" in rec.get("result", {}):
            c = rec["result"]["crystal"]
            crystals.append(Crystal(c["void"], c["identity"], c["prime"], 0.0, rec["id"]))
    if len(crystals) < 2:
        print("  Not enough crystal measurements for comparison.")
        return
    result = universality_test(crystals)
    print(f"  Universal: {result['universal']}")
    print(f"  Max distance: {result['max_distance']:.4f}")
    for pair, dist in zip(result["pairs"], result["distances"]):
        print(f"    {pair[0]} vs {pair[1]}: {dist:.4f}")

def _print_record(record):
    print(f"  ID: {record.id} | Run: {record.run} | Verdict: {record.verdict}")
    if "crystal" in record.result:
        c = record.result["crystal"]
        print(f"  Crystal: void={c['void']:.3f} identity={c['identity']:.3f} prime={c['prime']:.3f}")
    if "gini" in record.result:
        print(f"  Gini: {record.result['gini']:.4f}")
    if "onset_scale" in record.result:
        print(f"  Onset e*: {record.result['onset_scale']:.4f}")
