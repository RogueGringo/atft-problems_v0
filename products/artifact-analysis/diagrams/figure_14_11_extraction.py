#!/usr/bin/env python3
"""
Figure 14.11 — Full view of diagram D39-08-117c
Complete graph extraction from visual analysis.

Every node, every edge, every measurable feature.
This IS the data. The diagram IS the dataset.
"""

import json
import numpy as np

# ── NODE EXTRACTION ──────────────────────────────────────────────────────
# Positions normalized to [0,1] on the page. (0,0) = top-left.
# Size: 0=dot, 1=small, 2=medium, 3=large
# Rings: count of concentric inscription bands
# Fill: ratio of dark/filled area to total node area
# Glyph: central symbol identifier

nodes = {
    # === LARGE NODES (prime-class) ===
    "N1": {
        "name": "bold_chevron",
        "pos": [0.20, 0.30],      # left side, upper-middle
        "size": 3,                  # largest node in diagram
        "rings": 4,                 # 4+ concentric bands
        "fill_ratio": 0.7,         # mostly filled (dark center)
        "glyph": "angular_chevron", # bold angular arrow symbol
        "has_barcode": True,        # hash marks at periphery
        "inscription_density": 3,   # dense text on rings
        "class": "prime",
    },
    "N2": {
        "name": "wifi_triple",
        "pos": [0.72, 0.22],       # right side, upper
        "size": 3,
        "rings": 3,
        "fill_ratio": 0.8,         # very dark center
        "glyph": "triple_curve",    # wifi/triple-curve symbol
        "has_barcode": True,
        "inscription_density": 2,
        "class": "prime",
    },
    "N3": {
        "name": "orbital_mandala",
        "pos": [0.60, 0.65],       # center-right, lower
        "size": 3,
        "rings": 5,                 # most rings in diagram
        "fill_ratio": 0.6,
        "glyph": "concentric_mandala",
        "has_barcode": True,
        "inscription_density": 3,
        "class": "prime",
    },

    # === MEDIUM NODES (identity-class) ===
    "N4": {
        "name": "trident_center",
        "pos": [0.40, 0.35],       # center of diagram
        "size": 2,
        "rings": 2,
        "fill_ratio": 0.5,
        "glyph": "three_arm_trident",
        "has_barcode": False,
        "inscription_density": 2,
        "class": "identity",
    },
    "N5": {
        "name": "chevron_upper",
        "pos": [0.38, 0.13],       # upper center
        "size": 2,
        "rings": 2,
        "fill_ratio": 0.5,
        "glyph": "small_chevron",
        "has_barcode": False,
        "inscription_density": 1,
        "class": "identity",
    },
    "N6": {
        "name": "bottom_complex",
        "pos": [0.50, 0.88],       # bottom center
        "size": 2,
        "rings": 3,
        "fill_ratio": 0.6,
        "glyph": "small_mandala",
        "has_barcode": False,
        "inscription_density": 2,
        "class": "identity",
    },

    # === SMALL NODES (void-class / junction points) ===
    "J1": {
        "name": "junction_1",
        "pos": [0.35, 0.08],       # top, near N5
        "size": 0,
        "rings": 0,
        "fill_ratio": 1.0,         # solid dot
        "glyph": "dot",
        "has_barcode": False,
        "inscription_density": 0,
        "class": "void",
    },
    "J2": {
        "name": "junction_2",
        "pos": [0.52, 0.12],       # upper right of N5
        "size": 0,
        "rings": 0,
        "fill_ratio": 1.0,
        "glyph": "dot",
        "has_barcode": False,
        "inscription_density": 0,
        "class": "void",
    },
    "J3": {
        "name": "junction_3",
        "pos": [0.45, 0.50],       # mid-right
        "size": 1,
        "rings": 1,
        "fill_ratio": 0.3,
        "glyph": "small_circle",
        "has_barcode": False,
        "inscription_density": 0,
        "class": "void",
    },
    "J4": {
        "name": "junction_4",
        "pos": [0.52, 0.55],       # below J3
        "size": 0,
        "rings": 0,
        "fill_ratio": 1.0,
        "glyph": "dot",
        "has_barcode": False,
        "inscription_density": 0,
        "class": "void",
    },
    "J5": {
        "name": "junction_5",
        "pos": [0.48, 0.72],       # lower mid
        "size": 0,
        "rings": 0,
        "fill_ratio": 1.0,
        "glyph": "dot",
        "has_barcode": False,
        "inscription_density": 0,
        "class": "void",
    },
    "J6": {
        "name": "junction_6",
        "pos": [0.45, 0.80],       # above N6
        "size": 0,
        "rings": 0,
        "fill_ratio": 1.0,
        "glyph": "dot",
        "has_barcode": False,
        "inscription_density": 0,
        "class": "void",
    },
    "J7": {
        "name": "bean_shape",
        "pos": [0.12, 0.55],       # left side, isolated
        "size": 1,
        "rings": 0,
        "fill_ratio": 0.2,         # mostly outline
        "glyph": "bean",
        "has_barcode": False,
        "inscription_density": 0,
        "class": "void",
    },
}

# ── EDGE EXTRACTION ──────────────────────────────────────────────────────
# thickness: 0=absent, 1=thin, 2=medium, 3=heavy
# curvature: 0=straight, 1=slight, 2=moderate, 3=strong S-curve
# inscribed: whether the connection path has symbols on it

edges = [
    # Heavy connections (prime-class edges)
    {"from": "N1", "to": "N4", "thickness": 3, "curvature": 2, "inscribed": False},
    {"from": "N4", "to": "N3", "thickness": 3, "curvature": 3, "inscribed": False},
    {"from": "N4", "to": "N2", "thickness": 3, "curvature": 2, "inscribed": False},
    {"from": "N3", "to": "N6", "thickness": 3, "curvature": 3, "inscribed": False},

    # Medium connections (identity-class edges)
    {"from": "N5", "to": "N4", "thickness": 2, "curvature": 1, "inscribed": False},
    {"from": "N1", "to": "N5", "thickness": 1, "curvature": 1, "inscribed": False},
    {"from": "N5", "to": "J1", "thickness": 1, "curvature": 0, "inscribed": False},
    {"from": "N5", "to": "J2", "thickness": 1, "curvature": 0, "inscribed": False},
    {"from": "N2", "to": "J3", "thickness": 2, "curvature": 1, "inscribed": False},

    # Thin connections (void-class edges)
    {"from": "J3", "to": "J4", "thickness": 1, "curvature": 2, "inscribed": False},
    {"from": "J4", "to": "N3", "thickness": 1, "curvature": 1, "inscribed": False},
    {"from": "J4", "to": "J5", "thickness": 1, "curvature": 2, "inscribed": False},
    {"from": "J5", "to": "J6", "thickness": 1, "curvature": 2, "inscribed": False},
    {"from": "J6", "to": "N6", "thickness": 1, "curvature": 1, "inscribed": False},

    # Star/compass connection at top
    {"from": "N5", "to": "N1", "thickness": 1, "curvature": 1, "inscribed": True},
]

# ── ANALYSIS ─────────────────────────────────────────────────────────────

# Node class distribution
classes = {"void": 0, "identity": 0, "prime": 0}
for n in nodes.values():
    classes[n["class"]] += 1
total = sum(classes.values())

print("=" * 60)
print("FIGURE 14.11 — COMPLETE GRAPH EXTRACTION")
print("D39-08-117c")
print("=" * 60)

print(f"\nNodes: {total}")
for cls, count in classes.items():
    print(f"  {cls}: {count} ({count/total:.1%})")

# Complexity-weighted distribution
complexity = {"void": 0, "identity": 0, "prime": 0}
for n in nodes.values():
    weight = (n["size"] + 1) * (n["rings"] + 1) * (n["inscription_density"] + 1)
    complexity[n["class"]] += weight
total_complexity = sum(complexity.values())

print(f"\nComplexity-weighted:")
for cls, weight in complexity.items():
    print(f"  {cls}: {weight} ({weight/total_complexity:.1%})")

# Edge class distribution
edge_classes = {"void": 0, "identity": 0, "prime": 0}
for e in edges:
    if e["thickness"] >= 3:
        edge_classes["prime"] += 1
    elif e["thickness"] >= 2:
        edge_classes["identity"] += 1
    else:
        edge_classes["void"] += 1
total_edges = sum(edge_classes.values())

print(f"\nEdges: {total_edges}")
for cls, count in edge_classes.items():
    print(f"  {cls}: {count} ({count/total_edges:.1%})")

# Ring depth distribution (the z-axis)
ring_counts = [n["rings"] for n in nodes.values()]
print(f"\nRing depth distribution:")
for depth in range(max(ring_counts) + 1):
    count = ring_counts.count(depth)
    if count > 0:
        bar = "█" * (count * 4)
        print(f"  depth {depth}: {count} nodes {bar}")

# ── MULTI-CHANNEL ENCODING ──────────────────────────────────────────────
# Encode each node as a multi-channel {0,1,3} vector
# This is the diagram-as-signal representation

print(f"\n{'='*60}")
print("MULTI-CHANNEL {0,1,3} ENCODING")
print("Each node → (size, ring_depth, density, fill)")
print("=" * 60)

def to_013(value, thresholds):
    """Map a continuous value to {0,1,3} based on thresholds."""
    if value <= thresholds[0]:
        return 0
    elif value <= thresholds[1]:
        return 1
    else:
        return 3

encoded_nodes = []
for nid, n in nodes.items():
    ch_size = to_013(n["size"], [0, 1.5])          # 0→0, 1→1, 2-3→3
    ch_rings = to_013(n["rings"], [0.5, 2.5])       # 0→0, 1-2→1, 3+→3
    ch_density = to_013(n["inscription_density"], [0, 1.5])  # 0→0, 1→1, 2-3→3
    ch_fill = to_013(n["fill_ratio"], [0.3, 0.6])   # low→0, mid→1, high→3

    encoded = [ch_size, ch_rings, ch_density, ch_fill]
    encoded_nodes.append(encoded)
    print(f"  {nid:>3s} ({n['class']:>8s}): size={ch_size} rings={ch_rings} "
          f"density={ch_density} fill={ch_fill}  → {encoded}")

# Compute the {0,1,3} distribution of the encoded diagram
all_values = [v for enc in encoded_nodes for v in enc]
total_vals = len(all_values)
v0 = all_values.count(0) / total_vals
v1 = all_values.count(1) / total_vals
v3 = all_values.count(3) / total_vals

print(f"\n  DIAGRAM CRYSTAL (from node encoding):")
print(f"  void={v0:.3f}  identity={v1:.3f}  prime={v3:.3f}")
print(f"  As percentages: {v0*100:.1f} / {v1*100:.1f} / {v3*100:.1f}")

# Compare to our measured crystal
print(f"\n  COMPARISON:")
print(f"  English text crystal (L2=0): 22.2 / 41.7 / 36.1")
print(f"  Diagram crystal:             {v0*100:.1f} / {v1*100:.1f} / {v3*100:.1f}")
print(f"  Δ void:     {abs(v0 - 0.222)*100:+.1f}%")
print(f"  Δ identity: {abs(v1 - 0.417)*100:+.1f}%")
print(f"  Δ prime:    {abs(v3 - 0.361)*100:+.1f}%")

# Save the full extraction
extraction = {
    "source": "Figure 14.11 — D39-08-117c",
    "nodes": nodes,
    "edges": edges,
    "encoded_nodes": {nid: enc for nid, enc in zip(nodes.keys(), encoded_nodes)},
    "node_class_distribution": {k: v/total for k, v in classes.items()},
    "complexity_weighted": {k: v/total_complexity for k, v in complexity.items()},
    "edge_class_distribution": {k: v/total_edges for k, v in edge_classes.items()},
    "diagram_crystal": {"void": v0, "identity": v1, "prime": v3},
    "english_crystal": {"void": 0.222, "identity": 0.417, "prime": 0.361},
}

outpath = "/home/wb1/Desktop/Dev/atft-problems/products/artifact-analysis/diagrams/figure_14_11_data.json"
with open(outpath, "w") as f:
    json.dump(extraction, f, indent=2, default=str)
print(f"\nSaved to {outpath}")
