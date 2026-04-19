# Astronomy Unsolved Problems — ATFT Triage Design

**Date:** 2026-04-19
**Status:** Approved design, ready for plan.
**Author:** Blake Jones (with Claude)

## Goal

Produce a triage of astronomy's open problems ranked by how well the ATFT toolbox in this repo (persistent homology, typed sheaf Laplacians, onset scale ε*, {0,1,3} crystal kernel, cross-domain isomorphism analyzer) can attack them. The artifact is a decision aid for picking the next `problems/<slug>` project, modeled on the existing Millennium-problem write-ups.

## Source

- **Primary:** the current Wikipedia page *List of unsolved problems in astronomy*, fetched live via WebFetch at run time.
- **Augmentation:** problems the Wikipedia page omits but are relevant to this repo's toolbox — flagged with `"source": "augmented"` and a reason. Expected augmentations include specific dark-matter substructure tests, FRB repetition-pattern questions, and coronal reconnection topology questions. Final augmentation list is decided during Phase 1 execution.

## Pipeline

```
[Fetch Wikipedia page + apply augmentations]
        │
[PHASE 1 — Pre-filter]
  binary pass per problem: kept or cut with one-line reason
        │
        ▼
  ── GATE ──  survivors shown to user; user vetoes/adds before scoring
        │
[PHASE 2 — Deep-score survivors on 5 axes]
        │
[Rank; write top-3 ATFT translation sketches]
        │
[Emit astronomy_triage.md + astronomy_triage.json]
```

The gate is the control point — the user decides which problems deserve deep-scoring effort before it is spent.

## Phase 1 — Pre-filter criteria

A problem is **kept** iff it passes all three binary checks:

1. **Structured data exists or is simulatable.** Observational catalogs, lightcurves, sky maps, GW strain, redshift surveys, or a physically meaningful simulation regime producing point clouds, fields, or graphs. Cuts pure-philosophy or zero-handle questions (e.g., "what was before the Big Bang").
2. **A control parameter or ordering variable is identifiable.** Time, redshift, mass, temperature, Reynolds-analog, α-analog — something to sweep along, the way Navier-Stokes sweeps Re or SAT sweeps α. Cuts "why is X its value" coincidence problems with no sweep axis.
3. **A topological/geometric observable is plausibly informative.** The question is about structure, connectivity, phase change, clustering, singularity, symmetry, or anomaly — not about a single scalar. "What is H₀" fails; "why does the H₀ tension persist across independent probes" passes, because it is a consistency-of-structure question.

Each cut is annotated with the single criterion it failed plus a one-line reason. Survivors pass to Phase 2.

## Phase 2 — Deep-score rubric

Each survivor scored on 5 axes, **0–3 integer per axis**, total 0–15.

| Axis | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **Data availability** | no useful data | sparse / indirect | good public catalogs | rich + simulation-matched |
| **Order parameter clarity** | none obvious | exists but contested | well-defined single variable | natural family (like Re) |
| **Topological signature plausibility** | speculative | metaphor only | structure-level argument | near-isomorphic to a solved / studied case |
| **Field impact if moved** | niche | sub-field | field-wide | paradigm-level |
| **Novelty vs existing methods** | already standard | marginal | meaningful new angle | tools see something current methods can't |

**Isomorphism flag** (separate from the total, not summed):

- `0` — no match to any existing `problems/<slug>` entry.
- `1` — loose analogy.
- `2` — tight structural match (existing instrument likely transfers with minimal new code).

**Rationale:** 1–3 sentences per survivor explaining the scores, citing specific axes.

**Ranking:** primary sort by total (desc), tiebreak by isomorphism flag (desc).

## Outputs

Two files written, both under `docs/`:

### `docs/astronomy_triage.md`

Human-readable narrative. Sections in order:

1. **Methodology** — ~200 words. Source, filter criteria, rubric.
2. **Phase 1 — cut list.** Table: `problem | failed-criterion | one-line-reason`.
3. **Phase 2 — survivor scoreboard.** Table sorted by total: `rank | problem | data | order-param | topology | impact | novelty | total | iso-flag`.
4. **Top-3 translation sketches.** For the top 3 survivors only, a ~300-word translation into ATFT language following the shape of `problems/navier-stokes/README.md`: point cloud / control parameter / sheaf / detection target / what success looks like. These are the "pick one and go" candidates.
5. **Appendix — augmentations beyond Wikipedia.** The problems added, with reasons.

### `docs/astronomy_triage.json`

Structured, one record per problem (both cut and kept).

```json
{
  "id": "coronal-heating",
  "title": "Why is the solar corona hotter than the surface?",
  "category": "stellar",
  "source": "wikipedia",
  "phase1": {
    "kept": true,
    "failed_criterion": null,
    "reason": "Passes: SDO/Hinode data, temperature-vs-height ordering variable, reconnection-topology observable."
  },
  "phase2": {
    "scores": {"data": 3, "order_param": 2, "topology": 3, "impact": 2, "novelty": 3},
    "total": 13,
    "isomorphism_flag": 1,
    "isomorphism_ref": "navier-stokes",
    "rationale": "..."
  }
}
```

- `source` ∈ `"wikipedia"`, `"augmented"`.
- `phase1.failed_criterion` ∈ `1`, `2`, `3`, `null`.
- `phase2` is `null` for cut problems.
- `phase2.isomorphism_ref` is the `problems/<slug>` name or `null`.

### What is *not* produced

- **No** per-survivor `problems/<slug>/README.md` stubs. The top-3 sketches live inside the markdown only. Stubbing out survivor directories is a separate follow-up the user can trigger by picking one and invoking a new design pass.

## Non-goals

- Not running any ATFT experiment during this triage. No persistent homology, no sheaf computation, no simulation. Triage is desk work: judgment against the rubric, nothing more.
- Not committing to tackle any specific problem. The triage informs the next choice; the choice itself is a separate decision.
- Not a literature review. Translation sketches cite reasoning, not papers.
