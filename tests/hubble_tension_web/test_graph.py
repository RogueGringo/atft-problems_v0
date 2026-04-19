import numpy as np
import pytest


def test_build_typed_graph_produces_typed_edges():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph, EDGE_TYPES
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    envs = [Environment.VOID, Environment.WALL, Environment.WALL, Environment.FILAMENT]
    web = LocalCosmicWeb(positions=positions, environments=envs)
    nodes, edges = build_typed_graph(web, k=2)
    assert nodes == 4
    for src, dst, etype in edges:
        assert 0 <= src < 4 and 0 <= dst < 4
        assert etype in EDGE_TYPES


def test_edge_types_are_ordered_pair_of_environments():
    from problems.hubble_tension_web.graph import edge_type_for_pair
    from problems.hubble_tension_web.types import Environment
    t = edge_type_for_pair(Environment.WALL, Environment.VOID)
    assert t == "void-wall" or t == "wall-void"
    t2 = edge_type_for_pair(Environment.VOID, Environment.WALL)
    assert t == t2   # deterministic, symmetric


def test_graph_is_connected_for_large_k():
    from problems.hubble_tension_web.types import LocalCosmicWeb, Environment
    from problems.hubble_tension_web.graph import build_typed_graph, to_adjacency
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(30, 3))
    envs = [Environment.VOID] * 30
    web = LocalCosmicWeb(positions=positions, environments=envs)
    _, edges = build_typed_graph(web, k=8)
    A = to_adjacency(30, edges)
    reached = {0}
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for v in range(30):
                if A[u, v] and v not in reached:
                    reached.add(v); nxt.append(v)
        frontier = nxt
    assert len(reached) == 30
