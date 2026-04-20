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


def test_oriented_edge_type_is_order_sensitive():
    from problems.hubble_tension_web.graph import oriented_edge_type_for_pair
    from problems.hubble_tension_web.types import Environment
    t_vw = oriented_edge_type_for_pair(Environment.VOID, Environment.WALL)
    t_wv = oriented_edge_type_for_pair(Environment.WALL, Environment.VOID)
    assert t_vw == "void-wall"
    assert t_wv == "wall-void"
    assert t_vw != t_wv   # asymmetry is the point


def test_edge_types_constant_covers_all_ordered_pairs():
    from problems.hubble_tension_web.graph import EDGE_TYPES
    from problems.hubble_tension_web.types import Environment
    n_envs = len(list(Environment))
    assert len(EDGE_TYPES) == n_envs * n_envs   # 16 for 4 envs
    assert "void-wall" in EDGE_TYPES and "wall-void" in EDGE_TYPES


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
