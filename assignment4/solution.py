"""
CS 536 - Assignment 4: Maximum Concurrent Flow and Network Design
Part 2 - Arbitrary (hose-model) traffic MILP, implemented with the Gurobi
Python API.

Given a traffic matrix T in the hose polytope on n=8 nodes (row/col sums <= 4,
T_ii = 0), this script jointly chooses a directed d-regular topology
(d=4) and a multi-commodity flow routing that maximize the concurrent-flow
scalar alpha such that alpha * T is routable on the chosen topology.

Modelling note (aggregation by destination)
-------------------------------------------
A standard equivalence for multi-commodity flow lets us collapse the
per-pair variable f[s,t,i,j] into a per-destination variable
g[t,i,j] = sum_s f[s,t,i,j]. Any feasible g can be decomposed back into
per-source flows, so the optimal alpha is identical. This cuts the LP size
by a factor of n (here, 8x), which lets the model fit inside Gurobi's free
restricted license. With an unrestricted academic license the per-pair
formulation would work equally well -- we document the equivalence in the
report.

Usage:
    python3 solution.py              # runs three sample matrices
    python3 solution.py --help
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Core MILP solver
# ---------------------------------------------------------------------------
@dataclass
class SolverResult:
    alpha: float
    adjacency: np.ndarray            # shape (n, n), integer -- # parallel links
    dest_flows: Dict[int, np.ndarray]  # destination t -> flow matrix (n,n)
    model_status: int


def solve_best_topology(
    T: np.ndarray,
    d: int = 4,
    allow_parallel_links: bool = True,
    time_limit_s: float = 60.0,
    verbose: bool = False,
) -> SolverResult:
    """Solve the topology-and-routing MILP for a given traffic matrix T.

    Decision variables
    ------------------
    x[i,j]   : integer in [0, d], # directed links from i -> j (x[i,i] = 0)
    g[t,i,j] : continuous >= 0, aggregated flow towards destination t on (i,j)
    alpha    : continuous >= 0, concurrent-flow scalar (objective)

    Constraints
    -----------
    * x is a d-regular nonnegative integer matrix with zero diagonal
    * Per-edge capacity: sum_t g[t,i,j] <= x[i,j]
    * Flow conservation for each destination t:
        - At every source v != t:   out(g_t, v) - in(g_t, v) = alpha * T[v,t]
        - At destination t:         in(g_t, t) - out(g_t, t) = alpha * sum_v T[v,t]
    """
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    assert T.shape == (n, n)
    assert np.allclose(np.diag(T), 0.0), "T_ii must be 0"

    nodes = range(n)
    # Active destinations: anyone who receives some traffic
    dests: List[int] = [t for t in nodes if T[:, t].sum() > 0]

    m = gp.Model("best_topology")
    if not verbose:
        m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit_s

    # --- topology variables ---
    x_ub = d if allow_parallel_links else 1
    x = m.addVars(nodes, nodes, vtype=GRB.INTEGER, lb=0, ub=x_ub, name="x")

    # --- destination-aggregated flow variables ---
    g = m.addVars(dests, nodes, nodes, lb=0.0, name="g")

    # --- alpha ---
    alpha = m.addVar(lb=0.0, name="alpha")

    # --- no self-loops in topology, and no flow on self-links ---
    for i in nodes:
        m.addConstr(x[i, i] == 0, name=f"no_self_{i}")
    for t in dests:
        for i in nodes:
            m.addConstr(g[t, i, i] == 0, name=f"no_self_g_{t}_{i}")

    # --- d-regularity (in and out) ---
    for i in nodes:
        m.addConstr(
            gp.quicksum(x[i, j] for j in nodes) == d, name=f"outdeg_{i}"
        )
        m.addConstr(
            gp.quicksum(x[j, i] for j in nodes) == d, name=f"indeg_{i}"
        )

    # --- capacity: sum_t g[t,i,j] <= x[i,j] ---
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            m.addConstr(
                gp.quicksum(g[t, i, j] for t in dests) <= x[i, j],
                name=f"cap_{i}_{j}",
            )

    # --- flow conservation per destination ---
    for t in dests:
        total_in = float(T[:, t].sum())  # sum_v T[v,t]
        for v in nodes:
            out_v = gp.quicksum(g[t, v, w] for w in nodes if w != v)
            in_v = gp.quicksum(g[t, u, v] for u in nodes if u != v)
            if v == t:
                # destination absorbs total demand
                m.addConstr(in_v - out_v == alpha * total_in,
                            name=f"cons_{t}_dst")
            else:
                # source v injects its own demand alpha * T[v,t]
                m.addConstr(out_v - in_v == alpha * T[v, t],
                            name=f"cons_{t}_{v}")

    # --- objective ---
    m.setObjective(alpha, GRB.MAXIMIZE)
    m.optimize()

    # --- unpack solution ---
    adj = np.zeros((n, n), dtype=int)
    for i in nodes:
        for j in nodes:
            adj[i, j] = int(round(x[i, j].X))

    flows: Dict[int, np.ndarray] = {}
    for t in dests:
        F = np.zeros((n, n), dtype=float)
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                F[i, j] = g[t, i, j].X
        flows[t] = F

    return SolverResult(
        alpha=float(alpha.X),
        adjacency=adj,
        dest_flows=flows,
        model_status=m.Status,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_solution(
    T: np.ndarray, result: SolverResult, d: int = 4, tol: float = 1e-5
) -> None:
    """Sanity-check the LP output. Raises AssertionError on any violation."""
    n = T.shape[0]
    adj = result.adjacency
    alpha = result.alpha

    # Degree
    assert (adj.sum(axis=0) == d).all(), (
        f"in-degree violated: {adj.sum(axis=0)}"
    )
    assert (adj.sum(axis=1) == d).all(), (
        f"out-degree violated: {adj.sum(axis=1)}"
    )
    assert (np.diag(adj) == 0).all(), "self-loops present"

    # Capacity
    total_flow = np.zeros((n, n))
    for F in result.dest_flows.values():
        total_flow += F
    assert (total_flow <= adj + tol).all(), "capacity violated"

    # Conservation per destination
    for t, F in result.dest_flows.items():
        net_out = F.sum(axis=1) - F.sum(axis=0)  # out - in per node
        expected = np.zeros(n)
        for v in range(n):
            if v == t:
                expected[v] = -alpha * T[:, t].sum()
            else:
                expected[v] = alpha * T[v, t]
        assert np.allclose(net_out, expected, atol=tol), (
            f"conservation violated for destination {t}:\n"
            f"  net_out = {net_out}\n  expected = {expected}"
        )


# ---------------------------------------------------------------------------
# Example traffic matrices
# ---------------------------------------------------------------------------
def hose_uniform(n: int = 8, d: int = 4) -> np.ndarray:
    """T_ij = d/(n-1) off-diagonal; saturates row and column hose budgets."""
    T = np.full((n, n), d / (n - 1))
    np.fill_diagonal(T, 0.0)
    return T


def cyclic_permutation(n: int = 8, d: int = 4) -> np.ndarray:
    """T = d * cyclic-shift permutation; pure matching demand."""
    T = np.zeros((n, n))
    for i in range(n):
        T[i, (i + 1) % n] = float(d)
    return T


def mixed_matching(n: int = 8, d: int = 4) -> np.ndarray:
    """Two heavy flows + uniform background, still in the hose polytope."""
    T = np.zeros((n, n))
    # background
    bg = 1.0 / (n - 1)  # tiny uniform demand
    T[:] = bg
    np.fill_diagonal(T, 0.0)
    # two heavy flows, carefully chosen so row/col sums stay <= d
    heavy = [(0, 3, 3.0), (5, 2, 3.0)]
    for (s, t, w) in heavy:
        T[s, t] += w
    # Scale if needed so the row/column budget is respected
    row_max = T.sum(axis=1).max()
    col_max = T.sum(axis=0).max()
    s = min(d / row_max, d / col_max)
    T *= s
    return T


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_adjacency(adj: np.ndarray, label: str = "") -> None:
    n = adj.shape[0]
    print(f"\nAdjacency x (rows = source, cols = destination){' '+label if label else ''}:")
    print("    " + " ".join(f"{j:>2}" for j in range(n)))
    for i in range(n):
        print(f"{i:>2}: " + " ".join(f"{adj[i,j]:>2}" for j in range(n)))
    print(f"   row sums (out-deg): {adj.sum(axis=1).tolist()}")
    print(f"   col sums (in-deg) : {adj.sum(axis=0).tolist()}")


def summarize(name: str, T: np.ndarray, result: SolverResult) -> None:
    print("=" * 72)
    print(f"Scenario: {name}")
    print("-" * 72)
    print(f"Traffic matrix T (row sums = {T.sum(axis=1).round(3).tolist()}):")
    for row in T:
        print("  " + " ".join(f"{v:5.2f}" for v in row))
    print(f"\n  alpha*  = {result.alpha:.6f}")
    print_adjacency(result.adjacency)
    # routed demand check
    served = sum(F.sum() for F in result.dest_flows.values())
    # Count unique edges carrying flow (union across all destinations)
    edges_with_flow = np.zeros(result.adjacency.shape, dtype=bool)
    for F in result.dest_flows.values():
        edges_with_flow |= (F > 1e-9)
    unique_edges = edges_with_flow.sum()
    print(f"  total flow (sum |g|): {served:.3f}")
    print(f"  edges carrying flow : {int(unique_edges)} / "
          f"{int(result.adjacency.sum())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-parallel", action="store_true",
                   help="forbid parallel links (x_ij in {0,1})")
    p.add_argument("--verbose", action="store_true",
                   help="print Gurobi solver log")
    p.add_argument("--time-limit", type=float, default=60.0)
    args = p.parse_args()

    scenarios = [
        ("Cyclic permutation x 4 (integer hose)", cyclic_permutation()),
        ("Hose-uniform T_ij = 4/7",               hose_uniform()),
        ("Mixed matching + uniform background",   mixed_matching()),
    ]

    for name, T in scenarios:
        result = solve_best_topology(
            T,
            d=4,
            allow_parallel_links=not args.no_parallel,
            time_limit_s=args.time_limit,
            verbose=args.verbose,
        )
        verify_solution(T, result, d=4)
        summarize(name, T, result)


if __name__ == "__main__":
    main()
