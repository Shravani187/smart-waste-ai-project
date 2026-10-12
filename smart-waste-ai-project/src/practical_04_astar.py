"""Practical 4 — A* informed search for emergency navigation.

Grid with obstacles and movement costs.  Uses f(n) = g(n) + h(n) with the
Manhattan-distance heuristic (suitable for four-directional movement).
"""

from __future__ import annotations

import heapq

# 0 = free (cost 1), 1 = obstacle, 2 = rough terrain (cost 3)
GRID = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 2, 2, 1, 0],
    [0, 0, 0, 0, 0],
]
START = (0, 0)
TARGET = (4, 4)
MOVE_COST = {0: 1, 2: 3}
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, target):
    """Return (path, total_cost) or (None, None) if unreachable."""
    if grid[target[0]][target[1]] == 1:
        return None, None
    open_heap = [(heuristic(start, target), 0, start, [start])]
    best_g = {start: 0}
    visited = set()
    while open_heap:
        f, g, pos, path = heapq.heappop(open_heap)
        if pos in visited:
            continue
        visited.add(pos)
        if pos == target:
            return path, g
        for dr, dc in DIRECTIONS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                cell = grid[nr][nc]
                if cell == 1:
                    continue
                ng = g + MOVE_COST[cell]
                npos = (nr, nc)
                if ng < best_g.get(npos, float("inf")):
                    best_g[npos] = ng
                    heapq.heappush(
                        open_heap, (ng + heuristic(npos, target), ng, npos, path + [npos])
                    )
    return None, None


def dijkstra(grid, start, target):
    """Baseline: A* with h(n)=0 reduces to Dijkstra's algorithm."""
    if grid[target[0]][target[1]] == 1:
        return None, None
    heap = [(0, start, [start])]
    visited = set()
    while heap:
        g, pos, path = heapq.heappop(heap)
        if pos in visited:
            continue
        visited.add(pos)
        if pos == target:
            return path, g
        for dr, dc in DIRECTIONS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                cell = grid[nr][nc]
                if cell == 1:
                    continue
                heapq.heappush(heap, (g + MOVE_COST[cell], (nr, nc), path + [(nr, nc)]))
    return None, None


def render(grid, path=None):
    symbols = {0: ".", 1: "#", 2: "~"}
    display = [[symbols[c] for c in row] for row in grid]
    if path:
        for r, c in path[1:-1]:
            display[r][c] = "*"
    return "\n".join(" ".join(row) for row in display)


def run(grid=None, start=None, target=None):
    grid = grid or GRID
    start = start or START
    target = target or TARGET
    path, cost = astar(grid, start, target)
    dpath, dcost = dijkstra(grid, start, target)
    print("Grid (.=free, #=block, ~=rough cost 3):")
    print(render(grid))
    print("\nA* path:", path)
    print("A* total cost:", cost)
    print("Dijkstra path:", dpath)
    print("Dijkstra total cost:", dcost)
    print("\nA* grid with path:")
    print(render(grid, path))
    return (path, cost), (dpath, dcost)


if __name__ == "__main__":
    run()
