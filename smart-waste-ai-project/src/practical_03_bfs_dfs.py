"""Practical 3 — BFS and DFS for a rescue robot (unweighted grid).

Finds a path from a start cell to a target cell in a building grid with
blocked cells (obstacles).  BFS guarantees a shortest path in an unweighted
graph; DFS does not.
"""

from __future__ import annotations

from collections import deque

# Legend: '.' = free, '#' = blocked, 'S' = start, 'T' = target
GRID = [
    ["S", ".", ".", "#", "."],
    ["#", "#", ".", "#", "."],
    [".", ".", ".", ".", "."],
    [".", "#", "#", "#", "."],
    [".", ".", ".", ".", "T"],
]

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find(grid, symbol):
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == symbol:
                return (r, c)
    return None


def _neighbors(grid, pos):
    r, c = pos
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] != "#":
                yield (nr, nc)


def bfs(grid):
    """Breadth First Search. Returns (path, steps) or (None, None)."""
    start, target = _find(grid, "S"), _find(grid, "T")
    if start is None or target is None:
        return None, None
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos == target:
            return path, len(path) - 1
        for nb in _neighbors(grid, pos):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None, None


def dfs(grid):
    """Depth First Search. Returns (path, steps) or (None, None)."""
    start, target = _find(grid, "S"), _find(grid, "T")
    if start is None or target is None:
        return None, None
    stack = [(start, [start])]
    visited = {start}
    while stack:
        pos, path = stack.pop()
        if pos == target:
            return path, len(path) - 1
        for nb in reversed(list(_neighbors(grid, pos))):
            if nb not in visited:
                visited.add(nb)
                stack.append((nb, path + [nb]))
    return None, None


def render(grid, path=None):
    """Return a textual rendering of the grid with an optional path marked."""
    display = [row[:] for row in grid]
    if path:
        for r, c in path[1:-1]:
            display[r][c] = "*"
    return "\n".join(" ".join(row) for row in display)


def run(grid=None):
    grid = grid or GRID
    bfs_path, bfs_steps = bfs(grid)
    dfs_path, dfs_steps = dfs(grid)
    print("Building grid:")
    print(render(grid))
    print("\nBFS path:", bfs_path)
    print("BFS steps:", bfs_steps)
    print("DFS path:", dfs_path)
    print("DFS steps:", dfs_steps)
    print("\nBFS grid with path:")
    print(render(grid, bfs_path))
    print("\nDFS grid with path:")
    print(render(grid, dfs_path))
    return (bfs_path, bfs_steps), (dfs_path, dfs_steps)


if __name__ == "__main__":
    run()
