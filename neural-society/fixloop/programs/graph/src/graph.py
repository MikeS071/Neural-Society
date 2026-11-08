"""Unweighted undirected graphs as adjacency dicts."""

from collections import deque


def build(edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def shortest_path(adj, start, goal):
    """List of nodes from start to goal (inclusive), or None if unreachable."""
    if start == goal:
        return [start]
    prev = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in sorted(adj.get(node, ())):
            if nxt in prev:
                continue
            prev[nxt] = node
            if nxt == goal:
                path = [goal]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            queue.append(nxt)
    return None


def components(adj):
    """Connected components as sorted lists, ordered by their smallest node."""
    seen = set()
    out = []
    for start in sorted(adj):
        if start in seen:
            continue
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out.append(sorted(comp))
    return out
