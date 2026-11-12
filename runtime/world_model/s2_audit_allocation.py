"""Allocate observed eligible rows to case quotas without double-counting.

Identical candidate-to-case neighborhoods share one capacity node. Exact
maximum flow resolves competing cases; greedy assignment can strand a scarce
case even when a complete disjoint allocation exists.
"""

from collections import deque


def allocate_groups(groups, quotas):
    """Return an exact disjoint allocation from grouped matching row IDs.

    ``groups`` maps sorted case-ID tuples to distinct sorted row-ID sequences.
    Each row may contribute to at most one case. Quotas are upper bounds, so
    shortages remain visible rather than causing sampling with replacement.
    """
    seen = set()
    for cases, rows in groups.items():
        if not cases or set(cases) - set(quotas) or len(set(cases)) != len(cases):
            raise ValueError('candidate neighborhoods require distinct known case IDs')
        for row in rows:
            if row in seen:
                raise ValueError('a row cannot occur in two allocation groups')
            seen.add(row)
    if any(type(value) is not int or value < 0 for value in quotas.values()):
        raise ValueError('case quotas must be nonnegative integers')
    cases = sorted(quotas)
    neighborhoods = sorted(groups)
    case_nodes = {case: i + 1 for i, case in enumerate(cases)}
    first_group = len(cases) + 1
    sink = first_group + len(neighborhoods)
    graph = [[] for _ in range(sink + 1)]

    def edge(source, destination, capacity):
        forward = [destination, len(graph[destination]), capacity]
        backward = [source, len(graph[source]), 0]
        graph[source].append(forward)
        graph[destination].append(backward)
        return forward, capacity

    references = {}
    for position, neighborhood in enumerate(neighborhoods):
        node = first_group + position
        edge(0, node, len(groups[neighborhood]))
        for case in neighborhood:
            references[neighborhood, case] = edge(node, case_nodes[case], len(groups[neighborhood]))
    for case in cases:
        edge(case_nodes[case], sink, quotas[case])
    while True:
        parents = [None] * len(graph)
        parents[0] = (0, 0)
        queue = deque([0])
        while queue and parents[sink] is None:
            source = queue.popleft()
            for position, (destination, _, capacity) in enumerate(graph[source]):
                if capacity and parents[destination] is None:
                    parents[destination] = source, position
                    queue.append(destination)
                    if destination == sink:
                        break
        if parents[sink] is None:
            break
        amount, node = float('inf'), sink
        while node:
            source, position = parents[node]
            amount = min(amount, graph[source][position][2])
            node = source
        node = sink
        while node:
            source, position = parents[node]
            forward = graph[source][position]
            forward[2] -= amount
            graph[node][forward[1]][2] += amount
            node = source
    assigned = {case: [] for case in cases}
    for neighborhood in neighborhoods:
        rows, offset = sorted(groups[neighborhood]), 0
        for case in neighborhood:
            forward, original = references[neighborhood, case]
            count = original - forward[2]
            assigned[case].extend(rows[offset:offset + count])
            offset += count
    return assigned
