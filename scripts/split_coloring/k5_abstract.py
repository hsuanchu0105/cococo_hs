"""Exhaustive check: ALL abstract 5-segment configurations where every pair
crosses exactly once. Vertex 0's partner order fixed WLOG (relabeling).
An abstract configuration = per segment, an order of its 4 partners.
This is a superset of geometrically realizable configurations, so if all are
SAT, every real complete 5-segment arrangement is SAT."""
from itertools import permutations, product
from split_coloring_search import solve

others = [tuple(j for j in range(5) if j != i) for i in range(5)]
perms4 = list(permutations(range(4)))

# crossing ids: pair (i,j) i<j -> cid
pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
cid = {frozenset(p): k for k, p in enumerate(pairs)}

unsat_found = 0
count = 0
fixed0 = others[0]  # order (1,2,3,4) for segment 0
for p1 in perms4:
    o1 = tuple(others[1][k] for k in p1)
    for p2 in perms4:
        o2 = tuple(others[2][k] for k in p2)
        for p3 in perms4:
            o3 = tuple(others[3][k] for k in p3)
            for p4 in perms4:
                o4 = tuple(others[4][k] for k in p4)
                orders_partners = [fixed0, o1, o2, o3, o4]
                order = [[cid[frozenset((i, j))] for j in op]
                         for i, op in enumerate(orders_partners)]
                count += 1
                if solve(pairs, order) is None:
                    unsat_found += 1
                    print("ABSTRACT UNSAT:", orders_partners, flush=True)
    print(f"progress: outer {p1} done, {count} checked, {unsat_found} UNSAT", flush=True)
print(f"DONE: {count} abstract configurations, {unsat_found} UNSAT")
