"""Probe candidate sufficient conditions:
(a) every segment has <= 3 crossings          -> always colorable?
(b) triangle-free crossing graph              -> always colorable?
(c) does allowing 2 cuts per segment fix the 6-segment counterexample?
"""
import random, math
from itertools import combinations, product
from split_coloring_search import build, solve, random_segments

rng = random.Random(2026)

def max_deg(order): return max((len(o) for o in order), default=0)

def has_triangle(pairs, n):
    adj = [set() for _ in range(n)]
    for i, j in pairs: adj[i].add(j); adj[j].add(i)
    return any(adj[a] & adj[b] for a, b in pairs)

# (a) max degree <= 3
tested = bad = 0
for t in range(300000):
    n = rng.randint(6, 12)
    segs = random_segments(n, rng)
    pairs, order = build(segs)
    if not pairs or max_deg(order) > 3: continue
    if sum(len(o) for o in order) < 2 * (n - 1): continue  # want dense-ish
    tested += 1
    if solve(pairs, order) is None:
        bad += 1; print("(a) UNSAT with max degree 3!", segs); break
print(f"(a) max-3-crossings instances tested: {tested}, UNSAT: {bad}")

# (b) triangle-free crossing graph
tested = bad = 0
for t in range(300000):
    n = rng.randint(6, 14)
    segs = random_segments(n, rng)
    pairs, order = build(segs)
    if not pairs or has_triangle(pairs, n): continue
    if len(pairs) < n: continue  # want at least one cycle
    tested += 1
    if solve(pairs, order) is None:
        bad += 1; print("(b) UNSAT triangle-free!", segs); break
print(f"(b) triangle-free instances tested: {tested}, UNSAT: {bad}")

# (c) two cuts per segment on the counterexample
CE = [((0.51,0.71),(0.35,0.30)),((0.97,0.53),(0.38,0.61)),((0.69,0.24),(0.21,0.97)),
      ((0.33,0.50),(0.99,0.59)),((1.00,0.80),(0.51,0.14)),((0.43,0.86),(0.70,0.03))]
pairs, order = build(CE)
n = len(order); m = [len(o) for o in order]
idx = [dict() for _ in range(n)]
for i, o in enumerate(order):
    for p, c in enumerate(o): idx[i][c] = p
def color(i, pos, g1, g2, b): return b ^ (pos >= g1) ^ (pos >= g2)
found = None
for cuts in product(*[[(a, b) for a in range(m[i] + 1) for b in range(a, m[i] + 1)]
                      for i in range(n)]):
    for bvec in product((0, 1), repeat=n):
        if all(color(i, idx[i][c], *cuts[i], bvec[i]) != color(j, idx[j][c], *cuts[j], bvec[j])
               for c, (i, j) in enumerate(pairs)):
            found = (cuts, bvec); break
    if found: break
print("(c) counterexample with <=2 cuts/segment:", "SAT" if found else "UNSAT", found)
