"""Search for a counterexample to the one-cut two-coloring problem for segment
arrangements.

Model: each segment i gets a cut gap g_i in {0..m_i} (m_i = #crossings on it)
and a base color b_i; the color of segment i at its t-th crossing is
b_i XOR [t >= g_i].  Valid iff at every crossing the two segments differ.

For fixed gaps the b_i satisfy XOR constraints -> parity union-find.
We backtrack over gaps (segments ordered by degree desc).
"""
import random, math, sys
from itertools import combinations


def seg_cross(a, b, c, d):
    (x1, y1), (x2, y2) = a, b
    (x3, y3), (x4, y4) = c, d
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    den = dx1 * dy2 - dy1 * dx2
    if abs(den) < 1e-12:
        return None
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / den
    u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / den
    if 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9:
        return (t, u)
    return None


def build(segs):
    n = len(segs)
    per = [[] for _ in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = seg_cross(*segs[i], *segs[j])
            if r:
                cid = len(pairs)
                pairs.append((i, j))
                per[i].append((r[0], cid))
                per[j].append((r[1], cid))
    order = [[cid for _, cid in sorted(l)] for l in per]
    return pairs, order


class DSU:
    __slots__ = ("p", "r")

    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n  # parity to parent

    def find_par(self, x):
        # returns (root, parity from x to root), with path compression
        stack = []
        while self.p[x] != x:
            stack.append(x)
            x = self.p[x]
        root = x
        par = 0
        # process from the node nearest the root down to x itself,
        # accumulating parity to the root
        for y in reversed(stack):
            par ^= self.r[y]
            self.p[y] = root
            self.r[y] = par
        return root, par  # par == parity of stack[0] == original x (0 if x==root)

    def union(self, a, b, parity):
        ra, pa = self.find_par(a)
        rb, pb = self.find_par(b)
        if ra == rb:
            return (pa ^ pb) == parity
        self.p[ra] = rb
        self.r[ra] = pa ^ pb ^ parity
        return True

    def copy(self):
        d = DSU(0)
        d.p = self.p[:]
        d.r = self.r[:]
        return d


class NodeLimit(Exception):
    pass


def solve(pairs, order, node_limit=None):
    """Return solution (gaps dict) or None if UNSAT. Raises NodeLimit."""
    n = len(order)
    m = [len(o) for o in order]
    idx = [dict() for _ in range(n)]
    for i, o in enumerate(order):
        for pos, cid in enumerate(o):
            idx[i][cid] = pos
    nbr = [[] for _ in range(n)]
    for cid, (i, j) in enumerate(pairs):
        nbr[i].append((j, cid))
        nbr[j].append((i, cid))
    perm = sorted(range(n), key=lambda i: -m[i])
    g = [None] * n
    nodes = 0

    def rec(k, dsu):
        nonlocal nodes
        if k == n:
            return True
        s = perm[k]
        for gv in range(m[s] + 1):
            nodes += 1
            if node_limit and nodes > node_limit:
                raise NodeLimit()
            d2 = dsu.copy()
            ok = True
            for (j, cid) in nbr[s]:
                if g[j] is None:
                    continue
                par = 1 ^ (idx[s][cid] >= gv) ^ (idx[j][cid] >= g[j])
                if not d2.union(s, j, par):
                    ok = False
                    break
            if ok:
                g[s] = gv
                if rec(k + 1, d2):
                    return True
                g[s] = None
        return False

    if rec(0, DSU(n)):
        return list(g)
    return None


# ---------- instance generators ----------

def random_segments(n, rng):
    return [((rng.random(), rng.random()), (rng.random(), rng.random()))
            for _ in range(n)]


def tangent_lines(angles, R=100.0):
    """Segments tangent to unit circle at given angles, long enough to
    contain all pairwise crossings."""
    segs = []
    for th in angles:
        px, py = math.cos(th), math.sin(th)  # tangent point
        dx, dy = -math.sin(th), math.cos(th)  # direction
        segs.append(((px - R * dx, py - R * dy), (px + R * dx, py + R * dy)))
    return segs


def main():
    rng = random.Random(12345)
    unsat = []

    # 1) evenly spaced tangent lines, n = 3..9
    for n in range(3, 10):
        angles = [2 * math.pi * i / n + 0.01 * math.sin(i) for i in range(n)]
        pairs, order = build(tangent_lines(angles))
        try:
            sol = solve(pairs, order, node_limit=20_000_000)
        except NodeLimit:
            print(f"even tangents n={n}: NODE LIMIT")
            continue
        print(f"even tangents n={n}: {'SAT' if sol else 'UNSAT'}")
        if not sol:
            unsat.append(("even tangents", n, angles))

    # 2) random tangent angle sets
    for n in range(4, 9):
        trials = 2000 if n <= 6 else (400 if n == 7 else 60)
        bad = 0
        for t in range(trials):
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))
            # reject near-degenerate
            ok = all((angles[(i+1) % n] - angles[i]) % (2*math.pi) > 1e-3
                     for i in range(n))
            if not ok:
                continue
            pairs, order = build(tangent_lines(angles))
            try:
                sol = solve(pairs, order, node_limit=20_000_000)
            except NodeLimit:
                print(f"tangent n={n} trial {t}: NODE LIMIT")
                continue
            if not sol:
                bad += 1
                unsat.append(("tangent", n, angles))
                print(f"tangent n={n} trial {t}: UNSAT!  angles={angles}")
        print(f"random tangents n={n}: {trials} trials, {bad} UNSAT")

    # 3) random segments in unit square
    for n in range(4, 10):
        trials = 3000 if n <= 6 else (800 if n <= 8 else 200)
        bad = 0
        for t in range(trials):
            segs = random_segments(n, rng)
            pairs, order = build(segs)
            try:
                sol = solve(pairs, order, node_limit=20_000_000)
            except NodeLimit:
                print(f"segments n={n} trial {t}: NODE LIMIT")
                continue
            if not sol:
                bad += 1
                unsat.append(("segments", n, segs))
                print(f"segments n={n} trial {t}: UNSAT!  segs={segs}")
        print(f"random segments n={n}: {trials} trials, {bad} UNSAT")

    print("\nTotal UNSAT found:", len(unsat))
    if unsat:
        import json
        with open("unsat_instances.json", "w") as f:
            json.dump([(k, n, v) for (k, n, v) in unsat], f, default=list)


if __name__ == "__main__":
    main()
