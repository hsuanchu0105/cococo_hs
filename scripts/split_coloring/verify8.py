"""Independent exact verification of the 8-segment counterexample.

- crossings computed with exact rational arithmetic (Fraction)
- exhaustive enumeration over ALL gap vectors, feasibility of base colors
  checked by simple BFS 2-coloring (independent of the DSU solver)
"""
from fractions import Fraction as F
from itertools import product

SEGS = [((0.7365061723166438, 0.25945206102856655), (0.6096769602892887, 0.9789674010972261)),
        ((0.5103640367391336, 0.7070661150428175), (0.3479669544108014, 0.2982739737729422)),
        ((0.9748603323386892, 0.5267232797897903), (0.3760880935062113, 0.6118746073427236)),
        ((0.16999480295852742, 0.08291057954114833), (0.501114584841771, 0.08033089706064545)),
        ((0.6911020016664421, 0.24089234197712328), (0.21283115795078722, 0.9658151274608557)),
        ((0.3276131911368697, 0.4954051684780252), (0.9875020309365504, 0.5943155969029232)),
        ((0.9963071965477264, 0.8020388871057784), (0.5069539389665562, 0.13657873413495503)),
        ((0.4300828082074646, 0.8556151169416012), (0.6964690550673079, 0.03003413563118429))]

segs = [((F(a), F(b)), (F(c), F(d))) for ((a, b), (c, d)) in SEGS]
n = len(segs)

pairs = []
per = [[] for _ in range(n)]
for i in range(n):
    for j in range(i + 1, n):
        (x1, y1), (x2, y2) = segs[i]
        (x3, y3), (x4, y4) = segs[j]
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3
        den = dx1 * dy2 - dy1 * dx2
        if den == 0:
            continue
        t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / den
        u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / den
        if 0 < t < 1 and 0 < u < 1:
            cid = len(pairs)
            pairs.append((i, j))
            per[i].append((t, cid))
            per[j].append((u, cid))

order = [[cid for _, cid in sorted(l)] for l in per]
m = [len(o) for o in order]
print("crossings per segment:", m, " total crossings:", len(pairs))

# sanity: no ties in parameters along any segment (general position)
for i, l in enumerate(per):
    ts = sorted(t for t, _ in l)
    assert all(ts[a] != ts[a + 1] for a in range(len(ts) - 1)), f"tie on segment {i}"

# no three segments concurrent: all crossing POINTS distinct
pts = set()
for i in range(n):
    (x1, y1), (x2, y2) = segs[i]
    for t, cid in per[i]:
        pts.add((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
print("distinct crossing points:", len(pts), "(should equal total crossings)")

idx = [dict() for _ in range(n)]
for i, o in enumerate(order):
    for pos, cid in enumerate(o):
        idx[i][cid] = pos

total = 1
for mi in m:
    total *= mi + 1
print("gap vectors to enumerate:", total)

def feasible(g):
    # constraints b_i xor b_j = 1 ^ [idx_i>=g_i] ^ [idx_j>=g_j]; BFS 2-color
    adj = [[] for _ in range(n)]
    for cid, (i, j) in enumerate(pairs):
        par = 1 ^ (idx[i][cid] >= g[i]) ^ (idx[j][cid] >= g[j])
        adj[i].append((j, par))
        adj[j].append((i, par))
    col = [None] * n
    for s in range(n):
        if col[s] is not None:
            continue
        col[s] = 0
        stack = [s]
        while stack:
            v = stack.pop()
            for w, par in adj[v]:
                want = col[v] ^ par
                if col[w] is None:
                    col[w] = want
                    stack.append(w)
                elif col[w] != want:
                    return False
    return True

count_sat = 0
for g in product(*[range(mi + 1) for mi in m]):
    if feasible(g):
        count_sat += 1
        print("SAT with gaps", g)
        break
if count_sat == 0:
    print("UNSAT CONFIRMED: no valid split coloring exists for these 8 segments")

# --- shrinking: is any 7-segment sub-arrangement also UNSAT? ---
def solve_subset(keep):
    ksegs = [SEGS[i] for i in keep]
    # rebuild with floats is fine for solving; but use exact again
    ss = [((F(a), F(b)), (F(c), F(d))) for ((a, b), (c, d)) in ksegs]
    nn = len(ss)
    prs = []
    pr = [[] for _ in range(nn)]
    for i in range(nn):
        for j in range(i + 1, nn):
            (x1, y1), (x2, y2) = ss[i]
            (x3, y3), (x4, y4) = ss[j]
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x4 - x3, y4 - y3
            den = dx1 * dy2 - dy1 * dx2
            if den == 0:
                continue
            t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / den
            u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / den
            if 0 < t < 1 and 0 < u < 1:
                cid = len(prs)
                prs.append((i, j))
                pr[i].append((t, cid))
                pr[j].append((u, cid))
    orr = [[cid for _, cid in sorted(l)] for l in pr]
    mm = [len(o) for o in orr]
    ix = [dict() for _ in range(nn)]
    for i, o in enumerate(orr):
        for pos, cid in enumerate(o):
            ix[i][cid] = pos

    def feas(g):
        adj = [[] for _ in range(nn)]
        for cid, (i, j) in enumerate(prs):
            par = 1 ^ (ix[i][cid] >= g[i]) ^ (ix[j][cid] >= g[j])
            adj[i].append((j, par))
            adj[j].append((i, par))
        col = [None] * nn
        for s in range(nn):
            if col[s] is not None:
                continue
            col[s] = 0
            st = [s]
            while st:
                v = st.pop()
                for w, par in adj[v]:
                    want = col[v] ^ par
                    if col[w] is None:
                        col[w] = want
                        st.append(w)
                    elif col[w] != want:
                        return False
        return True

    for g in product(*[range(mi + 1) for mi in mm]):
        if feas(g):
            return True  # SAT
    return False

print("\nsub-arrangements (drop one segment):")
for drop in range(n):
    keep = [i for i in range(n) if i != drop]
    print(f"  drop segment {drop}: {'SAT' if solve_subset(keep) else 'UNSAT'}")
