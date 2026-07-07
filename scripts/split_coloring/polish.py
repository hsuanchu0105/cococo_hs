"""(a) round the 6-segment counterexample to 2-decimal coordinates and
re-verify UNSAT exactly; (b) test whether triangle parity constraints alone
are infeasible; (c) plot."""
from fractions import Fraction as F
from itertools import product, combinations

RAW = [((0.5103640367391336, 0.7070661150428175), (0.3479669544108014, 0.2982739737729422)),
       ((0.9748603323386892, 0.5267232797897903), (0.3760880935062113, 0.6118746073427236)),
       ((0.6911020016664421, 0.24089234197712328), (0.21283115795078722, 0.9658151274608557)),
       ((0.3276131911368697, 0.4954051684780252), (0.9875020309365504, 0.5943155969029232)),
       ((0.9963071965477264, 0.8020388871057784), (0.5069539389665562, 0.13657873413495503)),
       ((0.4300828082074646, 0.8556151169416012), (0.6964690550673079, 0.03003413563118429))]


def build(ss):
    nn = len(ss)
    prs, pr = [], [[] for _ in range(nn)]
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
    ix = [dict() for _ in range(nn)]
    for i, o in enumerate(orr):
        for pos, cid in enumerate(o):
            ix[i][cid] = pos
    return nn, prs, [len(o) for o in orr], ix, orr


def unsat(ss):
    nn, prs, mm, ix, _ = build(ss)

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

    return not any(feas(g) for g in product(*[range(mi + 1) for mi in mm]))


# ---- (a) round to 2 decimals, exact verify ----
for dec in (2, 3):
    q = 10 ** dec
    rounded = [((F(round(a * q), q), F(round(b * q), q)),
                (F(round(c * q), q), F(round(d * q), q)))
               for ((a, b), (c, d)) in RAW]
    nn, prs, mm, ix, orr = build(rounded)
    ok = len(prs) == 14 and unsat(rounded)
    print(f"rounded to {dec} decimals: crossings={len(prs)}, "
          f"{'UNSAT (counterexample survives)' if ok else 'NOT a counterexample'}")
    if ok:
        best = rounded
        break

nn, prs, mm, ix, orr = build(best)
print("\nfinal segments (exact rationals):")
for i, ((a, b), (c, d)) in enumerate(best):
    print(f"  S{i+1}: ({float(a):.2f},{float(b):.2f}) -- ({float(c):.2f},{float(d):.2f})")
print("crossings per segment:", mm)
print("crossing order along each segment (crossing id -> partner):")
part = {cid: (i, j) for cid, (i, j) in enumerate(prs)}
for i, o in enumerate(orr):
    partners = [(set(part[cid]) - {i}).pop() + 1 for cid in o]
    print(f"  along S{i+1}: crosses S{partners}")

# ---- (b) triangle-parity-only infeasibility? ----
# triangle {a,b,c}: parity condition = odd # of members switch strictly
# between their two triangle crossings.
tris = []
pairset = {frozenset(p): cid for cid, p in enumerate(prs)}
for a, b, c in combinations(range(nn), 3):
    if all(frozenset(x) in pairset for x in ((a, b), (a, c), (b, c))):
        tris.append((a, b, c))
print(f"\n#pairwise-crossing triangles: {len(tris)}")

def tri_ok(g):
    for (a, b, c) in tris:
        cnt = 0
        for (x, y, z) in ((a, b, c), (b, a, c), (c, a, b)):
            c1 = pairset[frozenset((x, y))]
            c2 = pairset[frozenset((x, z))]
            i1, i2 = ix[x][c1], ix[x][c2]
            lo, hi = min(i1, i2), max(i1, i2)
            if lo < g[x] <= hi:  # switch strictly between the two crossings
                cnt += 1
        if cnt % 2 == 0:
            return False
    return True

tri_feasible = any(tri_ok(g) for g in product(*[range(mi + 1) for mi in mm]))
print("triangle parity conditions alone:",
      "still satisfiable (need longer cycles for the proof)" if tri_feasible
      else "ALREADY INFEASIBLE -> triangles alone prove the counterexample")

# ---- (c) plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axp = plt.subplots(figsize=(7, 7))
cols = plt.cm.tab10.colors
for i, ((a, b), (c, d)) in enumerate(best):
    axp.plot([a, c], [b, d], color=cols[i], lw=2, label=f"S{i+1}")
    axp.annotate(f"S{i+1}", ((float(a) + float(c)) / 2 + 0.01,
                             (float(b) + float(d)) / 2 + 0.01),
                 color=cols[i], fontsize=11, fontweight="bold")
# crossings
for cid, (i, j) in enumerate(prs):
    (x1, y1), (x2, y2) = best[i]
    # recompute point
    (x3, y3), (x4, y4) = best[j]
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    den = dx1 * dy2 - dy1 * dx2
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / den
    px, py = x1 + t * dx1, y1 + t * dy1
    axp.plot([float(px)], [float(py)], "ko", ms=4)
axp.set_aspect("equal")
axp.set_title("6-segment counterexample: no one-cut 2-coloring")
fig.tight_layout()
fig.savefig("counterexample6.png", dpi=130)
print("\nwrote counterexample6.png")
