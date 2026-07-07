"""Greedily shrink the counterexample: repeatedly drop any segment whose
removal keeps the arrangement UNSAT. Exact arithmetic throughout."""
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


def build(idxs):
    ss = [tuple((F(a), F(b)) for (a, b) in SEGS[i]) for i in idxs]
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
    return nn, prs, [len(o) for o in orr], ix


def sat(idxs):
    nn, prs, mm, ix = build(idxs)

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

    return any(feas(g) for g in product(*[range(mi + 1) for mi in mm]))


cur = list(range(8))
changed = True
while changed:
    changed = False
    for d in list(cur):
        cand = [i for i in cur if i != d]
        if not sat(cand):
            cur = cand
            changed = True
            print("dropped", d, "-> still UNSAT with", cur)
            break

print("\nminimal (w.r.t. deletion) counterexample: segments", cur)
nn, prs, mm, ix = build(cur)
print("crossings per segment:", mm, " total:", len(prs))
print("crossing pairs (local indices):", prs)
for i in cur:
    print(f"  S{i}: {SEGS[i]}")
