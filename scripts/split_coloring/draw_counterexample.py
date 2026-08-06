"""Re-draw the 6-segment counterexample (new file, does not touch counterexample6.png).

Differences vs. polish.py's plot:
  * coordinates rescaled to a 0-10 box, equal aspect on both axes;
  * no title;
  * segment labels placed automatically: close to their own segment, clear of
    every other segment and of the other labels;
  * the 7 side-disjoint triangular faces of the pigeonhole proof filled yellow;
  * S3 rotated 6.5 deg counterclockwise about its lower-right endpoint, which
    blows up the tiny S1-S2-S3 face (area 0.008 -> 0.276) -- verified below to
    leave the instance a genuine counterexample;
  * free tails normalised to a fixed length past the outermost crossing: S3
    (upper left) and S6 (lower right) shortened, S2 (left) lengthened.

The script re-verifies the drawn instance exactly (Fractions): 14 crossings,
(S1,S5) the only non-crossing pair, no valid one-cut 2-coloring, and the
7-triangle pigeonhole certificate intact.
"""
from fractions import Fraction as F
from itertools import combinations, product
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# verified counterexample, 2-decimal coordinates rescaled to a 0-10 box
SEGS = [((5.1, 7.1), (3.5, 3.0)),
        ((9.7, 5.3), (3.8, 6.1)),
        ((6.9, 2.4), (2.1, 9.7)),
        ((3.3, 5.0), (9.9, 5.9)),
        ((10.0, 8.0), (5.1, 1.4)),
        ((4.3, 8.6), (7.0, 0.3))]

ROT_DEG = 6.5              # counterclockwise rotation of S3 about its endpoint 0
TAILS = {1: 1, 2: 1, 5: 1}  # segment index -> endpoint whose free tail is re-set
TAIL_LEN = 0.9              # how far past the outermost crossing that tail reaches
LABEL_ENDS = {2: 1}         # segment index -> endpoint its label is pinned to
BOX = 10.6                  # side of the square frame

# the 7 triangles of the pigeonhole proof (1-based, as in the README)
TRIANGLES = [(2, 3, 1), (1, 2, 6), (2, 4, 6), (2, 4, 5),
             (3, 5, 6), (3, 4, 6), (1, 3, 4)]
TRIS0 = [tuple(sorted(x - 1 for x in t)) for t in TRIANGLES]


# ---------------------------------------------------------------- geometry --
def build(ss):
    """crossing list, per-segment crossing order, crossing points."""
    n = len(ss)
    prs, pr, pts, par = [], [[] for _ in range(n)], {}, {}
    for i in range(n):
        for j in range(i + 1, n):
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
                pts[frozenset((i, j))] = (x1 + t * dx1, y1 + t * dy1)
                par[(i, j)], par[(j, i)] = t, u
    orr = [[c for _, c in sorted(l)] for l in pr]
    ix = [dict() for _ in range(n)]
    for i, o in enumerate(orr):
        for p, c in enumerate(o):
            ix[i][c] = p
    return n, prs, [len(o) for o in orr], ix, pts, par


def exact(ss, dec=2):
    q = 10 ** dec
    return [((F(round(a * q), q), F(round(b * q), q)), (F(round(c * q), q), F(round(d * q), q)))
            for ((a, b), (c, d)) in ss]


def rotate(seg, deg, pivot_end=0):
    cx, cy = seg[pivot_end]
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return tuple((round(cx + c * (x - cx) - s * (y - cy), 2),
                  round(cy + s * (x - cx) + c * (y - cy), 2)) for (x, y) in seg)


def set_tail(ss, i, end, margin):
    """put endpoint `end` of segment i exactly `margin` past its outermost crossing
    (shortens or lengthens the free tail; never removes a crossing)."""
    _, _, _, _, _, par = build(ss)
    ts = [t for (a, b), t in par.items() if a == i]
    (x1, y1), (x2, y2) = ss[i]
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    # t may leave [0,1] -- that lengthens the tail; the exact check below
    # confirms no crossing is gained or lost
    t = (max(ts) + margin / L) if end == 1 else (min(ts) - margin / L)
    new = (round(x1 + t * dx, 2), round(y1 + t * dy, 2))
    out = list(ss)
    out[i] = (ss[i][0], new) if end == 1 else (new, ss[i][1])
    return out


# ---------------------------------------------------------- certification --
def unsat(ss):
    n, prs, m, ix, _, _ = build(ss)

    def feas(g):
        adj = [[] for _ in range(n)]
        for cid, (i, j) in enumerate(prs):
            p = 1 ^ (ix[i][cid] >= g[i]) ^ (ix[j][cid] >= g[j])
            adj[i].append((j, p))
            adj[j].append((i, p))
        col = [None] * n
        for s in range(n):
            if col[s] is not None:
                continue
            col[s], st = 0, [s]
            while st:
                v = st.pop()
                for w, p in adj[v]:
                    want = col[v] ^ p
                    if col[w] is None:
                        col[w] = want
                        st.append(w)
                    elif col[w] != want:
                        return False
        return True

    return not any(feas(g) for g in product(*[range(mi + 1) for mi in m]))


def pigeonhole(ss):
    """the 7 triangles are faces with pairwise-disjoint sides, parities infeasible."""
    n, prs, m, ix, _, _ = build(ss)
    pairset = {frozenset(p): c for c, p in enumerate(prs)}
    sides = {}
    for t in TRIS0:
        for x in t:
            y, z = [v for v in t if v != x]
            if frozenset((x, y)) not in pairset or frozenset((x, z)) not in pairset:
                return False, f"triangle {tuple(v+1 for v in t)} is not a crossing triple"
            i1 = ix[x][pairset[frozenset((x, y))]]
            i2 = ix[x][pairset[frozenset((x, z))]]
            sides[(t, x)] = set(range(min(i1, i2) + 1, max(i1, i2) + 1))
    for x in range(n):
        for a, b in combinations([t for t in TRIS0 if (t, x) in sides], 2):
            if sides[(a, x)] & sides[(b, x)]:
                return False, f"sides overlap on S{x+1}"
    if any(all(sum(1 for x in t if g[x] in sides[(t, x)]) % 2 for t in TRIS0)
           for g in product(*[range(mi + 1) for mi in m])):
        return False, "7-triangle parity system is satisfiable"
    return True, "7 side-disjoint triangles, parity system infeasible"


# ------------------------------------------------------------ label layout --
def dist_point_seg(p, seg):
    (px, py), ((x1, y1), (x2, y2)) = p, seg
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def label_positions(segs, out=(0.30, 0.45, 0.60), side=(-0.30, 0.0, 0.30),
                    ends=LABEL_ENDS):
    """every label sits just beyond one endpoint of its segment, on the segment's
    own direction; the endpoint is chosen to clear the other segments and labels
    (unless pinned in `ends`), and a sideways nudge is used only when staying
    collinear would collide."""
    placed = []
    for i, ((x1, y1), (x2, y2)) in enumerate(segs):
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        others = [s for j, s in enumerate(segs) if j != i]

        cands = []
        for ext in out:
            for sd in side:
                if ends.get(i, 0) == 0:
                    cands.append(((x1 - ext * ux + sd * nx, y1 - ext * uy + sd * ny), sd))
                if ends.get(i, 1) == 1:
                    cands.append(((x2 + ext * ux + sd * nx, y2 + ext * uy + sd * ny), sd))

        def score(cs):
            c, sd = cs
            near_other = min(dist_point_seg(c, s) for s in others)
            near_label = min((math.dist(c, q) for q in placed), default=9e9)
            return (min(near_other, 1.5) + min(near_label, 1.2)
                    - 1.3 * dist_point_seg(c, segs[i]) - 1.2 * abs(sd))

        placed.append(max(cands, key=score)[0])
    return placed


def report(segs, tag):
    """exact re-verification of the drawn instance."""
    ex = exact(segs)
    n, prs, m, ix, pts, _ = build(ex)
    missing = [tuple(sorted(x + 1 for x in p)) for p in combinations(range(6), 2)
               if frozenset(p) not in {frozenset(q) for q in prs}]
    ok_p, why = pigeonhole(ex)
    print(f"--- {tag} ---")
    for i, ((a, b), (c, d)) in enumerate(segs):
        print(f"  S{i+1}: ({a:5.2f},{b:5.2f}) -- ({c:5.2f},{d:5.2f})   crossings: {m[i]}")
    print(f"crossings: {len(prs)} (expected 14); non-crossing pairs: {missing}")
    print(f"pigeonhole certificate: {ok_p} -- {why}")
    print(f"no valid one-cut 2-coloring (exact brute force): {unsat(ex)}")
    a, b, c = [pts[frozenset((TRIS0[0][i], TRIS0[0][j]))] for i, j in ((0, 1), (0, 2), (1, 2))]
    print(f"area of face S1S2S3: "
          f"{float(abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])))/2:.4f}")
    return pts


def draw(segs, pts, labels, fname, lim=BOX):
    fig, ax = plt.subplots(figsize=(7, 7))
    cols = plt.cm.tab10.colors

    for a, b, c in TRIANGLES:                          # the 7 proof faces
        tri = [pts[frozenset(p)] for p in ((a - 1, b - 1), (a - 1, c - 1), (b - 1, c - 1))]
        ax.fill([float(p[0]) for p in tri], [float(p[1]) for p in tri],
                facecolor="gold", alpha=0.45, zorder=0,
                edgecolor="goldenrod", lw=1.0)         # thin edge: some faces are slivers

    for i, ((a, b), (c, d)) in enumerate(segs):
        ax.plot([a, c], [b, d], color=cols[i], lw=2, zorder=2)

    for (px, py) in pts.values():
        ax.plot([float(px)], [float(py)], "ko", ms=4, zorder=3)

    for i, (lx, ly) in enumerate(labels):
        ax.text(lx, ly, f"S{i+1}", color=cols[i], fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax.set_aspect("equal")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks(range(0, int(lim) + 1))
    ax.set_yticks(range(0, int(lim) + 1))
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    print(f"wrote {fname}\n")


def centered(segs, labels, lim=BOX):
    """translate everything (2-decimal shift, so the arrangement is unchanged) so
    that the drawing -- segments and labels -- is centred in the frame.  Labels are
    carried along rather than re-solved, so both figures label the same ends."""
    xs = [c for s in segs for (c, _) in s] + [p[0] for p in labels]
    ys = [c for s in segs for (_, c) in s] + [p[1] for p in labels]
    ddx = round(lim / 2 - (min(xs) + max(xs)) / 2, 2)
    ddy = round(lim / 2 - (min(ys) + max(ys)) / 2, 2)
    print(f"centring shift: ({ddx:+.2f}, {ddy:+.2f})")
    return ([((a + ddx, b + ddy), (c + ddx, d + ddy)) for ((a, b), (c, d)) in segs],
            [(x + ddx, y + ddy) for (x, y) in labels])


# ------------------------------------------------------------------- build --
SEGS[2] = rotate(SEGS[2], ROT_DEG, pivot_end=0)
for i, end in TAILS.items():
    SEGS = set_tail(SEGS, i, end, TAIL_LEN)
LABELS = label_positions(SEGS)

draw(SEGS, report(SEGS, "as placed"), LABELS, "counterexample6_triangles.png")

SHIFTED, SHIFTED_LABELS = centered(SEGS, LABELS)
draw(SHIFTED, report(SHIFTED, "centred in the frame"), SHIFTED_LABELS,
     "counterexample6_triangles_centered.png")
