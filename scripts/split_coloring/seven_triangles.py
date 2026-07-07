"""Check the user's 7 triangles: are they valid pairwise-crossing triples with
pairwise-disjoint SIDES (open sub-segments between their two crossings)?
If yes: pigeonhole proof (7 triangles, 6 cuts)."""
from fractions import Fraction as F
from itertools import combinations, product

SEGS = [((F(51,100),F(71,100)),(F(35,100),F(30,100))),
        ((F(97,100),F(53,100)),(F(38,100),F(61,100))),
        ((F(69,100),F(24,100)),(F(21,100),F(97,100))),
        ((F(33,100),F(50,100)),(F(99,100),F(59,100))),
        ((F(1),F(80,100)),(F(51,100),F(14,100))),
        ((F(43,100),F(86,100)),(F(70,100),F(3,100)))]
n=6
prs=[]; pr=[[] for _ in range(n)]
for i in range(n):
    for j in range(i+1,n):
        (x1,y1),(x2,y2)=SEGS[i]; (x3,y3),(x4,y4)=SEGS[j]
        dx1,dy1=x2-x1,y2-y1; dx2,dy2=x4-x3,y4-y3
        den=dx1*dy2-dy1*dx2
        if den==0: continue
        t=((x3-x1)*dy2-(y3-y1)*dx2)/den; u=((x3-x1)*dy1-(y3-y1)*dx1)/den
        if 0<t<1 and 0<u<1:
            cid=len(prs); prs.append((i,j)); pr[i].append((t,cid)); pr[j].append((u,cid))

missing=[(i+1,j+1) for i in range(n) for j in range(i+1,n)
         if frozenset((i,j)) not in {frozenset(p) for p in prs}]
print("non-crossing pairs:", missing)

orr=[[c for _,c in sorted(l)] for l in pr]
ix=[dict() for _ in range(n)]
for i,o in enumerate(orr):
    for p,c in enumerate(o): ix[i][c]=p
pairset={frozenset(p):c for c,p in enumerate(prs)}

TRIS=[(1,6,2),(3,2,1),(1,4,3),(4,6,3),(6,5,3),(4,2,6),(2,4,5)]
TRIS=[tuple(sorted(x-1 for x in t)) for t in TRIS]

# each side of triangle {a,b,c} on segment x: gap interval (lo, hi] between
# the positions of its two triangle crossings on x
sides={}  # (tri, seg) -> set of gaps
valid=True
for t in TRIS:
    for x in t:
        y,z=[v for v in t if v!=x]
        if frozenset((x,y)) not in pairset or frozenset((x,z)) not in pairset:
            print("INVALID TRIANGLE (missing crossing):", tuple(v+1 for v in t)); valid=False; continue
        i1,i2=ix[x][pairset[frozenset((x,y))]],ix[x][pairset[frozenset((x,z))]]
        lo,hi=min(i1,i2),max(i1,i2)
        sides[(t,x)]=set(range(lo+1,hi+1))  # gaps strictly inside

print("all 7 are pairwise-crossing triples:", valid)
print("\nsides as gap-sets per segment:")
for x in range(n):
    used=[(t,sides[(t,x)]) for t in TRIS if (t,x) in sides]
    print(f"  S{x+1}:", ", ".join(f"T{TRIS.index(t)+1}{{{tuple(v+1 for v in t)}}}:gaps{sorted(g)}" for t,g in used))

# pairwise disjointness per segment
disjoint=True
for x in range(n):
    ts=[t for t in TRIS if (t,x) in sides]
    for a,b in combinations(ts,2):
        if sides[(a,x)] & sides[(b,x)]:
            print("OVERLAP on S%d between"%(x+1), a, b); disjoint=False
print("\nall sides pairwise disjoint:", disjoint)

# independent confirmation: parity constraints of just these 7 triangles are UNSAT
m=[len(o) for o in orr]
def ok(g):
    for t in TRIS:
        cnt=sum(1 for x in t if g[x] in sides[(t,x)])
        if cnt%2==0: return False
    return True
feasible=any(ok(g) for g in product(*[range(mi+1) for mi in m]))
print("7-triangle parity system satisfiable:", feasible, "(False = pigeonhole proof confirmed)")
