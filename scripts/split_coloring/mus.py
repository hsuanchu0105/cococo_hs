"""Minimal infeasible subset of triangle parity constraints for the rounded
6-segment counterexample."""
from fractions import Fraction as F
from itertools import product, combinations

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
orr=[[c for _,c in sorted(l)] for l in pr]
m=[len(o) for o in orr]
ix=[dict() for _ in range(n)]
for i,o in enumerate(orr):
    for p,c in enumerate(o): ix[i][c]=p
pairset={frozenset(p):c for c,p in enumerate(prs)}
tris=[t for t in combinations(range(n),3)
      if all(frozenset(x) in pairset for x in combinations(t,2))]

def feasible(subset):
    for g in product(*[range(mi+1) for mi in m]):
        ok=True
        for (a,b,c) in subset:
            cnt=0
            for (x,y,z) in ((a,b,c),(b,a,c),(c,a,b)):
                i1,i2=ix[x][pairset[frozenset((x,y))]],ix[x][pairset[frozenset((x,z))]]
                lo,hi=min(i1,i2),max(i1,i2)
                if lo<g[x]<=hi: cnt+=1
            if cnt%2==0: ok=False; break
        if ok: return True
    return False

sub=list(tris)
assert not feasible(sub)
changed=True
while changed:
    changed=False
    for t in list(sub):
        cand=[x for x in sub if x!=t]
        if not feasible(cand):
            sub=cand; changed=True; break
print("minimal infeasible triangle set (%d triangles):"%len(sub))
for t in sub: print("  {S%d, S%d, S%d}"%(t[0]+1,t[1]+1,t[2]+1))
# print the interval (positions) of each triangle pair on each segment for the proof
print("\nfor each segment: order of partners and, per listed triangle, the interval the cut must (parity) relate to")
for i,o in enumerate(orr):
    partners=[(set(prs[c])-{i}).pop()+1 for c in o]
    print(f"  S{i+1} partner order: {partners}")
