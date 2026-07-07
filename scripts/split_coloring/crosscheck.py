"""Cross-validate the DSU backtracking solver against an independent
brute-force enumerator, and validate returned solutions directly."""
import random
from itertools import product
from split_coloring_search import build, solve, random_segments

def brute(pairs, order):
    n=len(order); m=[len(o) for o in order]
    idx=[dict() for _ in range(n)]
    for i,o in enumerate(order):
        for p,c in enumerate(o): idx[i][c]=p
    for g in product(*[range(mi+1) for mi in m]):
        for b in product((0,1),repeat=n):
            if all((b[i]^(idx[i][c]>=g[i])) != (b[j]^(idx[j][c]>=g[j]))
                   for c,(i,j) in enumerate(pairs)):
                return g,b
    return None

rng=random.Random(7)
mismatch=0
for t in range(1500):
    segs=random_segments(rng.choice([4,5]),rng)
    pairs,order=build(segs)
    s1=solve(pairs,order)
    s2=brute(pairs,order)
    if (s1 is None)!=(s2 is None):
        mismatch+=1; print("MISMATCH",segs)
    if s1 is not None:
        # validate solver's own solution directly
        n=len(order); idx=[dict() for _ in range(n)]
        for i,o in enumerate(order):
            for p,c in enumerate(o): idx[i][c]=p
        # solver returns gaps only; recover b by 2-coloring check = reuse brute with fixed g
        found=False
        for b in product((0,1),repeat=n):
            if all((b[i]^(idx[i][c]>=s1[i])) != (b[j]^(idx[j][c]>=s1[j]))
                   for c,(i,j) in enumerate(pairs)):
                found=True; break
        if not found:
            mismatch+=1; print("BAD SOLUTION",segs,s1)
print("mismatches:",mismatch,"(out of 1500 instances)")
