import random, math
from itertools import product
from split_coloring_search import build, solve

rng = random.Random(999)
def chords(n):
    # random chords of a circle -> dense crossings
    segs=[]
    for _ in range(n):
        a,b = rng.uniform(0,2*math.pi), rng.uniform(0,2*math.pi)
        segs.append(((math.cos(a),math.sin(a)),(math.cos(b),math.sin(b))))
    return segs

bad=0; trials=200000
for t in range(trials):
    segs = chords(5)
    pairs, order = build(segs)
    if len(pairs) < 8:   # want dense instances
        continue
    if solve(pairs, order) is None:
        bad+=1
        print("UNSAT at n=5!", segs)
        break
print(f"n=5 chords: {t+1} trials examined, {bad} UNSAT")
