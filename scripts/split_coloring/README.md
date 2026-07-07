# One-cut two-coloring of segment arrangements

**Problem.** Given k line segments in general position (no three concurrent), each
segment gets ONE cut point; the two pieces are colored blue/green (equivalently: the
color along a segment changes at most once). A coloring is *valid* if at every
crossing the two segments have different colors. In routing terms: 2 layers, at most
one via per wire, every crossing must be between different layers.

**Result (computer-assisted, 2026-07-07).**
- k ≤ 5: a valid coloring ALWAYS exists — even for pseudosegments. Proof: exhaustive
  check of all 331,776 abstract complete-crossing 5-segment configurations (vertex-0
  partner order fixed WLOG by relabeling), all satisfiable; non-crossing pairs reduce
  to the complete case by abstractly appending the missing crossing at both segments'
  far ends (adds constraints only).
- k = 6: counterexample exists (see `counterexample6.png`), verified UNSAT by exact
  rational brute force over all 211,680 cut placements. Hence counterexamples exist
  for every k ≥ 6 (pad with far-away segments). **6 is the exact threshold.**

**Counterexample** (all pairs cross except S1–S5; 14 crossings):

    S1 (0.51,0.71)–(0.35,0.30)   S2 (0.97,0.53)–(0.38,0.61)   S3 (0.69,0.24)–(0.21,0.97)
    S4 (0.33,0.50)–(0.99,0.59)   S5 (1.00,0.80)–(0.51,0.14)   S6 (0.43,0.86)–(0.70,0.03)

**Key lemma (triangle parity).** In any pairwise-crossing triple, an odd number (1 or 3)
of the segments must have their cut strictly between their two triangle crossings —
otherwise the three inequality constraints form an odd cycle.

**Hand proof of the counterexample (pigeonhole, due to Hsuanchu).** The 7 triangles
{126},{123},{134},{346},{356},{246},{245} have pairwise-disjoint SIDES: every side is
the sub-segment between two CONSECUTIVE crossings (they are triangular faces of the
arrangement), and the 21 (segment, gap) slots are all distinct (`seven_triangles.py`).
By the lemma each triangle needs >= 1 cut strictly inside one of its sides; each of the
6 cuts occupies one gap, so it serves at most one triangle; 6 < 7. Contradiction.
(An alternative certificate: the 8 triangles through S6 — {126},{136},{236},{246},
{256},{346},{356},{456} — have jointly infeasible parity constraints, see `mus.py`.)

## Files

| file | purpose |
|---|---|
| `split_coloring_search.py` | core solver (backtracking over cut gaps + parity union-find) and the random search that first found UNSAT instances at k=8,9 |
| `verify8.py` | independent exact-rational verification of the original 8-segment instance; drop-one analysis |
| `shrink.py` | greedy shrinking of the 8-segment instance to the 6-segment counterexample |
| `polish.py` | rounds coordinates to 2 decimals, re-verifies exactly, checks triangle-parity-only infeasibility, draws `counterexample6.png` |
| `mus.py` | minimal infeasible subset of triangle parity constraints (the 8 triangles above) |
| `k5_abstract.py` | exhaustive check of all abstract complete 5-segment configurations (the k ≤ 5 proof) |
| `hunt5.py` | 200k random dense 5-segment instances, all SAT (sanity for the k=5 result) |
| `crosscheck.py` | cross-validates the backtracking solver against an independent brute-force enumerator on 1500 random instances |
| `seven_triangles.py` | verifies the 7-triangle pigeonhole proof: valid triples, pairwise-disjoint sides, infeasibility |
| `sufficient_probe.py` | empirical probes of sufficient conditions: max-3-crossings (18,369 SAT / 0 UNSAT), triangle-free crossing graph (2,120 SAT / 0 UNSAT), 2 cuts/segment fixes the counterexample |

## Sufficient conditions & open questions

Provable sufficient conditions for colorability:
- k <= 5 (the exhaustive theorem above);
- bipartite crossing graph (color each class monochromatically, no cuts needed);
- every segment has <= 2 crossings (constraints decouple: a segment with 2 crossings
  realizes all four color patterns).

Checking the pigeonhole witness (>= k+1 side-disjoint triangular faces) is easy: each
segment piece borders <= 2 faces, so the face-conflict graph is planar with max degree 3
— greedy already gives N/4 side-disjoint faces, and the remaining small cases are exact.
For general (non-face) triangles the packing problem is likely NP-hard. The witness is
one-sided: its absence proves nothing.

Conjectures (empirical, `sufficient_probe.py`, 2026-07-07; no counterexample found):
- every segment has <= 3 crossings  => colorable  (18,369 dense random instances SAT);
- triangle-free crossing graph      => colorable  (2,120 instances SAT — weaker evidence,
  obstructions would need overloaded odd cycles of length >= 5);
- 2 cuts per segment always suffice (the 6-segment counterexample becomes SAT with 2 cuts,
  using second cuts on only 4 of 6 segments; m-1 cuts trivially suffice).

Open: is colorability NP-complete? It is in NP (the coloring certifies SAT) but not known
to be in coNP — that would need a complete family of short UNSAT witnesses (pigeonhole +
parity certificates are candidates; completeness unknown).

## Running

Run from this directory (helpers import `split_coloring_search`):

    python3.11 verify8.py           # exact verification, ~min
    python3.11 k5_abstract.py       # exhaustive k=5, ~min
    python3.11 seven_triangles.py   # pigeonhole proof check, seconds
    python3.11 sufficient_probe.py  # sufficient-condition probes, ~10 min
