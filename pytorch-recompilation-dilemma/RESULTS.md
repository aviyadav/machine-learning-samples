# Measured results

Reproduce with:

```bash
python -m recompilation_dilemma all --profile smoke --samples 120
```

**Environment:** Windows, CPU only (`torch 2.14.0+cpu`, Python 3.14), Inductor
backend with the MSVC toolchain loaded. Profile `smoke`: 120 requests, rows in
`[128, 1024)`, a 2-layer 64-wide version of the article's GNN.

Only the *recording* machine is Windows; the command is identical everywhere and
nothing in the package is platform-conditional. The one OS-specific part of the
experiments is the async mitigation's affinity and priority levers, which are
Linux facilities -- above, their absence is recorded rather than faked, and the
GIL-interval lever that does work everywhere is the one the article found most
impactful. On Linux the async numbers are the ones that would change, and the
`article` profile becomes the interesting one to run on a GPU.

**Budget:** no served request may exceed **50ms**; at least **90%** of served
requests must use the compiled graph. No speedup is required in this profile —
see the note at the end.

## Summary

```
#  mitigation                                                recompiles  compiled  max step  speedup  verdict
-  --------------------------------------------------------  ----------  --------  --------  -------  -------
0  eager (no compilation)                                    -           0%        8ms       1.00x    FAIL
0  torch.compile (as-is)                                     8           100%      5570ms    0.80x    FAIL
1  Preload an existing compilation cache                     8           100%      146ms     0.84x    FAIL
2  Rely on the recompilation cache limit                     0           0%        13ms      0.50x    FAIL
3  Control compilation behaviour with Dynamo configuration   0           100%      1ms       0.90x    PASS
4  Seed the graph compiler during a warm-up phase            1           100%      803ms     0.90x    FAIL
5  Execute recompilations on a designated background thread  1           0%        5ms       0.31x    FAIL
6  Fix the computation graph with ahead-of-time compilation  -           -         -         -        not run
```

## The problem

```
  eager      : mean 0.76ms, max 7.76ms, compiled-path coverage 0%  -> FAIL
  torch.compile: mean 0.90ms (steady), max 5569.61ms, speedup 0.80x  -> FAIL
  tail spikes : 8 recompilations, worst step is 7290x the mean eager step
  dilemma reproduced: eager never uses the compiled graph, torch.compile
  misses the tail bound, and each fails a different part of the budget
```

Eight requests — the catalog topologies — each blocked the serving loop on a
full compilation. The worst was **5.57 seconds**, about 7290x the average eager
request. That is the whole problem: 118 of 120 requests were fine, and the tail
was catastrophic.

## Option by option

| # | Result | What the numbers say |
| --- | --- | --- |
| 1 | **FAIL** — 8 recompilations, max **146ms** | The preloaded cache cut the worst recompilation from 5570ms to 146ms, a **97% reduction**. It did not remove it: the guards are not part of the artifacts, so every catalog topology still triggered a recompilation. Closest of the failures, and the article's own conclusion — a big improvement, not a fix. |
| 2 | **FAIL** — 0 recompilations, **0%** coverage | With `cache_size_limit=1` the tail is bounded (13ms) precisely *because* nothing is compiled. With the limit at 2 the picture inverts: 94% coverage, 1 recompilation, 520ms. The limit moves the trade, it does not remove it. |
| 3 | **PASS** — 0 recompilations, max **1.26ms** | `use_duck_shape=False` plus `mark_unbacked` on the variable dimensions removes the guard failures, so there is nothing left to recompile. |
| 3b | *brittle* — 6 recompilations, max **597ms** | Same configuration, same catalog, two entries swapped. Nothing about the configuration changed; the *guards* did. This is the caveat made measurable. |
| 4 | **FAIL** — 1 recompilation at the first served request, max **803ms** | All 10 catalog topologies were warmed (7.5s of start-up). The first request after warm-up still compiled: Dynamo did not serve it from a warm-up graph. |
| 5 | **FAIL** — tail fixed, **0%** coverage | The blocking wait is gone — the worst served request is 4.8ms — but the background compilation outlasts the entire serving loop, so the wrapper fell back to eager for 118 of 120 requests. The technique needs enough traffic per compilation to pay off; this workload does not have it. Tuning the GIL interval is still visible and still matters: the eager-served max went from 167ms to 4.8ms. |
| 6 | **not run** | `torch.export` succeeded, and its boundary behaviour is confirmed below, but AOTInductor packaging fails on this platform. |

### Option 6's boundary check

The export itself works, and so does the property the article claims for it —
an input outside the declared bounds is **rejected**, not silently handled:

```
  boundary behaviour: rejected the request: AssertionError: Guard failed: x.size()[0] <= 1024
```

Packaging and loading the AOT model then fail with an `InductorError`
(`FileNotFoundError` inside AOTInductor's bundle writer on Windows). Because
`aoti_load_package` is a native entry point that can take the interpreter down
with a segmentation fault rather than raising, this step runs in a child process
(`recompilation_dilemma/solutions/_aot_child.py`); the worst case is the reported
"not run" above instead of an aborted run. On the article's Linux/GPU
environment this is the step that would produce the measurement.

## Two environment caveats

**The speedup dimension is not exercised here.** On this machine Inductor
roughly ties with eager for a model this size — repeated calibration runs landed
anywhere between 0.80x and 1.34x, which is not a signal, it is noise. Rather
than pick a threshold that flips between runs, the CPU profile leaves
`min_speedup` unset and enforces the two checks that do not depend on the
hardware: the tail bound and compiled-path coverage. The `article` profile sets
`min_speedup=1.10`, which is where the article's 1.18x measurement belongs.
Coverage is what carries the "do not just give up on compiling" half of the
dilemma instead, and it discriminates exactly as intended: it is what fails
Option 2 and Option 5.

**Absolute compile times are CPU times.** ~0.75s per compilation here versus a
much larger figure on the article's L40S — but the ratio that matters, one
compilation against one request, is what produces the dilemma in both places.

## Reproducibility notes

The recompile *count* is stable across runs (8 for the unmitigated run, 0 for
Option 3's tuned ordering). The *magnitude* of a spike varies with how long a
compilation takes, which on a shared CPU varies by roughly 2-5x between runs;
one earlier run measured the unmitigated worst step at 5.4s and another at
10.6s. Which request the post-warm-up recompile lands on shifts between request
0 and request 1 for the same reason. Neither changes any verdict.
