# The PyTorch Recompilation Dilemma, in code

A runnable project built from **"Overcoming the PyTorch Recompilation Dilemma in
Real-Time Inference Workloads"** (Chaim Rand, Medium, Sep 2026).

The article describes a problem and then lists six ways to work around it. This
project contains **both halves as code**: a module that reproduces the problem and
states it as a machine-checkable predicate, and one module per mitigation that is
measured against that same predicate.

---

## The dilemma

`torch.compile` makes a serving model faster on average. Every Dynamo guard
violation, however, triggers a **recompilation**: a blocking, multi-second
compilation that lands in the middle of the request loop. For real-time inference
that spike is worse than not compiling at all, so teams fall back to eager
execution and give up the speedup.

The project encodes this as a budget that must be satisfied on every axis at once
(`config.LatencyBudgetSpec`):

| Constraint | Meaning | Fails when |
| --- | --- | --- |
| `max_step_ms` | no served request may be slower than this | a recompilation spike |
| `min_compiled_ratio` | at least this fraction of requests must use the compiled graph | you stopped compiling to protect the tail |
| `min_speedup` | *optional* steady-state speedup over eager | the speedup was given up |

Drop the second constraint and "stop compiling" passes the tail bound trivially;
drop the first and naive `torch.compile` wins. The dilemma is that **no single
configuration satisfies all of them**, and `problem.Dilemma.demonstrated` is a
computed boolean rather than a claim in a docstring.

The third constraint is optional on purpose. Whether Inductor beats eager at all
is a property of the hardware -- on a laptop CPU it roughly ties -- so the `smoke`
profile leaves it unset and enforces the two checks that are hardware
independent, while the `article` profile sets it. [`RESULTS.md`](RESULTS.md) has
the measurements behind that decision.

---

## Layout

```
recompilation_dilemma/
├── config.py            RunConfig, the two profiles, and LatencyBudgetSpec
├── model.py             the toy message-passing GNN, verbatim from the article
├── dataset.py           the synthetic graph workload + the 10-pattern catalog
├── harness.py           compiler state, recompilation accounting, serving loops
├── metrics.py           Step/LatencyReport/Verdict -- where the budget is judged
├── runtime.py           portable timing, CPU affinity, thread priority
├── problem.py           THE PROBLEM: eager vs naive torch.compile
├── cli.py               `problem`, `solution <name>`, `all`, `list`
└── solutions/
    ├── preload_cache.py     Option 1 -- load a previous run's cache artifacts
    ├── cache_size_limit.py  Option 2 -- cap recompilations, accept eager fallback
    ├── dynamo_config.py     Option 3 -- duck-shape + mark_unbacked, and its limits
    ├── warmup_seed.py       Option 4 -- compile every topology up front
    ├── async_background.py  Option 5 -- compile on a background thread
    └── aot_export.py        Option 6 -- torch.export + AOTInductor instead of JIT
```

---

## Running it

The article's environment is Linux + CUDA, and that is the target platform: the
project has no Windows-only code path. It runs as-is on Linux (GPU or CPU) and
on macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# The problem on its own
python -m recompilation_dilemma problem --profile smoke

# One mitigation
python -m recompilation_dilemma solution dynamo-config --profile smoke

# The problem and all six mitigations, with a summary table and a verdict each
python -m recompilation_dilemma all --profile smoke
```

`run-cmd.sh` wraps the last of those, finds the virtual environment itself, and
forwards its arguments, so `./run-cmd.sh`, `./run-cmd.sh problem` and
`./run-cmd.sh all --samples 40` all do the obvious thing:

```bash
./run-cmd.sh
```

Every command also accepts the options after the subcommand
(`... all --profile smoke`), so the order above is a convenience, not a rule.

Two profiles (`config.Profile`):

* `article` -- the article's exact model (768-wide, 16 layers) and dataset
  (300 samples, 1000-4000 rows). Wants a GPU.
* `smoke` -- the same structure at CPU scale, calibrated so the trade-off is
  still visible. Use `--samples N` to shorten it further. This is the default
  when no CUDA device is present.

`torch.compile` needs a C++ toolchain for its Inductor backend -- `gcc`/`g++` on
Linux, MSVC on Windows. If you do not have one, everything still runs with
`--backend eager`, which exercises the whole recompilation machinery (guards,
cache limits, fallbacks, warm-up, background compilation) without generating
machine code:

```bash
python -m recompilation_dilemma all --profile smoke --backend eager
```

The first kernel Inductor launches also compiles Triton's C shim. On glibc 2.42 and
newer that compile prints a `_POSIX_C_SOURCE` redefinition warning, because the shim
includes `<cuda.h>` before `Python.h` and therefore reaches glibc's default feature
level first. `harness.silence_python_header_warnings()` puts CPython's headers first,
the order they are written for; it does nothing on any other platform, and deleting
the one call that installs it costs nothing but the warning.

Compiler artifacts and the packaged AOT model are written under `.rd-cache/`
relative to the directory you run from; each run clears and repopulates it, and
the resolved path is printed in the header of every run. Add it to your
`.gitignore` if it is not already there.

### Platform notes

| Platform | Status |
| --- | --- |
| Linux + CUDA | The article's environment. Both profiles available; `article` needs a GPU to finish in reasonable time. |
| Linux + CPU | Fully supported. Use `smoke`; Inductor works if a C/C++ compiler is installed. |
| macOS | Supported. Use `smoke`; a CUDA device is never required. |
| Windows | Supported. See below -- Inductor needs one extra step. |

The three OS-level levers the async mitigation uses (`os.sched_setaffinity`,
`os.setpriority`, and per-thread affinity) are Linux facilities. Where they do
not exist, `runtime.InterferenceControls` records that they were unavailable
instead of the experiment failing; the GIL interval tuning, which the article
identifies as the most impactful lever, works everywhere.

#### Windows (optional helper)

Inductor shells out to `cl.exe` and inherits whatever environment it is given; it
does not locate Visual Studio itself, so a plain `python -m ...` fails with
`cl is not found`. `scripts/with_msvc_env.py` loads `vcvars64.bat` into the child
process:

```bash
python scripts/with_msvc_env.py -m recompilation_dilemma all --profile smoke
```

This helper is a convenience, not a dependency: on Linux and macOS it is a
transparent passthrough, and nothing in `recompilation_dilemma/` imports or
references it.

---

## The six mitigations

Each one is a real implementation measured against the same budget. `tradeoff`
and `caveat` are fields on `SolutionOutcome`, so the code says what each one
costs and when it breaks, not just what it bought.

| # | Mitigation | Mechanism | What it costs | When it breaks |
| --- | --- | --- | --- | --- |
| 1 | Preload cache | move Inductor/AOTAutograd code caches between processes | nothing, but only helps topologies a previous run compiled | guards are *not* part of the artifacts, so recompilations still fire -- just more cheaply |
| 2 | Cache size limit | stop compiling once `cache_size_limit` variants exist | requests that miss fall back to eager | needs tuning to the shape distribution; too low and almost nothing is compiled |
| 3 | Dynamo configuration | stop the guards from firing (`use_duck_shape=False`, `mark_unbacked`) | nothing at run time | the configuration fits the sample *ordering* that produced the guard messages |
| 4 | Warm-up seeding | compile every expected topology before serving | start-up time, and requires knowing the topologies | any topology the warm-up misses recompiles mid-flight |
| 5 | Background compilation | move the compilation to a worker thread, serve eagerly meanwhile | foreground/worker interference for the duration of the compile | only helps when compilations are rare relative to traffic |
| 6 | AOT export | `torch.export` with bounded shapes + `aoti_compile_and_package` | shape bounds are permanent; out-of-bounds input **fails** | less flexible than `torch.compile`; may require code changes |

Two of these deserve a note on how they are reported:

* **Option 3** measures the tuned ordering *and* the same catalog with two entries
  swapped. That second measurement is the point: the configuration removes the
  guards that were reported for one ordering, and a different ordering
  reintroduces recompilations -- including a stride mismatch no configuration
  hint anticipates.
* **Option 6** runs the packaging and loading in a child process
  (`solutions/_aot_child.py`). `aoti_load_package` is a native entry point, and a
  failure there can take the interpreter down with a segmentation fault rather
  than raising; isolating it means the worst case is a reported
  "unavailable in this environment" instead of an aborted run.

---

## Results

[`RESULTS.md`](RESULTS.md) has the full measured output of a run on this machine:
which mitigations met the budget, which did not, and what the numbers say about
each. In short, on a CPU-only box:

* the dilemma reproduces -- eager fails on coverage, and naive `torch.compile`
  misses the tail bound by roughly 100x;
* **Option 3** passes, and its swapped-ordering variant shows how brittle that is;
* **Option 1** cuts the worst recompilation by 97% and still fails, because
  preloading code artifacts does not preload guards;
* **Options 2 and 5** bound the tail and fail on coverage instead -- the trade
  each of them makes, written out as a number.

---

## Attribution

The problem statement, the model, the workload, and the six mitigation
strategies come from the article. The code here is a working implementation: the
original listing is Linux+CUDA specific in places (`os.sched_setaffinity`,
`os.setpriority`, `torch.cuda.synchronize()`) and omits the packaging needed to
run as a project, so those parts are written portably and the recompilation
accounting is made explicit.
