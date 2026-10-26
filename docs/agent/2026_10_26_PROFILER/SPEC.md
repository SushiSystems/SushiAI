# SushiProfiler: the stack's one profiler

**Status:** Open — written 2026-10-26 for the owner's approval; nothing is built until he approves it.

## 1. Intent

The owner asked on 2026-10-26 for a profiler that measures everything measurable while any
SushiAI model trains or runs inference, and keeps what it measured. He suspects the arithmetic of
SushiBLAS's young kernels first: YOLOv5n trains at about 5 s per iteration on the RTX 3080 Ti at
20-46% GPU utilisation, and nothing in the stack says where the time goes.

He then widened it, in five rulings the same day:

1. The work is not for SushiAI alone. What SushiRuntime and SushiBLAS gain stays there and is
   used there, so each library profiles itself at its own tier.
2. SushiEngine's trace layer moves into a new repository, SushiProfiler, licensed like the rest
   of the stack (PolyForm Noncommercial 1.0.0), and becomes the profiler of every Sushi library.
3. SushiProfiler is general purpose and configurable. It is shaped neither by SushiAI nor by
   SushiEngine: any C++17 program, inside the stack or outside it, profiles itself with it. It
   knows no model, no kernel, no frame and no Sushi library.
4. It is optimised and does not carry over the engine's defects; section 5 lists each change.
5. The design is worked out to the interfaces; sections 5 to 9.

He left the remaining engineering calls to the orchestrator; section 13 lists them.

## 2. Goals, non-goals, anti-goals

Goals:

- One answer to "where did the time of this period go", for any period a program marks: a
  training step, an inference batch, a benchmark iteration, an engine frame.
- Host time and device time on one clock, so a gap on the device is blamed on what the host was
  doing at that moment.
- For SushiBLAS kernels: achieved FLOP/s and bytes/s against ceilings measured on the device,
  which answers the owner's question about the arithmetic directly.
- A run record kept on disk, so two runs are compared with medians and intervals, and a change to
  a kernel shows as a measured difference.
- Measured costs: a scope costs one relaxed load and a branch while off, at most 25 ns while on,
  and nothing when compiled out (section 10).

Non-goals for this work:

- Hardware counters, PC sampling and stall reasons (Nsight Compute, CUPTI). The record leaves room
  for them; collecting them is later work.
- Optimisation what-if simulation (Daydream, Coz): phase P7.
- A GUI. Perfetto and the terminal are the viewers; SushiEngine keeps its editor window.

Anti-goals, which make the work a failure even if it runs:

- Numbers that change the program they measure without saying so. A profiled SushiRuntime run
  submits node by node instead of replaying one command graph; the record states the execution
  layer of every period.
- A host clock around an asynchronous launch reported as kernel time.
- Model, kernel or engine vocabulary in SushiProfiler. Each tier names its own zones.
- Profiler work done on the thread being measured, beyond writing one event.
- A second profiler in any repository after this one lands.

## 3. Prior art

The citations below were checked against the venue or arXiv page on 2026-10-26. Tool behaviour is
cited from the vendor's documentation page.

### 3.1 Tools

| Tool | What we take | What we leave |
| --- | --- | --- |
| PyTorch `torch.profiler` on Kineto ([docs](https://docs.pytorch.org/docs/2.14/profiler.html)) | The `wait / warmup / active / repeat` window over step markers; opt-in flags that each add cost; grouping of averages | Python dispatch hooks |
| Holistic Trace Analysis ([repository](https://github.com/facebookresearch/HolisticTraceAnalysis)) | Temporal breakdown; idle time split by cause; short-kernel and launch statistics; trace diff | Multi-rank communication |
| `torch.utils.benchmark` ([docs](https://docs.pytorch.org/docs/2.14/benchmark_utils.html)) | Warm up, synchronise before reading the clock, keep raw times, take the median | |
| XProf and `jax.profiler` ([XProf](https://github.com/openxla/xprof), [JAX](https://docs.jax.dev/en/latest/profiling.html)) | A step-time overview; asynchronous dispatch waited on before a trace ends | gRPC capture servers |
| CUPTI ([docs](https://docs.nvidia.com/cupti/main/main.html)) | Buffered records, sorted and joined after the fact | Vendor lock-in |
| Nsight Systems and Compute ([Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)) | Host ranges projected onto the device timeline; speed-of-light percentages; per-precision roofline | Kernel replay, which changes clocks and caches |
| intel-llvm XPTI ([framework](https://github.com/intel/llvm/blob/sycl/xptifw/doc/XPTI_Framework.md)), unitrace ([tool](https://github.com/intel/pti-gpu/tree/master/tools/unitrace)) | A stub that costs nothing with no listener; queued, submitted and executing intervals per kernel | Level Zero metrics |
| ONNX Runtime profiling ([docs](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)) | Each device kernel attached to the host node that launched it | |
| Ultralytics `results.speed` ([docs](https://docs.ultralytics.com/modes/predict/)) | A cheap per-stage time that is always on | |
| Perfetto ([docs](https://perfetto.dev/docs/)) | Opens Chrome trace JSON offline; SQL over a trace | Its protobuf writer, for now |
| SushiEngine `foundation/profiling` (in-house, built 2026-10-09) | The ring, the clock, the TSC calibration, the thread-retirement guards, the zone tree fold, the Chrome writer | What section 5 replaces |

### 3.2 Literature

Performance models:

- S. Williams, A. Waterman, D. Patterson. *Roofline: an insightful visual performance model for
  multicore architectures.* CACM 52(4), 2009. [doi](https://doi.org/10.1145/1498765.1498785).
  Attainable FLOP/s is the lesser of the peak and bandwidth times arithmetic intensity; a kernel
  below both ceilings is bound by latency, launch cost or occupancy.
- C. Yang, R. Gayatri, T. Kurth, P. Basu, S. Williams. *An empirical roofline methodology for
  quantitatively assessing performance portability.* P3HPC at SC18, 2018.
  [pdf](https://crd.lbl.gov/assets/Uploads/p3hpc-roofline-final.pdf). Ceilings measured by
  microbenchmark on the device.
- C. Yang, T. Kurth, S. Williams. *Hierarchical Roofline analysis for GPUs.* CUG 2019; CCPE
  32(20), 2020. [doi](https://doi.org/10.1002/cpe.5547). One ceiling per memory level and
  precision.
- N. Ding, S. Williams. *An instruction roofline model for GPUs.* PMBS at SC19, 2019.
  [abstract](https://www.dcs.warwick.ac.uk/pmbs/pmbs19/PMBS19/abstracts/paper2.html). Kernels with
  few FLOPs.
- Y. Wang, C. Yang, S. Farrell. *Hierarchical Roofline performance analysis for deep learning
  applications.* arXiv 2009.05257, 2020. [arXiv](https://arxiv.org/abs/2009.05257). One point per
  kernel, sized by its share of time.
- H. Qi, E. R. Sparks, A. Talwalkar. *Paleo: a performance model for deep neural networks.* ICLR
  2017. [openreview](https://openreview.net/forum?id=SyVVJ85lg). FLOPs and bytes from shapes;
  predicted beside measured.

Training workloads:

- H. Zhu et al. *TBD: Benchmarking and analyzing deep neural network training.* IISWC 2018.
  [arXiv](https://arxiv.org/abs/1803.06905). Device busy time and FP32 utilisation are two numbers.
- H. Zhu, A. Phanishayee, G. Pekhimenko. *Daydream: accurately estimating the efficacy of
  optimizations for DNN training.* USENIX ATC 2020.
  [page](https://www.usenix.org/conference/atc20/presentation/zhu-hongyu).
- G. X. Yu, T. Grossman, G. Pekhimenko. *Skyline: interactive in-editor computational performance
  profiling for deep neural network training.* UIST 2020.
  [arXiv](https://arxiv.org/abs/2008.06798). Time against batch size separates fixed from
  per-sample cost.
- G. X. Yu, Y. Gao, P. Golikov, G. Pekhimenko. *Habitat: a runtime-based computational performance
  predictor for deep neural network training.* USENIX ATC 2021.
  [page](https://www.usenix.org/conference/atc21/presentation/yu).
- M. Jeon, S. Venkataraman, A. Phanishayee et al. *Analysis of large-scale multi-tenant GPU
  clusters for DNN training workloads.* USENIX ATC 2019.
  [page](https://www.usenix.org/conference/atc19/presentation/jeon). Idle device time from input,
  host stalls and synchronisation.
- M. Wang, C. Meng, G. Long et al. *Characterizing deep learning training workloads on
  Alibaba-PAI.* IISWC 2019. [arXiv](https://arxiv.org/abs/1910.05930). Jobs bound by many small
  kernels.
- P. Mattson et al. *MLPerf Training benchmark.* MLSys 2020.
  [arXiv](https://arxiv.org/abs/1910.01500).
- S. Sridharan et al. *Chakra: advancing performance benchmarking and co-design using standardized
  execution traces.* arXiv 2305.14516, 2023. [arXiv](https://arxiv.org/abs/2305.14516).
- C. Curtsinger, E. D. Berger. *Coz: finding code that counts with causal profiling.* SOSP 2015.
  [doi](https://doi.org/10.1145/2815400.2815409).

Kernel analysis:

- K. Zhou, X. Meng, R. Sai, J. Mellor-Crummey. *GPA: a GPU performance advisor based on
  instruction sampling.* CGO 2021. [arXiv](https://arxiv.org/abs/2009.04061).
- K. Zhou, L. Adhianto et al. *Measurement and analysis of GPU-accelerated applications with
  HPCToolkit.* Parallel Computing 108, 2021. [arXiv](https://arxiv.org/abs/2109.06931). Device
  time attributed to the host context that launched it.
- Z. Jia, M. Maggioni, B. Staiger et al. *Dissecting the NVIDIA Volta GPU architecture via
  microbenchmarking.* arXiv 1804.06826, 2018. [arXiv](https://arxiv.org/abs/1804.06826).
- L. Zheng et al. *Ansor: generating high-performance tensor programs for deep learning.* OSDI
  2020. [page](https://www.usenix.org/conference/osdi20/presentation/zheng).

Measurement method:

- T. Hoefler, R. Belli. *Scientific benchmarking of parallel computing systems: twelve ways to
  tell the masses when reporting performance results.* SC 2015.
  [doi](https://doi.org/10.1145/2807591.2807644).
- A. Georges, D. Buytaert, L. Eeckhout. *Statistically rigorous Java performance evaluation.*
  OOPSLA 2007. [pdf](https://dri.es/files/oopsla07-georges.pdf).
- T. Kalibera, R. E. Jones. *Rigorous benchmarking in reasonable time.* ISMM 2013.
  [doi](https://doi.org/10.1145/2464157.2464160).
- T. Mytkowicz, A. Diwan, M. Hauswirth, P. F. Sweeney. *Producing wrong data without doing
  anything obviously wrong!* ASPLOS 2009.
  [page](https://sape.inf.usi.ch/publications/asplos09.html).

## 4. Where things stand

### 4.1 SushiRuntime

- `RuntimeConfig::profiling` (default false) builds the SYCL queues with `enable_profiling`.
- `RunReport` carries `node_events` (submit, start and end on the event clock, launched and live
  width per dispatch), `ClockPairing` to `steady_clock`, the `ExecutionLayer` that ran, and
  host-to-device and device-to-host bytes.
- With profiling on, a chainable plan runs node by node on one in-order queue; without it, it may
  replay one `command_graph` per step.

### 4.2 SushiBLAS

No timing outside `benchmark_harness` (ten iterations, a printed median). No FLOP or byte count.
Every kernel is recorded through `TaskRecorder::insert` with a task name and an operation id.

### 4.3 SushiAI

- `sa train` builds a runtime with profiling off, and the training program calls `run_step`
  directly, so `Trainer::fit`'s per-node sum never runs.
- No host phase is timed.
- The anchor-free step runs the forward part on the device, reads ports back, assigns targets on
  the host, writes the loss leaves and runs the rest. That pause can explain low utilisation as
  well as slow kernels can.
- Lowering keeps one running total of the tasks it emitted (`LoweringContext::count`), not a
  count per IR node.

### 4.4 SushiEngine

`modules/foundation/profiling`, about 4,700 lines: an invariant-TSC clock calibrated to
`steady_clock`; a single-producer ring of 24-byte events per thread; a recorder that drains the
rings every 5 ms; zones with groups and compile-time levels; device tracks; a live consumer beside
the file capture; a Chrome trace writer; `FrameSampler` and `ZoneTreeFolder`; an optional Tracy
backend; a test that the zone thread allocates nothing. `execution/run_report_trace.hpp` turns a
SushiRuntime `RunReport` into device spans.

### 4.5 SushiDSP

A real-time audio library on the CPU, depending on SushiLog alone. `AudioNode::process()` runs on
the audio thread and may not allocate, lock, call the kernel or throw. Its `benchmark_harness`
times 20 warm-up and 200 timed repetitions and keeps the minimum, which hides the slow blocks that
cause dropouts; `solver_counters.hpp` counts solves, failed solves and the worst iteration count;
four room and GUI tests time code by hand against millisecond bounds; `timing_gate.hpp` skips a
bound in an unoptimised build. Gates are fractions of a 256-frame block (2.67 ms, 2.0 ms, 0.15 ms
in its design documents). No load meter, dropout count or jitter is measured.

### 4.6 SushiTrack

A C++17 multi-object tracker called once per frame, with a C ABI, a Python binding, a Python
inference loop and a benchmark runner; it depends on SushiLog alone. The one C++ timing sums
`steady_clock` microseconds around `update` and prints a mean (`mot_runner.cpp:67-71`); the Python
loop times with `time.time()`; recorded results carry no date, commit or machine. Its backlog
items RW-11 and RW-12 cannot close without a per-frame cost to compare. Per-frame counts (objects
in and out, matches per stage) are measured nowhere.

## 5. What changes from the engine, and why

| Engine | Defect | SushiProfiler |
| --- | --- | --- |
| A zone writes a begin and an end event, 48 bytes, and the capture balances them: drops unmatched ends, closes open zones | Two ring writes per zone, and a repair pass that exists only because of the split | A zone writes one complete event at its end: begin, end, site, name, 24 bytes. Nesting is rebuilt by sorting. No unmatched end can exist. A zone still open at stop is reported as open in the record, with its begin |
| Events hold `const ZoneSite*`, a pointer into the image that declared the zone | Invalid after that image unloads; cannot be written to a file | A site registers once, on first execution, into a registry that copies its strings and returns a 32-bit `SiteID`. Events hold the id |
| One `zone_gate` for every zone; level and group only at compile time | Cannot turn on one group or one level without a rebuild | The gate stays one relaxed load while off. While on, a byte per site, set from the configuration, says whether that site records |
| The drainer wakes every 5 ms | Wakes a core 200 times a second with nothing to do | The drainer sleeps until a period ends, a ring crosses three quarters full, or a stop. A timed wait of one second is the backstop |
| `FrameSampler::mark_frame` drains and folds on the frame thread | The fold is charged to the frame it measures | `mark_period` writes one marker event. The consumer thread folds |
| Offsets inside a frame are `uint32_t` nanoseconds, held at 4.29 s | A 5 s YOLO step is clipped | 64-bit nanoseconds everywhere in the record |
| A device span that ended before the frame began is counted late and dropped | SushiRuntime reports a step's spans after the step; they would all be late | A source states the period each span belongs to. Nothing is dropped for arriving late |
| A device span holds a name and four times | No op, size, FLOPs or bytes; no link to the host zone that launched it | A span carries an attribute block and the thread and time of its submit, which places it under the host zone open at that moment |
| Tracy is a compile-time backend that replaces the recorder | A Tracy build leaves the sampler empty | Tracy, when built, is one more consumer of the same events (phase P7) |
| Modules register through static initialisers | A static library drops the translation unit and the registration with it | Each library exports one `register_profiler_parts(Registry&)` function its host calls |
| The level is one public compile definition for every target that links the module | One library cannot be compiled at a different density than another | Each library has its own level macro and option: `SP_`, `SR_`, `SB_`, `SA_PROFILE_LEVEL` |
| Capture and analysis end at the editor window | No record, no statistics, no comparison | Sections 6 and 8 |

Kept unchanged: the ring with its cached tail, the TSC read with its
tightest-bracket calibration, the two thread-retirement guards for the MSVC ABI, chunked storage
allocated ahead by the consumer, the fold's exact accounting, and the allocation test.

## 6. SushiProfiler in detail

### 6.1 Modules

A repository shaped like SushiLog: `modules/`, the `sp` CLI, the documentation tree, a hub
manifest, pinned by tag from every library that uses it. It depends on the C++17 standard library and
Threads alone, so a program outside the stack takes it without taking anything else. It logs
nothing: every failure returns a status, and drops are counted in the record. Namespace
`SushiProfiler`.

| Tier | Module | Holds |
| --- | --- | --- |
| foundation | `clock` | `TraceClock`, `CalibrationPoint`, `Calibration` |
| foundation | `events` | `TraceEvent`, `DeviceSpan`, `Counter`, `AttributeBlock`, ids |
| capture | `sites` | `ZoneSite`, the site registry, the per-site enable bytes, name interning |
| capture | `rings` | `ThreadRing`, thread registration and retirement |
| capture | `hub` | The process-wide gate and the drainer handoff (section 6.3) |
| capture | `zones` | `Zone`, `SP_ZONE`, `SP_ZONE_NAMED`, `SP_COUNTER`, `SP_THREAD_NAME` |
| capture | `lock_wait` | `ContendedLock` and `SP_ZONE_LOCK_WAIT`: a lock that opens a zone only when it waits |
| capture | `periods` | `PeriodTimeline` and `mark_period` |
| record | `record` | `RunRecord` and its JSON reader and writer |
| record | `statistics` | Steady-state check, median, percentiles, bootstrap interval |
| record | `analysis` | The `Analysis` interface, the generic analyses (section 8) and `AnalysisRunner`, which runs the named analyses over a record |
| record | `record_writers` | The `RecordWriter` interface, given a finished record: `terminal`, `record` |
| record | `compare` | Two or more records side by side |
| session | `config` | `ProfilerConfig` and its JSON reader |
| session | `registry` | `Registry`: factories for sources, analyses, writers and environment contributors, by name |
| session | `window` | The capture schedule over one timeline's periods |
| session | `sources` | The `Source` and `SpanSink` interfaces |
| session | `environment` | The `EnvironmentContributor` interface and the host facts SushiProfiler reads itself (OS, CPU, threads) |
| session | `fold` | Events to one zone tree per thread per period |
| session | `stream_writers` | The `StreamWriter` interface, given each period as it closes: `chrome_trace` |
| session | `consumer` | The consumer thread: drains, folds, collects from sources, appends to the record, feeds stream writers |
| session | `session` | `Session`: starts and stops one consumer with the parts the configuration names |
| tools | `sp_tool` | The program behind `sp show` and `sp compare` |

The record tier does not depend on the session tier, so `sp show` analyses a file with no session
running. A lower tier never includes a higher one; the layering checker the other repositories run
holds.

Attribute keys are a contract between the library that writes them and the analysis that reads
them. Each library declares its keys as constants in one header and lists them on one page,
`docs/reference/PROFILER_KEYS.md`; the record stores the key names, so a renamed key shows as
missing, not as a wrong number.

### 6.2 Events

```cpp
using SiteID = std::uint32_t;   // 0 is no site
using NameID = std::uint32_t;   // 0 is the site's own name
using TrackID = std::uint32_t;

struct TraceEvent               // 24 bytes, trivially copyable
{
    std::uint64_t begin_ticks;
    std::uint64_t end_ticks;    // equal to begin_ticks for a marker or a counter
    SiteID site;                // high 4 bits: kind (zone, marker, counter, period)
    NameID name;                // a counter stores its value in end_ticks as the bits of a double
};

struct DeviceSpan               // written by a source, never on a hot path
{
    std::uint64_t queued_ns;    // submit time, converted by the source onto the steady clock
    std::uint64_t start_ns;     // likewise
    std::uint64_t end_ns;       // likewise
    std::uint64_t period;       // the period the work belongs to
    std::uint32_t uncertainty_ns; // the source's clock-conversion uncertainty
    TrackID track;
    NameID name;
    std::uint32_t submit_thread;
    std::uint32_t attributes;   // offset into the period's attribute arena, or none
};
```

A device clock is converted by its source, which knows the pairing; SushiRuntime's carries the
pairing's `uncertainty_ns` into each span. `device_idle` treats a gap shorter than the uncertainty
of its two spans as unattributed rather than guessing.

An attribute block is a run of `(KeyID, kind, value)` triples in a per-period arena: integers,
doubles and interned strings. SushiBLAS writes `flops`, `bytes_read`, `bytes_written`, `dtype`;
SushiRuntime writes `node`, `operation`, `launched_width`, `live_width`; SushiAI writes `ir_node`
and `op`. SushiProfiler reads no key by name except in the analyses that say so.

### 6.3 The hot path

```cpp
#define SP_ZONE(name_literal) \
    static const ::SushiProfiler::ZoneSite SP_UNIQUE(site){name_literal, __func__, __FILE__, \
                                                         __LINE__, nullptr}; \
    ::SushiProfiler::Zone SP_UNIQUE(zone)(SP_UNIQUE(site))
```

- `Zone`'s constructor loads the gate (`std::atomic<std::uint32_t>`, relaxed). Zero means off: the
  zone stores nothing and its destructor tests one member.
- While on, it reads the site's id, registering the site on first use under a mutex, then the
  site's enable byte. If enabled, it reads the clock once; the destructor reads it again and pushes
  one `TraceEvent`.
- The push is the engine's single-producer ring write with its cached tail. A full ring counts a
  drop. Crossing three quarters full sets a flag and notifies the drainer, once per crossing.
- A thread's ring is created when the thread first records, or ahead of time through
  `register_this_thread(capacity)` so the first zone of a hot thread does not allocate.
- At level 0 the macros expand to nothing. Level 1 is stage zones, level 2 fine zones, level 3
  per-element detail. The runtime level in the configuration can lower what the compiled level
  allows, never raise it.

### 6.4 Records other than zones

The surveys of SushiDSP and SushiTrack on 2026-10-26 found measurements that are not a time range
on one thread. Each is a kind of `TraceEvent`, written through the same ring with the same cost
rules, and none carries a client's vocabulary.

| Kind | Macro or call | Event fields | Serves |
| --- | --- | --- | --- |
| Counter | `SP_COUNTER(site, value)`; the site declares a unit and an operation: `add`, `set`, `max`, `min` | time, value as the bits of a double | Counts per period (objects in and out, solver iterations); a worst case is `max`, not a sum |
| Value | `SP_VALUE(site, value)`; the site declares a unit | time, value | Samples that are not times (a level in dB, a queue depth); every sample kept, so a distribution is possible |
| Instant | `SP_INSTANT(site)` | time | An occurrence: a buffer underrun a device reports, a tracker losing an object |
| External time | `record_instant_at(site, steady_ns)`, `record_zone_at(site, begin_ns, end_ns)` | caller's times on the steady clock | A clock outside the profiler: a device callback, another library's own timer |
| Flow | `SP_FLOW_BEGIN(site, id)`, `SP_FLOW_STEP(site, id)`, `SP_FLOW_END(site, id)` | time, the caller's 64-bit id | Latency of one item across threads: a parameter handed to a real-time thread, a frame through a pipeline. Written as Chrome flow events |

A zone, counter or value may carry a lane, `SP_ZONE_IN(site, lane_id)`, when one thread serves
several independent streams: the record and the Chrome trace show each lane as its own row under
that thread. A lane id is interned like a name.

A counter or value site is a static object, registered like a zone site. Real-time rules hold for
every kind: no allocation after `register_this_thread`, no lock, no system call, a bounded cost, a
full ring drops and counts. The clock read is a user-space instruction on both platforms; the
record states the clock and its measured cost per read.

### 6.5 Periods and the window

A timeline is a named sequence of periods: `"train.step"`, `"bench.iteration"`, `"engine.frame"`.
`PeriodTimeline::mark()` pushes one period event on the caller's ring and returns the new period's
index. Several timelines may run at once; the configuration names the one the window follows.

A timeline may carry a budget: `PeriodTimeline(name, budget_ns)`, or `mark(budget_ns)` when it
changes per period. A real-time block's budget is its duration; a frame's is its interval. The
`budget` analysis reports each period's used fraction, the count over budget and the worst period.

A period may carry tags, `timeline.tag(key, value)`, written once per period off the hot path:
the input sequence, the variant under test. The record keeps them and `sp compare` groups by them.

The window is PyTorch's schedule: `skip` periods are ignored, then cycles of `wait` (nothing
recorded), `warmup` (recorded, marked warm-up), `active` (recorded), repeated `repeat` times, 0
meaning until the session stops. The gate opens at the first warm-up period and closes after the
last active one, so periods outside the window cost what an off profiler costs.

### 6.5 The session

```cpp
class Session final
{
    public:
        static Result<std::unique_ptr<Session>> start(const ProfilerConfig& config,
                                                       const Registry& registry);
        PeriodTimeline& timeline(std::string_view name);
        Result<RunRecord> stop();                 // joins the consumer; returns the record
};
```

`stop` returns the record and nothing else. The host then calls
`AnalysisRunner::run(record, config.analyses)` and each `RecordWriter`; `sp show` takes the same
path from a file. Stream writers are the only writers that run inside the session.

- One session runs per process; a second `start` is refused with a status naming the first.
- `start` builds the sources, analyses and writers the configuration names from the registry,
  refusing an unknown name with the list of known ones.
- The consumer thread drains rings, folds each closed period into one tree per thread with the
  engine's fold arithmetic on 64-bit offsets, asks each source for that period's spans and
  counters, and appends the period to the record. Writers that stream (the Chrome trace) write
  each period as it closes, so memory holds one window, not the run.
- Dependencies arrive through `start`; the only process-wide state is the hub, because a macro at
  any call site must reach it.

### 6.6 Sources

```cpp
class Source
{
    public:
        virtual ~Source() = default;
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        virtual Status begin(SourceContext& context) = 0;   // tracks, keys, clock
        virtual void collect(std::uint64_t period, SpanSink& sink) = 0;
        virtual void end() = 0;
};
```

A source that receives its data on another thread, as SushiRuntime's does after each run, queues
it and hands it over in `collect`. `collect` runs on the consumer thread.

A span's period is the index of the configured timeline's current period when the work was
submitted: the source reads `PeriodTimeline::current()` (one relaxed load) when it is handed a
run, so the two runs of one anchor-free step both land in that step. `collect(period)` returns
every span tagged with that period; a span tagged with a later period waits for it. A repeated
dispatch of one node keeps its `invocation` as an attribute, so no two dispatches collapse.

An `EnvironmentContributor` adds named facts to the record's `environment` when the session
starts: SushiRuntime's adds the device, driver and backend, and each library adds its own name,
tag and commit.

### 6.7 Configuration

`ProfilerConfig` is one value every host fills from its own settings. Its JSON form:

```json
{
    "sushiprofiler": 1,
    "enabled": true,
    "level": 2,
    "groups": ["Data", "Device", "Host stage"],
    "timeline": "train.step",
    "window": { "skip": 0, "wait": 0, "warmup": 3, "active": 20, "repeat": 1 },
    "sources": { "sushiruntime.device": {}, "sushiai.attribution": {} },
    "analyses": ["breakdown", "device_idle", "top_spans", "roofline"],
    "writers": {
        "terminal": { "top": 20 },
        "record": { "path": "run.sprec.json" },
        "chrome_trace": { "path": "run.trace.json" }
    },
    "output": "build/profiles/yolov5n",
    "buffers": { "ring_events": 65536, "attribute_bytes": 1048576 }
}
```

The reader refuses an unknown key and a value out of range, naming the key. The format has a page
in `docs/reference/` and a version field.

## 7. The run record

One JSON file, `<name>.sprec.json`, version 1:

- `environment`: host, CPU, OS, device name and driver, SYCL backend, the build id and tag of every
  Sushi library that registered parts, thread counts, the configuration as read;
- `sites`, `names`, `keys`, `tracks`: the interned tables, written once;
- `periods`: per period its index, timeline, begin and end, whether warm-up, the execution layer
  its source reported, and per thread the zone tree (site, name, parent, inclusive, self, calls);
- `spans`: per period, structure of arrays (`queued`, `start`, `end`, `track`, `name`,
  `submit_thread`, `uncertainty`, then each attribute key as a column), so a key's name is written
  once per period, not once per span;
- `samples`: per period, the counters, values, instants and flow points, as columns per site;
- `tags`: the run's tags from the configuration and each period's own;
- `build`: per contributing library, its build type, optimisation and sanitisers;
- `clock`: the clock used, its resolution and its measured cost per read;
- `drops`: dropped events per thread, dropped spans, zones open at stop.

Analyses read a record, never the live program, so `sp show` re-analyses an old record with a
newer analysis.

## 8. Analyses

Generic, in SushiProfiler:

| Name | Answers |
| --- | --- |
| `breakdown` | Per period, self time per group on the timeline's thread, plus time no zone covered |
| `device_idle` | Per period, device busy and idle time; each idle gap attributed to the host zone open on the submitting thread during the gap, else to "launch" if the next span was queued before the gap began, else "unattributed" |
| `top_spans` | Spans grouped by name and by any attribute the configuration names, with count, total, median and share |
| `short_spans` | Count and total of spans under a threshold (10 µs by default) against the launch floor when one is known |
| `overhead` | Events and bytes recorded, drops, and the cost of the profiler measured against a baseline record |
| `budget` | Per timeline with a budget: used fraction per period, periods over budget, the worst period and its zones |
| `distribution` | Per period time, zone, counter and value: p50, p90, p99, maximum, a log-bucketed histogram, and jitter as the median absolute difference between successive periods |
| `counters` | Per counter site, its operation applied per period (sum, last, maximum, minimum), then the distribution over periods |
| `flows` | Per flow site, the latency from begin to end and between steps, as a distribution |

In SushiBLAS, `roofline`: per span with `flops` and bytes, achieved FLOP/s and bytes/s, arithmetic
intensity, and the fraction of the bound ceiling from a ceilings file.

Every timing is reported as median, bootstrap 95% interval (2,000 resamples, fixed seed, so the
same record gives the same text), minimum, p95 and coefficient of variation over the active
periods. A steady-state check (MSER-5 over period time) says whether the warm-up was long enough
and recommends a length when it was not.

`sp compare a b` reports each row's ratio of medians with its interval and marks a change only
when the two intervals do not overlap. It refuses two records whose build environment differs
(optimised against not, sanitised against not) unless told `--allow-build-mismatch`, and it
groups rows by period tags when both records carry them.

Checks turn a record into a verdict. A check is data, not a string to parse:

```json
{ "subject": "timeline:dsp.block", "statistic": "p99", "at_most_ns": 2670000 }
```

`sp check <record> <checks.json>` exits non-zero when a check fails, naming it, and
`check(record, checks)` returns the same verdicts to a C++ test. A library's timing tests and
benchmark gates then read a record instead of timing code by hand.

The record schema is a published, versioned file format with a validator, `sp validate`. A program
in another language, such as SushiTrack's Python pipeline, writes a record by that schema and every
analysis, comparison and check reads it like any other.

## 9. The libraries above

### 9.1 SushiRuntime

- `sushiruntime.device`: a source that converts each run's `node_events` through `ClockPairing`
  into spans, with `node`, `invocation`, `operation` and the widths as attributes, and reports the
  run's execution layer for the period.
- `NodeEventRecord` carries no operation id today. P3 adds to `RunReport` a node table, one row
  per registered node with its name and `OperationID`, indexed by `node_index`. The change only
  adds a field; no existing member moves.
- SushiProfiler is an optional dependency of each library: an option (`SR_ENABLE_PROFILER`, and
  likewise in SushiBLAS and SushiAI) defaults to on; off, the zone macros expand to nothing and the
  sources are not built, and the library builds without SushiProfiler present.
- `SR_ZONE` in plan build, submit, wait and poll, under the group `Runtime`.
- The graph asks for node events only while the hub wants device spans.

### 9.2 SushiBLAS

- A cost rule per operation id, giving FLOPs and bytes from the operands' shapes and types,
  written as span attributes. A test proves every recorded operation has a rule or is declared
  uncosted, as SushiAI's autograd table proves backward rules.
- `device_ceilings`, a benchmark `sb bench` runs: FMA loops per precision, a streaming copy, an
  empty kernel. It writes `<device>.ceilings.json`.
- `roofline`, the analysis of section 8.
- The benchmark harness marks iterations on the timeline `bench.iteration`, so every benchmark
  writes a record.

### 9.3 SushiAI

- `SA_ZONE` groups: `Data` (read, decode, augment), `Host` (feed, assignment, loss read, measure),
  `Copy` (input copy), `Device` (the step's runs and waits), `Optimizer`, `Evaluation` (decode,
  NMS, mean average precision).
- The timelines `train.step` and `inference.batch`.
- `sushiai.attribution`: a source that adds `ir_node` and `op` to each span. Lowering records,
  per IR node, the range of runtime registration indices it emitted (`first`, `count`), kept by
  the compiled step. `NodeEventRecord::node_index` is the registration index, so the join is a
  lookup. A test lowers a graph whose node emits three tasks and checks all three map back to it.
- Two runs answer the owner's question together. A run with `level` 1 and runtime profiling off
  measures the true step time and its host groups on the production execution path. A run with
  runtime profiling on gives the kernel breakdown on the node-by-node path. `sp compare` of the two
  states the cost of the second, so its findings are read with that cost in view.
- A `profile` section in the training and inference run files holding a `ProfilerConfig`. Absent,
  nothing records and the gate never opens.
- `sa profile show` and `sa profile compare`, forwarding to `sp`.

## 10. Performance budget

Measured by benchmarks in SushiProfiler, on the owner's machine, before P2 closes:

| Case | Budget |
| --- | --- |
| Zone, compiled at level 0 | No instruction |
| Zone, gate closed | One relaxed load and one branch; under 2 ns |
| Zone, recording | Under 25 ns, two clock reads and one 24-byte store |
| Zone thread | No allocation after `register_this_thread` |
| Drainer while idle | Under one wake-up a second |
| YOLOv5n step with stage zones on | Under 1% slower than with the profiler off, measured |

## 11. Measurement rules

1. Device time comes from SYCL event profiling, never from a host clock around a submit.
2. Warm-up periods are excluded by the window and reported with their count.
3. Medians with intervals, never means.
4. A comparison names an improvement only when intervals do not overlap.
5. The record holds the environment in full.
6. The profiler's own cost is in every record that has a baseline.

## 12. Phases

Each phase lands with its tests, module READMEs and changelog lines in the repository it names,
and each repository releases before the one above pins it.

| Id | Phase | Repository | Accepted when |
| --- | --- | --- | --- |
| P1 | Skeleton; `clock`, `events`, `sites`, `rings`, `hub`, `zones`, `periods` | SushiProfiler | Zone and ring tests pass; the allocation test passes; the section 10 benchmarks meet their budget |
| P2 | `config`, `registry`, `window`, `sources`, `environment`, `fold`, `consumer`, `session`, `record`, `statistics`, `analysis`, the writers, `compare`, checks, `sp show / compare / check / validate` | SushiProfiler | A test program's periods, counters, values, instants and flows give a record, a terminal summary and a Chrome trace that Perfetto opens; two records compare; a check fails when it should; a record round-trips and validates |
| P3 | The device source and runtime zones | SushiRuntime | A profiled two-node graph yields two spans on the session clock, inside their period |
| P4 | Cost rules, `device_ceilings`, `roofline`, the harness timeline | SushiBLAS | `sb bench` writes a record whose GEMM spans carry FLOPs and a ceiling fraction |
| P5 | Zones, timelines, attribution, run-file section, `sa profile` | SushiAI | A profiled YOLOv5n run writes a record that splits a step into its groups and names the IR node of each kernel |
| P6 | The first finding: where YOLOv5n's step goes, written up with the records | SushiAI | The report names the share of each group and the kernels that bound the step |
| P7 | The benchmark harness writes records over a budget timeline per block; the solver counters become counter sites; the hand-timed tests and `timing_gate` become checks | SushiDSP | `sd bench` writes a record with p99 block time and budget fraction, and the four hand-timed tests read checks |
| P8 | `mot_runner` marks a period per frame with zones per stage of `update` and counters per stage; the evaluator reads the record instead of a derived mean | SushiTrack | The benchmark tables are produced from records that carry date, commit and machine, and RW-11 and RW-12 have a number to compare |
| P9 | Later: a C ABI and a Python record writer; Tracy consumer; what-if from the dependency graph; memory timeline; hardware counters | all | Separate designs |

P7 and P8 follow P2's release and do not wait on P3 to P6.

SushiEngine's move onto SushiProfiler is not in this table. It is proposed to the session working
there, with a shim from `SE_ZONE` to `SP_ZONE` that keeps the editor's Profiler window working.

## 13. Calls made without the owner

- The name SushiProfiler, alias `sp`, macros with each library's own prefix.
- No dependency on SushiLog or any other Sushi library.
- `FrameProfiler`, with its script-module ledger, stays in the engine as a consumer.
- One complete event per zone instead of a begin and an end.
- Chrome trace JSON for the timeline; a JSON run record with columns for spans. A binary format
  waits until a measured record is too large.
- The configuration refuses unknown keys.
- MSER-5 for the steady-state check and a fixed-seed bootstrap for intervals.
- SushiProfiler is an optional dependency, on by default, in every library that uses it.
- SushiRuntime's `RunReport` gains a node table, an additive change.
- Findings are drawn from two runs, one on the production path and one profiled, compared.
- Counters, values, instants, external times, flows and lanes are event kinds of the one ring, not
  separate mechanisms.
- Checks are JSON data, not an expression language.
- Python reaches SushiProfiler through the record schema first; a binding waits for P7.

## 14. Open questions

- Whether SushiRuntime's registration index counts every task lowering emits, including tasks
  SushiBLAS records for scratch or late pairs. P5's test answers it; if not, lowering passes the IR
  node id in the task description, a SushiBLAS change.
- Whether a period on the command-graph replay path can be timed as one span, so an unprofiled run
  still has a device total.
- The repository's licence text and hub catalog entry follow SushiLog's; the GitHub repository
  itself is created by the owner.
