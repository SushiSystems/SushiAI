# examples

Two programs and two configuration folders.

| Path | What it is |
| --- | --- |
| `example.cpp` | A teaching file: one training run, read top to bottom |
| `demo_mlp.cpp` | The program behind `sa demo mlp` |
| [`heavy_mlp/`](heavy_mlp/README.md) | Three JSON files for `sushiai train` and `sushiai inference`: a deep MLP on the rings dataset |
| [`mnist_cnn/`](mnist_cnn/README.md) | Three JSON files for the same two commands: a convolutional network on MNIST |

The sections below hold the reasoning that the source files cite.

## example.cpp

The example declares a model, traces it into IR, differentiates the IR, compiles the result
once and replays it per batch. It uses MNIST when the IDX files are in `data/mnist` and a
seeded synthetic problem otherwise, so it runs anywhere with no download and no fixture, and
it prints which of the two it picked.

The model is an aggregate list. A parameter is discoverable because the aggregate declares
it; nothing has to remember to register it.

Tracing the forward pass, adding cross-entropy and differentiating are all IR-to-IR
transforms. No device work happens until the `Trainer` is constructed. Its constructor lowers
and compiles the step, which is the only compile of the run, and the report printed at the
end has to agree: the program exits with a failure when the compile count is not 1.

## demo_mlp.cpp

### Why it is a separate program from example.cpp

The two programs look alike and are two binaries on purpose.

`example.cpp` is a teaching file. It is meant to be read top to bottom and edited by whoever
is learning the API. Its output has no contract, and its shape is hardcoded so that nothing
distracts from the six steps of a training run.

`demo_mlp.cpp` is a product surface. `sa demo mlp` is a supported command: its flags and its
printed lines are a contract, `tests/CMakeLists.txt` pins the compile-count line so the demo
cannot rot unnoticed, and `docs/README.md` quotes its real output.

Pointing the CLI at the example binary would make every edit to a teaching file a change to
a shipped command, and would leave the demo with no flags at all. One extra translation unit
is the price of keeping those two lifecycles apart.

### Data source

The demo trains on MNIST when the training split's IDX files are present, and on the seeded
synthetic problem otherwise. Both paths come from `include/SushiAI/data/dataset.hpp`, where
the absence of the IDX files is not an error. The demo only chooses between the two and
records which one it got.

`DataChoice` carries the provenance note together with the dataset. "Which data did this run
see" is the first question anyone asks of a training number, and the answer must come from
the code path that chose the data, never from a second path that recomputes it.

The reported step count comes from the dataset and not from `--batches`. The MNIST path caps
itself at what the files hold, so the two can differ, and the schedule printed has to be the
one that runs.

### The two activations

`ActivationKind` has two entries where the library has more activations, because these two
are the two kinds. ReLU's gradient reads its own output, so it folds into the GEMM epilogue
with nothing else written. GELU's gradient reads its input, so it folds only because
`FusedForm::GemmBiasGelu` spills that input to a second buffer.

Running both from one binary turns the cost of that spill into a measurement. The fusion
report, `ops per step` and the `Buffers` line are printed for whichever activation was asked
for, and comparing `--activation gelu` with `--activation gelu --no-fuse gemm+bias+gelu` is
the A/B that shows what the entry buys.

### Settings and their defaults

The defaults reproduce the run quoted in `docs/README.md`, so `sa demo mlp` with no flags is
the run the documentation describes.

`learning_rate` is a `std::optional` and not a 0 sentinel, so `--lr 0` is a rejected value
and not a silent request for the default.

`fusion` is a flag and not a rebuild. The fused and unfused paths are supposed to produce the
same numbers and differ only in launch count, and the way to show that is to run both from
one binary against one SushiBLAS. Anyone reproducing the figures in `docs/README.md` needs
the same measurement.

`without` is the per-form half of `--fusion`. Fusion's payoff is a launch count, and a launch
count is worth something only if the wall clock agrees. Switching one form off and measuring
again is how a user finds out which catalog entry pays on their model and their device.

`mixed_precision` is off by default, and the default carries weight: every number the demo
prints, and every number quoted in `docs/README.md`, is an fp32 number, and the existence of
mixed precision must not move them. It is a flag for the same reason `--fusion` is one: the
question is what the difference is on one machine, one dataset and one seed, and only two
runs of the same binary answer it. The development machine has no low-precision matrix unit,
so the difference worth measuring there is the loss curve and never the wall clock. See
ground truth #23 in [`docs/design/ROADMAP.md`](../docs/design/ROADMAP.md).

`loss_scale` is off by default. With it off no node is emitted, so the graph, the operation
count and every number the demo prints are what they were before loss scaling existed. On
this workload fp16 without loss scaling neither stalls nor diverges, and both arms converge
to the same loss. The flag exists to make the machinery runnable and inspectable; the demo
does not need it. See `include/SushiAI/optim/loss_scaler.hpp`.

### Command line parsing

`parse_command_line` is a hand-written loop and not a parsing library. The whole grammar is
`--flag value`, every flag takes exactly one value, and a dependency for that would cost more
than it saves. Because every flag takes one value, the check for a missing value is written
once, in the `value` lambda, and not at each branch.

`validate` checks the extents in the demo, before the library sees them, so the message names
the flag the user typed and not an internal extent.

`parse_int`, `parse_seed` and `parse_real` report a malformed value and an out-of-range value
with one message. The distinction means nothing to whoever typed the value.

`--loss-scale` is checked only for positivity in the demo. That the value is a power of two
is `LossScaleState`'s rule, stated in `include/SushiAI/optim/loss_scaler.hpp`, and a
duplicate check in the demo would say it less well.

### Learning-rate defaults

The default learning rate is per optimizer: 3e-3 for AdamW and 0.2 for SGD. 3e-3 is a
sensible AdamW step and a uselessly small SGD one, so a shared default would make
`--optimizer sgd` look broken when it is only different. SGD runs with momentum 0.9, and the
report prints it.

### The configuration report

`human_bytes` prints bytes below a kibibyte. Integer KiB alone reads as "0 KiB instead of
0 KiB" on the two-step run the CTest smoke test uses, which makes the planner look broken
when it is working on a tiny model.

`describe_fusion` prints what the pass fused and what it refused. Every way the fusion pass
can be wrong produces a run that still converges, so the report shows the decisions. A reader
who sees "gemm+bias+relu x1, gemm+bias x1" next to "declined sushiai.gelu: its gradient
differentiates at its input" can tell whether the model got what it should have. The one form
that writes two values names the spilled value on its fused line. Without that note a reader
cannot tell the safe fold from the silently wrong one, because the two look identical in
every other number printed.

`describe_precision` prints the cast pass's result for the same reason: a conversion in the
wrong place is a run that still converges. Under fp32 it prints one line saying that every
operation computes in the graph's own type, which keeps the banner the same shape in both
modes.

`describe_arenas` shows the memory planner's saving. The `Buffers` line prints what was
allocated next to what one allocation per tensor would have cost, and the `Arenas` line says
which arenas account for the difference. When the planner declines to overlap live ranges,
for example on a graph too large for the reachability closure, the report prints that reason
and not a silent 0%.

### One template, two models

`train_and_report` is a template because the model's type is its structure. `NN::Sequential`
is a variadic aggregate, so the same MLP with GELU in place of ReLU is a different type and
not a different value, and `--activation` picks between two instantiations in `main`.
Everything from the trace to the final compile-count line lives in the template, so the two
arms cannot drift into printing different things.

Tracing, adding cross-entropy, differentiating, inserting casts and fusing are all IR-to-IR.
No device work has happened and nothing has been compiled when `build_classifier` returns.

Both precision policies are constructed and a pointer chooses between them. A policy object
is a handful of bytes with no device behind it, and holding one of each settles the lifetime
question: the chosen policy has to outlive `build_classifier` and the `Trainer`'s
constructor, and both run later in the same scope.

The `LossScaler` is declared before the `if (scaling)` block and not inside it, so it
outlives the trainer. The trainer records the optimizer's update, and with it the scaler's
screens, into the compiled plan.

The `Trainer`'s constructor lowers and compiles the step. The demo drives the epochs itself,
one `fit()` call each, so the per-epoch trajectory is a measurement and the compile count is
counted across every one of those calls.

The trainer walks the graph against the same precision policy the graph was built with. This
is the last refusal before the graph becomes device work. Under fp32 the check is the
existing rule that one element type holds everywhere, so it costs one walk and asserts what
was already true.

### Printing the result as one block

The trajectory is collected in a string and printed after training. SushiRuntime logs at
INFO to stdout in a Release build. Its level is a compile-time constant, so a consumer cannot
turn it down, and a report emitted line by line comes out shredded between the log lines.
Held back and printed in one write, the result survives intact.

The compile count is the last line on screen. Every step replayed one compiled plan, and
that line is the number that says so.

### Phases inside the functions

The source marked the phases of `train_and_report` and `main` (memory and initialisation,
the one compile, training, data, model) with separator comments. The comments are gone and
the functions were not split.
