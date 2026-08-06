# A convolutional layer set: ResNet-18 and the YOLOv3-tiny body

Status: design, approved 2026-08-07.
Repositories touched: **SushiAI** (layers, IR, autograd) and **SushiBLAS** (kernels).

This design adds the operation set, kernels and differentiation rules a
convolutional network needs, and assembles two models on top of it: ResNet-18
trained end to end, and the YOLOv3-tiny **body** run forward. It does not add a
detection head.

---

## 1. Scope

### In

- **SushiBLAS**: a `spatial` domain (`im2col`, `col2im`, max/average pooling,
  nearest-neighbour upsample, channel concatenation, per-channel affine), one
  segment-axis generalisation of `Reduce::AxisSpan`, and a layout-parametric
  GEMM bias epilogue.
- **SushiAI**: twenty new operation identities, their shape inference, their
  backward rules, a rematerialisation pass for `IM2COL`, and the layers
  `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool`, `Flatten`,
  `Upsample2d`, `Concat`, `LeakyReLU`, `BasicBlock`.
- **A restructuring of `src/autograd/backward.cpp`** into per-domain translation
  units, done *before* any new rule is written.
- **A `Buffer` concept** in the module system, for non-learnable checkpointed
  state (BatchNorm's running statistics).
- **Named children** in the module walk, alongside the existing index naming.
- **A data pipeline** for CIFAR-10, with host-side augmentation.
- **Parameter-name and tensor-layout parity with PyTorch**, so a `state_dict`
  converts to a checkpoint without reshaping logic.

### Out, and what each would actually need

- **The YOLO detection head.** Anchor assignment, a multi-part loss (box,
  objectness, class), NMS and mAP. Its own spec: none of it intersects the layer
  set, and the box-loss choice (MSE / GIoU / CIoU) is a design argument in its
  own right. Without it, YOLOv3-tiny can be run forward and its feature maps
  compared against a reference, but not trained.
- **Mixed precision for convolution.** `graph/cast_insertion.hpp`,
  `graph/precision.hpp` and `optim/loss_scaler.hpp` already exist; binding
  convolution to that policy is additive and deferred. Note that
  `autograd/gradcheck.hpp` documents itself as incompatible with mixed precision
  by design, so the two land in that order and not the reverse.
- **Native NCHW execution.** See CV-3.
- **Grouped and depthwise convolution.** Grouped convolution is expressible as
  G convolutions over channel slices — IR-level composition, no new attribute —
  and depthwise convolution is pathological for `im2col` + GEMM (K = R·S = 9 is
  a degenerate GEMM). Both wait for a measurement.
- **Winograd.** Not foreclosed; see CV-2.

### Relationship to `2026-08-07-config-driven-training-design.md`

That design is committed (`4ce8b90`) and **not yet implemented** — `config/`,
`nn/ilayer.hpp` and `io/` do not exist in the tree. This design therefore does
**not** depend on it, and the two can proceed in either order:

- Layers here are compile-time aggregates satisfying `NN::Module`, which is what
  `Sequential` already consumes. They need nothing from `LayerStack`.
- §11 lists the `NN::LayerRegistry` rows and factory functions each layer needs
  once that work lands. That spec anticipated this: it names `"conv2d"` and
  `"route"` as future rows and commits that `apps/train.cpp` will not change
  when they arrive.
- Checkpoint **file I/O** is that spec's `SACP` format. This design adds no
  second format; it constrains only names and layouts (§10).

---

## 2. Decisions

### CV-1 — Convolution is `IM2COL` + `MATMUL` in the IR, not an opaque node

`GraphBuilder::conv2d()` emits two nodes: `IM2COL(x) → col`, then
`MATMUL(col, W, transpose_b=true)`. The weight is used through a `RESHAPE` view
from `[C_out, R, S, C_in]` to `[C_out, R·S·C_in]`, which is contiguous and
therefore free.

What this buys, and each is a thing not written rather than a thing written:

| | |
|---|---|
| Half of convolution's backward | `matmul_backward` already produces both dgrad and wgrad. The only new rule is `im2col_backward`. |
| Convolution + bias + activation fusion | `graph/fusion.cpp` already matches `MATMUL`+`BIAS_ADD`+`RELU` → `GEMM_BIAS_RELU`. **Zero new catalog entries.** |
| Memory planning | The `col` buffer is a real `Value` with a live range, visible to the planner — not a hidden allocation inside a kernel. |
| The `MATMUL` rank-2 constraint | `src/graph/builder.cpp:82` rejects non-rank-2 operands. The `col` matrix is `[N·H_o·W_o, C_in·R·S]` and the weight view is `[C_out, C_in·R·S]`. Both rank-2: the constraint is not touched, and no batched matmul is needed. (The detection head would need one; it is out of scope.) |

**1×1 stride-1 convolution emits no `IM2COL` at all** — `im2col` is the identity
transform there, so `conv2d()` recognises the case and emits `RESHAPE` +
`MATMUL`. ResNet-18's projection shortcuts are exactly this shape.

### CV-2 — This does not foreclose implicit GEMM or Winograd

Both are algorithmically different from `im2col` + GEMM, and both are reachable
**as fusion-catalog rewrites of the `IM2COL`+`MATMUL` pair**, using the mechanism
`graph/fusion.hpp` already provides. The IR does not change to admit them. This
matters because 3×3 stride-1 convolutions dominate both models in scope and
Winograd F(2×2,3×3) uses ~2.25× fewer multiplies — a real future gain that this
decision keeps available rather than trades away.

### CV-3 — One native layout: NHWC. The other is a boundary transpose, not a second implementation

Activations are NHWC. This is a reversal of an earlier position in this design's
own review, and the reasoning is recorded because the reversal is the useful
part.

**Why NHWC:**

- The convolution GEMM becomes `[N·H·W, C_in·R·S] × [C_in·R·S, C_out]` — the
  *same call shape as `Linear`*, with M = N·H·W large. NCHW gives
  `[C_out, C_in·R·S] × [C_in·R·S, N·H·W]`, whose M is C_out (64 in ResNet-18's
  first layers), too small to tile well against an 8×8 register block.
- The channel index becomes the epilogue's `col` argument, so the existing
  `GEMM_BIAS` / `GEMM_BIAS_RELU` family applies to convolution **verbatim**. In
  NCHW the channel is `row` and every fused recorder would need a second form.
- `im2col` copies contiguous runs of `C_in` elements rather than gathering single
  strided elements.
- BatchNorm's per-channel statistics are a reduction over a zero-copy rank-2
  `[N·H·W, C]` view, which the existing `Reduce::AxisSpan` handles unchanged.

**Why only one:** supporting both natively was costed at ~1.4×, not the ~1.0×
first claimed. The kernels genuinely collapse — a layout, an axis and a strided
view all become host-computed stride numbers, exactly as `Reduce::AxisSpan` and
`Detail::strides_for` already do. What does **not** collapse is the IR: the
`col` matrices of the two layouts are transposes of each other, so the following
`MATMUL` carries a different (M,N,K) assignment, `conv2d()` emits structurally
different IR, and every fused lowering recorder becomes layout-aware. Paying
that permanently with no measurement contradicts this stack's own culture — the
`K_CPU_TWO_STAGE_MIN_ELEMENTS` threshold in `docs/ARCHITECTURE.md §7` is
documented as *"measured, not assumed"* and states its crossover point.

**What a transpose does and does not give.** A `LAYOUT_CONVERT` node permutes
NCHW to NHWC at the graph boundary and back — two O(N·C·H·W) passes per step,
immaterial beside the convolutions, and exactly what PyTorch's `channels_last`
does. That is **interoperability**: it lets NCHW data and NCHW checkpoints in.
It is *not* an NCHW implementation, because the point of one would be that
kernels address memory in NCHW order, which converting away defeats. Native
NCHW remains a pure addition behind the seam, gated on a measurement.

**The seam is still built layout-parametric** — `Spatial::WindowSpan` names no
layout, and `Kernels::ChannelBiasEpilogue` indexes `bias[row·rb + col·cb]` with
host-set `(rb,cb)`. Adding NCHW later is filling in a second stride assignment,
not re-cutting an abstraction.

### CV-4 — Layout is a graph-level invariant

`Graph::Value` carries a `Layout` for checking, but the invariant is graph-wide
and `Graph::finalise()` proves it. `LAYOUT_CONVERT` is the only node permitted to
change it.

A free-floating per-value layout would make every operation's contract "every
combination of my operands' layouts", a combinatorial surface nobody tests.
SushiBLAS's per-tensor `Core::Layout` is not a counter-precedent: mixed-layout
GEMM is a real BLAS use case, whereas a layout change in the middle of a
convolutional network is a bug.

### CV-5 — Full 2-D convolution geometry, in exactly eight attribute slots

`Graph::Attributes::CAPACITY` goes from 4 to 8. Eight is `TaskMetadata`'s
`K_MAX_PARAMS`, verified in `SushiRuntime/graph/task_types.hpp:146` — a hard
ceiling, and the new value sits exactly on it.

The eight are `kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w,
pad_top, pad_left`. **`pad_bottom` and `pad_right` are absent because the kernel
never needs them.** The kernel addresses input as
`h_in = h_o·stride_h − pad_top + r·dilation_h` and bounds-checks against
`[0, H)`; the trailing pads affect only *how many output rows exist*, which is
`H_o`, which the host computes and writes into the output `Value`'s `Shape`:

```
H_o = floor((H + pad_top + pad_bottom − dilation_h·(kernel_h − 1) − 1) / stride_h) + 1
```

So asymmetric padding costs **zero extra slots** over symmetric, and no geometry
restriction survives: non-square kernels, anisotropic stride and dilation, and
independent padding on all four sides are all expressible. This is what makes
YOLOv3-tiny's 2×2 stride-1 max-pool — padded on the right and bottom only, to
hold 13×13 → 13×13 — a supported case rather than a special one.

**The escape hatch is designed now and not walked through.** Nine knobs do not
fit; `groups` would be the ninth. Grouped convolution has an IR-composition
answer (CV-1's philosophy: G convolutions over channel slices), but 3-D
convolution does not. An operation exceeding eight uses a `NodeId`-keyed side
table on `Graph`, leaving `Attributes` as the inline fast path. Naming it here
means the first such operation does not force a simultaneous redesign of
`Attributes`, its `TaskMetadata` mapping, and every lowering recorder.

### CV-6 — `col2im` and every pooling backward are gathers, never atomic scatters

SushiBLAS links `sushiruntime_deterministic_fp` PUBLIC. A floating-point atomic
add has a combination order that varies between runs, so writing `col2im` as
"each column entry atomically adds into its pixel" would silently break the
library's bit-reproducibility guarantee — the same argument
`docs/ARCHITECTURE.md §7` gives for rejecting `sycl::reduction` in the folding
reductions.

Every backward in the `spatial` domain is therefore written so that **each output
element gathers its own contributions in a fixed order**: `col2im` walks the
column entries that read a given pixel, `maxpool2d_backward` walks the windows
that could have selected a given pixel, `upsample2d_backward` sums the fixed
number of outputs that read a given input.

### CV-7 — BatchNorm's running statistics are not IR nodes

`ir.hpp`'s central invariant is that values are single-assignment and no
operation mutates its inputs — the property that lets differentiation skip
aliasing analysis entirely. BatchNorm mutates `running_mean` and `running_var`
every step, which is a direct violation if expressed as an IR node.

They are instead **recorded tasks outside the differentiated graph**, in the class
`ops.hpp` already describes for `ADAMW_UPDATE` and `SGD_UPDATE`: *"Not IR
operations: no node carries these and no backward rule looks them up."* The
identity `BATCHNORM_UPDATE_STATS` is declared in `Ops::ALL` so the collision
proof covers it, and it is marked non-differentiable (CV-9).

This is recorded as a decision rather than a detail because **getting it wrong is
silent**. If the statistics are never updated, training runs perfectly — the loss
falls, nothing throws, nothing goes NaN — and the model produces garbage the
moment it is evaluated. That is the same failure genre as the `GEMM_BIAS_GELU`
warning and the `execute_reduction` bare-pointer regression, both of which this
codebase documents at length.

**Verification:** an integration test asserts `running_mean` changes across two
training steps and that an evaluation-mode forward pass over the training batch
approximates the training-mode output. A test that only checks training loss
cannot distinguish this bug from correctness.

### CV-8 — Training and evaluation are two compiled steps

BatchNorm's two modes are different graphs — one computes batch statistics, the
other is an affine map reading the running statistics — and a branch is not
expressible in a graph that is compiled once and replayed. `CompiledStep`'s
compile count is a pinned guarantee, so a mode flag cannot be a graph input.

`Train::TrainStep` and `Train::EvalStep` are therefore two `CompiledStep`s over
one parameter set and one pool. Parameters and buffers are shared by
construction (both bind the same tensors); activation arenas are not, since
their liveness differs.

This is built as a **mechanism, not a BatchNorm patch**: a layer declares which
mode it is being traced for, and dropout — the next operation with this property
— reuses it unchanged.

### CV-9 — Differentiability is declared and proved at compile time

`Ops::OpName` gains a `differentiable` flag, and a `static_assert` proves that
every operation so marked has a rule in the backward table.

This is the third instance of a discipline the codebase already applies twice —
`Ops::all_ids_unique()` and `fusion.cpp`'s `every_fused_activation_survives()` —
and it closes the one real risk in adding twenty operations at once: declaring an
identity and forgetting its rule is today not a compile error but a silent gap.

Placing the flag in `core/ops.hpp` is not a layering inversion: that file already
carries `ACTIVATIONS` and `GradientSource`, which are purely autograd concerns.

### CV-10 — Rematerialisation is `IM2COL`-specific, and named as such

The `col` buffer must survive to backward for the weight gradient. Kept, ResNet-18
at batch 128 does not fit 6 GB: the first block's buffer alone is
128·32·32·64·9·4 bytes ≈ 302 MB, and there are twenty convolutions.

After differentiation, a pass clones each `IM2COL` node next to its backward
consumer, so the buffer's lifetime drops from `ACTIVATION` to `SCRATCH` and only
one is live at a time. The cost is one extra gather pass, negligible beside the
GEMM it feeds. `ir.hpp` anticipates the mechanism: *"inserting a rematerialised
copy."*

**It is deliberately not a general framework.** A general rematerialisation pass
would have one client, no cost model, and would be making a memory policy
decision without memory information — the planner is what knows whether the
budget is exceeded, and unconditional rematerialisation is a free slowdown when
it is not. `IM2COL` is exempt from needing that model because its buffer is
always enormous relative to its recompute cost. Planner-driven rematerialisation
is a separate design question.

---

## 3. Where each piece lives

| Layer | Responsibility | Contents |
|---|---|---|
| **SushiBLAS** `engine/spatial/` | Windowed and strided movement over an N-D grid | `im2col`, `col2im`, `pool_max`, `pool_avg`, `upsample_nearest`, `concat_channels`, `channel_affine` |
| **SushiBLAS** `engine/reduce_axis.hpp` | Per-channel and per-sample statistics | a second **segment** axis on `AxisSpan` |
| **SushiBLAS** `include/SushiBLAS/kernels/blas/level3/` | Layout-parametric channel broadcast | `ChannelBiasEpilogue` |
| **SushiAI** `core/ops.hpp` | Identity | eleven `OpName`s, `differentiable` flag |
| **SushiAI** `graph/builder.hpp` | Shape inference, node emission | `conv2d`, `max_pool2d`, … |
| **SushiAI** `autograd/rules/` | Differentiation, one domain per translation unit | `spatial.cpp`, `norm.cpp`, `shape.cpp` |
| **SushiAI** `nn/layers/` | What a user writes | `Conv2d{3, 64, 7, /*stride*/2, /*pad*/3}` |

The word "convolution" does not appear in SushiBLAS. `im2col` is a gather,
`pool_max` is a windowed reduction, `channel_affine` is a broadcast elementwise
— all three are in the vocabulary that library already has. Parameter ownership,
training mode and differentiation are SushiAI's.

---

## 4. SushiBLAS: `engine/spatial/`

### `Spatial::WindowSpan`

Modelled on `Reduce::AxisSpan`, and for the same reason: it carries element
counts and strides only, so the layout, the axis order and any non-unit stride a
view inherited are resolved once on the host and the device sees none of them.
One kernel per operation, no layout branch in the inner loop, no second
instantiation.

It carries the batch/channel/row/column strides of the input, the window
geometry from CV-5, and the output extents. `window_span_for(const Tensor&,
const ConvGeometry&, const Tensor& out)` builds it and is the single place
layout is interpreted.

### The operations

| Operation | Shape | Note |
|---|---|---|
| `im2col` | `[N,H,W,C]` → `[N·H_o·W_o, C·R·S]` | Padding is a parameter, **not a separate op** — that saves a full-tensor copy forward and a crop backward. Out-of-bounds reads write zero. |
| `col2im` | inverse | Gather (CV-6). |
| `pool_max` | → `[N,H_o,W_o,C]` + an **INT64** index tensor | The second output is the argmax. INT64 for the reason `execute_index_reduction` exists: an index is not a number, and is exact in fp32 only below 2²⁴. |
| `pool_max_backward` | | Gather over the windows that could have selected each input pixel (CV-6). |
| `pool_avg` / `pool_avg_backward` | | Global average pooling is **not** here — it is an axis reduction, not a sliding window. |
| `upsample_nearest` | `[N,H,W,C]` → `[N,sH,sW,C]` | Forward gather; backward sums the fixed `s²` outputs reading each input. |
| `concat_channels` / `slice_channels` | | Copies between a channel range and a whole tensor, in both directions; they are each other's backward. One strided-copy kernel serves both, sharing `WindowSpan`'s addressing. Not expressed as a zero-copy `Tensor::slice()` view: the gradient a slice produces has to be a materialised operand for the next node, and a graph whose values are sometimes views and sometimes buffers is a distinction the memory planner would have to carry everywhere. |
| `channel_affine` | `y = x·γ[c] + β[c]` | BatchNorm's apply half, and the *whole* of BatchNorm in evaluation mode. |

Every one of them gets **both** a pure-C++ reference kernel and a SYCL kernel, so
`VERIFY` mode has a ground truth. Convolution's reference is direct convolution —
which is why a "direct convolution" approach was never an alternative to CV-1, it
was always inside it.

**A known conservatism, recorded rather than pre-optimised:** in NHWC a channel
range is strided, not a contiguous byte interval, so two `concat_channels` writes
produce overlapping `Tensor::region()`s and the dependency tracker serialises
them. Correct, and possibly slower than necessary. Measure before addressing.

### `AxisSpan` gains a second **segment** axis

Not a second folded axis — the distinction decides which loop changes.

BatchNorm over NHWC needs nothing new: `[N·H·W, C]` is a zero-copy rank-2 view
and folding axis 0 gives `[C]` through the existing five numbers.

Global average pooling does need it: `[N,H,W,C] → [N,C]` has segments indexed by
the pair `(n,c)`, whose offset is `n·HWC + c`, and no single `segment_stride`
expresses that. `segments2` and `segment_stride2` are added; the folded side stays
one stride, and every existing caller passes `segments2 = 1`.

`axis_span_for`'s documented rank-2 restriction (*"the launch shape generalises,
the operand validation here does not"*) is lifted only as far as this needs.

### Mean and sum-of-squares need no new accumulator

`Dispatch::execute_folding_axis_reduction` already takes a `transform` and a
`finalize`. Σx² is `transform = x·x`. BatchNorm's two statistics are two calls,
each reusing the library's fixed-order deterministic fold. A single-pass fused
accumulator returning a pair is an optimisation, not a requirement, and is
deferred until measured.

---

## 5. SushiAI: operation identities

```
Structural   RESHAPE, LAYOUT_CONVERT
Convolution  IM2COL, COL2IM
Pooling      MAXPOOL2D, MAXPOOL2D_BACKWARD, AVGPOOL2D, AVGPOOL2D_BACKWARD
Spatial      SPATIAL_MEAN, SPATIAL_MEAN_BACKWARD, UPSAMPLE2D, UPSAMPLE2D_BACKWARD
Structure    CONCAT_CHANNELS, SLICE_CHANNELS
Norm         BATCHNORM2D, BATCHNORM2D_BACKWARD, CHANNEL_AFFINE
Non-IR       BATCHNORM_UPDATE_STATS                      (CV-7)
Activation   LEAKY_RELU, LEAKY_RELU_BACKWARD
```

There is **no `CONV2D` identity** (CV-1) and **no residual-add identity** —
residual addition is the existing `ADD`.

`SPATIAL_MEAN_BACKWARD` is a distinct operation and its absence was a real gap in
an earlier draft of this design. Global average pooling's backward broadcasts an
`[N,C]` gradient over `[N,H,W,C]` scaled by `1/(H·W)`. That is a **per-(n,c)**
broadcast; `BIAS_ADD` broadcasts a row vector and `channel_affine` broadcasts
per-channel, and neither is this.

`SLICE_CHANNELS` was the same class of gap, caught in this document's own review.
`CONCAT_CHANNELS`'s backward extracts each operand's sub-range from the incoming
gradient, and no slicing identity existed to express it. It is declared as an
operation in its own right rather than as a concat-specific backward, because it
is independently reachable: Darknet's route layer with a group index is exactly a
channel slice, which is what YOLOv4-tiny's CSP blocks use.

`LEAKY_RELU` joins `Ops::ACTIVATIONS` with `GradientSource::OUTPUT`. For α > 0,
`leaky_relu(x) > 0` exactly when `x > 0`, so reading the output rather than the
input is a **bit-identical substitution**, not an approximation — the same
argument the table already records for `RELU`. It therefore inherits the free-
fusion property: an ordinary use count licenses the fold with no
activation-specific knowledge in the fusion pass. Its α is one attribute slot.

`Ops::ALL` must list every one of these, or the collision proof silently does not
cover them. This is a checklist item in the plan, not an assumption.

---

## 6. Autograd: one domain per translation unit

### What is wrong today, precisely

`src/autograd/backward.cpp` is 600 lines and changes for **four** distinct
reasons: a new operation, a change to the traversal, a change to the tape
interface, a change to the loss-scaling policy. With this design's rules it
would pass 1200.

**The core design is right and is not touched.** `GradientTape` is the only
interface a rule sees — a rule does not name `Differentiator`, so the dependency
is inverted and the interface is as narrow as a rule needs. Rules are free
functions; dispatch is a flat table. What is wrong is only that these live in one
file.

### The split

```
include/SushiAI/autograd/
  backward.hpp      differentiate() — the one public entry
  tape.hpp          GradientTape — the only interface a rule sees
  rule.hpp          BackwardRule, RuleEntry, the constexpr concat helper
  rules/
    elementwise.hpp   linalg.hpp   activation.hpp   loss.hpp
    spatial.hpp       norm.hpp     shape.hpp                     (new)
    all.hpp           inline constexpr RULES = concat(...)

src/autograd/
  differentiator.cpp  traversal only: run / contribute / skip / resolve
  tape.cpp            GradientTape methods
  rules/
    elementwise.cpp   linalg.cpp   activation.cpp   loss.cpp
    spatial.cpp       norm.cpp     shape.cpp                     (new)
```

Each domain header exposes an `inline constexpr std::array<RuleEntry, N>`;
`all.hpp` concatenates them at compile time. **Nothing self-registers.** The
existing header's argument stands and is preserved verbatim: `sushiai` is a
static library, so a translation unit whose only job is to run a registration
constructor resolves no symbol and the linker drops it — silently, and only in a
downstream consumer's build. Taking the address of a function declared in a
domain header and defined in its `.cpp` is an address constant expression, so the
table stays `constexpr` and `find_rule` stays a compile-time walk.

Adding an operation to a domain changes **only that domain's file**. Adding a
domain is one line in `all.hpp`.

### Rules are not merged

- `im2col_backward` is convolution's **only** new rule. `matmul_backward`
  produces dgrad and wgrad. No monolithic `conv2d_backward` is written.
- `maxpool2d_backward` and `avgpool2d_backward` are separate functions, not one
  `pool_backward` behind a mode flag. Their mathematics shares nothing: one
  scatters through a recorded index, the other broadcasts a constant divisor.
- `batchnorm2d_backward` is its own rule and is added to nothing.

`activation_backward` remaining shared across five activations is
**parameterisation, not conflation**: their structure is identical —
`[dy, primal] → dx`, differing only in the identity looked up in `ACTIVATIONS` —
and writing it five times would be duplication with no separation gained.

### Ordering

The restructuring is step 6 of §12, **before** any new rule is written. Beyond
not doing the work twice, this removes the option of "put it here for now and
move it later": every new rule is born in the right file.

---

## 7. The module system: two additions

### `Buffer` — non-learnable checkpointed state

`fields()` enumerates `Parameter` members only. BatchNorm's `running_mean` and
`running_var` are checkpointed, are mutated by the forward pass in training mode,
and must **never** receive a gradient or enter the optimizer.

Making them `Parameter`s with `requires_grad = false` is wrong: the optimizer's
walk would still find them. A distinct `Buffer` type and a `buffer("running_mean",
&BatchNorm2d::running_mean)` field form make the wrong state unrepresentable.

`for_each_buffer` mirrors `for_each_parameter`'s three-arm structure, including
the `HasParameterWalk` arm the config-driven design adds for runtime layer lists.
Checkpoint I/O walks both; the optimizer walks only parameters.

### Named children, alongside index naming

`visit_children(index, child)` is correct for `Sequential` and `LayerStack`, and
the config-driven design depends on the two naming identically — that equivalence
is what its checkpoint round-trip test proves, and this design does not weaken it.

`visit_named_children()` is added for composites whose members have distinct
names. This is not a competing convention; it is what PyTorch does. `layer1.0.conv1.weight`
is an index (`0`, from a `Sequential`) *and* names (`layer1`, `conv1`, from named
members). Index naming for lists, name naming for structs — the two compose.

---

## 8. Layers

```
Conv2d{in_ch, out_ch, kernel, stride, pad, dilation, use_bias}
BatchNorm2d{features, eps, momentum}      parameters γ β; buffers running_mean running_var
MaxPool2d{kernel, stride, pad}
AvgPool2d{kernel, stride, pad}
GlobalAvgPool{}
Flatten{}
Upsample2d{scale}
Concat{}                                   Darknet "route"
LeakyReLU{alpha}
BasicBlock{...}                            two conv-bn paths plus a shortcut
```

`BasicBlock` needs no new mechanism: it satisfies `Module` through its own
`forward()` (which has a branch — `Sequential` cannot express that, a hand-written
`forward` can) and `ModuleContainer` through `visit_named_children`.

---

## 9. Data pipeline

CIFAR-10's binary format is raw bytes — one label byte and 3072 pixel bytes per
record — so **no image decoder is needed**. Normalisation uses the existing
elementwise operations.

**Augmentation runs on the host**, during batch assembly: random crop with
padding, and horizontal flip. Two reasons, and this is a deliberate choice with a
stated threshold rather than an unnoticed gap:

1. Augmentation has no backward pass — it is on the input side of the graph.
2. A horizontal flip is a negative-stride copy, and `Reduce::AxisSpan` documents
   that all its strides are non-negative. Doing it on-device means a new
   operation.

At 32×32×3 = 3 KB per image the host cost is immaterial. It should move to the
device when image area times batch size makes the copy comparable to a
convolution — roughly at 224×224 inputs, which is where this pipeline would go
next and is therefore worth restating then rather than pre-solving now.

The augmentation RNG is separate and explicitly seeded; a run that cannot be
reproduced is not a run this stack accepts.

YOLOv3-tiny's dataset — COCO or VOC, with box labels — is not here. It belongs
with the detection-head spec.

---

## 10. PyTorch parity, correctly scoped

PyTorch's `.pth` is a Python pickle. Parsing it in C++ is not a reasonable
undertaking, and inventing a second checkpoint format beside the config-driven
design's `SACP` would be worse.

**So the deliverable is naming and layout parity, not a file reader**, such that a
`state_dict` → `SACP` conversion is a small Python script containing no reshaping
logic:

- Parameter names match PyTorch's for the same architecture (§7's index-and-name
  composition is what makes this achievable).
- Convolution weights are stored on disk as PyTorch stores them,
  `[C_out, C_in, R, S]`, and permuted once to `[C_out, R, S, C_in]` at load —
  and back at save. The file format is unchanged by the layout decision.
- BatchNorm follows PyTorch's convention: `weight`/`bias` for γ/β, and
  `running_mean`/`running_var` as buffers.

Darknet's `.weights` for YOLOv3-tiny is a flat binary and needs no pickle reader,
but is ordered by layer rather than named — so its converter is separate and
order-driven.

Verified by a load → save → byte-compare test.

---

## 11. Rows to add when the config-driven design lands

Nothing here depends on that work; this is the list it will consume.

- `NN::LayerRegistry`: `"conv2d"`, `"batchnorm2d"`, `"maxpool2d"`, `"avgpool2d"`,
  `"global_avgpool"`, `"flatten"`, `"upsample"`, `"route"`, `"leaky_relu"` — one
  table row and one factory each, the factory being the only thing that knows
  which keys its layer accepts.
- `Data::DatasetRegistry`: `"cifar10"`.
- `Train::ObjectiveRegistry`: unchanged; `"yolo"` belongs to the head spec.

---

## 12. Verification

| Test | What it catches | Tier |
|---|---|---|
| `gradcheck` over every new backward rule | Analytic derivative against finite differences. The single highest-value autograd test, and `autograd/gradcheck.hpp` already exists (fp64, and documented as incompatible with mixed precision — which is why mixed precision comes after). | integration |
| `VERIFY` mode over every new kernel | SYCL against the reference (direct convolution, direct pooling). | integration |
| **Determinism**: `col2im` and `pool_max_backward` run twice, compared **byte for byte** | An accidental atomic add (CV-6). Nothing else catches it. | integration |
| Shape inference over the full geometry | Asymmetric padding, dilation, non-square kernels, and the `H_o` formula of CV-5. | unit |
| BatchNorm statistics change across two steps; eval-mode forward approximates train-mode | CV-7's silent failure. A loss-only test cannot distinguish it. | integration |
| Two compiled steps share parameters; each reports one compile | CV-8. | integration |
| Rematerialisation reduces peak arena bytes | CV-10 actually firing, rather than being written and not reached. | integration |
| Checkpoint load → save → byte compare | §10. | integration |
| ResNet-18 trains and the loss falls | End to end. | integration |
| YOLOv3-tiny body forward against a reference | End to end. | integration |

**The last row needs a reference to exist.** Obtaining a dump of Darknet's or
PyTorch's intermediate tensors for YOLOv3-tiny is a **step in the plan**, not an
assumption — without it, "it looks like it works" is the strongest claim
available, and this design does not accept that as verification.

Test names carry the `Unit_` / `Integration_` prefixes the suite's
`gtest_discover_tests` filters rely on.

---

## 13. Build order

SushiBLAS's steps are independent of SushiAI and are verified by their own suite,
so when something is wrong later the question "is the convolution wrong or is the
autograd wrong" is never asked — the kernels are green before SushiAI touches
them.

1. **SushiBLAS** — `spatial`: `im2col` / `col2im`, reference and SYCL kernels, tests
2. **SushiBLAS** — pooling, upsample, concat, `channel_affine`
3. **SushiBLAS** — `AxisSpan`'s second segment axis
4. **SushiBLAS** — `ChannelBiasEpilogue`
5. **SushiAI** — `Attributes::CAPACITY` 4 → 8; `Value` carries `Layout`; `finalise()` proves uniformity
6. **SushiAI** — operation identities, `differentiable` flag and its `static_assert`, shape inference in `GraphBuilder`
7. **SushiAI** — **the autograd restructuring** (§6), before any new rule
8. **SushiAI** — the new backward rules
9. **SushiAI** — the `IM2COL` rematerialisation pass
10. **SushiAI** — `Buffer`, `for_each_buffer`, `visit_named_children`
11. **SushiAI** — `TrainStep` / `EvalStep` (CV-8) and `BATCHNORM_UPDATE_STATS` (CV-7)
12. **SushiAI** — the layers, then `BasicBlock`
13. ResNet-18; train it
14. YOLOv3-tiny body; obtain a reference dump; compare
15. CIFAR-10 data pipeline
16. Checkpoint naming and layout parity, plus the conversion script

---

## 14. Consequences to record

- `docs/ARCHITECTURE.md` (SushiAI) gains the `spatial` operation family, the
  autograd file split, and CV-7's statement that BatchNorm statistics live
  outside the IR.
- `docs/ARCHITECTURE.md` (SushiBLAS) §7 gains the `spatial` domain and its
  determinism argument (CV-6), and §2's layer diagram gains the domain.
- The config-driven design's "Out → CNN / YOLO" section becomes partially
  historical and should reference this document rather than be deleted — it
  records what was true when it was written.
- `Graph::Attributes`'s "more than a handful of scalar knobs is a design smell"
  comment needs amending: convolution genuinely has eight, and CV-5 records why
  that is geometry rather than smell.
