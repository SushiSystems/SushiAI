# YOLO26 on SushiAI: plan

**Status:** In progress. Wave 1 of the host-lane units; written and syntax-checked only.

Requirements are in [`SPEC.md`](SPEC.md). Nothing below has been built or run.

## IR operations

Eleven identities were added to `include/SushiAI/core/ops.hpp`: nine operations the model and the
loss use, and two gradient operations (`SILU_BACKWARD`, `SOFTPLUS_BACKWARD`). Each has a
`GraphBuilder` emitter and,
where it lies on a gradient path, a rule in the table. None has a lowering: `src/graph/lowering.cpp`
is frozen until the SushiBLAS v0.1.0 migration, so a graph that holds any of them compiles to
the "no lowering" throw until then.

Abbreviations: `ops` = `include/SushiAI/core/ops.hpp`, `bh` = `include/SushiAI/graph/builder.hpp`,
`el` = `sushiblas: engine/math/elementwise.hpp`, `nl` = `sushiblas: engine/math/nonlinear.hpp`,
`lg` = `sushiblas: engine/logic/logic.hpp`. Kernel line numbers are the declaration lines read on
2026-10-07.

Rules shared by every emitter here: all operands have one shape and one dtype, there is no
broadcasting, and the result is declared with `declare_like` on the first data operand, so it
inherits that operand's layout. `Graph::finalise` rejects a node whose operands sit in two
layouts. The result is a new value; the in-place kernels below need a copy in their lowering,
as `EXP` has today.

Under the mixed-precision policy the eleven identities stay in the graph's own precision, and so
does every node their rules append. `narrows_under_mixed_precision` (`src/graph/precision.cpp`)
names its narrowing operations one by one and admits an activation only when its gradient reads
the forward output; `SILU` and `SOFTPLUS` read their input. That file was not changed.
`Unit_Backward.NoRuleOfTheElementwiseOperationsAppendsANodeThatNarrows` pins the second half: it
differentiates through each of the eight forward operations and fails if an appended node narrows.

In the formulas, `dy` (or `dc`) is the gradient of the result, and `[p]` is 1 where `p` holds and
0 elsewhere.

### DIV

- Identity: `Ops::DIV`, `"sushiai.div"`, differentiable.
- Emitter: `ValueId GraphBuilder::div(ValueId a, ValueId b) const`.
- Shape rule: `a` and `b` share shape and dtype; the result copies `a`.
- Gradient: `q = dc / b`; `da = q`; `db = -(q * c)`, with `c` the forward result. Because
  `c = a / b`, this equals `-dc * a / b^2` and never squares `b`. Emitted as `DIV`, `MUL`, `SCALE`.
- Kernel: `div(A, B, C)`, `el:100`.
- Lowering: not written, waits for the migration.

### MINIMUM

- Identity: `Ops::MINIMUM`, `"sushiai.minimum"`, differentiable.
- Emitter: `ValueId GraphBuilder::minimum(ValueId a, ValueId b) const`.
- Shape rule: `a` and `b` share shape and dtype; the result copies `a`.
- Gradient: `m = [a <= b]`; `da = select(m, dc, 0)`; `db = select(m, 0, dc)`. Emitted as
  `LESS_EQUAL`, `ZERO`, and one `SELECT` per operand that needs a gradient.
- Tie rule: where `a == b` the whole gradient goes to `a`, the first operand, as SPEC T7 states.
  The reference differs. Run in the pinned environment (torch 2.14.1+cpu), `torch.min(a, b)`,
  `torch.max(a, b)` and `torch.maximum` each give `dc / 2` to both operands at a tie; the CIoU
  calls at `U/utils/metrics.py:149-159` are these. A tie between two box coordinates has
  measure zero for real-valued predictions, so stage c decides whether it matters. If it does,
  the rule becomes `da = 0.5 * dc * ([a <= b] + 1 - [b <= a])`, `db = dc - da`, which needs
  both masks.
- Kernel: `min(A, B, C)`, `el:285`.
- Lowering: not written, waits for the migration.

### MAXIMUM

- Identity: `Ops::MAXIMUM`, `"sushiai.maximum"`, differentiable.
- Emitter: `ValueId GraphBuilder::maximum(ValueId a, ValueId b) const`.
- Shape rule: `a` and `b` share shape and dtype; the result copies `a`.
- Gradient: `m = [b <= a]`; `da = select(m, dc, 0)`; `db = select(m, 0, dc)`. Same nodes as
  `MINIMUM`, with the comparison's operands swapped.
- Tie rule: where `a == b` the whole gradient goes to `a`. The reference splits it, as under
  `MINIMUM`. The clamp at zero is a separate case; see "CIoU clamp" below.
- Kernel: `max(A, B, C)`, `el:295`.
- Lowering: not written, waits for the migration.

### ABS

- Identity: `Ops::ABS`, `"sushiai.abs"`, differentiable.
- Emitter: `ValueId GraphBuilder::abs(ValueId x) const`.
- Shape rule: the result copies `x`.
- Gradient: `sign = [0 <= x] - [x <= 0]`; `dx = dy * sign`. Emitted as `ZERO`, two `LESS_EQUAL`,
  `SUB`, `MUL`.
- At zero: both masks are 1, so `sign` is 0 and `dx` is 0. The task's `select(x >= 0, dy, -dy)`
  would give `dy` there; the sign form was chosen because the map states the gradient as
  `dy * sign(x)` (SPEC section 1). The reference agrees: in the pinned environment the gradient
  of `x.abs()` at `x = 0` is 0.
- Kernel: `abs(t)`, in place, `el:156`.
- Lowering: not written, waits for the migration.

### ATAN

- Identity: `Ops::ATAN`, `"sushiai.atan"`, differentiable.
- Emitter: `ValueId GraphBuilder::atan(ValueId x) const`.
- Shape rule: the result copies `x`.
- Gradient: `dx = dy / (x * x + 1)`. The one is `constant_like(x, 1)`. Emitted as `MUL`,
  `CONSTANT`, `ADD`, `DIV`: four kernels and two temporaries per `ATAN`. CIoU has two `atan`
  calls per level and branch (`U/utils/metrics.py:166`).
- Kernel: `atan(t)`, in place, `el:211`.
- Lowering: not written, waits for the migration.

### SOFTPLUS

- Identity: `Ops::SOFTPLUS`, `"sushiai.softplus"`, differentiable; and `Ops::SOFTPLUS_BACKWARD`,
  `"sushiai.softplus_backward"`, not differentiable.
- Emitters: `ValueId GraphBuilder::softplus(ValueId x) const`; the gradient goes through
  `GraphBuilder::activation_backward`, as `SILU`'s does.
- Shape rule: the result copies `x`; `SOFTPLUS_BACKWARD` needs `dy` and `x` of one shape and dtype.
- Gradient: `dx = SOFTPLUS_BACKWARD(dy, x) = dy * sigmoid(x)`, with `x` the forward input. One node.
- Table row: `{SOFTPLUS, SOFTPLUS_BACKWARD, GradientSource::INPUT}` in `Ops::ACTIVATIONS`. The
  generic `activation_backward` rule serves it.
- Why a gradient identity and not `SIGMOID` then `MUL`: `SIGMOID` reads its output in the table,
  so it narrows under mixed precision, and the classification loss would compute its gradient
  factor in the low type against ruling D8. The task text prescribed the composition; this
  departs from it and is listed under Open for the owner.
- Fusion: not foldable, for the reason given under `SILU`.
- Kernels: `softplus(t)`, in place, `nl:185`; `softplus_backward(dy, x, dx)`, `nl:195`.
- Lowering: not written for either identity, waits for the migration.

### SILU

- Identity: `Ops::SILU`, `"sushiai.silu"`, differentiable; and `Ops::SILU_BACKWARD`,
  `"sushiai.silu_backward"`, not differentiable.
- Emitters: `ValueId GraphBuilder::silu(ValueId x) const`; the gradient goes through the
  existing `GraphBuilder::activation_backward(OpID, ValueId dy, ValueId primal, const Attributes&)`.
- Shape rule: the result copies `x`; `SILU_BACKWARD` needs `dy` and `x` of one shape and dtype.
- Gradient: `dx = SILU_BACKWARD(dy, x) = dy * s * (1 + x * (1 - s))`, `s = sigmoid(x)`, with `x`
  the forward input.
- Table row: `{SILU, SILU_BACKWARD, GradientSource::INPUT}` in `Ops::ACTIVATIONS` (seven rows).
  The generic `activation_backward` rule serves it.
- Fusion: not foldable. No catalog entry in `src/graph/fusion.cpp` ends in `SILU`, so
  `every_fused_activation_survives` skips it and its assertion holds without a spill entry. An
  entry that folds `SILU` later needs a spill of the pre-activation, as `GEMM_BIAS_GELU` has.
- Kernels: `silu(t)`, in place, `nl:149`; `silu_backward(dy, x, dx)`, `nl:159`.
- Lowering: not written for either identity, waits for the migration.

### SELECT

- Identity: `Ops::SELECT`, `"sushiai.select"`, differentiable.
- Emitter: `ValueId GraphBuilder::select(ValueId condition, ValueId a, ValueId b) const`.
- Shape rule: `condition`, `a` and `b` share shape and dtype. The condition holds 0 or 1 in the
  dtype of `a`; the emitter cannot check its contents. The result copies `a`.
- Forward: `out = condition != 0 ? a : b`.
- Gradient: `da = select(condition, dy, 0)`; `db = select(condition, 0, dy)`; nothing to
  `condition`. Emitted as `ZERO` and one `SELECT` per branch that needs a gradient.
- Kernel: `where(condition, A, B, out)`, `lg:115`.
- Lowering: not written, waits for the migration.

### LESS_EQUAL

- Identity: `Ops::LESS_EQUAL`, `"sushiai.less_equal"`, a gradient barrier.
- Emitter: `ValueId GraphBuilder::less_equal(ValueId a, ValueId b) const`.
- Shape rule: `a` and `b` share shape and dtype; the result, 1 where `a <= b` and 0 elsewhere,
  copies `a`, dtype included.
- Why this one: it is the only comparison the rules above need. `MINIMUM` uses `[a <= b]`,
  `MAXIMUM` uses `[b <= a]`, and `ABS` uses both orders against zero.
- Gradient: none, and no rule. The mask requires no gradient whatever its operands need, so in
  `mul(less_equal(p, q), x)` with `p` depending on a parameter `MUL`'s rule emits the gradient
  of `x` and nothing for the mask. `Unit_Backward.AComparisonMultipliedIntoAProductEmitsNothingForTheMask`
  pins that. The mechanism is under "Gradient barriers" below.
- Kernel: `less_equal(A, B, result)`, `lg:110`.
- Lowering: not written, waits for the migration.

### Supporting change

`Ops::is_activation_backward(OpID)` reads the gradient column of `Ops::ACTIVATIONS`.
`GraphBuilder::activation_backward` uses it in place of its list of five identities, so a row
added to the table needs no edit in the builder.

### Tests

Device-free, unbuilt and never run.

- `tests/functional/unit/test_elementwise_ops.cpp`, suite `Unit_ElementwiseOps`: identity and
  operand order, shape, dtype and layout of every emitter; rejection of unequal shapes and dtypes,
  for the two input-reading gradients too; `finalise` on a graph of every emitter, on binary
  operands in two layouts and on a `SELECT` condition in another layout than its branches.
- `tests/functional/unit/test_backward.cpp`: the nodes each rule emits and the values they read,
  with one operand or both needing a gradient; that no appended node narrows.
- `tests/functional/unit/test_ops.cpp`, `test_rule_table.cpp`: the eleven identities, their flags,
  the `SILU` and `SOFTPLUS` table rows, and that the comparison has no rule.

### Open

- Owner decision: `SOFTPLUS` differentiates through `SOFTPLUS_BACKWARD`, not the `SIGMOID`, `MUL`
  composition the task text gave. See the `SOFTPLUS` section for the reason.
- Ruling D8 lists the applied SiLU among what narrows. `SILU` reads its input, so
  `narrows_under_mixed_precision` keeps it wide, and `precision.cpp` is not in this wave.
- No finite-difference case exists for eight rules: `DIV`, `MINIMUM`, `MAXIMUM`, `ABS`, `ATAN`,
  `SOFTPLUS`, `SELECT`, `SILU`. `Unit_Backward.EveryRuleInTheTableIsNamedByANumericTest` holds
  them in a list named `unproven`, whose size is fixed by `K_UNPROVEN_RULES`; each leaves it when
  a case in `tests/functional/integration/test_autograd_numeric.cpp` names it. That test's name
  still says every rule is proved; `tests/functional/unit/README.md` cites the name, so it was
  kept.
- The tie rule of `MINIMUM` and `MAXIMUM` differs from the reference's even split; stage c
  decides whether to change it.
- No README was in this wave's files, so the notes above are the only record of the missing
  lowerings; `tests/functional/unit/README.md` and the graph module's README do not mention the
  new operations, the `unproven` list or `Unit_ElementwiseOps`.
- `tests/functional/unit/test_elementwise_ops.cpp` has no licence block;
  `tools/licensing/write_license_block.py` writes it.

## Loss roots and stop-gradient

Written and syntax-checked only; nothing below has been built or run.

### Loss root

`Autograd::differentiate(graph, loss, wrt, seed_factor, provenance)` accepts a one-element `loss`
produced by any operation that has a backward rule.

- Seed: `one = constant_like(loss, 1)`, one `CONSTANT` node in the root's shape, element type and
  layout. `CONSTANT` has no lowering until the migration, so a graph with a seeded root does not
  compile until then.
- Seed factor: `seed = DYNAMIC_SCALE(one, seed_factor)` when a factor is given, `seed = one`
  otherwise. The call is `GradientTape::apply_seed_factor`, the same one `NLL_LOSS`'s rule uses.
- Rule: `d(loss)/d(loss) = seed`. The seed is recorded as the root's one contribution, so the
  root's rule reads it through `output_grad(0)` like any other gradient. For
  `loss = w * (t1 + t2)` this gives `d(t1) = d(t2) = SCALE(seed, w)`.
- `NLL_LOSS` is the exception. `Autograd::starts_from_implicit_seed(op)`
  (`include/SushiAI/autograd/rules/loss.hpp`) is true for it alone; for such a root nothing is
  emitted before the walk, and its rule applies the factor itself as before. Its emitted nodes do
  not change: fused `EXP, SUB, SCALE[, DYNAMIC_SCALE]`, unfused `SCALE[, DYNAMIC_SCALE]`.
- `Differentiator::resolve` no longer special-cases the root. An `NLL_LOSS` root has no
  contribution and resolves to `NO_VALUE`, as it did.
- Provenance: the `CONSTANT` and the `DYNAMIC_SCALE` have the root's producer as origin.

Root checks, in this order. The first two are in `Differentiator::run`, the rest in
`Differentiator::require_seedable_root`, and all seven are raised before any node is appended:

| Root | Message names |
| --- | --- |
| more than one element | "needs a scalar to differentiate", the shape |
| no producer | "nothing produces it" |
| produced by `LEAF` | "it is a leaf", `sushiai.leaf` |
| requires no gradient | "requires no gradient", the producer, a gradient barrier such as `STOP_GRADIENT` as a possible cause |
| not the producer's first result | "not the first result", the producer |
| element type not `HALF`, `FLOAT32` or `FLOAT64` | "element type is", the type, the producer |
| producer has no rule | "no backward rule", the producer |

Every rule reads `output_grad(0)`, so a seed on a second result (the `MAXPOOL2D` index, the
`BATCHNORM2D` statistics) would be read by nobody; that root throws. A root that a gradient
barrier produces (`STOP_GRADIENT`, `LESS_EQUAL`, `CONSTANT`, `ZERO`) reports "requires no
gradient", because that check comes before the rule check.

A throw raised by a rule during the walk comes after the seed was appended, so the graph keeps a
`CONSTANT` it does not use. Two roots reach that today: a one-element
`LOG_SOFTMAX` root, whose diagnostic-only rule throws its own text, and a sum that has an
`NLL_LOSS` among its terms.

An `NLL_LOSS` root that requires no gradient does not throw: it returns `NO_VALUE` for every
request and appends nothing, which `Unit_Backward.ADeadLossPathEmitsNothingAtAll` pins. The two
kinds of root differ here on purpose, to leave that test as it is; see Open.

`NLL_LOSS` as a term of a sum still throws in its own rule (`src/autograd/rules/loss.cpp`,
"differentiable only as the value passed to differentiate()"). That restriction was not in this
package.

### STOP_GRADIENT

- Identity: `Ops::STOP_GRADIENT`, `"sushiai.stop_gradient"`, not flagged differentiable. It has
  no backward rule, and that is deliberate: the reverse walk never asks for one.
- Emitter: `ValueId GraphBuilder::stop_gradient(ValueId x) const`.
- Shape rule: the result copies the shape, element type and layout of `x`.
- Forward: `y = x`.
- Gradient: `dx = 0` whatever `dy` is, expressed as no contribution at all.
- Mechanism: it is a gradient barrier; see "Gradient barriers" below. `Ops::is_gradient_barrier(op)`
  (`ops.hpp`) has two readers. `Graph::propagate_requires_grad` (`ir.hpp`) skips a barrier node,
  so its result and everything computed from it alone keep `requires_grad == false` and their
  rules emit nothing for it. `Differentiator::differentiate_node` returns at a barrier before it
  looks for a rule. That return is a guard: every rule under `src/autograd/rules/` checks `input_requires_grad`
  before it contributes, so no rule sends a gradient to a barrier's result today. The test
  `Unit_LossRoots.AContributionToAStopGradientResultIsDroppedWithoutAskingForARule` sets
  `requires_grad` on the result by hand to reach it; without the return it throws "No backward
  rule for sushiai.stop_gradient".
- A value read both directly and through a `STOP_GRADIENT` receives the direct contribution
  only; no `ACCUMULATE` is emitted for it.
- Use in YOLO26: the features entering the one-to-one head
  (`ultralytics/nn/modules/head.py:196`, `xi.detach()`), and CIoU's `alpha`
  (`ultralytics/utils/metrics.py:167-169`).
- Lowering: not written, waits for the migration. It is an alias of the operand's storage, or a
  copy if the planner cannot let two values share a buffer; the choice belongs to the lowering.

Text for the graph module's README, which was not in this package's files:

> `STOP_GRADIENT(x)`: forward identity. No gradient flows from its result to `x`. It has no
> backward rule on purpose: `Ops::is_gradient_barrier` makes `Graph::finalise` leave its result
> without `requires_grad`, and the reverse walk steps over the node. No lowering yet; it waits for
> the SushiBLAS v0.1.0 migration (an alias or a copy).

### Files

- `include/SushiAI/core/ops.hpp`: `STOP_GRADIENT`, its entry in `ALL`, `is_gradient_barrier`.
- `include/SushiAI/graph/ir.hpp`: the barrier in `propagate_requires_grad`.
- `include/SushiAI/graph/builder.hpp`, `src/graph/builder.cpp`: `stop_gradient`.
- `include/SushiAI/autograd/rules/loss.hpp`: `starts_from_implicit_seed`.
- `src/autograd/differentiator.hpp`, `.cpp`: `require_seedable_root`, `seed_root`, the two root
  checks in `run`, the barrier in `differentiate_node`, `resolve` without the root case.
- `include/SushiAI/autograd/backward.hpp`: the parameter text of `differentiate`.
- `tests/functional/unit/test_loss_roots.cpp`: suite `Unit_LossRoots`, 25 tests.
- `tests/functional/unit/test_backward.cpp`: one expectation that `STOP_GRADIENT` has no rule.
- Not changed: `tape.hpp`, `tape.cpp`, `rules/loss.cpp`, `rules/shape.hpp`, `rules/shape.cpp`,
  `test_provenance.cpp`. Its expectations hold: every graph in it has an `NLL_LOSS` root.

### Waits for the migration

- The `STOP_GRADIENT` lowering (alias or copy). Until then a graph holding one compiles to the
  "no lowering" throw.
- The seed's `CONSTANT` lowering. `DYNAMIC_SCALE` is lowered today.

### Open

- `tests/functional/unit/test_loss_roots.cpp` has no licence block and is in no build list.
- Owner decision: a seeded root that requires no gradient throws, an `NLL_LOSS` root in the same
  state returns nothing. Making them agree means changing
  `Unit_Backward.ADeadLossPathEmitsNothingAtAll` or dropping the new throw.
- Owner decision: a one-element parameter passed as the root throws ("it is a leaf"), although
  its gradient is the seed itself.
- `SPEC.md` line 53 describes `STOP_GRADIENT` as "a registered rule that returns without
  contributing". What is built is the barrier of line 54; `SPEC.md` is outside this package's
  files.
- The README paragraph quoted above is in no README; the graph module's README is outside this
  package's files. `tests/functional/unit/README.md` has no section for `test_loss_roots.cpp`.
- No numeric test covers a seeded root; a finite-difference case over an `ADD` root belongs in
  `tests/functional/integration/test_autograd_numeric.cpp` after the build is allowed.

## Constants, gradient barriers and the loss graph

Written and syntax-checked only; nothing below has been built or run. The torch figures were read
by running the pinned environment (`sushiai-yolo26-ref`, torch 2.14.1+cpu) on three-element
tensors.

### CONSTANT

- Identity: `Ops::CONSTANT`, `"sushiai.constant"`, a gradient barrier. It reads no operand.
  Attribute slot 0 is the fill value.
- Emitters: `ValueId GraphBuilder::constant(Shape shape, double value, DType dtype = FLOAT32) const`
  and `ValueId GraphBuilder::constant_like(ValueId x, double value) const`, beside `zeros` and
  `zeros_like`. `constant` yields an NHWC activation, as `zeros` does. `constant_like` copies the
  shape, element type and layout of `x` and does not read it.
- Element type: half, float32 or float64; anything else throws. An integer fill would need a
  rounding rule for the `double` attribute, and no caller needs one.
- Callers: `Differentiator::seed_root` (`constant_like(loss, 1)`) and `atan_backward`
  (`constant_like(x, 1)`). They were the only two places that built `EXP(zeros_like(...))`; a grep
  of `src/` and `include/` for `zeros` finds no third. CIoU's `eps` (`U/utils/metrics.py:145-169`,
  five additions), its `1` and its `1 + eps` are each one `constant_like`.
- `ZERO` stays its own identity. `zeros` and `zeros_like` emit it as before, and so do the rules
  that route a gradient (`MINIMUM`, `MAXIMUM`, `SELECT`, `ABS`, `SUM_ROWS`, `SLICE_CHANNELS`,
  `CHANNEL_AFFINE`). `ZERO` has a lowering in the frozen `src/graph/lowering.cpp:909-915` and
  `CONSTANT` has none, so folding `ZERO` into `CONSTANT(0)` now would stop graphs that compile
  today. After the migration `ZERO` can become `CONSTANT` with value 0 and one lowering.
- Kernel: `random().constant(t, value)`, `sushiblas: engine/math/random.hpp:89`, the call `ZERO`
  lowers to with value 0. Whether it accepts a HALF tensor was not read.
- Lowering: not written, waits for the migration.
- An `NLL_LOSS` root emits what it emitted before. `Differentiator::run` calls `seed_root` only
  when `starts_from_implicit_seed` is false, and `nll_loss_backward` builds no constant: its
  `EXP` reads the log-probabilities. `Unit_LossRoots.AFusedCrossEntropyRootEmitsNoSeedNode` and
  `Unit_Backward.TwoLayerMlpProducesTheExpectedBackwardTopology` pin the node lists.

### Gradient barriers

One rule covers every operation that passes no gradient. It is stated at
`Ops::gradient_barrier_op` in `ops.hpp`: an identity declared through it carries
`OpName::blocks_gradient`, its result never requires a gradient, and it has no backward rule.

- Readers: `Ops::is_gradient_barrier(op)` reads the flag from `Ops::ALL`.
  `Graph::propagate_requires_grad` skips a barrier node; `Differentiator::differentiate_node`
  returns at one before it looks for a rule.
- Proofs at compile time: `ops.hpp` asserts that no identity is both differentiable and a
  barrier; `autograd/rules/all.hpp` asserts that no barrier has a row in `RULES`.
- The four barriers: `STOP_GRADIENT`, `LESS_EQUAL`, `CONSTANT`, `ZERO`.
- `LESS_EQUAL` lost `less_equal_backward` and its row; `ELEMENTWISE_RULES` has 12 rows. The dead
  `MUL` that the rule cost is gone, and the test that pinned it is replaced by
  `Unit_Backward.AComparisonMultipliedIntoAProductEmitsNothingForTheMask`.

What the identities not flagged differentiable do, read from `ir.hpp`, `differentiator.cpp` and
the rule tables, and why only four are barriers:

| Identity | On a gradient path today | Barrier |
| --- | --- | --- |
| `LEAF` | no operand; a parameter's flag is set by `GraphBuilder::parameter` | no: a parameter leaf requires a gradient |
| `ZERO`, `CONSTANT` | no operand, so the flag is never set and `differentiate_node` returns before the rule lookup | yes: the flag restates this, and no graph changes |
| `LESS_EQUAL` | had an empty rule; its mask required a gradient | yes |
| `STOP_GRADIENT` | was the one barrier | yes |
| `CAST` | passes `requires_grad`; throws "No backward rule" if reached | no: a barrier would turn that throw into a silently missing gradient |
| `DYNAMIC_SCALE`, `LOG_SOFTMAX` | pass `requires_grad`; throw (`LOG_SOFTMAX` with its own text) | no, same reason |
| the twelve `*_BACKWARD` operations | emitted by rules after `finalise`; a second differentiation over them throws "No backward rule" | no: second-order use must keep failing loudly until each has a rule |
| `GEMM_BIAS*`, `EXP_SUB_SCALE` | written by fusion, after differentiation | no |
| `ADAMW_UPDATE`, `SGD_UPDATE`, `BATCHNORM_UPDATE_STATS` | written beside the graph; never on a path | no |

No graph that existed before `LESS_EQUAL` changes. `propagate_requires_grad` only ever sets the
flag, a `ZERO` node has no operand to set it from, and `has_backward_rule(ZERO)` was already
false, so the two readers take the same branch for `ZERO` as before by a different test.

### CIoU clamp

`.clamp_(0)` on the intersection sides (`U/utils/metrics.py:149-151`) is not `RELU`, which
`SPEC.md` line 65 and T5 name. `narrows_under_mixed_precision` is true for `RELU` and
`RELU_BACKWARD` (`src/graph/precision.cpp:62-67`), so part of the loss would compute in the low
type, against ruling D8. `MAXIMUM`, `CONSTANT`, `LESS_EQUAL`, `ZERO` and `SELECT` all stay wide.

- The task's form: `MAXIMUM(x, constant_like(x, 0))`. The mask is `[0 <= x]`, which holds at
  `x = 0`, and a tie goes to the first operand, so at exactly zero `dx = dy`.
- The reference: the gradient of `x.clamp(0)`, `x.clamp_min(0)` and `(x * 1).clamp_(0)` at
  `x = 0` is 0, the same as `torch.relu`. `torch.maximum(x, zeros)` gives 0.5 there.
- The form that matches the reference: `MAXIMUM(constant_like(x, 0), x)`, the constant first. The
  mask is `[x <= 0]`, a tie goes to the constant, which is dropped, and `dx = 0` at exactly zero.
  The forward value is the same either way. This is the order to use in the loss; it departs
  from the operand order the task text gave, for the measured reason above.
- Tests: `Unit_Backward.MaximumOfXAndAZeroConstantRoutesATieToX` and
  `Unit_Backward.MaximumOfAZeroConstantAndXGivesTheClampedOperandNothingAtATie` pin both orders
  by their masks; the second also fails if any node of the graph narrows.
- Emitted for the gradient: `LESS_EQUAL`, `ZERO`, `SELECT`, three nodes against `RELU_BACKWARD`'s
  one.

### Identities without a lowering

Thirteen. A graph that holds any of them compiles to the "no lowering" throw until the
migration. Kernel lines are declaration lines in `sushiblas: include/SushiBLAS/engine/`, read on
2026-10-07. The in-place kernels need a copy of the operand first, as `EXP` has today.

| Identity | SushiBLAS kernel |
| --- | --- |
| `DIV` | `div(A, B, C)`, `math/elementwise.hpp:100` |
| `MINIMUM` | `min(A, B, C)`, `math/elementwise.hpp:285` |
| `MAXIMUM` | `max(A, B, C)`, `math/elementwise.hpp:295` |
| `ABS` | `abs(t)`, in place, `math/elementwise.hpp:156` |
| `ATAN` | `atan(t)`, in place, `math/elementwise.hpp:211` |
| `SOFTPLUS` | `softplus(t)`, in place, `math/nonlinear.hpp:185` |
| `SOFTPLUS_BACKWARD` | `softplus_backward(dy, x, dx)`, `math/nonlinear.hpp:195` |
| `SILU` | `silu(t)`, in place, `math/nonlinear.hpp:149` |
| `SILU_BACKWARD` | `silu_backward(dy, x, dx)`, `math/nonlinear.hpp:159` |
| `LESS_EQUAL` | `less_equal(A, B, result)`, `logic/logic.hpp:110` |
| `SELECT` | `where(condition, A, B, out)`, `logic/logic.hpp:115` |
| `STOP_GRADIENT` | `copy(x, y)`, `blas/level1.hpp:65`, the call `RESHAPE` lowers to; or an alias of the operand's storage if the planner allows one |
| `CONSTANT` | `constant(t, value)`, `math/random.hpp:89` |

Not read: whether `less_equal` writes its mask in the operands' element type, which `SELECT`
and the `ABS` rule assume, and whether `where` reads a floating condition.

### NLL_LOSS as a term of a sum

`NLL_LOSS` still cannot be a term of a sum. `nll_loss_backward` (`src/autograd/rules/loss.cpp:46-86`)
never reads `output_grad(0)`: it writes `-targets / N`, or `(softmax - targets) / N` in the fused
pair, as if the gradient arriving at its result were one, and it applies the seed factor
itself. Inside `w * nll + other` it would drop `w` and apply the factor a second time, so the
rule throws unless its result is the root ("differentiable only as the value passed to
differentiate()"). Lifting that means multiplying its contribution by the one-element
`output_grad(0)`; it was left alone so that an `NLL_LOSS` root keeps its node list.

The detection loss does not need it. Its class term is binary cross-entropy with logits on soft
targets, `SUB(SOFTPLUS(x), MUL(x, t))` (`U/utils/loss.py:358, 451-454`), its box terms are CIoU
and L1, and no term is a softmax cross-entropy. The root is an `ADD` of weighted one-element
sums, which the explicit seed serves.

### Tests

- `tests/functional/unit/test_ops.cpp`: the identity list has 64 names; the four barriers, that
  none is differentiable, and that `LEAF`, `CAST` and the gradient operations are not barriers.
- `tests/functional/unit/test_precision.cpp`: the thirteen identities are in `stays`, each
  checked against `src/graph/precision.cpp:50-70`; the two lists cover 64.
- `tests/functional/unit/test_diagnostics.cpp`: `ALossThatNoRuleCanSeedIsRejectedRatherThanReturningNothing`
  is replaced by three tests: a one-element `MATMUL` root is seeded, a root that requires no
  gradient is refused, and a root whose producer has no rule is refused.
- `tests/functional/unit/test_rule_table.cpp`: no barrier has a rule.
- `tests/functional/unit/test_elementwise_ops.cpp`: the two `CONSTANT` emitters, the refused
  element types, and that `zeros` still writes `ZERO`.
- `tests/functional/unit/test_backward.cpp`, `test_loss_roots.cpp`: the seed and the `ATAN` rule
  as one `CONSTANT`; roots a barrier produces; the mask and the constant on a parameter path.

### Open

- A graph with a seeded root compiled before this change (`ZERO`, `EXP` and `DYNAMIC_SCALE` are
  lowered) and does not until `CONSTANT` is lowered. `NN::CrossEntropyLoss`, the one loss the
  library ships, ends in `NLL_LOSS` and is not affected.
- Owner decision: `OpName::blocks_gradient` and `Ops::gradient_barrier_op` are new public names in
  `ops.hpp`, and `GraphBuilder::constant` refuses integer and complex element types.
- Owner decision: the clamp's operand order, constant first, against the task text.
- `tests/functional/unit/README.md` lines 170, 185, 556, 1843 and 1900 describe tests this
  package changed; that file is outside this package's files. So are `SPEC.md` lines 53, 65 and
  T5.
- No finite-difference case names `CONSTANT` on a path, and none is needed: it has no rule.

## Detection host lane

Written and syntax-checked only; nothing below has been built or run. The module's facts are in
`include/SushiAI/detect/README.md`. This section lists what the package carries forward.

### Open

- Owner decision: `detection_row.hpp` holds the row columns and two limits of other units,
  `K_MAX_DETECTIONS` (`Detect.max_det`) and `K_VALIDATION_CONFIDENCE` (the validator's default).
  Moving the two to `detect/detection_limits.hpp` adds a header and a line in `SushiAI.h`.
- Owner decision: `keep_confident_detections` does not port the `[:max_det]` cut of
  `ultralytics/utils/nms.py:77`; its header states `k <= K_MAX_DETECTIONS` as a precondition. A
  `max_detections` parameter would port the cut and changes the signature.
- Owner decision: `GroundTruthBatch::box_counts` is removed, because nothing but
  `pad_ground_truth` read it. The function now allocates a `[B]` cursor per call. The field comes
  back if a consumer for the counts is named.
- Owner decision: the public names an earlier round added without asking:
  the signature of `select_nms_detections`, `keep_confident_detections`, `K_POINT_VALUES`,
  `K_VALIDATION_CONFIDENCE`, `PredictionShape`, `AnchorLeaves` and `build_anchor_leaves`.
- No reference comparison covers a removal by `suppress_overlaps`. In the fixture no two
  candidates overlap above 0.7, and at a lower threshold the result depends on torch's order of
  equal scores (282 distinct values in 1542). It needs weights whose scores differ, which is a new
  fixture.
- `torchvision.ops.nms` was not compared with `TorchNMS.nms`. The reference environment holds
  torchvision 0.29.1, but the exporter does not import it and its kernel is compiled.
- `raise` in `src/detect/assigner.cpp` takes the square root for 0.5, the value for 1 and the
  product for 2. Whether torch's scalar `pow` special-cases the same exponents was not confirmed;
  the reference tests compare within `1e-5`.
- The error-message needles of the tests assume how `SushiBLAS::Core::format` prints a float
  (`1.5`, `-1`). The formatter was not read; the first test run settles it.
- `near_reference` and `equal_reference` stay in `tests/common/reference_fixture.hpp`, so a hand
  test includes the checkpoint reader. Moving them waits on `tests/common/array_compare.hpp`
  being granted.
- `Unit_DetectAnchorLeaves` and `Unit_DetectLevelRows` sit in `test_detect_anchors.cpp`,
  `Unit_DetectGroundTruth` and `Unit_DetectTargetLeaves` in `test_detect_assigner.cpp`, and
  `Unit_DetectTopK` in `test_detect_postprocess.cpp`. Splitting them waits on five test files
  being granted and on their lines in `tests/CMakeLists.txt`.
- Build lists: `src/CMakeLists.txt` lacks `detect/anchor_leaves.cpp`, and `SushiAI.h` lacks
  `detect/anchor_leaves.hpp` and `detect/prediction_shape.hpp`.
- The 34 files of the package have no licence block; `tools/licensing/write_license_block.py`
  writes it.
- No changelog entry is written, and `tests/functional/unit/README.md` has no section for the
  four detect test files. Both are outside this package's files.

## Inference: the forward pass against the reference

Planned on 2026-10-09 from the tree after the layout move; nothing below is written, built or
run. `U:` is the pinned Ultralytics installed in the conda environment `sushiai-yolo26-ref`; a
"region" was read around the cited line. `MF` is `tests/fixtures/yolo26/MANIFEST.json`, `mj` is
`applications/examples/yolo26/model.json`, `sb:` is `D:\Projects\sushiblas\modules`.

### Measured

**Row 10, `c2psa`** (`mj:22`: filters 256, hidden 128, repeats 1, heads 2), at batch 1, N = 400.

| Step | Reference | Result |
| --- | --- | --- |
| `cv1`, Conv 1x1 with BN and SiLU | `U:nn/modules/block.py:1479-1480` region, `Conv(c1, 2c)` | `[1,20,20,256]` |
| split in two | `block.py:1492` | `a`, `b`: `[1,20,20,128]` |
| `m.0.attn.qkv`, Conv 1x1, BN, no activation | `block.py:1307` region; `h = dim + 2 * heads * key_dim` | `[1,20,20,256]` |
| per head (2): q, k, v on the channel axis, head-major | `block.py:1323`; `head_dim = 64`, `key_dim = 32` | q, k `[1,20,20,32]`, v `[1,20,20,64]`; head `h` starts at channel `128 * h` |
| tokens | view to `[B, heads, kd, N]` | q, k `[1,400,32]`, v `[1,400,64]` by reshape, no permute in NHWC |
| scale | `block.py:1304`, `key_dim ** -0.5`, applied to q before the product (`:1331`) | 0.17677669529663687 |
| scores, then softmax on the last axis | `block.py:1331` | `[1,400,400]` |
| product with v | `block.py:1332` | `[1,400,64]`, then `[1,20,20,64]`; heads joined to `[1,20,20,128]` |
| `pe`, depthwise 3x3, stride 1, pad 1, BN, no activation, over v of both heads joined | `block.py:1309`, `:1332` | `[1,20,20,128]`, added to the line above |
| `proj`, Conv 1x1, BN, no activation | `block.py:1308` region | `[1,20,20,128]` |
| `b = b + attn(b)` | `PSABlock.forward` | `[1,20,20,128]` |
| `ffn.0` Conv 1x1 to 256 with SiLU, `ffn.1` Conv 1x1 to 128 without | `block.py:1370` | `[1,20,20,128]`; `b = b + ffn(b)` |
| `cv2` over `a` and `b` joined | `block.py:1494` | `[1,20,20,256]` |

State, in file order (`MF:2842`, records 165 to 199 of `weights.sacp`): `cv1`, `cv2`,
`m.0.attn.qkv`, `.proj`, `.pe`, `m.0.ffn.0`, `m.0.ffn.1`, each with `conv.weight`, `bn.weight`,
`bn.bias`, `bn.running_mean`, `bn.running_var`. Weights: `cv1`, `cv2` `[256,1,1,256]`; `qkv`
`[256,1,1,128]`; `proj` `[128,1,1,128]`; `pe` `[3,3,128]`; `ffn.0` `[256,1,1,128]`; `ffn.1`
`[128,1,1,256]`. Row 22 (`mj:34`) holds the same `PSABlock` under `m.0.1`, after a
`Bottleneck(128, 128, 64)` under `m.0.0` (records 345 to 389), with 2 heads and no `heads` key.

**Operations the attention needs.** Every identity of `ops.hpp` has a lowering row today
(`modules/core/graph/include/SushiAI/graph/lowering/all.hpp`: 62 rows, 3 unlowered).

| Need | Emitter and shape rule | Gradient rule | Lowering | SushiBLAS kernel |
| --- | --- | --- | --- | --- |
| `[B,H,W,C]` to tokens | `reshape`, `builder.cpp:1113` | exists | copy, `lowering/shape.cpp:60` | `copy` |
| transpose or permute | not needed: `matmul` takes two flags | | | none exists in `sb:operations` |
| batched product | **missing**: `matmul` throws unless both operands are rank 2 (`builder.cpp:168`) | `matmul_backward` builds its results through `matmul` with flags (`autograd/source/rules/linalg.cpp:34-66`), so it serves rank 3 once the emitter does | exists unchanged: one `gemm` (`lowering/linalg.cpp:49`) | `gemm` is batched above rank 2, contiguous operands, the output sets the batch count (`sb:operations/blas/include/SushiBLAS/blas/level3.hpp:45-53`; `blas/README.md:399-405`) |
| softmax on the last axis | **no emitter**; composable as `reshape`, `log_softmax` (rank 2, `builder.cpp:445`), `exp`, `reshape` | missing: `LOG_SOFTMAX` is `op()` (`ops.hpp:249`); inference does not need it | exists for each part (`lowering/loss.cpp:43`, 9 tasks, then `exp`) | no softmax kernel; `max_along`, `ger`, `exp`, `gemv`, `log` |
| scale by a constant | `scale`, `builder.cpp:291` | exists | exists | `scal` |
| depthwise convolution | **missing**: no identity, no emitter; `conv2d` requires weight `C_in` equal to input C (`builder.cpp:663`) | missing | missing | **exists**: `window_dot`, `window_dot_backward`, `window_dot_weights_backward`, taps `[kernel_h, kernel_w, C]` (`sb:operations/spatial/include/SushiBLAS/spatial/spatial.hpp:127-150`), the fixture's weight layout |
| split and join on channels | `slice_channels` `builder.cpp:918`, `concat_channels` `:876` | exist | exist | exist |

**Row 23, `detect`** (`mj:35`, reads rows 16, 19, 22; `U:nn/modules/head.py:123-143` region).
Per level, with input widths 64, 128, 256: a box branch `Conv 3x3 (x to 16)`, `Conv 3x3 (16 to
16)`, a biased 1x1 convolution to 4; a class branch `DWConv 3x3 (x) + Conv 1x1 (x to 80)`,
`DWConv 3x3 (80) + Conv 1x1 (80 to 80)`, a biased 1x1 convolution to 80. `DWConv` is a depthwise
convolution with BN and SiLU. The fixture shows direct regression: `reg_max` is 1, the box
outputs have 4 channels and there is no DFL state. The head exists twice, as `cv2`, `cv3` and as
`one2one_cv2`, `one2one_cv3` (records 390 to 593, from `MF:6485`); training detaches the
features that enter the second copy (`head.py:196`), evaluation does not. Raw outputs in
`forward.sacp`: `<branch>.level<l>.boxes` `[1,H,W,4]` and `.scores` `[1,H,W,80]`, logits, for
`one2many`, `one2one` and H = W = 80, 40, 20. The `detect` module decodes host arrays:
`decode_xyxy` and `decode_xywh` read `[B,A,4]` distances and `[A,2]` anchor points, `scale_boxes`
an `[A]` stride (`modules/vision/detect/include/SushiAI/vision/detect/box_coding.hpp:42-71`), and
`select_top_detections` reads `[B,A,4]` xyxy boxes and `[B,A,C]` probabilities
(`detections_topk.hpp:58`), so the levels are joined and the sigmoid applied first.

**`forward.sacp`** (`MF:9856`; 21 tensors, all float32): `features.level0..2` (`[1,80,80,64]`,
`[1,40,40,128]`, `[1,20,20,256]`, the outputs of rows 16, 19 and 22), the 12 raw outputs,
`<branch>.decoded.boxes` `[1,8400,4]` and `.scores` `[1,8400,80]` (one-to-many in xywh pixels,
one-to-one in xyxy pixels, scores after the sigmoid), `one2one.detections` `[1,300,6]` and
`one2many.suppressed.detections` `[300,6]`. A parity test can compare three intermediate tensors
and everything from the head on. All three features depend on row 10, and no tensor isolates it.

Tolerances of the earlier fixture tests, all absolute (`tests/common/reference_fixture.hpp:238`):
0 for top-k and suppression rows (`test_detect_postprocess.cpp:52`), `1.3e-4` for a decode from
reference raw outputs, `1.6e-5` for the round trip, `1e-6` for IoU (`test_detect_boxes.cpp:50-59`).

`weights.sacp` holds 594 records named `model.<row>...`; `ModelState` names are the same without
the prefix, so one rule serves: `IO::RenameRule{"model.", ""}`. `import_weights` fails on a
record the model lacks and on a tensor the file lacks (`modules/layers/io/source/import_weights.cpp:105`,
`:135`), so only the whole model imports. The batch normalisation is not folded: every layer
carries `running_mean` and `running_var`, drawn at random by the exporter, with epsilon `1e-3`.

**The `inference` program** stops when a model publishes more than one output
(`applications/programs/source/inference.cpp:112-114`) and counts classes over labelled vectors.
A detection report needs an image source, the letterbox and its inverse, the decode, top-k, the
confidence cut and a box output format. The first and the last are public interfaces nobody has
designed, so the report is a later plan; this one ends at the parity test.

### Units

Hierarchy order; the dispatch is `model effort`; `vb/` is `modules/vision/blocks/`, and a bare
header name is under `vb/include/SushiAI/vision/blocks/`. Each block type below also declares
`static constexpr auto fields()`, `Graph::ValueId forward(const NN::Trace& trace, Graph::ValueId x)`
and `Shape output_shape(const Shape& input) const`, as `SPPF` does.

**U1. Depthwise identities and emitters.** `haiku high`. Files:
`modules/core/core/include/SushiAI/core/ops.hpp`, `modules/core/graph/include/SushiAI/graph/builder.hpp`,
`modules/core/graph/source/builder.cpp`, `modules/core/core/tests/test_ops.cpp`, new
`modules/core/graph/tests/test_depthwise_ops.cpp`.
```cpp
inline constexpr OpName K_DEPTHWISE_CONV2D = differentiable_op("sushiai.depthwise_conv2d");
inline constexpr OpName K_DEPTHWISE_CONV2D_BACKWARD_INPUT = op("sushiai.depthwise_conv2d_backward_input");
inline constexpr OpName K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT = op("sushiai.depthwise_conv2d_backward_weight");
ValueId GraphBuilder::depthwise_conv2d(ValueId x, ValueId weight, const ConvGeometry& g) const;
ValueId GraphBuilder::depthwise_conv2d_backward_input(ValueId dy, ValueId weight, Shape input_shape, const ConvGeometry& g) const;
ValueId GraphBuilder::depthwise_conv2d_backward_weight(ValueId dy, ValueId x, const ConvGeometry& g) const;
```
Shape rule: `x` rank 4 NHWC `[N,H,W,C]`; `weight` rank 3 `[g.kernel_h, g.kernel_w, C]` of the same
element type; result `[N, out_h, out_w, C]` from `window_extents`, padded symmetrically as
`conv2d` is, geometry in the attribute slots `maxpool2d` uses. Tests (`Unit_DepthwiseOps`): the
identity, operands, shape and layout of each emitter; refusal of a rank-4 weight, a wrong tap or
channel extent, an NCHW input; `test_ops.cpp` lists 68 identities. Acceptance: both suites pass.

**U2. Rank-3 `MATMUL` and the softmax emitter.** `haiku high`. Waits on U1, which holds the
same files. Files: `builder.hpp`, `builder.cpp`, `ops.hpp` (the comment of `K_MATMUL`), new
`modules/core/graph/tests/test_batched_matmul.cpp`, new `tests/integration/test_attention_operations.cpp`.
```cpp
ValueId GraphBuilder::matmul(ValueId a, ValueId b, bool transpose_a = false, bool transpose_b = false) const;
ValueId GraphBuilder::softmax(ValueId x) const;                 // over the last axis, rank >= 2
```
`matmul` accepts two rank-3 operands whose first extents agree and yields `[B, m, n]`; mixed
ranks throw. `softmax` emits `reshape` to `[elements / last, last]`, `log_softmax`, `exp` and
`reshape` back; it adds no identity and has no gradient. Tests (`Unit_BatchedMatmul`): the four
flag combinations at rank 3; refusal of mixed ranks and unequal batches; the four nodes of
`softmax`; `matmul_backward` over a rank-3 node emits rank-3 products.
`Integration_AttentionOperations` (device): a `[2,3,4]` by `[2,5,4]` transposed product and a
`[2,3,3]` softmax against values computed in double, `1e-6` absolute. Acceptance: both pass.

**U3. Depthwise lowerings.** `haiku high`. Waits on U1. Files:
`modules/core/graph/include/SushiAI/graph/lowering/spatial.hpp`,
`modules/core/graph/source/lowering/spatial.cpp`, `modules/core/graph/tests/test_lowering_table.cpp`,
`tests/integration/test_lowered_operations.cpp`. `Status lower_<name>(const LoweringContext& context)`
for `depthwise_conv2d` (`window_dot`), `depthwise_conv2d_backward_input` (`window_dot_backward`)
and `depthwise_conv2d_backward_weight` (`window_dot_weights_backward`); `K_SPATIAL_LOWERINGS` has
15 rows. Tests: the table holds 65 rows; a 3x3, stride 1, pad 1 forward over `[1,4,4,2]` against
three nested loops in the test, `1e-6`. Acceptance: both pass.

**U4. Depthwise gradient rule.** `haiku high`. Waits on U1. Files:
`modules/core/autograd/include/SushiAI/autograd/rules/spatial.hpp`,
`modules/core/autograd/source/rules/spatial.cpp`, `modules/core/autograd/tests/test_rule_table.cpp`,
`modules/core/autograd/tests/test_precision.cpp`, `tests/integration/test_autograd_numeric.cpp`.
`void depthwise_conv2d_backward(const GradientTape& tape)` contributes the two gradient emitters
of U1 to `x` and to the weight, each only when its operand needs a gradient; `K_SPATIAL_RULES`
has 9 rows. Tests: the row exists; the three identities stay wide under mixed precision; one
finite-difference case over input and weight. Acceptance: that case passes.

**U5. `NN::DepthwiseConv2d`.** `haiku high`. Waits on U1. Files: new
`modules/layers/nn/include/SushiAI/nn/depthwise_conv2d.hpp`, `modules/layers/nn/source/depthwise_conv2d.cpp`,
`modules/layers/nn/tests/test_depthwise_conv2d.cpp`; `modules/layers/nn/README.md`.
```cpp
struct DepthwiseConv2d
{
    int64_t channels = 0;
    Graph::ConvGeometry geometry{};
    Parameter weight{};     // [kernel_h, kernel_w, channels]
    static constexpr auto fields();                       // field("weight", ...)
    [[nodiscard]] static DepthwiseConv2d same_padded(int64_t in_channels, int64_t out_channels,
                                                     int64_t kernel, int64_t stride);
    Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x);
    Shape output_shape(const Shape& input) const;
};
```
Shaped as `Conv2d`; `same_padded` refuses unequal widths. State name: `weight`. Tests
(`Unit_DepthwiseConv2d`): the node it emits, the declared weight shape, `output_shape` at strides
1 and 2, the refusals, the state walk. Acceptance: the suite passes.

**U6. `Attention` and the depthwise aliases.** `opus low`. Waits on U2, U5. Files:
`conv_norm.hpp`, `vb/tests/test_block_conv_norm.cpp`; new `attention.hpp`,
`vb/source/attention.cpp`, `vb/tests/test_block_attention.cpp`.
```cpp
using DWConv = ConvNorm<NN::DepthwiseConv2d, NN::SiLU>;
using DWConvLinear = ConvNorm<NN::DepthwiseConv2d, NN::Identity>;
struct Attention            // fields: child "qkv", "proj", "pe"
{
    int64_t heads = 0, head_dim = 0, key_dim = 0;   // head_dim = channels / heads, key_dim = head_dim / 2
    ConvLinear qkv;                                 // channels -> channels + 2 * heads * key_dim
    ConvLinear proj;
    DWConvLinear pe;
    Attention(int64_t channels, int64_t head_count, NN::NormOptions norm);
};
```
`forward` follows the table under "Measured" and scales q before the product. Tests
(`Unit_BlockAttention`), for two heads: 6 slices, 2 scales, 4 products, 2 softmax groups, 1 add;
the operands and flags of each product; the scale against `1 / sqrt(32)`; state names and shapes
against manifest records 175 to 189; refusal of an indivisible width. Acceptance: it passes.

**U7. `PSABlock` and `C2PSA`.** `haiku high`. Waits on U6. Files: new `psa_block.hpp`,
`c2psa.hpp`, `vb/source/psa_block.cpp`, `vb/source/c2psa.cpp`,
`vb/tests/test_block_psa_block.cpp`, `vb/tests/test_block_c2psa.cpp`; `vb/README.md`.
```cpp
struct PSABlock             // fields: child "attn", "ffn"; forward: x += attn(x); x += ffn(x)
{
    Attention attn;
    NN::Sequential<Conv, ConvLinear> ffn;                 // c -> 2c -> c
    PSABlock(int64_t channels, int64_t heads, NN::NormOptions norm);
};
struct C2PSA                // fields: child "cv1", "cv2", "m"
{
    int64_t hidden = 0;
    Conv cv1;               // in -> 2 * hidden
    Conv cv2;               // 2 * hidden -> out
    NN::ModuleList<PSABlock> m;
    C2PSA(int64_t in_channels, int64_t out_channels, int64_t hidden_channels, int64_t repeats, int64_t heads, NN::NormOptions norm);
};
[[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c2psa_node(NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);
```
Keys of a `c2psa` row: `filters`, `hidden`; optional `repeats` (1), `heads`
(`max(hidden / 64, 1)`), `bn_epsilon`, `bn_momentum`; `filters` must equal the input width.
Tests: the two adds of `PSABlock` and what each reads; `C2PSA` chains off its second slice;
state against records 165 to 199; the factory builds row 10. Acceptance: both suites pass.

**U8. `DetectHead`.** `opus low`. Waits on U6. Files: new `detect_head.hpp`,
`vb/source/detect_head.cpp`, `vb/tests/test_block_detect_head.cpp`.
```cpp
using BoxBranch = NN::Sequential<Conv, Conv, NN::Conv2d>;
using ClassStage = NN::Sequential<DWConv, Conv>;
using ClassBranch = NN::Sequential<ClassStage, ClassStage, NN::Conv2d>;
class DetectHead final : public NN::Node   // overrides forward, walk_state, input_count, ports
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<DetectHead>> create(NN::ShapeSpan levels, int64_t classes, NN::NormOptions norm);
};
[[nodiscard]] Result<std::unique_ptr<NN::Node>> make_detect_node(NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);
```
State: `cv2.<l>.<0..2>`, `cv3.<l>.<0..1>.<0..1>`, `cv3.<l>.2`, and the same under `one2one_cv2`
and `one2one_cv3`. Widths: box branch `max(16, C0 / 4, 4)`, class branch
`max(C0, min(classes, 100))`. Ports, four per level, in the fixture's order and names:
`one2many.level<l>.boxes`, `one2many.level<l>.scores`, then the two `one2one` groups. A training
trace passes each feature through `stop_gradient` before the second copy; an evaluation trace
does not. Keys of a `detect` row: `classes`; optional `bn_epsilon`, `bn_momentum`. Tests
(`Unit_BlockDetectHead`): port names and shapes; the 204 state names and shapes against records
390 to 593; three `STOP_GRADIENT` nodes in training, none in evaluation. Acceptance: it passes.

**U9. The `bottleneck_psa` row of `c3k2`.** `haiku medium`. Waits on U7. Files: `c3k2.hpp`,
`vb/source/c3k2.cpp`, `vb/tests/test_block_c3k2.cpp`. `make_c3k2_of_bottleneck_psa`, a
`NodeFactory`, builds a `C3k2` over `NN::Sequential<Bottleneck, PSABlock>` of `Bottleneck(hidden,
hidden, hidden / 2)` and `PSABlock(hidden, max(hidden / 64, 1))`, and is the `"bottleneck_psa"` row
of `K_C3K2_INNER`. Tests: the factory builds row 22; state against records 345 to 389.
Acceptance: `Unit_BlockC3k2` passes.

**U10. Registration and the model file.** `haiku medium`. Waits on U7, U8. Files:
`modules/vision/registration/source/registration.cpp`, `.../registration/tests/test_registration.cpp`,
`.../registration/README.md`, `applications/examples/yolo26/README.md`,
`modules/layers/nn/tests/test_node_registry.cpp`. `register_nodes` adds
`{"c2psa", &Blocks::make_c2psa_node}` and `{"detect", &Blocks::make_detect_node}`. Tests: seven
rows in order; `model.json` builds whole and publishes twelve names; the walked state equals the
594 manifest names without `model.`, shapes included. Acceptance: that name test passes.

**U11. Forward parity.** `opus low`. Waits on U3, U9, U10. File: new
`tests/integration/test_yolo26_forward.cpp`, suite `Integration_Yolo26Forward`, device. It loads
`model.json`, publishes rows 16, 19 and 22 beside the head, builds the evaluation graph for
`[1, 640, 640, 3]`, imports `weights.sacp` through `{"model.", ""}`, feeds `image` of
`input.sacp` and runs one `EvaluationStep`. A case is skipped when the fixture is absent.
Acceptance: the four cases pass at the tolerances of question 3; a miss is reported, not tuned.

| Case | Compares |
| --- | --- |
| features | rows 16, 19, 22 against `features.level0..2` |
| raw outputs | the twelve ports against their records |
| decodes | the host decode of our raw outputs (sigmoid, `make_anchors`, `join_level`, `decode_xywh` or `decode_xyxy`, `scale_boxes`) against `<branch>.decoded.*` |
| detections | the score column of `select_top_detections` against column 4 of `one2one.detections`; boxes and classes are not compared, because near-equal scores reorder rows |

### Waves and builds

| Wave | Units | Closes on | The orchestrator adds |
| --- | --- | --- | --- |
| 1 | U1 | syntax check | `modules/core/graph/CMakeLists.txt`: `tests/test_depthwise_ops.cpp` |
| 2 | U2, U3, U4, U5 | **build 1**, `sushiai_tests` | graph: `tests/test_batched_matmul.cpp`; `modules/layers/nn/CMakeLists.txt`: the source and the test of U5; `SushiAI/SushiAI.h`: `nn/depthwise_conv2d.hpp`; `tests/CMakeLists.txt`: `integration/test_attention_operations.cpp` |
| 3 | U6 | syntax check | `modules/vision/blocks/CMakeLists.txt`: `source/attention.cpp`, `tests/test_block_attention.cpp`; `SushiAI/vision/vision.h`: `blocks/attention.hpp` |
| 4 | U7, U8 | syntax check | blocks: three sources, three tests; `vision.h`: three headers |
| 5 | U9, U10, U11 | **build 2**, both test binaries | `tests/CMakeLists.txt`: `integration/test_yolo26_forward.cpp` in `sushiai_vision_tests`, the first root file that binary takes |

`cmake/SushiAILayers.cmake` gains nothing: no module is added. Build 1 finds a wrong kernel call
or gradient in a four-element test and not under 23 rows; one build at the end gives that up.
The head stays beside the blocks (model-graph ruling 6a), so `vision/detect` keeps to `core`.

### Questions for the owner

1. **Rank-3 products.** `MATMUL` is documented as 2-D. (a) Widen it, as U2 plans: its lowering
   and rule already serve rank 3. (b) A new `BATCH_MATMUL` identity. Recommendation: a.
2. **Depthwise gradients now or later.** (a) Three identities now, as U1, U3 and U4 plan; the
   kernels exist. (b) The forward identity alone as `op()`; U4 falls away, a training graph
   throws "No backward rule", and the `differentiable` flag changes later. Recommendation: a.
3. **Parity tolerance.** No run exists to take a number from. (a) Fix it now: `1e-4` absolute
   on features and raw outputs, the existing `1.3e-4` on decoded boxes, `1e-6` on probabilities.
   (b) Measure in build 2 and pin what the owner approves. Recommendation: a.
4. **More reference tensors.** (a) Leave the fixture; a fault in row 10 shows only from row 16.
   (b) Rerun the exporter with the outputs of rows 9, 10 and 21 added (1.4 MB on a 41.9 MB set
   under a 64 MiB ceiling), as a unit before wave 5 and an approved fixture change. Recommendation: b.

### Not in this plan, and not verified

- Not planned: the detection report, initialisers (the head's biases among them), a softmax
  gradient, a registry row for `depthwise_conv2d`.
- That the SushiBLAS the build resolves has `window_dot`. SushiAI's lowerings include
  `<SushiBLAS/engine/spatial.hpp>` (`lowering/norm.cpp:36`); the sibling tree, which is being
  moved, has the header at `SushiBLAS/spatial/spatial.hpp` with no commit. Build 1 settles it,
  and whether `gemm` runs on a batch. The cost of `log_softmax` over `[400, 400]` is unmeasured.
- That `NN::Sequential` names its children `0`, `1` and takes a nested `Sequential` and a plain
  `Conv2d` (`layers.hpp:181-300`, signatures only); the state tests of U7 to U9 pin it. How a
  biased 1x1 `Conv2d` is built, the accessor of a trace's mode and the `ModelSpec` field of the
  outputs were not read, which is why U8 and U11 are `opus low`.
