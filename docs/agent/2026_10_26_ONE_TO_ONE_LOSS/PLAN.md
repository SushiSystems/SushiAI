# The one-to-one loss of an anchor-free head: plan

**Status:** Open — written 2026-10-26; the owner asked that day for the loss to be written before the profiler, after `sa train` refused YOLO26 because its one-to-one copy received no gradient.

## 1. The goal

A model that publishes a one-to-one copy of its anchor-free head, such as the YOLO26 example,
trains both copies in one step. Every parameter of the copy receives a gradient, so the trainer's
check that each parameter is on a path from the loss passes.

## 2. What the reference does

Ultralytics' `E2ELoss` at commit `ce13994` (`U/utils/loss.py`) builds the detection loss twice:
the one-to-many copy with `tal_topk=10`, the one-to-one copy with `tal_topk=7, tal_topk2=1`. The
total is `o2m * L_one2many + o2o * L_one2one`. `o2m` starts at 0.8 and falls in a straight line to
0.1 over `epochs - 1` updates, `max(1 - x / max(epochs - 1, 1), 0) * (0.8 - 0.1) + 0.1`, and `o2o`
is `1 - o2m`. Its logged items are those of the one-to-one copy. The head feeds the one-to-one copy
detached features, which SushiAI's `DetectHead` already does with `stop_gradient`.

## 3. Rulings

| Subject | Ruling |
| --- | --- |
| Scope | Training only; inference keeps reading the one-to-many ports (owner, 2026-10-26: write the loss now) |
| Keys | `one_to_many_weight_start` and `one_to_many_weight_end` in the `anchor_free` objective section, defaults 0.8 and 0.1, each in 0..1 |
| Weights in the graph | Two `[1]` leaves the step writes each run, so a change of weight recompiles nothing |
| Epoch count of the schedule | The epochs the run reaches, `first_epoch + epochs`; a resumed run counts from epoch 0 |
| Reported terms | Each term is the weighted sum of the two copies' terms, so the reported total is the loss the graph differentiates. This differs from the reference, which logs the one-to-one terms |

## 4. Tasks

| Id | Task | Accepted when |
| --- | --- | --- |
| O1 | `FreeAssignmentStage` takes a `FreeBranch`; one-to-one leaves are named `free.one_to_one.<kind>` | a one-to-one stage names its leaves apart and a model without the copy is refused |
| O2 | `FreeBranchWeights`, its reader and `one_to_many_weight` | the schedule tests pin 0.8, 0.7, 0.1 over eight epochs |
| O3 | `FreeObjective` declares both stages' leaves and the two weight leaves and emits the weighted sum | the one-to-one step test moves every weight |
| O4 | `FreeTrainingStep` keeps one branch per copy; `run_free_training` sets the weight per epoch | the YOLO26 example trains ten epochs and writes its checkpoint |
