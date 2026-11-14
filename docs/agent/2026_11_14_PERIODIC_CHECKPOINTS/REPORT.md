# Periodic checkpoints: report

**Status:** Open — built and measured; the next commit closes it

## What was built

T1 reads `checkpoint_every` (commit `02058a02`). T2 and T3 add `progress_to_checkpoint` and
`checkpoint_after_update` to `training_run`, the `after_update` hook of the epoch loop and an
`after_update` option of the anchor and anchor-free loops, which return a `Status` so a failed
write ends the run without a throw; the three rows call it, and a file with a `cluster` section
refuses the key.

## One run, measured

2026-11-14, debug tree, YOLOv3-tiny from the imported weights over `coco128`, batch size 4,
2 epochs of 32 updates, rate 1e-5 with a warm-up of 10 updates.

| Run | What it did |
| --- | --- |
| Reference | `checkpoint` only; 27 s; epoch losses 24.310044 and 22.718462; `ref.sushi` of 70 848 398 bytes |
| Interrupted | `checkpoint_every` of 20 batches; a checkpoint of the same size appeared during the run and the process was killed at once; no `.partial` file stayed |
| Resumed | `resume` from that file, 2 epochs; epoch losses 24.310044 and 22.718462; `resumed.sushi` |

`cmp ref.sushi resumed.sushi` reported no difference. This is one model, one kill point and one
machine; it is not a claim for every run, and no test pins it.

## What is tested

The rule is tested as `Small_CheckpointSchedule`, the hook of the epoch loop in `test_epoch_loop.cpp`
and the hook of the anchor and the anchor-free loops in their loop tests, each with the refusal of a
failed hook. `Medium_InferenceProgram` runs the tiny fixture for 2 epochs with `batches: 1` and
finds a whole checkpoint, no partial file and a report that reads it. No test compares a resumed
run with an uninterrupted one; the measured run above is the only evidence of that.
