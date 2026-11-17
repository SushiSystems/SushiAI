# tests/layers/train/test_trainer/test_unwritten_leaf.cpp

Suites: `Medium_TrainerUnwrittenLeaf`.

A `Linear` over three features is trained through `Train::Trainer` by a scripted
`Train::InMemoryObjective`. It declares the leaves `targets` and `type_mask_weights` as the
`CrossEntropyObjective` with a row mask does, emits that objective's loss, and its scripted feed
fills `targets` alone. The trainer
compiles on a device, so the suite is `Medium_`; the marking itself is tested `Small_` in
`test_leaf_writes.cpp`.

## RunStepRefusesALeafTheFeedSkipped

`run_step` over a batch of four samples fails with `InvalidDescription`, "'type_mask_weights'" and
"unwritten" (RW-111). The feed returned success, so the refusal comes from the trainer's check,
before the step ran.
