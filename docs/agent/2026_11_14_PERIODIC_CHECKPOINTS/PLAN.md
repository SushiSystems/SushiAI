# Periodic checkpoints: plan

**Status:** Open — planned, nothing written
**Designs:** DATASETS

## 1. Problem

`sushiai train` writes its checkpoint once, after the last epoch (`applications/programs/source/training_program.cpp`).
A run that stops in epoch k, by a crash, a kill or a full disk, loses every update it made. The
loop could resume from a batch: `epoch_loop.cpp` starts at `first_epoch` and `first_batch`, and
`IO::RunProgress` holds the epoch and the batch the next update starts from. Only the writing in
the middle of a run is missing. RW-71 states one half of it, a save at a batch whose written index is
that batch.

## 2. Decisions

| Subject | Ruling |
| --- | --- |
| The key | `checkpoint_every` in the training file, an object with exactly one member, `epochs` or `batches`, a positive integer. An absent key keeps today's behaviour: one checkpoint at the end |
| The file version | 4. The key is the first of version 4, as `cluster` was of 2 and `deterministic` of 3: `K_TRAIN_VERSION` becomes 4 and `K_TRAIN_CHECKPOINT_EVERY_VERSION` names it. A file of an older version that names the key is refused |
| What the write holds | The model state, the run state and the progress of the next update, through `save_run_checkpoint`, so a resume from the file goes on at that epoch and batch |
| Where it goes | The one `checkpoint` path of the file, replaced each time. The write is the atomic one of `committed_file.hpp`, so the previous file stays whole until the new one is complete |
| The count | `batches: N` writes after every N-th update of the run, counting from the first update of this run, not from the epoch; `epochs: N` writes after the last batch of every N-th epoch of the run. The final checkpoint is written as today and carries the same progress as a periodic one would |
| What does not take it | A run row that does not drive `run_epoch_loop`, and a cluster run, refuse the key with `Unsupported`, naming it. A key that is ignored silently is a defect |
| The cost | A write stalls the loop for the time of the write; the page of the training file says so with the size a YOLOv3-tiny checkpoint has, 70 MB |

## 3. Tasks

Each task has one criterion. The build and the test run come once, after the three are written.

| Id | Task | Accepted when |
| --- | --- | --- |
| T1 | Read and check `checkpoint_every` | A config test accepts `{"epochs": 2}` and `{"batches": 3}` into the spec, and refuses an empty object, both members, a member of 0 or below, a member that is no integer, an unknown member, and a version 3 file that names the key |
| T2 | Call a hook from the epoch loop | A loop test over 3 epochs of 4 batches records the epoch and the next batch at each call: for `batches: 5` the calls come after updates 5 and 10, for `epochs: 2` after the last batch of epoch 1 |
| T3 | Write it from `sushiai train` | A program test runs the tiny fixture with `batches: 1` and loads the checkpoint with `load_run_checkpoint`: its progress is the epoch and the batch after the last update, and a run of the same file with `resume` set to it ends where the uninterrupted run ended |

T4 closes the work: the training file page, the programs README, the changelog and RW-71 and
the entry T1 opens for the cluster and the fallback run.

## 4. Not in this work

- Keeping a number of old checkpoints: one file, replaced.
- Writing at a time interval: a clock makes a run depend on the machine, and the loop does not read one.
- A cluster run: its replicas hold the state, and which of them writes is a design of
  `docs/design/LAYERED_EXECUTION/README.md`.
