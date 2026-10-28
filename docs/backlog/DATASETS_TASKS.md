# Datasets tasks

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-59 | P2 | unmeasured | Run `Batch` with roles and `vision/codecs` | both pass, with `test_learning_curve`, `test_determinism` and `test_compile_once` |  | `docs/design/DATASETS/README.md` | 2026-10-25 |
| RW-60 | P2 | unmeasured | See the loader allocate from two threads | `Medium_DatasetsReaderSection.WorkersReadTheSameBatchesAhead` passes in a recorded run |  |  | 2026-10-25 |
| RW-61 | P2 | unmeasured | Run a death test inside an engine fixture | `ABatchOutsideTheEpochStopsTheProcess` is seen to pass engine start-up in its child |  |  | 2026-10-19 |
| RW-63 | P2 | open | Give the imaging functions a status | they return a status, and `fit_sample` reports each error as what it is |  |  | 2026-10-19 |
| RW-64 | P2 | open | Average on reduction when resizing | `resize_bilinear` does not alias a large image fitted to a small input |  |  | 2026-10-19 |
| RW-66 | P2 | open | Give the decoders one exception boundary | the four functions of `vision/codecs` share one unit, and `raw_decoder` leaves `decode_image.cpp` |  |  | 2026-10-19 |
| RW-67 | P2 | open | Test a 16-bit PNG without a `gAMA` chunk | the case exists |  |  | 2026-10-19 |
| RW-68 | P2 | open | Let a collator say it reads no mask | a collator that reads no mask stops a mask being decoded |  |  | 2026-10-19 |
| RW-71 | P2 | open | Checkpoint inside an epoch | `sushiai train` saves at a batch and the written batch index is that batch |  |  | 2026-10-19 |
| RW-72 | P2 | open | Test the cluster loop and the trainer run from a start other than zero | a test starts the cluster loop and the `Trainer` fallback run past epoch 0 and past batch 0, as `test_epoch_loop.cpp` does |  |  | 2026-10-25 |
| RW-73 | P2 | open | Let `Trainer::fit` resume past epoch 0 | `fit` begins at the epoch it is given |  |  | 2026-10-19 |
| RW-74 | P2 | open | Write version 2 of the weight file from the YOLO26 exporter | `tools/reference/export_yolo26_reference.py` writes version 2 |  |  | 2026-10-19 |
| RW-75 | P2 | open | Keep the old file when a save is refused | a refused save leaves the target as it was |  |  | 2026-10-19 |
| RW-76 | P2 | open | Let an objective name its metrics before it measures | a sum of two terms computes mean IoU once |  |  | 2026-10-19 |
| RW-77 | P2 | unmeasured | Run segmentation | the objectives, the head and the overfit test pass |  |  | 2026-10-25 |
| RW-78 | P2 | open | Upsample bilinearly in the graph | a head can have boundaries finer than its scale (G7 of the graph capabilities plan) |  |  | 2026-10-19 |
| RW-79 | P2 | open | Take several polygons per object in a label | an outline of more than one part loads |  |  | 2026-10-19 |
| RW-80 | P2 | open | Give the `default` pipeline keys for its probabilities, gains and mosaic epochs | each is a key |  |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
