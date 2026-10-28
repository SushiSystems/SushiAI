# Memory planning speed

## Remaining work

| ID | Pri | State | Title | Done when | Blocked by | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-100 | P2 | open | Write the scratch rules of the rows that have none | the rows of `lowering/norm.cpp`, `loss.cpp`, `linalg.cpp` and `random.cpp` that call `LoweringContext::scratch()`, and the shared `ones()` tensors, have… |  |  | 2026-10-19 |
| RW-101 | P2 | open | Find why a full-size training step leaves Darknet | a Darknet dump of each layer's output under batch statistics and a case that walks them show where YOLOv4 and YOLOv7 leave it |  |  | 2026-10-19 |
| RW-102 | P2 | open | Close the step time against Darknet | a full-size step takes the time of a Darknet iteration on the same card |  |  | 2026-10-19 |
| RW-103 | P2 | open | Make the one-batch overfit of YOLOv7 hold | `Large_DarknetYolov7Training.TheCostOfOneBatchFallsUnderATenth` passes, the cause of the instability found and the bound ruled by the owner |  |  | 2026-10-19 |
| RW-104 | P2 | unmeasured | Measure where YOLOv3's training step leaves Darknet, layer by layer | a per-layer dump under batch statistics shows whether the difference grows over the 75 convolutions or enters at one |  |  | 2026-10-19 |
| RW-105 | P2 | blocked | Feed a batch without two copies | a step writes the batch into the plan's input buffers without two `blas().copy()` calls | sushiruntime RW-59 |  | 2026-10-19 |

## Known issues

| ID | Origin | Symptom | Workaround | Fix | Source | Opened |
| --- | --- | --- | --- | --- | --- | --- |
