# Report: the first full gating run of the export branch and the large Darknet cases

**Status:** Open — written 2026-11-25; RW-103, RW-193 and RW-197 hold what the runs left
**Designs:** MODEL_IMPORT

Branch `feat/model-export` at `a4b01c7e`, debug build, `SA_REFERENCE_DATA_DIR` set to the Darknet reference data.

## Gating run (`sa test -l gating`)

73 of 73 test sets passed in 256 s. This is the first full run that holds the export tests: the three ONNX fixture tests (which expect the refusal of row 6 'flatten'), the CLI test and the program tests all passed. `ThePlanOfTheStepChangesNoValue` (RW-192) did not fail in this run; one green run does not close it, since it failed under load only.

Slow medium shards in that run, against their 240 s timeout:

| Shard | Time |
| --- | --- |
| `medium_shard_8of17` | 232 s |
| `medium_shard_15of17` | 130 s |
| `medium_shard_5of17` | 102 s |

RW-193 still holds: shard 8of17 ended 8 s before its timeout.

## Large Darknet run (`sa test -l large -f "Large_Darknet*"`)

The filter `Large_Darknet` without the wildcard matched no case (KI-19). With the wildcard, 42 test sets ran in 469 s.

- 36 cases passed: the four forward cases and the eight training cases of each of YOLOv3, YOLOv4 and YOLOv7, against the dumps. Each took 0.3 to 12 s.
- 3 cases timed out and none failed: `TheCostOfOneBatchFallsUnderATenth` of the three models took 129, 155 and 184 s against the 120 s limit. The cases did not reach their verdict. Filed as RW-197; RW-103 stays open.

## Backlog

- Closed RW-116.
- Opened RW-197.
- Still open from this run: RW-103, RW-193, RW-192.

## Not done

The tiny models' large cases do not exist; their cases are `Medium_` and ran in the gating run. The overfit curves of v3-tiny and v4-tiny are still unmeasured.
