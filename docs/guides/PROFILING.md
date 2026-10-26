# Profiling a run

How to record where a training step or an inference batch spends its time, and how to read the record. SushiAI measures through SushiProfiler alone: the programs `sushiai train` and `sushiai inference` take one flag, `--profile <config>`, and the records are read with `sa profile`.

## Recording a run

```bash
sa train     train.json --profile profile.json
sa inference model.json model.sushi eval.json --profile profile.json
```

`profile.json` is a SushiProfiler configuration; SushiProfiler's configuration format reference lists its keys. A run profiled on the production path looks like this:

```json
{
    "sushiprofiler": 1,
    "timeline": "train.step",
    "window": { "skip": 0, "wait": 0, "warmup": 3, "active": 20, "repeat": 1 },
    "sources": { "sushiruntime.device": {} },
    "analyses": ["breakdown", "device_idle", "roofline"],
    "writers": {
        "terminal": { "top": 20 },
        "record": { "path": "run.sprec.json" }
    },
    "output": "build/profiles/mlp"
}
```

Without `--profile` no session opens, nothing is written and every zone stays closed. With it:

- A path that does not exist, a file that is not JSON and an unknown key stop the program before the first step. The message names the file and quotes SushiProfiler.
- A run that fails still writes its record, and the program reports the failure of the run, not the result of the record.

## What a record holds

Each training step is one period of the timeline `train.step`, and each inference batch one period of `inference.batch`. The `timeline` key says which of the two the window follows. The step that is running when the run ends has no end mark of its own, so the program ends it with one last mark; a window longer than the run records every step.

The zones of a period belong to a group, and `breakdown` adds them up by group.

| Group | Stages | What it times |
| --- | --- | --- |
| `data` | `read` | `dataset.read` |
| `host` | `feed`, `readback`, `assign`, `loss`, `measure` | Host work around the device: leaves written for a step, outputs read back, targets assigned, the loss and the metrics |
| `copy` | `input` | The copy of a batch into the step's input buffer |
| `device` | `run` | The step's replay, which waits for its device work |
| `optim` | `schedule` | The learning rate set before a step |
| `eval` | `decode`, `nms`, `ap` | Box decoding, suppression and the average-precision update of a detector |
| `io` | `checkpoint` | A checkpoint saved or loaded |
| `compile` | `lower`, `runtime` | Lowering the graph and the runtime's plan, once per compiled step |
| `eager` | `partition`, `plan`, `compile`, `execute`, `flush` | The stages of an eager step; `flush` holds the others |
| `train` | `fit` | `Trainer::fit` as a whole |

A zone outside every period, such as `io.checkpoint` before the first step, is counted as outside and is not in a period.

## Which kernel belongs to which node

With the `sushiruntime.device` source, each device span carries two attributes. `graph` is the label of the step's graph: `train.step`, `train.lead` for the part before a host stage, `eval.step`, `replica.step`, or `eager.step`. `origin` names the IR node the kernel was recorded for, as `<node id>:<operation>`; the optimizer, the gradient exchange and the carried state have the origin `step.trailing`. In an eager step the origin starts with the region, as `r3/12:matmul`, because each region numbers its nodes from 0.

The `roofline` analysis groups spans by both, so two steps with equal node numbers stay apart. The device idle analysis names the innermost zone open on the submitting thread over each gap, which is how a step that waits on its host stage shows up.

## Reading a record

```bash
sa profile record show build/profiles/mlp/run.sprec.json --analysis breakdown --top 10
sa profile record compare before.sprec.json after.sprec.json
```

`sa profile <args>` runs SushiProfiler's `sp <args>` from the project root. When `sp` is not on the path the command says so and where to install it.

## What the build decides

| Setting | Default | Effect |
| --- | --- | --- |
| `SA_ENABLE_PROFILER` | `ON` | `OFF` links no SushiProfiler, expands every zone to nothing, and makes `--profile` refuse with a message that names the option |
| `SA_PROFILE_LEVEL` | `2` | The compile-time ceiling. `0` removes every zone; `1` keeps the stage zones; `2` also keeps fine zones. The `level` key of a configuration can lower it at run time and never raise it |

Both are written into a generated header, so a change of level rebuilds the sources that expand a zone and nothing else.

## Not covered

- The distributed path has the same zones as the single-device one, but a worker runs no session, so only the master's record holds them. SushiRuntime times the distributed layer.
- `sa profile` forwards to `sp` and has no command of its own.
