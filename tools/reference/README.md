# tools/reference

`export_yolo26_reference.py` runs Ultralytics at commit
`ce1399439b93c490b648b55dedd0a6334f0199da` and writes what it computes to
`tests/fixtures/yolo26/`. Stages b and c of `docs/agent/2026_10_07_YOLO26/SPEC.md` compare
SushiAI against these files. Every number in them comes from a run; none is typed.

## Running

The exporter needs Python 3.11 with torch, numpy and the pinned Ultralytics. On the owner's
machine that is the conda environment `sushiai-yolo26-ref`.

```
conda run -n sushiai-yolo26-ref python tools/reference/export_yolo26_reference.py tests/fixtures/yolo26 --seed 0
```

| Argument | Meaning |
| --- | --- |
| `OUTPUT_DIR` | Where the five SACP files and `MANIFEST.json` go |
| `--seed N` | Seed of the weights, the BatchNorm draws and the input image; default 0 |
| `--bn-state perturbed` | Default. Draws every BatchNorm weight, bias, running mean and running variance from the seed |
| `--bn-state initial` | Keeps torch's initial BatchNorm state: weight 1, bias 0, mean 0, variance 1 |
| `--external-weights DIR` | Writes `weights.sacp` to `DIR`; the manifest marks it `external` |

One run builds the model, does one eval-mode forward pass, one train-mode forward pass with one
loss evaluation and one backward. It takes about 6 seconds on two CPU threads. The exporter
calls `torch.set_num_threads(2)`, turns `torch.use_deterministic_algorithms` on, sets
`YOLO_OFFLINE=1` and replaces the socket connect and lookup calls with ones that fail, so a
download attempt ends the run with exit code 1. Two runs with the same seed on the same machine
produce byte-identical files. Building `DetectionModel` runs one extra 256 px forward inside
Ultralytics to find the strides (`ultralytics/nn/tasks.py:517`).

The exporter refuses to put more than 64 MiB (67108864 bytes) into `OUTPUT_DIR` and asks for
`--external-weights`. Git ignores `tests/fixtures/yolo26/*.sacp` and tracks only `MANIFEST.json`,
so the ceiling bounds the working tree and not the repository. A seed 0 run writes 41918762
bytes. The exporter exits 1 with one line on stderr on any failure, 0 otherwise.

The torch-free parts (the SACP encoder, the layout rule, the manifest, the writer and the
arithmetic of the running-variance candidates) are tested by
`tools/tests/test_export_yolo26_reference.py`.

## What the files hold

| File | Contents |
| --- | --- |
| `weights.sacp` | Every floating state-dict entry under its Ultralytics name, as the model stood before either pass |
| `input.sacp` | `image`, `[1, 640, 640, 3]` NHWC, uniform draws in [0, 1); torch receives `image.permute(0, 3, 1, 2)` |
| `forward.sacp` | Eval mode, BatchNorm on the running statistics in `weights.sacp`: three feature maps, twelve raw head outputs, both decodes, the top-k detections, the rows suppression keeps at a lowered confidence |
| `loss.sacp` | Train mode, BatchNorm on batch statistics: ground truth, loss items per branch, the inputs and the outputs of label assignment per branch, total loss, three gradients, one BatchNorm's running statistics after the pass |
| `batchnorm.sacp` | Train mode: two BatchNorm passes, each with its input, parameters, running statistics before and after, output, and two candidate values of the updated running variance |
| `MANIFEST.json` | Versions, seed, per file SHA-256 and size, per tensor name, dtype, shape, source and min, max, mean |

### Layouts

| Tensor | Reference shape | Written shape | Rule |
| --- | --- | --- | --- |
| Convolution weight, groups 1 | `[C_out, C_in, kh, kw]` | `[C_out, kh, kw, C_in]` | `permute(0, 2, 3, 1)` |
| Depthwise weight, groups = channels | `[C, 1, kh, kw]` | `[kh, kw, C]` | `squeeze(1).permute(1, 2, 0)` |
| Convolution bias, BatchNorm weight, bias, running mean, running variance | `[C]` | `[C]` | Unchanged |
| Feature map, raw head output per level | `[B, C, H, W]` | `[B, H, W, C]` | `permute(0, 2, 3, 1)` |
| Decoded boxes and scores | `[B, 4 + nc, A]` | `[B, A, 4]` and `[B, A, nc]` | `permute(0, 2, 1)`, then split |
| Assignment inputs and outputs | `[B, A]`, `[B, A, 4]`, `[B, A, nc]` | Same | Unchanged |

The first three rows are the "Weight importer" block of the SPEC. A state-dict tensor of any
other kind is written unpermuted and listed under `unspecified_layout` in the manifest; YOLO26n
has none. `num_batches_tracked` is a rank-0 integer counter and is not written; the manifest
lists the 114 skipped names.

Ultralytics concatenates the levels on the anchor axis (`ultralytics/nn/modules/head.py:176`).
The exporter cuts a raw head output back into its levels, so anchor `h * W + w` of a level is
element `[b, h, w, :]`. Decoded tensors and assignment outputs keep the concatenated axis:
levels in order, offsets 0, 6400 and 8000 at 640 px, recorded under `levels` in the manifest.

A foreground mask is boolean in torch and SACP has no boolean dtype, so it is written as
`int32` zeros and ones. A gradient takes the layout of the parameter it belongs to.

### Assigner inputs

A forward hook on each branch's `TaskAlignedAssigner` captures the call's arguments and its
result. `<branch>.assignment.input.scores` is the first argument, `pred_scores.detach().sigmoid()`,
and `<branch>.assignment.input.boxes` the second, `pred_bboxes.detach() * stride_tensor`, in pixel
xyxy (`ultralytics/utils/loss.py:440-441`). `anchors.points` is the third. The ground truth the
assigner received is not written padded; `batch.batch_idx`, `batch.cls` and `batch.bboxes` hold
the rows it was padded from.

### Validation row counts

`forward.sacp` holds no output at the validator's confidence: with the seeded weights the largest
class score is `1.9e-4`, the reference validator keeps no row of either head, and SACP cannot hold
an empty tensor. The manifest records the run under `context.validation` of `forward.sacp` instead: the
arguments `ultralytics/models/yolo/detect/val.py:168-176` passes to `non_max_suppression`, which
suppression function it selects (`TorchNMS.nms`, or `torchvision.ops.nms` when torchvision is
imported), the rows kept per head and each head's largest score. The
confidence, 0.001, is typed from `ultralytics/engine/validator.py:145`, where the validator sets
it inside a call that needs a dataset.

### Suppression rows

`one2many.suppressed.detections`, `[300, 6]`, holds the rows `non_max_suppression` keeps of the
first image's one-to-many decode with the same arguments and `conf_thres` lowered to `1.8e-4`.
1542 of the seeded scores pass that confidence. `context.suppression` of `forward.sacp` records
the arguments, the candidate count and the row count, and the exporter fails when no score
passes. No two candidates overlap above the IoU threshold of 0.7, so the rows are the 300
highest-scoring candidates; candidates of equal score come in the order torch's `argsort` gave
them. Each row is x1, y1, x2, y2 in pixels, the score and the class index.

### Running variance factor

`batchnorm.sacp` answers one question: whether torch moves the running variance toward the
biased batch variance or toward that variance times n/(n-1), where n is the number of values a
channel's statistics are taken over, `N * H * W`. `loss.sacp` cannot answer it, because
`model.0.bn` has n = 102400 and the two forms differ by less than float32 resolves.

The file holds two cases. Every name starts with `batchnorm_factor.<case>.`.

| Case | Layer | Reference input | n |
| --- | --- | --- | --- |
| `unit` | One `torch.nn.BatchNorm2d(3)` passed through `initialize_weights` (`ultralytics/utils/torch_utils.py:674`), which sets eps 0.001 and momentum 0.03 | `[2, 3, 2, 2]`, standard normal draws | 8 |
| `model.9.cv1.bn` | That layer of YOLO26n, in the train-mode pass `loss.sacp` comes from | `[1, 128, 20, 20]` | 400 |

The unit layer's weight, bias and running statistics are drawn from the ranges of
`batch_norm_perturbation`, by a generator seeded with `--seed` that nothing else uses. A forward
pre-hook and a forward hook on the layer copy the tensors, so the model case costs no extra pass.

| Suffix | Holds | Layout |
| --- | --- | --- |
| `input`, `output` | What the layer received and returned | NHWC |
| `weight`, `bias` | The layer's parameters | `[C]` |
| `before.running_mean`, `before.running_var` | The running statistics when the pass began | `[C]` |
| `after.running_mean`, `after.running_var` | The running statistics torch left | `[C]` |
| `candidate.biased.running_var` | `before * (1 - momentum) + variance * momentum` | `[C]` |
| `candidate.unbiased.running_var` | `before * (1 - momentum) + variance * momentum * n / (n - 1)` | `[C]` |

`variance` is the biased variance of `input` over batch, height and width. The exporter computes
both candidates in float64 from the float32 tensors and rounds the result to float32.
`context.cases.<case>` of `batchnorm.sacp` records n, momentum, eps and, per candidate, its
factor, the largest absolute difference from `after.running_var` over the channels and whether
every channel is equal in float32. `matching_candidate` names the candidate whose difference is
at most `match_tolerance`, which is 4 float32 epsilons of the largest `after.running_var`. The
exporter fails unless exactly one candidate is that close.

The seed 0 run names `unbiased` in both cases. The differences it recorded:

| Case | `biased` | `unbiased` | `match_tolerance` |
| --- | --- | --- | --- |
| `unit` | 8.2741e-3 | 0, equal in float32 | 5.2864e-7 |
| `model.9.cv1.bn` | 1.3888e-5 | 1.1921e-7 | 6.9510e-7 |

## SACP layout as written

SACP is SushiAI's checkpoint container. The exporter writes version 1 and nothing else, with
every integer little-endian and no padding or alignment between fields.

| Field | Type | Value | Taken from |
| --- | --- | --- | --- |
| Magic | `u32` | `0x50434153`, the bytes `SACP` | `include/SushiAI/io/checkpoint_format.hpp:48`, written at `src/io/checkpoint_format.cpp:128` |
| Version | `u32` | 1 | `checkpoint_format.hpp:57`, `checkpoint_format.cpp:129` |
| Tensor count | `u32` | At most 4096 | `checkpoint_format.hpp:60`, `checkpoint_format.cpp:130` |

Then, per tensor, in order:

| Field | Type | Value | Taken from |
| --- | --- | --- | --- |
| Name length | `u32` | Bytes in the name, at most 256 | `checkpoint_format.hpp:63`, `checkpoint_format.cpp:148` |
| Name | bytes | UTF-8, no terminator | `checkpoint_format.cpp:149` |
| Dtype | `u32` | The `DataType` enumerator's value | `checkpoint_format.cpp:150` |
| Rank | `u32` | At most 6 | `checkpoint_format.cpp:151`; limit at `include/SushiAI/core/common.hpp:52` |
| Extents | `i64` per dimension | Each positive | `checkpoint_format.cpp:152-153`; checked at `:222` |
| Payload | bytes | Elements row-major, element count times element size | `checkpoint_format.cpp:155-156`, size at `checkpoint_format.hpp:88-91` |

`u32` is four bytes, low byte first (`checkpoint_format.cpp:64-70`); `i64` is eight bytes, low
byte first, two's complement (`checkpoint_format.cpp:73-80`).

Dtype codes follow the order of `SushiBLAS::Core::DataType`
(`D:\Projects\sushiblas\include\SushiBLAS\core\common.hpp:49-58`), with element sizes from
`include/SushiAI/core/common.hpp:63-69`: `half` 0, `float32` 1, `float64` 2, `complex32` 3,
`complex64` 4, `int32` 5, `int64` 6. The exporter writes `float32`, `int32` and `int64`.

Whether the C++ reader accepts a rank-0 record was not checked, so the one scalar,
`total_loss`, is written with shape `[1]`.

## Manifest format

`format_version` is 1. Keys are `snake_case`.

| Key | Holds |
| --- | --- |
| `ultralytics_commit`, `ultralytics_version`, `torch_version`, `python_version` | Read from the installed packages; the run fails if the commit is not the pinned one |
| `seed`, `thread_count`, `deterministic_algorithms` | The run's settings, read back from torch |
| `files[].name`, `sha256`, `size_bytes`, `location` | One entry per SACP file; `location` is `output_directory` or `external` |
| `files[].context` | The conditions of that file: mode, BatchNorm statistics used, level geometry, gains, assigner settings, box formats, validation row counts, suppression arguments and counts, running-variance candidates |
| `files[].tensors[]` | `name`, `dtype`, `shape`, `original_shape`, `layout`, `source`, `min`, `max`, `mean` |

`source` is the Ultralytics file, the first line of the function and its qualified name, found
with `inspect` at run time. `min`, `max` and `mean` are computed from the array whose bytes are
written; the mean is accumulated in float64.

| Version | Shipped in | Change |
| --- | --- | --- |
| 1 | Unreleased | First version |
