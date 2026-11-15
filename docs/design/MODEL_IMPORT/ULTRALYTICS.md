# Ultralytics

`sushiai import ultralytics` reads an Ultralytics checkpoint (`yolo26n.pt` first, then
`yolov5n.pt`, `yolov8n.pt` and `yolo11n.pt`) and writes the same three files
`sushiai import darknet` writes. It carries out the `.pt` part of phase 9 and pulls forward the
two readers of phase 5 that it needs. RW-182 built YOLO26n; RW-183 added YOLOv5n (and Ultralytics'
YOLOv5nu), YOLOv8n and YOLO11n.

## Rulings

These were made on 2026-11-15, under the owner's ruling of 2026-10-12 that every foreign format is
read by SushiAI's own readers and takes no new dependency.

| Subject | Ruling |
| --- | --- |
| How a `.pt` is read | Natively, through `ZipArchive`, `read_pickle` and `TorchArchive` in `io`. `tools/reference/export_yolo26_reference.py` stays an oracle that runs outside the product and is never a step of the import |
| Where the architecture comes from | The `yaml` dictionary the checkpoint pickles inside its model object. The import writes `model.json` from it, so no hand-written model file is needed |
| Who owns the name map | Nobody, because there is none. The vision blocks name their children as Ultralytics does (`cv1`, `m`, `one2one_cv3`), so a tensor's path below the layer list `model` is its record name: `model.2.m.0.cv1.conv.weight` is `2.m.0.cv1.conv.weight`. The module table owns the architecture only |
| How the import is checked | Three checks, listed under "Verification" below |

## What a `.pt` holds

An Ultralytics checkpoint is a PyTorch zip archive of stored entries. Its `<folder>/data.pkl` is a
protocol 2 pickle whose top value is a dictionary:

- `model` holds a pickled `DetectionModel` in half precision.
- `ema` holds the EMA copy, or None in a released file.
- `train_args` holds the training arguments.

A tensor is a call of `_rebuild_tensor_v2`. Its arguments are a persistent id that names the entry
`data/<key>` and its storage type, then the storage offset, the size and the stride. A parameter
wraps such a tensor in `_rebuild_parameter`.

A module object is a NEWOBJ of its class and a BUILD of its attribute dictionary. The importer
reads four of the attributes: `_parameters`, `_buffers`, `_non_persistent_buffers_set` and
`_modules`, plus `groups` of a convolution and `yaml` of the top object. The released
`yolo26n.pt` names 28 globals, and every one is an Ultralytics or `torch.nn.modules` class or a
PyTorch rebuilder.

## The bricks

| Brick | Where | Responsibility |
| --- | --- | --- |
| `ZipArchive` | `io` | The entries of a zip file held in memory, plain or zip64, as views. It reads stored entries only and refuses a compressed one by name |
| `read_pickle` | `io` | A pickle of protocol 2 to 5 into a tree of `PickleValue` nodes. It calls nothing, and admits a global only when the caller's list names it |
| `pickle_to_value` | `io` | The plain data of a tree into a `Config::Value` |
| `TorchArchive` | `io` | A PyTorch zip: the pickle tree, and each tensor read at its stride into row-major float32, from half, float32 or float64 |
| `copy_kernel_last`, `copy_depthwise_kernel_last` | `io` | A weight `[out, in, kh, kw]` to `[out, kh, kw, in]`, and `[C, 1, kh, kw]` to `[kh, kw, C]`. Both importers call them |
| `read_ultralytics_checkpoint` | `vision/importers` | The walk of the module tree: the `yaml` description and every floating tensor by path |
| Module table | `vision/importers` | One row per Ultralytics module type, each in its own unit: Conv, C3, C2f, C3k2, SPPF, C2PSA, nn.Upsample, Concat and Detect |
| `describe_ultralytics_model` | `vision/importers` | The backbone and the head, through the table, into a model document and its task, with the `TensorRules` that rename and drop tensors |
| `list_ultralytics_tensors` | `vision/importers` | The tensors as float32 weight records in SushiAI's layouts, named by the rules |
| `import_ultralytics_files` | `vision/importers` | The `ImportedModel`, and the `ultralytics` row of the importer registry |

`TorchArchive` admits PyTorch's own rebuilders and storages. The importer adds the module prefixes
`ultralytics.`, `models.common.`, `models.yolo.` (the YOLOv5 repository) and `torch.nn.modules.`,
whose classes become plain objects holding their BUILD
dictionary. Any other name is refused, and the message names it.

Neither importer includes a header of the other. Both share `ImportedModel`, its writer, the
importer registry, the model file writer and `copy_kernel_last`.

## The module table

A row has two parts: the Ultralytics module name, and the function that writes the SushiAI row
from the entry's arguments. The function follows Ultralytics' own model builder:

- A width is capped at the scale's channel cap, multiplied by the width factor and rounded up to
  a multiple of eight.
- A repeat count above 1 is multiplied by the depth factor and rounded half to even.
- A C3k2 takes C3k as its inner block at the scales m, l and x.
- `nc`, `anchors` and `None` in the arguments are resolved. A file without `scales` states its
  factors as `depth_multiple` and `width_multiple`.
- A Conv that states its padding keeps it.
- The checkpoint's module classes are read beside the `yaml`. The one class the translation
  reads is the activation after SPPF's first convolution: Ultralytics' current source builds it
  linear, and YOLO26n pickles it so, while `yolov5nu.pt`, `yolov8n.pt` and `yolo11n.pt` pickle a
  SiLU. The `sppf` row says so with `activate_narrowing`. With no checkpoint the translation
  builds what the source builds.

A module with no row is refused, and the message lists every such module of the description with
the entries that use it. The Detect row writes `reg_max`, `end2end` and the class branch, plain
before C3k2 and depthwise after it, and drops the fixed `dfl` ramp. A Detect with anchors, as
YOLOv5 writes it, becomes one 1x1 `conv2d` per level published as `head_<level>` and the
`anchor_detection` task, its anchors as fractions of the input side; the level's tensors
`<detect>.m.<level>.` are renamed to that convolution's row and the `anchors` buffer is dropped.

## Verification

1. **The model file.** The descriptions of `yolo26n.pt`, `yolov8n.pt` and `yolo11n.pt` translate
   to the text of `model.json` in `applications/examples/yolo26`, `yolov8n` and `yolo11n`
   (`tests/vision/importers/test_ultralytics_model.cpp`).
2. **The weights.** The exporter also writes `model.pt`, the same seeded model that `torch.save`
   wrote. Every tensor the import reads from it equals, bit for bit and in its layout, the record
   of `weights.sushi` (`tests/vision/importers/test_ultralytics_import/test_import.cpp`). The forward pass on
   those weights is already compared with Ultralytics' `forward.sacp` by the registration suite,
   so no parity helper moved out of `darknet_forward_cases`.
3. **The detections.** `tests/vision/importers/test_ultralytics_import/test_detections.cpp`, a `large` test,
   imports `yolo26n.pt`, `yolov8n.pt`, `yolo11n.pt` and `yolov5nu.pt` and runs them as
   `sushiai inference` does over the 16 coco128 images, resized to 640 by 640, that
   `tools/reference/export_ultralytics_detections.py` gave Ultralytics. On 2026-11-15 every image
   gave Ultralytics' count and classes, each score within 2048 u and each corner within 2048 u of
   the side, once our corners are clipped to the image as Ultralytics clips them (KI-22): 39, 45,
   47 and 40 detections. The test found that SPPF's first activation differs between checkpoints.
4. **The original YOLOv5n.** `yolov5n.pt` of YOLOv5 v7.0 imports into the anchor model; its
   detections have not been compared with the YOLOv5 repository's own.

The released `.pt` files are downloaded on the owner's machine and are never committed.

## Left out

- `train_args` is not translated, so `train.json` states no recipe and the import notes it.
- A compressed zip entry is refused. PyTorch writes stored entries, so a deflate reader waits for
  `.npz` (phase 9).
