# Ultralytics

The plan of `sushiai import ultralytics`, which reads an Ultralytics checkpoint (`yolo26n.pt`
first, then `yolov5n.pt`, `yolov8n.pt` and `yolo11n.pt`) and writes the same three files
`sushiai import darknet` writes. It carries out the `.pt` part of phase 9 and pulls the two
readers of phase 5 it needs forward. RW-181 asked for it, RW-182 builds YOLO26n and RW-183 adds
the other three.

## Rulings

Made on 2026-11-15 under the owner's ruling of 2026-10-12, which says every foreign format is
read by readers of SushiAI's own and takes no new dependency.

| Subject | Ruling |
| --- | --- |
| How a `.pt` is read | Natively: `ZipArchive` and `PickleReader` in `io`, as "The readers" of the README states them. `tools/reference/export_yolo26_reference.py` stays an oracle that runs outside the product, never a step of the import |
| Where the architecture comes from | The `yaml` dictionary the checkpoint pickles inside its model object. The import writes `model.json` from it, so no hand-written model file is needed and none is trusted |
| Who owns the name map | The module table of the Ultralytics importer. Each row names an Ultralytics module type, the SushiAI row it becomes and the sub-paths of its tensors. No separate map file and no table of whole names |
| How the import is checked | Three checks, in "Verification" below: the written model file against `applications/examples/yolo26/model.json`, the forward pass against the fixture Ultralytics wrote, and the detections of `sushiai inference` on coco128 against Ultralytics' own run |

## What a `.pt` holds

An Ultralytics checkpoint is a PyTorch zip archive. Its `data.pkl` is a pickle whose top value is
a dictionary. Its `model` key holds a pickled `DetectionModel`, in half precision and with the
EMA weights, and `train_args` holds the training arguments. A tensor is a `persistent_id` that
names an entry `data/<key>` of the archive, plus its storage offset, shape and strides.

A module object is pickled as its class name and a `BUILD` of its attribute dictionary. The
importer reads three of the attributes: `_parameters`, `_buffers` and `_modules`. The walk of
`_modules` from the top object gives each tensor its path (`model.2.m.0.cv1.conv.weight`). The
`yaml` attribute of the top object gives the architecture.

## The bricks

| Brick | Where | Responsibility |
| --- | --- | --- |
| `ZipArchive` | `io` | The entries of a zip file, stored and deflated; offsets into the file, no copy |
| `PickleReader` | `io` | A pickle of protocol 2 to 5 into a tree of values. It runs no code |
| `TorchArchive` | `io` | The entries of a PyTorch zip: the pickle tree with each tensor as a view into its storage entry. Shared with `sushiai import weights` |
| `UltralyticsCheckpoint` | `vision/importers` | The module tree of the `model` key: the `yaml` dictionary and the tensors, by path |
| `ultralytics_module_table` | `vision/importers` | One row per Ultralytics module type, each in its own unit (`ultralytics_module_conv.cpp`, `..._c3k2.cpp`, `..._sppf.cpp`, `..._c2psa.cpp`, `..._upsample.cpp`, `..._concat.cpp`, `..._detect.cpp`) |
| `translate_ultralytics_model` | `vision/importers` | The `yaml` rows into a `Config::ModelSpec` and the `anchor_free_detection` task section, through the table |
| `store_ultralytics_weights` | `vision/importers` | The tensors into weight records: the record name from the table, the layout from the record, half precision widened to `float32` |
| `import_ultralytics` | `vision/importers` | The `ImportedModel`, and the `ultralytics` row of the importer registry |

`PickleReader` accepts the callables of the README's table and, for this importer, also each
`ultralytics.nn.*` and `torch.nn.modules.*` class name. Those become plain objects holding their
`BUILD` dictionary, since nothing of them runs. Any other name is refused, and the message names
it. The list sits in the reader's accepted-name table, so a new family adds rows, not code.

### Shared with the Darknet importer, written once

| Part | Home | State |
| --- | --- | --- |
| `ImportedModel`, its writer, the importer registry | `io` | Exists; the Darknet importer uses it |
| The model file writer and the task section writer | `config`, `vision/detect` | Exists (phases 1 and 2) |
| Layout permutation of a weight into its record | `io` | Moves out of `darknet_weight_layout.hpp`; Darknet and Ultralytics both permute `OIHW` and depthwise weights |
| The loading of a reference tensor and the measured comparison | `vision/import_parity`, a test-support module | Moves out of `darknet_forward_cases`; that module keeps only what reads a `.cfg` |

Neither importer includes a header of the other.

## The module table

A row has three parts: the Ultralytics class name, the function that writes the SushiAI row from
the `yaml` arguments, and the pairs of sub-path and record name, such as `conv.weight` with the
convolution weight and `bn.running_mean` with the running mean. A composite row (`C3k2`, `C2PSA`,
`Detect`) lists its inner modules by index, so `m.0.cv1` maps through the bottleneck row and not
through a whole name. A module type with no row is refused, and the message lists every such type
of the file with the indices that use it.

YOLO26n needs seven rows. YOLOv8n and YOLO11n add `C2f` and `C3k`, and the `Detect` row with the
distribution focal head. YOLOv5n needs `C3` and the anchor-based `Detect`, which turns into the
`anchor_detection` task section.

## Verification

1. **The model file.** `sushiai import ultralytics yolo26n.pt` writes a `model.json` that parses
   to the same `ModelSpec` as `applications/examples/yolo26/model.json`. This is a `small` test on
   a `.pt` fixture of a few kilobytes, written by `tools/reference/` from `yolo26n.yaml` at a
   width that keeps it small.
2. **The forward pass.** The weights imported from that fixture give the six feature maps and the
   twelve head outputs of `forward.sacp`. The bound is measured the way the Darknet parity suites
   measure theirs: the worst error over the tolerance helper's bound, printed as `MEASURED` and
   fixed from the first run.
3. **The detections.** On the owner's machine, `sushiai inference` on coco128 with the imported
   `yolo26n.pt` is compared with Ultralytics' own predictions at the same thresholds, dumped by a
   reference script outside the product. Each reference box is matched by IoU, and the share
   matched and the worst score difference are measured and then bounded. This is a `large` test.

The real `.pt` files are downloaded on the owner's machine and are never committed.

## Order of work

| Entry | Holds |
| --- | --- |
| RW-182 | `ZipArchive`, `PickleReader`, `TorchArchive`; the shared layout and parity moves; the four YOLO26n bricks and seven table rows; checks 1 to 3 |
| RW-183 | The rows for YOLOv5n, YOLOv8n and YOLO11n, a fixture and checks 1 to 3 for each |

The importer writes no `train.json` values it cannot read from `train_args`. A value with no key
is a note, as in phase 4.
