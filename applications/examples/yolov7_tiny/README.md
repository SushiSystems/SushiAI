# yolov7_tiny: training a detector from a model file, a weight file and a folder of images

`train.json` describes a run of `sushiai train` over a YOLOv7-tiny model: the anchor objective,
a YOLO-layout dataset read from disk, SGD under a step schedule, and the weights the run starts
from. There is no C++ in this directory. `applications/programs/source/train.cpp` reads the
file, as it does for [`mnist_cnn`](../mnist_cnn/README.md).

No run has been made with this file, and the directory does not hold everything the run needs.
The values in `train.json` are untuned first guesses.

## What has to be on disk

| File | Where it comes from | In this directory |
| --- | --- | --- |
| `train.json` | Written by hand | Yes |
| `model.json` | `sushiai import darknet yolov7-tiny.cfg yolov7-tiny.weights --output <directory>` | No |
| `weights.sushi` | The same import step | No |
| `datasets/coco128/` | A dataset in the YOLO layout, copied by the user to the repository root | No |

`sushiai import` is written and has not been built or run;
[`docs/design/MODEL_IMPORT/README.md`](../../../docs/design/MODEL_IMPORT/README.md) describes it as phase 4.
The import step writes `model.json` with a `task` section of type `anchor_detection`, which
states the classes, the anchors and the heads; the `anchor` objective reads that section, so a
model file without it is refused.

The step also writes a `train.json` of its own, and refuses to replace a file that is already
in its output directory. Import into another directory and copy the two files here:

```bash
sa import darknet yolov7-tiny.cfg yolov7-tiny.weights --output build/yolov7_tiny
cp build/yolov7_tiny/model.json build/yolov7_tiny/weights.sushi applications/examples/yolov7_tiny/
```

The written `train.json` differs from the one here in four ways. Its `data` is `null`, since
no `.cfg` names a dataset. Its `objective` states every training option of the heads. Its
`schedule` holds the `.cfg`'s rate, burn-in, steps and scales. Its `checkpoint` is
`trained.sushi`. The import prints a note for each key of the `.cfg` it could not place, such
as `batch`, `subdivisions` and `max_batches`.

`model` and `weights` are relative paths, and `modules/core/config/source/train_config.cpp`
resolves both against the directory of the configuration file.

## The data

The `data` block names a reader and a task instead of a `type`:

| Key | Value here | Meaning |
| --- | --- | --- |
| `reader` | `yolo_txt` | The on-disk layout |
| `task` | `detection` | The roles a batch carries: `boxes`, `classes` and `valid`. Images are fitted as the model file's `fit` says |
| `root` | `../../../datasets/coco128` | The directory of the dataset, against the directory of `train.json` |
| `split` | `train` | The image list `root/train.txt`, or the folder `root/images/train` |
| `batch_size` | `8` | Images per batch; the images past the last whole batch are not read |
| `input` | `[416, 416]` | Height and width every image is fitted to |
| `max_objects` | `128` | Object slots per image; an image with more objects stops the run |
| `classes` | `80` | Classes of the labels; a label at or above it is refused with its file and line |

`input` followed by 3 channels has to equal `input.shape` of `model.json`, and `classes` has to
equal the `classes` of its `task` section; the driver stops with an error when either differs.

A YOLO-layout folder holds the images and one text file per image:

```
datasets/coco128/
    images/train/000000000009.jpg
    labels/train/000000000009.txt
```

Each row of a label file is `class cx cy w h`, in fractions of the image. An image without a
label file has no object. The reader finds the label file by replacing the last `images` folder
on the path with `labels`. The images may be `.jpg`, `.jpeg` or `.png`.

`root` resolves against the directory of the configuration file, so
`../../../datasets/coco128` is `<repository>/datasets/coco128` wherever `sa` is called from.
Write an absolute `root` to keep the dataset elsewhere.
The repository ships no dataset and downloads none.

## The rate

`schedule` states the learning rate: a warm-up of 100 updates up to `base_rate`, which then
holds. A file with a `schedule` states its rate there only, so `optimizer` has no
`learning_rate`; the driver refuses a file that states both. `steps` and `scales` add the
iterations at which the rate is multiplied.

## Running it

```bash
sa build
sa train applications/examples/yolov7_tiny/train.json
```

The driver prints a header, then a line for the first epoch, every fifth epoch and the last
one. Each line carries the summed cost of the last update of the epoch, the share of the boxes
in it and the rate that update ran at. It ends with a `compiles:` line and a `wrote` line that
names `yolov7_tiny.sushi`, written beside `train.json`. That file starts a later run when
`weights` names it.

This file quotes no cost, precision or timing, because none has been measured.

## Limits

- No sample is augmented and no epoch is shuffled. Every image is decoded each time its batch
  is read.
- One set of training options covers every head; the keys of `objective` are those of
  `modules/vision/detect/include/SushiAI/vision/detect/anchor/head_training_section.hpp`, and a
  key left out takes its default.
- The run writes one checkpoint, after the last epoch. A run started from it restarts the
  schedule at its first update.
- `sushiai inference` prints the detections of a model whose file has an `anchor_detection`
  task, over a dataset split and with a fixed score threshold of 0.25. It has not been run.
- An unknown key in any block is an error.
