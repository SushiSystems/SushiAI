# yolov3_tiny: import, train and run YOLOv3-tiny from its Darknet files

Two JSON files for `sa train` and `sa inference` over a YOLOv3-tiny model that
`sa import darknet` writes. There is no C++ in this directory. The model file and the weight
file are not committed: they come from the import step into `build/yolov3_tiny`, and both
files here point at them.

No run has been made with `train.json`; its values are the import's recipe with a data block
added. `eval.json` was written from the same data block.

## What has to be on disk

| File | Where it comes from | In this directory |
| --- | --- | --- |
| `train.json` | Written by hand from the import's own `train.json` | Yes |
| `eval.json` | Written by hand | Yes |
| `build/yolov3_tiny/model.json` | `sa import darknet` | No |
| `build/yolov3_tiny/weights.sushi` | The same import step | No |
| `yolov3-tiny.cfg`, `yolov3-tiny.weights` | The Darknet project | No |
| A YOLO-layout dataset | `D:/Projects/.sushisystems/datasets/coco128` here | No |

## Import

```bash
sa import darknet yolov3-tiny.cfg yolov3-tiny.weights --output build/yolov3_tiny
```

The import writes `model.json` (43 rows, two outputs, a `task` section of type
`anchor_detection`), `weights.sushi` and a `train.json` whose `data` is `null`. It refuses to
replace a file already in the directory unless `--force` is given. Keep that `train.json`
where it is: the one in this directory is the same file with the `data` block filled and the
paths changed.

## Where the paths point

| Key | Value | Resolves to |
| --- | --- | --- |
| `model` | `../../../build/yolov3_tiny/model.json` | Against the directory of `train.json`, the repository's `build/yolov3_tiny` |
| `weights` | `../../../build/yolov3_tiny/weights.sushi` | The same folder |
| `checkpoint` | `../../../build/yolov3_tiny/trained.sushi` | The same folder; the run writes the file there |
| `data.root` | `D:/Projects/.sushisystems/datasets/coco128` | Absolute; the `data` block is passed through as written and is not resolved against the file |

Edit `data.root` to where the dataset lies on your machine. The dataset follows the YOLO
layout: `images/train2017/*.jpg` with one `labels/train2017/*.txt` per image, each row
`class cx cy w h` in fractions of the image.

## The data

| Key | Value | Meaning |
| --- | --- | --- |
| `reader` | `yolo_txt` | The on-disk layout |
| `task` | `detection` | The roles a batch carries: `boxes`, `classes` and `valid` |
| `split` | `train2017` | The image list `root/train2017.txt`, or the folder `root/images/train2017` |
| `batch_size` | `16` | Images per batch; images past the last whole batch are not read |
| `input` | `[416, 416]` | Height and width every image is fitted to |
| `max_objects` | `128` | Object slots per image |
| `classes` | `80` | Classes of the labels |

`input` with 3 channels has to equal `input.shape` of `model.json`, and `classes` has to equal
the `classes` of its `task` section. `eval.json` carries the same block, so the evaluation
split is the training split: it measures fit, not generalisation.

## Training

The `objective` and `schedule` are the values the import took from the `.cfg`. Both are
written for Darknet's `max_batches`, so the schedule steps at iterations 400000 and 450000,
which 10 epochs of coco128 at batch 16 (80 updates) never reach; the warm-up of 1000 updates does not finish either, so the whole run trains under the warm-up ramp.

```bash
sa train applications/examples/yolov3_tiny/train.json
```

## Inference

```bash
sa inference build/yolov3_tiny/model.json build/yolov3_tiny/trained.sushi \
    applications/examples/yolov3_tiny/eval.json
```

The report prints one line per detection (class, score, box as `x1 y1 x2 y2` in pixels of the
network input) at a score threshold of 0.25, and the mean average precision at IoU 0.5 over
the split. To see the detections of the imported weights before any training, pass
`build/yolov3_tiny/weights.sushi` in place of the checkpoint.

This file quotes no cost, precision or timing, because none has been measured.
