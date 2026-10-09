# Darknet-era YOLO: plan

**Status:** Draft, 2026-10-09, waiting on the owner's answers below. Nothing is written or built. Requirements, measurements and formats are in `SPEC.md` beside this file; section numbers refer to it.

## 1. Rulings and questions

Ruled on 2026-10-09: training is in scope, with a forward-parity part before a training part in every model (Q2); the published weights may be downloaded.

The plan is written under the recommendation of each open question. An answer that differs changes the units in the last column.

| # | Question | Recommendation | Alternatives | Changes |
| --- | --- | --- | --- | --- |
| Q1 | Where the anchor unit lives | Its own module `vision/detect_anchor`, namespace `SushiAI::Vision::DetectAnchor`, depending on `vision/detect` for ground truth, IoU and NMS. `detect` already has `anchors.hpp` and `anchor_leaves.hpp` for anchor points (SPEC 2.4); a module of its own keeps anchor boxes apart from them | (a) Files inside `vision/detect`: `DOMAIN_MODULES.md` gives one module per task and both are detection; the cost is two meanings of "anchor" in one folder. (b) The name `vision/detect_grid`, which also describes YOLOv1, a model with cells and no anchors | D02, D27, T02, T03, skeleton |
| Q3 | Order of the models | v3-tiny, v3, v4-tiny, v4, v2, v1 | v2 before v4 | Milestone order |
| Q4 | How Darknet is obtained and run | Option C of the table below | A, B | D05, T06 |
| Q5 | What the importer produces | A `Config::ModelSpec` that `NN::build_model` turns into a graph (SPEC 2.3). The importer needs no engine, and a Darknet model can also be written as a model file. It needs rows in `nn` for what Darknet uses | A `ModelGraph` built with `ModelGraph::add`, no new rows; the importer then owns node construction and its checks | D06, D07, D14, D19, D23, D29 |
| Q6 | How much of YOLOv1 belongs to 0.1.0 | Leave `yolov1.cfg` for after the tag and say so in `PROGRAMME.md`: it needs a `local` operation in the core tier, dropout in training, a loss of its own and the largest weights file. If v1 must be in, take `yolov1-tiny.cfg`, which has no `local` **(confirm)** | (a) Full v1, after the YOLO26 waves. (b) v1 out of the ruling | M6 |
| Q7 | Public shape of the trailing padding on `MaxPool2d` | Two fields, `trailing_pad_h` and `trailing_pad_w`; row key `trailing_padding`, an integer or `[h, w]`. `ConvGeometry` is not touched | (a) `pad_bottom`, `pad_right` in `ConvGeometry`: it edits `builder.hpp` and every emitter, so it waits for YOLO26. (b) A boolean `keep_extent`, which covers this one pool only | D01, D06 |
| Q8 | Where the downloads live, and what a test does without them | A directory outside the repository named by `SA_REFERENCE_DATA_DIR`, with `darknet/<file>` below it; the tool downloads into it and checks the SHA-256 the manifest pins. A parity test is reported skipped when it is absent. `testing` rule 4 forbids a test that depends on a file outside the repository, so this needs the owner's exception | `yolov3-tiny.weights` (35 MB) git-ignored under `tests/fixtures/darknet/yolov3_tiny/`, as the YOLO26 SACP files are; the larger ones still need the directory | D05, parity units |
| Q9 | Which Darknet is the reference | AlexeyAB's repository at one pinned commit: its batch-norm epsilon is the form `BatchNorm2d` has (SPEC 4.3), and it runs v4 | pjreddie's for v1 to v3; its epsilon form needs a second fold in the core tier | D05, D07 |
| Q10 | How distance-IoU suppression enters `vision/detect` | A free function beside `suppress_overlaps` and a `SuppressionCriterion` field in `NmsConfig` | Suppress per class in the caller and leave `detect` alone | D18 |
| Q11 | May `vision/importers` depend on `vision/detect_anchor` | Yes: a translated model carries each head's options, so there is one options type | The importer returns the raw keys and a program builds the options | D07, T01 |
| Q12 | What is shared with YOLO26's detection objective, and where the anchor objective lives | The loss emitter and the assigner in `vision/detect_anchor` ("loss terms" belong to the task module, `DOMAIN_MODULES.md:72`). The `Train::Objective` implementation beside YOLO26's, wherever stage 5 puts it (`DOMAIN_MODULES.md:123-125`). YOLO26's CIoU emitter exposed as a function of `vision/detect` so v4 calls it | A second CIoU graph here; an objective inside `detect_anchor`, which then depends on `train` | T03, T04, T09 |
| Q13 | The class term of `region` (v2) | Ask after YOLO26 lands: softmax cross-entropy as a term of a sum needs a rule the core tier refuses today (SPEC 3) | Train v2 with a per-class logistic term and record that it departs from Darknet | T14 |
| Q14 | How far training goes for 0.1.0 | Per model: target parity, gradient parity and overfitting one batch (SPEC 5.3). No full COCO run and no reproduction of Darknet's augmentation | A full run of YOLOv3-tiny to a stated mAP, which needs Darknet's loader semantics and days of the 1060 | T07 and the later parity units |

Q4, with the weights downloaded in every option:

| Option | What it is | Cost |
| --- | --- | --- |
| A | A prebuilt `darknet` binary the owner supplies; the tool reads `detector test -ext_output -out result.json` | Nothing to maintain. Detections only, at about six printed digits; no head tensor and nothing for training |
| B | AlexeyAB's repository built outside this repository as a library and driven through its Python binding | No patch. Reading a layer's output or delta through `ctypes` depends on the layout of Darknet's `network` and `layer` structs, which the binding does not declare in full **(confirm)** |
| C | The same build, CPU only, without OpenCV, with one patch kept in `tools/reference/darknet/` that writes the network input, chosen layer outputs and, in training mode, each head's delta and chosen weight gradients as raw `float32`; `export_darknet_reference.py` turns them into SACP and the manifest | One patch of about 150 lines against one pinned commit, listed in `NOTICE.md`. Every tensor of SPEC 5.1 and 5.3 comes from Darknet's own arithmetic |

## 2. Modules

| Module | Tier | Holds | Depends on |
| --- | --- | --- | --- |
| `vision/detect_anchor` (new) | vision | Options, shape arithmetic, decode; from training on, the assigner and the loss emitter | PUBLIC `vision/detect`, and `graph` from T03; PRIVATE `core`; EXTERNAL_PUBLIC `SushiRuntime::SushiRuntime` |
| `vision/importers` (new) | vision | The `.cfg` reader, the translation to a `ModelSpec`, the `.weights` reader | PUBLIC `core config nn vision/detect_anchor` |
| `layers/nn` (grown) | layers | Trailing padding on `MaxPool2d`; `Add`; `ChannelSlice`; `Mish`; `SpaceToDepth`; their rows | unchanged |
| `vision/detect` (grown, M4) | vision | Distance-IoU suppression | unchanged |
| `vision/imaging` (grown, M2) | vision | The inverse of a letterbox for boxes | unchanged |

## 3. Orchestrator steps

Workers do not write build files (dispatch rule 3).

**O1, the skeleton, before wave 1.** It edits `vision.h`, which the YOLO26 workflow must also edit (SPEC 6), so it lands after YOLO26's umbrella lines or in the same hands.

- `cmake/SushiAILayers.cmake`: rows `vision/detect_anchor  vision` and `vision/importers  vision` in `SUSHIAI_MODULE_TIERS`; the same two rows in `tools/layering/check_layering.py`.
- `modules/vision/CMakeLists.txt`: `add_subdirectory(detect_anchor)` after `detect`, `add_subdirectory(importers)` after `blocks`.
- `modules/vision/detect_anchor/CMakeLists.txt`, `modules/vision/importers/CMakeLists.txt`: one `sushiai_add_module(NAME ... TIER vision SOURCES ... TESTS ...)` each, with the dependencies of section 2.
- `modules/vision/umbrella/CMakeLists.txt`: both keys added to `PUBLIC_DEPENDS`. `modules/vision/umbrella/include/SushiAI/vision/vision.h`: one include per new public header as each wave is integrated; the configure fails on a header left out.
- A README stub in each new module, replaced by the worker that fills it.

**After each wave.** Sources and tests appended to the `CMakeLists.txt` of `layers/nn`, `vision/detect`, `vision/imaging` and the two new modules; integration tests appended to the vision list of `tests/CMakeLists.txt` (near line 208; the variable was not read); the `.gitignore` line `tests/fixtures/darknet/*/*.sacp`; `NOTICE.md` for the Darknet patch. **At close.** The changelog lines, `PROGRAMME.md` stage 6, `REMAINING_WORK.md`, `docs/design/DARKNET_YOLO.md` drawn from `SPEC.md`, and the manual pages the work makes false.

## 4. Units

"Now" means the unit edits no file the YOLO26 workflow edits and needs nothing YOLO26 has yet to land. Paths are relative to the repository root; `nn/` stands for `modules/layers/nn/`, `da/` for `modules/vision/detect_anchor/`, `im/` for `modules/vision/importers/`, and `inc/` inside each for its `include/SushiAI/...` folder. Dispatch rows: `haiku medium` (spec-complete, 1 or 2 files), `haiku high` (spec-complete, more files), `opus low` (integration or judgment).

### M1: YOLOv3-tiny forward pass

**D01. `MaxPool2d` trailing padding.** `layers/nn`. Now. `haiku high`.
- Files: `nn/inc/max_pool2d.hpp`, `nn/source/max_pool2d.cpp`, `nn/tests/test_max_pool2d.cpp`, `tests/integration/test_max_pool_trailing.cpp` (new).
- Surface, two fields of `struct MaxPool2d`: `int64_t trailing_pad_h = 0;` and `int64_t trailing_pad_w = 0;`, the padding rows below and columns right beyond the symmetric pad. `output_shape` uses `Graph::output_extent(H, pad_top, pad_top + trailing_pad_h, dilation_h, kernel_h, stride_h)`; `forward` calls `GraphBuilder::maxpool2d_with_extents`. A negative value and an all-padding window throw as an illegal window does today.
- Tests: 13x13, 2x2, stride 1, trailing 1 gives 13x13 and trailing 0 gives 12x12; 416 at stride 2 gives 208 either way; on an engine, an all-negative input keeps the true maximum in the last row and column, which pins that a padded tap never wins.
- Acceptance: the last row and column of a 2x2 stride-1 pool over a known 3x3 input equal the maxima of the truncated windows.

**D02. The anchor head decode.** `vision/detect_anchor`. Now. `haiku high`.
- Files: `da/inc/anchor_head_options.hpp`, `da/inc/anchor_head_shape.hpp`, `da/inc/anchor_head_decode.hpp`, the three `da/source/*.cpp` of the same names, `da/tests/test_anchor_head_decode.cpp`, `da/README.md`.

```cpp
struct AnchorSize { float width = 0.0f; float height = 0.0f; };
enum class AnchorUnits : std::uint8_t { NetworkPixels, GridCells };
enum class ClassActivation : std::uint8_t { Sigmoid, Softmax };
enum class ScoreGate : std::uint8_t { Objectness, Score };

struct AnchorHeadOptions
{
    std::vector<AnchorSize> anchors;   // the anchors of this head, mask applied
    std::int64_t num_classes = 0;
    AnchorUnits anchor_units = AnchorUnits::NetworkPixels;
    ClassActivation class_activation = ClassActivation::Sigmoid;
    ScoreGate score_gate = ScoreGate::Objectness;
    float scale_xy = 1.0f;
};
[[nodiscard]] Status validate_anchor_head(const AnchorHeadOptions& options);

struct NetworkSize { std::int64_t height = 0; std::int64_t width = 0; };
struct AnchorHeadShape { std::int64_t batch = 0; std::int64_t height = 0; std::int64_t width = 0; };
[[nodiscard]] std::int64_t count_head_channels(const AnchorHeadOptions& options) noexcept;
[[nodiscard]] std::int64_t count_head_rows(const AnchorHeadShape& shape, const AnchorHeadOptions& options) noexcept;

void decode_anchor_head(SushiRuntime::span<const float> raw, const AnchorHeadShape& shape,
                        const AnchorHeadOptions& options, NetworkSize network, float confidence_threshold,
                        SushiRuntime::span<float> boxes, SushiRuntime::span<float> scores);
```

- `raw` is `[B, H, W, A * (5 + C)]`; `boxes` receives `[B, H * W * A, 4]` centre and size in network pixels and `scores` `[B, H * W * A, C]`, row `(h * W + w) * A + a`, by SPEC 4.4. Both are what `Detect::select_nms_detections` reads; a caller joins the heads on the row axis. The `yolo` form is the defaults, the `region` form `GridCells`, `Softmax`, `Score`. Shaped as `vision/detect` is: free functions on spans, a status only where values come from a file.
- Tests: zero logits on a 1x1 grid give the cell centre, the anchor size and scores of 0.25; `scale_xy` 1.2 moves a saturated offset `(s - 1) / 2` past the cell edge; `GridCells` divides by the grid; `Softmax` rows sum to the objectness; each gate zeroes what SPEC 4.4 says; the row order on a 2x3 grid with two anchors; the refusals of `validate_anchor_head`.
- Acceptance: every formula of the `yolo` and `region` decodes is pinned by a case whose expected value follows from logits of zero or of large magnitude, to 1e-6 absolute.

**D03. The `.cfg` reader.** `vision/importers`. Now. `haiku high`.
- Files: `im/inc/darknet_config.hpp`, `im/source/darknet_config.cpp`, `im/tests/test_darknet_config.cpp`.

```cpp
struct DarknetOption { std::string key; std::string value; std::int64_t line = 0; };
struct DarknetSection { std::string type; std::int64_t line = 0; std::vector<DarknetOption> options; };
struct DarknetConfig { DarknetSection net; std::vector<DarknetSection> layers; };  // index = Darknet layer index
[[nodiscard]] Result<DarknetConfig> parse_darknet_config(std::string_view text, std::string_view origin);
[[nodiscard]] Result<DarknetConfig> load_darknet_config(const std::filesystem::path& path);
```

- The grammar of SPEC 4.1 and nothing about what a section means.
- Tests: both comment forms, blank lines, spaces around `=` and in lists, CRLF; a key before any section, a line without `=`, a repeated key and a first section other than `[net]` each fail with `InvalidDescription` naming origin and line; a missing file gives `FileNotFound`.
- Acceptance: a text with every lexical form of SPEC 4.1 parses to the sections, keys and lines the test states.

**D04. The `.weights` reader.** `vision/importers`. Now. `haiku high`.
- Files: `im/inc/darknet_weight_slot.hpp`, `im/inc/darknet_weights.hpp`, `im/source/darknet_weights.cpp`, `im/tests/test_darknet_weights.cpp`, `tests/integration/test_darknet_weights_state.cpp`.

```cpp
enum class DarknetTensorLayout : std::uint8_t { Vector, ConvolutionWeight, ConnectedWeight, ConnectedAfterMap };
struct DarknetWeightSlot { std::string state_name; DarknetTensorLayout layout = DarknetTensorLayout::Vector; Shape source_shape; std::int64_t layer = 0; };
struct DarknetWeightsHeader { std::int32_t major = 0; std::int32_t minor = 0; std::int32_t revision = 0; std::uint64_t seen = 0; };

[[nodiscard]] Result<DarknetWeightsHeader> read_darknet_header(std::istream& stream);
void convert_darknet_tensor(SushiRuntime::span<const float> source, const DarknetWeightSlot& slot, SushiRuntime::span<float> converted);
[[nodiscard]] Result<DarknetWeightsHeader> load_darknet_weights(const std::filesystem::path& path,
    SushiRuntime::span<const DarknetWeightSlot> slots, NN::ModelState& state);
```

- Layouts: `Vector` `[n]` and `ConnectedWeight` `[out, in]` are copied; `ConvolutionWeight` `[out, in, kh, kw]` becomes `[out, kh, kw, in]`; `ConnectedAfterMap` `[out, C * H * W]` becomes `[out, H * W * C]`. A slot names the `ModelState` entry that receives the tensor, the shape as Darknet stores it (`[out, C, H, W]` for `ConnectedAfterMap`) and the Darknet layer, for diagnostics. The slots are in file order; the reader does not know what a layer is. Strict both ways (SPEC 4.2). Bytes are assembled field by field.
- Tests: both header forms; a version above 1000 refused; each layout on a tensor whose elements are their own indices; a file the test writes into the build tree fills a two-layer state; one float short, one float over and a slot of another shape each give `Incompatible` naming the layer.
- Acceptance: a weight written as `[out, in, kh, kw]` index values arrives in the state as `[out, kh, kw, in]` with every element at the permuted position.

**D05. The reference tool, forward.** `tools/reference`. Now. `opus low`.
- Files: `tools/reference/export_darknet_reference.py`, `tools/reference/darknet/dump_layers.patch`, `tools/reference/darknet/README.md`, `tools/tests/test_export_darknet_reference.py`, the Darknet section of `tools/reference/README.md`.
- Surface: `export_darknet_reference.py OUTPUT_DIR --model NAME --darknet-dir DIR --data-dir DIR --image FILE [--thresh 0.25] [--nms 0.45] [--layers I,J]`; it downloads the cfg and the weights into `--data-dir` when absent and refuses a file whose SHA-256 differs from a pinned one. Exit 1 with one line on stderr on failure. It imports the SACP encoder and the manifest writer of the YOLO26 exporter; if that needs a shared module under `tools/reference/`, the worker reports it and stops, since it moves code the YOLO26 exporter owns.
- The worker reads Darknet's source at the pinned commit and replaces every **(confirm)** of SPEC 4 the forward depends on with the file and line it found. It runs nothing from Darknet; the owner builds and runs it.
- Tests, Darknet-free: the NCHW to NHWC rule, the decoded-row order, the manifest of a synthetic run, the refusal above the ceiling, the hash check.
- Acceptance: on a synthetic directory of raw dumps the tool writes `input.sacp`, `forward.sacp` and a manifest whose hashes and shapes the test recomputes.

**D06. Row key `trailing_padding`.** `layers/nn`. Now. Waits on D01. `haiku medium`.
- Files: `nn/source/layer_registry.cpp`, `nn/tests/test_node_registry.cpp`. `maxpool2d` accepts the key, an integer or `[h, w]`, default 0; `avgpool2d` refuses it. Acceptance: `{"type": "maxpool2d", "kernel": 2, "stride": 1, "trailing_padding": 1}` over `[13, 13, 256]` builds and reports `[13, 13, 256]`.

**D07. The translation.** `vision/importers`. Now. Waits on D02, D03, D04, D06. `opus low`.
- Files: `im/inc/darknet_model.hpp`, `im/source/darknet_model.cpp`, `im/source/darknet_layer_table.{hpp,cpp}`, `im/source/darknet_layer_{convolutional,maxpool,upsample,route,yolo}.cpp`, `im/tests/test_darknet_model.cpp`, `im/README.md`.

```cpp
enum class DarknetSuppression : std::uint8_t { Greedy, DistanceIou };
struct DarknetHead { std::string output; std::int64_t layer = 0; DetectAnchor::AnchorHeadOptions options; DarknetSuppression suppression = DarknetSuppression::Greedy; float suppression_beta = 1.0f; };
struct DarknetModel { Config::ModelSpec spec; DetectAnchor::NetworkSize network; std::vector<DarknetHead> heads; std::vector<DarknetWeightSlot> slots; std::vector<std::int64_t> row_of_layer; };
[[nodiscard]] Result<DarknetModel> translate_darknet_model(const DarknetConfig& config);
```

- `DarknetHead::output` is the published output's name, `"head.<k>"`, and `layer` the index of its `yolo` or `region` section. `spec` holds rows of `nn` types and publishes the heads; `heads` is in file order, `slots` in `.weights` order, and `row_of_layer[i]` is the row that holds Darknet layer `i`'s output.
- One private translator per section type, found through a table of `{type, function}`: a new section type is a new source file and one row. Unknown types and unknown keys are refused by SPEC 4.1. A `[convolutional]` becomes `conv2d`, then `batchnorm2d` with `bn_epsilon` 1e-5 when normalised, then the activation row unless it is `linear`.
- Tests: the v3-tiny cfg written out in the test (input data, not a reference) gives the row count, the `from` of each `route`, two outputs and their options, each derived by SPEC 4.1; `build_model` over `register_layers` accepts the spec and reports heads `[13, 13, 255]` and `[26, 26, 255]`; every slot name is an entry of a `ModelState` created from the traced model; the slots' float count is the count the cfg's shapes give; an unknown section and an unknown key fail naming the line.
- Acceptance: the translated v3-tiny spec builds, and every slot name is an entry of its state.

**D08. YOLOv3-tiny forward parity.** Root tests. Now to write; runs with the owner's fixture. Waits on D05, D07. `opus low`.
- Files: `tests/integration/test_darknet_yolov3_tiny.cpp`, `tests/fixtures/darknet/README.md`.
- Tests against `forward.sacp`, each tolerance stated: layers 11 and 12; both raw heads; decoded boxes and scores per head; the detections after `select_nms_detections` at Darknet's thresholds, row for row. Skipped without the fixture or the weights.
- Acceptance: the library's detections for the fixture image equal Darknet's.

### M1T: YOLOv3-tiny training

**T02. The anchor assigner.** `vision/detect_anchor`. Now: host code. `opus low`.
- Files: `da/inc/anchor_loss_options.hpp`, `da/inc/anchor_targets.hpp`, `da/inc/anchor_assigner.hpp`, `da/source/anchor_loss_options.cpp`, `da/source/anchor_assigner.cpp`, `da/tests/test_anchor_assigner.cpp`.

```cpp
enum class BoxLoss : std::uint8_t { SquaredError, CompleteIou };
enum class ObjectnessTarget : std::uint8_t { One, Iou };

struct AnchorLossOptions
{
    std::vector<AnchorSize> all_anchors;    // every anchor of the model, for the best-anchor match
    std::vector<std::int64_t> mask;         // the anchors of all_anchors this head owns
    float ignore_threshold = 0.7f;
    float match_threshold = 1.0f;           // iou_thresh; 1 assigns the best anchor only
    BoxLoss box_loss = BoxLoss::SquaredError;
    ObjectnessTarget objectness_target = ObjectnessTarget::One;
    float box_weight = 1.0f; float object_weight = 1.0f; float no_object_weight = 1.0f; float class_weight = 1.0f;
    std::int64_t prior_pull_images = 0;     // region: images during which boxes are pulled to their anchor
};
[[nodiscard]] Status validate_anchor_loss(const AnchorLossOptions& loss, const AnchorHeadOptions& head);

struct AnchorTargets { std::vector<float> target; std::vector<float> logistic_weight; std::vector<float> square_weight; };

void assign_anchor_targets(SushiRuntime::span<const float> decoded_boxes, const Detect::GroundTruthBatch& truth,
                           std::uint64_t images_seen, const AnchorHeadShape& shape, const AnchorHeadOptions& head,
                           const AnchorLossOptions& loss, NetworkSize network, AnchorTargets& targets);
```

- `decoded_boxes` is what `decode_anchor_head` wrote for this head from the forward-only pass. The three leaves each have the raw head's shape, `[B, H, W, A * (5 + C)]`; `logistic_weight` is zero where no cross-entropy term applies and `square_weight` where no squared-error term does, so the loss needs no slicing: the rule of SPEC 4.5 fills them element by element. This unit implements the `yolo` form with `SquaredError` and `One`; the fields the later forms read exist from the start so the structure does not change.
- Tests: one truth on a 2x2 grid with three anchors assigns the cell and anchor SPEC 4.5 names, with the stated targets and weight `2 - w * h`; a truth whose best anchor is outside the mask assigns nothing; a prediction above the ignore threshold has zero objectness weight; two truths in one cell; an empty truth list gives objectness terms only.
- Acceptance: for a hand-placed truth every non-zero element of the three leaves is the value SPEC 4.5 gives.

**T06. The reference tool, training.** `tools/reference`. Now. Waits on D05. `opus low`.
- Files: the five of D05. Adds `--truth FILE` and writes `training.sacp` (SPEC 5.3); the patch gains a driver that feeds the dumped input and truth with the network in training mode and no augmentation. The worker replaces every **(confirm)** of SPEC 4.5 with the file and line it found. Acceptance: on synthetic dumps the tool writes `training.sacp` and its manifest entry with the `context` keys of SPEC 5.3.

**T01. Training keys of the cfg.** `vision/importers`. Now. Waits on D07, T02. `haiku high`.
- Files: `im/inc/darknet_training.hpp`, `im/source/darknet_training.cpp`, `im/source/darknet_layer_yolo.cpp`, `im/inc/darknet_model.hpp`, `im/tests/test_darknet_training.cpp`.

```cpp
struct DarknetSchedule
{
    double learning_rate = 0.001; double momentum = 0.9; double decay = 0.0005;
    std::int64_t burn_in = 0; std::vector<std::int64_t> steps; std::vector<double> scales;
    std::int64_t batch = 1; std::int64_t subdivisions = 1; std::int64_t max_batches = 0;
};
[[nodiscard]] Result<DarknetSchedule> read_darknet_schedule(const DarknetSection& net);
```

- `DarknetHead` gains `DetectAnchor::AnchorLossOptions loss`; `DarknetModel` gains `DarknetSchedule schedule`. A `policy` other than `steps` is refused. Acceptance: the v3-tiny cfg yields `ignore_threshold` 0.7, both masks, all six anchors and the schedule its `[net]` states.

**T03. The loss emitter.** `vision/detect_anchor`. After YOLO26 waves: it needs the loss root, `CONSTANT` and the scalar reduction YOLO26's loss uses, whose emitter name is read at dispatch. `opus low`.
- Files: `da/inc/anchor_loss.hpp`, `da/source/anchor_loss.cpp`, `da/tests/test_anchor_loss.cpp`, `tests/integration/test_anchor_loss_gradient.cpp`.

```cpp
struct AnchorLossLeaves { Graph::ValueId target; Graph::ValueId logistic_weight; Graph::ValueId square_weight; };
[[nodiscard]] AnchorLossLeaves declare_anchor_loss_leaves(const Graph::GraphBuilder& builder, Graph::ValueId raw);
[[nodiscard]] Graph::ValueId emit_anchor_loss(const Graph::GraphBuilder& builder, Graph::ValueId raw,
                                              const AnchorLossLeaves& leaves, double scale);
```

- `sum(logistic_weight * (softplus(raw) - raw * target)) + 0.5 * sum(square_weight * (raw - target)^2)`, times `scale` (1 / batch). Emitters: `softplus`, `mul`, `sub`, `add`, `scale` and the reduction. No new operation.
- Tests: the node list; on an engine, the gradient at `raw` equals `logistic_weight * (sigma(raw) - target) + square_weight * (raw - target)` to a stated tolerance, and is exactly zero where both weights are zero.
- Acceptance: the gradient at the raw head equals that expression on a seeded input.

**T05. The `steps` schedule with burn-in.** After YOLO26 waves: it is one schedule of the scheduler object YOLO26 introduces (SPEC 2.6); module and files follow that object. `haiku medium`. Acceptance: the rate at batches 0, `burn_in / 2`, `burn_in` and one past each step equals SPEC 4.5's formula.

**T04. The anchor objective.** After YOLO26 waves: it implements the extended `Train::Objective` (several outputs, declared leaves, the host stage between the two replays) that YOLO26's detection objective defines. Module by Q12; files follow YOLO26's objective. `opus low`.
- Surface: `class AnchorObjective final : public Train::Objective`, built from the `NetworkSize` and, per head, an `AnchorHeadOptions`, an `AnchorLossOptions` and the output name; it names no importer type. It declares the three leaves per head, sums `emit_anchor_loss` over the heads, and in the host stage decodes each head, calls `assign_anchor_targets` and fills the leaves.
- Acceptance: a two-head model trains one step and the leaves fed equal what `assign_anchor_targets` returns for the forward-only predictions.

**T07. YOLOv3-tiny training parity.** Root tests. After YOLO26 waves. Waits on T01 to T06. `opus low`.
- File: `tests/integration/test_darknet_yolov3_tiny_training.cpp`. Against `training.sacp`: the leaves against Darknet's delta pattern and values; the gradient at each raw head against `-delta / batch`; the two weight gradients; then 200 steps on the one batch.
- Acceptance: the gradient at both raw heads equals Darknet's, and the loss on the one batch falls below a tenth of its first value.

### M2: YOLOv3

**D09. `Add` node.** `layers/nn`. Now. `haiku high`. Files: `nn/inc/add.hpp`, `nn/source/add.cpp`, `nn/tests/test_add.cpp`. Shaped as `Concat` is, over `GraphBuilder::add`:

```cpp
class Add final : public Node   // all five members public
{
    [[nodiscard]] static Result<std::unique_ptr<Add>> create(ShapeSpan inputs);
    [[nodiscard]] Status forward(const Trace& trace, ValueSpan inputs, std::vector<Graph::ValueId>& outputs) override;
    [[nodiscard]] std::size_t input_count() const noexcept override;
    void walk_state(std::string_view prefix, const StateVisitor& visitor) override;
    [[nodiscard]] SushiRuntime::span<const Port> ports() const noexcept override;
};
[[nodiscard]] Result<std::unique_ptr<Node>> make_add_node(ShapeSpan inputs, const Config::Value& attributes, std::string_view context);
```

Tests: one, three and unequal inputs and any key are `InvalidDescription`. Acceptance: two `[H, W, C]` inputs trace to one `K_ADD` of that shape.

**D10. Inverse letterbox for boxes.** `vision/imaging`. Now. `haiku medium`. Files: `modules/vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp`, `source/letterbox.cpp`, `tests/test_imaging_letterbox_inverse.cpp`. `[[nodiscard]] PixelBox invert_letterbox(const PixelBox& box, const LetterboxPlan& plan) noexcept;` with `x = (x - plan.left) / plan.ratio`, `y` likewise with `plan.top`, nothing clipped. Acceptance: forward then inverse returns the source box within 1e-4 pixels for a wide, a tall and an unscaled plan.

**D11. `shortcut` translator.** `vision/importers`. Now. Waits on D07. `haiku medium`. Files: `im/source/darknet_layer_shortcut.cpp`, `im/source/darknet_layer_table.cpp`, `im/tests/test_darknet_shortcut.cpp`. An `add` row reading the layer before and layer `from`; a non-linear `activation` adds its row; unequal shapes refused. Acceptance: a three-layer cfg translates to an `add` row whose `from` names the two rows, for a negative and an absolute `from`.

**D12. YOLOv3 forward parity.** Root tests. Waits on D09, D11, D14. `opus low`. File: `tests/integration/test_darknet_yolov3.cpp`; as D08, three heads, and the layers either side of the first `shortcut`. Acceptance: the detections equal Darknet's.

**T08. YOLOv3 training parity.** After YOLO26 waves. Waits on D12, T07. `opus low`. File: `tests/integration/test_darknet_yolov3_training.cpp`; as T07 over three heads, which pins that a truth is assigned in exactly one head. Acceptance: the gradient at the three raw heads equals Darknet's.

### M3: YOLOv4-tiny

**D13. `ChannelSlice` layer.** `layers/nn`. Now. `haiku high`. Files: `nn/inc/channel_slice.hpp`, `nn/source/channel_slice.cpp`, `nn/tests/test_channel_slice.cpp`. Shaped as `Upsample2d` is, over `GraphBuilder::slice_channels`: `struct ChannelSlice { int64_t offset = 0; int64_t count = 0; static constexpr auto fields(); Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const; Shape output_shape(const Shape& input) const; };`. Acceptance: `{offset 32, count 32}` over `[H, W, 64]` reports `[H, W, 32]` and traces one `K_SLICE_CHANNELS`; a range past the channels throws.

**D14. Rows `add` and `slice_channels`.** `layers/nn`. Now. Waits on D09, D13. `haiku medium`. Files: `nn/source/layer_registry.cpp`, `nn/tests/test_node_registry.cpp`. Acceptance: both types build through `make_node` and each refuses an unknown key.

**D15. Grouped `route`.** `vision/importers`. Now. Waits on D07. `haiku medium`. Files: `im/source/darknet_layer_route.cpp`, `im/tests/test_darknet_route_groups.cpp`. With `groups > 1` each source gets a `slice_channels` row before the join. Acceptance: `layers=-1, groups=2, group_id=1` over 64 channels gives one row with offset 32 and count 32; a count `groups` does not divide is refused.

**D16. YOLOv4-tiny forward parity.** Root tests. Waits on D14, D15. `opus low`. File: `tests/integration/test_darknet_yolov4_tiny.cpp`; as D08 with `scale_x_y` 1.05 from the cfg. Acceptance: the detections equal Darknet's.

**T09. CIoU box term and multi-anchor match.** `vision/detect_anchor`. After YOLO26 waves; Q12. `opus low`. Files: `da/inc/anchor_box_loss.hpp`, `da/source/anchor_box_loss.cpp`, `da/source/anchor_assigner.cpp`, `da/tests/test_anchor_box_loss.cpp`. `[[nodiscard]] Graph::ValueId emit_anchor_box_loss(const Graph::GraphBuilder& builder, Graph::ValueId raw, Graph::ValueId target_boxes, Graph::ValueId box_mask, const AnchorHeadShape& shape, const AnchorHeadOptions& head, NetworkSize network, double weight);` decodes the box entries in the graph (`sigmoid`, `exp`, constants for the cell offsets and anchors) and calls YOLO26's CIoU emitter with its guard for unassigned rows. The assigner honours `match_threshold` and, for `CompleteIou`, writes the box targets as boxes and zero box weights in the other two leaves. Acceptance: the gradient at the raw head of a one-truth case equals the finite-difference gradient of `1 - CIoU`.

**T10. v4 training keys.** `vision/importers`. Now. Waits on T01. `haiku medium`. Files: `im/source/darknet_layer_yolo.cpp`, `im/tests/test_darknet_training_v4.cpp`. `iou_loss`, `iou_normalizer`, `cls_normalizer`, `obj_normalizer`, `iou_thresh` into `AnchorLossOptions`. Acceptance: the v4-tiny cfg's values reach the options.

**T11. YOLOv4-tiny training parity.** After YOLO26 waves. Waits on D16, T09, T10. `opus low`. File: `tests/integration/test_darknet_yolov4_tiny_training.cpp`. Acceptance: as T07.

### M4: YOLOv4

**D17. `Mish` layer.** `layers/nn`. Now; `softplus`, `tanh`, `mul` rechecked at dispatch. `haiku high`. Files: `nn/inc/mish.hpp`, `nn/source/mish.cpp`, `nn/tests/test_mish.cpp`, `tests/integration/test_mish_values.cpp`. Shaped as `SiLU` is: `struct Mish { static constexpr auto fields(); Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const; Shape output_shape(const Shape& input) const; };`. Darknet cuts its softplus off at a threshold **(confirm)**; the test names the largest difference that causes. Acceptance: the values at -20, -1, 0, 1, 20 equal `x * tanh(log1p(exp(x)))` computed in double, 1e-6 relative.

**D18. Distance-IoU suppression.** `vision/detect`. Now. Q10. `haiku high`. Files: `modules/vision/detect/include/SushiAI/vision/detect/distance_suppression.hpp`, `source/distance_suppression.cpp`, `include/.../detections_nms.hpp`, `source/detections_nms.cpp`, `tests/test_detect_distance_suppression.cpp`.

```cpp
enum class SuppressionCriterion : std::uint8_t { Iou, DistanceIou };
void suppress_overlaps_by_distance(SushiRuntime::span<const float> boxes, SushiRuntime::span<const float> scores,
                                   float threshold, float beta, std::vector<std::int64_t>& kept);
```

`NmsConfig` gains `SuppressionCriterion criterion = SuppressionCriterion::Iou` and `float distance_beta = 1.0f`. The criterion is IoU minus the squared centre distance over the squared enclosing diagonal, raised to `beta` **(confirm)**. Acceptance: with the default criterion every existing `detect` test is unchanged, and of two boxes with equal IoU the one with the farther centre is kept.

**D19. Row `mish`.** `layers/nn`. Waits on D17. `haiku medium`. Files: `nn/source/layer_registry.cpp`, `nn/tests/test_node_registry.cpp`. Acceptance: `{"type": "mish"}` builds and refuses a key.

**D20. `mish` and the suppression keys.** `vision/importers`. Now. Waits on T10. `haiku medium`. Files: `im/source/darknet_layer_convolutional.cpp`, `im/source/darknet_layer_yolo.cpp`, `im/tests/test_darknet_yolov4_keys.cpp`. Acceptance: `activation=mish` adds a `mish` row, and `nms_kind=greedynms` with `beta_nms` reaches the `DarknetHead`.

**D21. YOLOv4 forward parity.** Root tests. Waits on D18 to D20. `opus low`. File: `tests/integration/test_darknet_yolov4.cpp`; as D08 plus the three SPP pool outputs. Acceptance: the detections equal Darknet's.

**T12. YOLOv4 training parity.** After YOLO26 waves. Waits on D21, T11. `opus low`. File: `tests/integration/test_darknet_yolov4_training.cpp`. Acceptance: as T07, three heads.

### M5: YOLOv2

**D22. `SpaceToDepth` layer.** `layers/nn`. Now; `im2col_with_extents`, `reshape`, `slice_channels`, `concat_channels` rechecked at dispatch. `opus low`: the element order comes from reading Darknet's `reorg` (SPEC 4.1). Files: `nn/inc/space_to_depth.hpp`, `nn/source/space_to_depth.cpp`, `nn/tests/test_space_to_depth.cpp`, `tests/integration/test_space_to_depth_values.cpp`. `struct SpaceToDepth { int64_t block = 2; static constexpr auto fields(); Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const; Shape output_shape(const Shape& input) const; };`. If Darknet's order cannot be expressed by these emitters the worker stops and reports; a new operation is the owner's decision. Acceptance: `[26, 26, 64]` becomes `[13, 13, 256]`, and on an index-valued input every element sits where the cited Darknet rule puts it.

**D23. Row `space_to_depth`.** `layers/nn`. Waits on D22. `haiku medium`. Files: `nn/source/layer_registry.cpp`, `nn/tests/test_node_registry.cpp`. Key `block`. Acceptance: the row builds.

**D24. `reorg` and `region` translators.** `vision/importers`. Now. Waits on T01. `haiku high`. Files: `im/source/darknet_layer_reorg.cpp`, `im/source/darknet_layer_region.cpp`, `im/source/darknet_layer_table.cpp`, `im/tests/test_darknet_region.cpp`. `region` gives a head with `GridCells`, `Softmax`, `Score` and loss options with the four scales, `rescore` and the anchor pull. Acceptance: the v2 cfg translates to one head with five anchors in cells.

**D25. YOLOv2 forward parity.** Root tests. Waits on D23, D24. `opus low`. File: `tests/integration/test_darknet_yolov2.cpp`; as D08 plus the `reorg` output. Acceptance: the detections equal Darknet's.

**T13. `region` assignment.** `vision/detect_anchor`. Now: host code. Waits on T02. `opus low`. Files: `da/source/anchor_assigner.cpp`, `da/tests/test_anchor_assigner_region.cpp`. `ObjectnessTarget::Iou`, the four weights, `prior_pull_images`, anchors in cells. Acceptance: for a hand-placed truth every non-zero element of the leaves is the value SPEC 4.5 gives for `region`, before and after `prior_pull_images`.

**T14. `region` class term.** After YOLO26 waves; Q13. `opus low`. Files and surface wait for the answer: either a rule in the core tier or a second emitter `emit_anchor_class_loss` in `da/`. Acceptance: the gradient at the class logits equals `softmax(tc) - target` times the class weight on assigned rows and zero elsewhere.

**T15. YOLOv2 training parity.** After YOLO26 waves. Waits on D25, T13, T14. `opus low`. File: `tests/integration/test_darknet_yolov2_training.cpp`. Acceptance: as T07, one head.

### M6: YOLOv1, subject to Q6

**D26. `local` layer.** `core` and `layers/nn`. After YOLO26 waves: it adds an operation to `ops.hpp` and an emitter to `builder.*`, with its lowering and gradient rule. `opus low`; the operation's design comes to the owner first, and a `high` dispatch only with his approval. Files and signature are not fixed here. Acceptance: a `local` layer over a known input equals the sum written out per position.

**D27. `detection` decode.** `vision/detect_anchor`. Now. `haiku high`. Files: `da/inc/cell_head_options.hpp`, `da/inc/cell_head_decode.hpp`, `da/source/cell_head_decode.cpp`, `da/tests/test_cell_head_decode.cpp`.

```cpp
enum class SizeCoding : std::uint8_t { Linear, SquareRoot };
struct CellHeadOptions { std::int64_t side = 7; std::int64_t boxes_per_cell = 2; std::int64_t num_classes = 20; SizeCoding size_coding = SizeCoding::Linear; };
void decode_cell_head(SushiRuntime::span<const float> raw, std::int64_t batch, const CellHeadOptions& options, NetworkSize network,
                      float confidence_threshold, SushiRuntime::span<float> boxes, SushiRuntime::span<float> scores);
```

Acceptance: each term of the `detection` decode of SPEC 4.4 is pinned by a one-cell case.

**D28. `connected`, `local`, `dropout`, `detection` translators.** `vision/importers`. Waits on D07, D26. `opus low`. Files: four `im/source/darknet_layer_*.cpp`, `im/source/darknet_layer_table.cpp`, `im/inc/darknet_model.hpp` (a `cell_heads` list), `im/tests/test_darknet_yolov1.cpp`. Acceptance: the v1 cfg translates, with the connected slot marked `ConnectedAfterMap`.

**D29. Row `local`.** `layers/nn`. Waits on D26. `haiku medium`. Files: `nn/source/layer_registry.cpp`, `nn/tests/test_node_registry.cpp`. Acceptance: the row builds.

**D30. YOLOv1 forward parity.** Root tests. Waits on D27 to D29. `opus low`. File: `tests/integration/test_darknet_yolov1.cpp`. Acceptance: the detections equal Darknet's.

**T16. `detection` assigner and loss.** `vision/detect_anchor`. After YOLO26 waves. `opus low`. Files: `da/inc/cell_assigner.hpp`, `da/inc/cell_loss.hpp`, their two sources, `da/tests/test_cell_loss.cpp`. Shaped as T02 and T03 are: three leaves at the raw output's shape, one emitter. Acceptance: the gradient at the raw output equals minus the delta SPEC 4.5 describes for a one-cell case.

**T17. Dropout in training.** `layers/nn`. After YOLO26 waves; not measured (SPEC 3). `opus low`. Files and surface come with the design. Acceptance: the kept fraction and the scale of the kept values equal the stated probability on a seeded mask.

**T18. YOLOv1 training parity.** After YOLO26 waves. Waits on D30, T16, T17. `opus low`. File: `tests/integration/test_darknet_yolov1_training.cpp`. Acceptance: as T07, with dropout switched off in both.

## 5. Waves and builds

Files within a wave are disjoint. A build is the owner's; so is each Darknet run.

| Wave | Units | Waits on | Closes with |
| --- | --- | --- | --- |
| 0 | O1 | Q1, Q5, Q7, Q11; YOLO26's umbrella lines | nothing |
| 1 | D01, D02, D03, D04, D05 | wave 0; Q4, Q8, Q9 for D05 | syntax checks |
| 2 | D06, D07 | wave 1 | syntax checks |
| 3 | D08 | wave 2; the Darknet forward run for `yolov3_tiny` | **B1**: both suites, then D08 with the fixture. Ends M1 |
| 4 | T02, T06 | B1 | syntax checks |
| 5 | T01 | wave 4 | syntax checks |
| 6 | T03, T05 | YOLO26's loss root, reduction and scheduler built; Q12 | syntax checks |
| 7 | T04 | wave 6; YOLO26's objective and two-pass trainer built | syntax checks |
| 8 | T07 | wave 7; the Darknet training run for `yolov3_tiny` | **B2**. Ends M1T |
| 9 | D09, D10, D11, D13, D15 | B1 | syntax checks |
| 10 | D14, D12, D16, T08, T10 | wave 9; B2 for T08; Darknet runs for `yolov3`, `yolov4_tiny` | syntax checks |
| 11 | D17, D18, D20, T09 | wave 10; Q10 | syntax checks |
| 12 | D19, D21, T11, T12 | wave 11; Darknet runs for `yolov4` | **B3**. Ends M2, M3, M4 and their training parts |
| 13 | D22, D24, T13 | B3 | syntax checks |
| 14 | D23, D25, T14 | wave 13; Q13; Darknet runs for `yolov2` | syntax checks |
| 15 | T15 | wave 14 | **B4**. Ends M5 and M5T |
| 16 | D26, D27, D28 | B4; Q6 | syntax checks |
| 17 | D29, D30, T16, T17 | wave 16 | syntax checks |
| 18 | T18 | wave 17; Darknet runs for `yolov1` | **B5**. Ends M6 and M6T |

Forty-eight units: thirty forward (D01 to D30) and eighteen training (T01 to T18). Eighteen waves after the skeleton, five builds. Thirty-four units are "now": D01 to D25, D27 to D30 in the sense that they edit no YOLO26 file (D28, D29 and D30 wait on D26), and T01, T02, T06, T10, T13. Fourteen are "after YOLO26 waves": D26, T03, T04, T05, T07, T08, T09, T11, T12, T14 to T18. Waves 9 and 10 minus T08 can run before waves 6 to 8 when YOLO26 is late; the forward milestones do not wait for training.

Shared files, one editor per wave: `layer_registry.cpp` (D06, D14, D19, D23, D29); `darknet_layer_table.cpp` (D07, D11, D24, D28); `darknet_layer_yolo.cpp` (D07, T01, T10, D20, in waves 2, 5, 10, 11); `anchor_assigner.cpp` (T02, T09, T13); the reference tool (D05, T06).

B1 is the first time any of this compiles, and it compiles the whole tree, so it cannot happen before the YOLO26 working tree builds.

## 6. Not planned here

- A full training run on COCO and Darknet's data loader with its augmentation (Q14).
- A program under `applications/` that takes an image file and prints detections. The parity tests start from Darknet's input tensor; an image path needs Darknet's float resize or an accepted difference from it.
- A fused `K_MISH`, and a batch-norm fold in pjreddie's epsilon form.
- `yolov2-tiny`, `yolov3-spp` and the scaled YOLOv4 cfgs (`new_coords=1`).

## 7. Not verified

- Nothing was built, run or downloaded. No Darknet source was read: every **(confirm)** in `SPEC.md` is from memory, and D05 and T06 replace them with citations. The costliest if wrong: the maxpool offset rule, the `reorg` element order, the two score gates of the `yolo` decode, AlexeyAB's epsilon form, the `seen` width per file, whether `detector test` letterboxes, and all of SPEC 4.5, where the claim that the deltas are the gradient of a cross-entropy on logits carries the whole loss design.
- The weights file sizes are recalled. The YOLOv3-tiny and YOLOv3 counts agree with the parameter counts those models are known by; the others have no second source here.
- What YOLO26 will expose: the name of its scalar reduction, the shape of its extended objective and host stage, its scheduler object and whether its CIoU emitter is callable from another module. Its SPEC and PLAN were read; its code for these is not written.
- Whether `Tensor::data()` is host-writable on every device, the exact state names `build_model` produces, and which list of `tests/CMakeLists.txt` takes a cross-module vision test.
- Whether `RESHAPE` of an `IM2COL` result carries a layout the next NHWC emitter accepts (D22), and whether a leaf-weighted sum over a rank-4 value has a reduction today (T03).
- Whether any of the six cfgs uses a key or an activation outside SPEC 4.1.
- The Darknet licence and the licence of `data/dog.jpg`.
- Rule 6 of the dispatch block had nothing to check: this task wrote two Markdown files.
