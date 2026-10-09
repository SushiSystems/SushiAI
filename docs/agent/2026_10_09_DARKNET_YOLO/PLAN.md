# Darknet-era YOLO: plan

**Status:** Draft, revised 2026-10-09 to the owner's rulings of that day; waiting on the answers of section 1 and on `REPORT.md`. Nothing is written or built. Requirements, measurements and formats are in `SPEC.md` beside this file; section numbers refer to it.

## 1. Rulings and questions

Settled on 2026-10-09, by the owner or by the engineer he delegated them to:

| Subject | Ruling |
| --- | --- |
| Scope | `yolov3-tiny`, `yolov3`, `yolov4-tiny`, `yolov4`, `yolov7-tiny`, `yolov7`, in that order. YOLOv1 and YOLOv2 are out |
| Training | In scope for every model, a forward-parity part before a training part |
| Reference | AlexeyAB's Darknet, built outside the repository, CPU only, patched to dump; deleted when every reference exists. `REPORT.md` holds the dump layout and settles every **(REPORT)** mark of the SPEC |
| Anchor unit | Its own module, `vision/detect_anchor`, namespace `SushiAI::Vision::DetectAnchor` |
| Importer | `vision/importers`; it emits a `Config::ModelSpec` that `NN::build_model` turns into a graph |
| Suppression | The criterion is a field of the configuration, `NmsConfig` |
| Shared with YOLO26 | The CIoU emitter and the home of the objective. Neither is in the tree (SPEC 2.6), so this plan writes the emitter and asks for the home (Q4) |
| Outside files | Weights and cfgs under `SA_REFERENCE_DATA_DIR` (`darknet/<file>` below it); reference tensors git-ignored under `tests/fixtures/darknet/<model>/`. A test without its files is reported skipped |

Open. The plan is written under each recommendation; a different answer changes the units in the last column.

| # | Question | Recommendation | Alternative | Changes |
| --- | --- | --- | --- | --- |
| Q1 | The outside files are an exception to rule 4 of the `testing` skill. Is it granted? | Yes, for the parity tests of this work and of YOLO26 only, each skipped by name when a file is missing | Commit the reference tensors: 14 MB for `yolov7` alone | D15 to D20, T10 to T15 |
| Q2 | The patch dumps the input, the `[yolo]` outputs and the detections. Training parity needs a second dump in training mode (SPEC 5.3), and a deleted patch cannot regenerate a reference. | Extend the patch before the build is deleted so it also dumps each head's `delta` for one fixed batch, and keep the patch text in `tools/reference/darknet/` with a `NOTICE.md` entry | No training dump: training is then checked against SPEC 4.5 as the REPORT corrects it and against finite differences, never against Darknet's numbers | D11, T06, T10 to T15 |
| Q3 | How far training goes | Per model: assignment-target parity, loss parity, gradient parity and overfitting one batch. No reason against it was found, with one limit: loss parity holds only where the REPORT shows that Darknet's printed terms are the minimised ones | A full run of YOLOv3-tiny on COCO to a stated mAP, which needs Darknet's loader and augmentation and days of the GTX 1060 | T10 to T15 |
| Q4 | Home of the detection objective and of the step that runs a forward-only pass, the host assigner and the training replay. `Train::Objective` has no host stage (SPEC 2.6) | A module `vision/training`, which `DOMAIN_MODULES.md:123-125` leaves open, depending on `train` and `vision/detect_anchor`; YOLO26's objective lands beside it. The step composes `Train::EvaluationStep` and `Train::TrainingStep` and changes neither | (a) Inside `vision/detect_anchor`, which then depends on `train`. (b) A host-stage hook on `Train::Objective`, a change to a public interface of the layers tier | T09, O1 |
| Q5 | Home and shape of the learning-rate schedule | `core/optim`: a value type `StepSchedule` with `rate_at(iteration)`, read by the caller that sets `SGD`'s rate. It names nothing of Darknet | A function in `vision/importers`, where no second model family could reach it | T05 |
| Q6 | Public shape of the trailing padding on `MaxPool2d` | Two fields, `trailing_pad_h` and `trailing_pad_w`; row key `trailing_padding`, an integer or `[h, w]`. `ConvGeometry` is not touched | `pad_bottom`, `pad_right` in `ConvGeometry`, which edits `builder.hpp` and every emitter | D01, D05 |
| Q7 | May `vision/importers` depend on `vision/detect_anchor` and `vision/detect` | Yes: a translated model carries each head's options and its suppression criterion, so there is one options type | The importer returns the raw keys and a program builds the options | D12, T01 |
| Q8 | How the plan runs | The waves of section 5 | Inline, in the unit order of section 4 | Section 5 |

## 2. Modules

| Module | Tier | Holds | Depends on |
| --- | --- | --- | --- |
| `vision/detect_anchor` (new) | vision | Head options, shape arithmetic, decode, the assigner, the loss emitters | PUBLIC `vision/detect`, `graph`; PRIVATE `core`; EXTERNAL_PUBLIC `SushiRuntime::SushiRuntime` |
| `vision/importers` (new) | vision | The `.cfg` reader, the translation to a `ModelSpec`, the `.weights` reader, the training settings | PUBLIC `core`, `config`, `nn`, `vision/detect`, `vision/detect_anchor` |
| `vision/training` (new, Q4) | vision | The anchor objective and its two-pass step | PUBLIC `train`, `nn`, `vision/detect_anchor` |
| `layers/nn` (grown) | layers | Trailing padding on `MaxPool2d`; `Add`; `ChannelSlice`; `Mish`; their rows | unchanged |
| `vision/detect` (grown) | vision | Distance-IoU suppression and the criterion field; the CIoU emitter | `graph` added if absent; not read |
| `vision/imaging` (grown) | vision | The inverse of a letterbox for a box | unchanged |
| `core/optim` (grown, Q5) | core | `StepSchedule` | unchanged |

## 3. Orchestrator steps

Workers do not write build files (dispatch rule 3).

**O1, the skeleton, before wave 1.**

- `cmake/SushiAILayers.cmake` and `tools/layering/check_layering.py`: rows `vision/detect_anchor`, `vision/importers`, `vision/training`, each `vision`.
- `modules/vision/CMakeLists.txt`: one `add_subdirectory` per new module, after what it depends on.
- A `CMakeLists.txt` in each new module: one `sushiai_add_module(NAME ... TIER vision SOURCES ... TESTS ...)` with the dependencies of section 2.
- `modules/vision/umbrella/CMakeLists.txt`: the three keys in `PUBLIC_DEPENDS`. `vision.h`: one include per new public header as each wave is integrated.
- A README stub in each new module, replaced by the worker that fills it.

**After each wave.** Sources and tests appended to the `CMakeLists.txt` of every module touched; root tests appended to `sushiai_tests` or to `sushiai_vision_tests` (`tests/CMakeLists.txt`, beside `integration/test_yolo26_forward.cpp`); the `.gitignore` line `tests/fixtures/darknet/*/*.sacp`; the definition that hands `SA_REFERENCE_DATA_DIR` to the tests.

**At close.** The changelog lines, `PROGRAMME.md` stage 6, `REMAINING_WORK.md`, `docs/design/DARKNET_YOLO.md` drawn from `SPEC.md`, `DOMAIN_MODULES.md` for the new modules, and the manual pages the work makes false.

## 4. Units

Every unit is free: none waits on another piece of work. Paths are relative to the repository root; `nn/` stands for `modules/layers/nn/`, `de/` for `modules/vision/detect/`, `da/` for `modules/vision/detect_anchor/`, `im/` for `modules/vision/importers/`, `tr/` for `modules/vision/training/`, and `inc/` inside each for its `include/SushiAI/...` folder. Dispatch rows: `haiku medium` (spec-complete, 1 or 2 files), `haiku high` (spec-complete, more files), `opus low` (integration or judgment).

### 4.1 Layers and bricks

**D01. `MaxPool2d` trailing padding.** `layers/nn`. `haiku high`.
- Files: `nn/inc/max_pool2d.hpp`, `nn/source/max_pool2d.cpp`, `nn/tests/test_max_pool2d.cpp`, `tests/integration/test_max_pool_trailing.cpp` (new).
- Surface, two fields of `struct MaxPool2d`: `int64_t trailing_pad_h = 0;` and `int64_t trailing_pad_w = 0;`, the padding rows below and columns right beyond the symmetric pad. `output_shape` adds them to the trailing side; `forward` calls `GraphBuilder::maxpool2d_with_extents`. A negative value and an all-padding window throw as an illegal window does today.
- Tests: 13x13, 2x2, stride 1, trailing 1 gives 13x13 and trailing 0 gives 12x12; 416 at stride 2 gives 208 either way; on an engine, an all-negative input keeps the true maximum in the last row and column.
- Acceptance: the last row and column of a 2x2 stride-1 pool over a known 3x3 input equal the maxima of the truncated windows.

**D02. `Add` node.** `layers/nn`. `haiku high`. Files: `nn/inc/add.hpp`, `nn/source/add.cpp`, `nn/tests/test_add.cpp`. Shaped as `Concat` is, over `GraphBuilder::add`:

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

**D03. `ChannelSlice` layer.** `layers/nn`. `haiku high`. Files: `nn/inc/channel_slice.hpp`, `nn/source/channel_slice.cpp`, `nn/tests/test_channel_slice.cpp`. Shaped as `Upsample2d` is, over `GraphBuilder::slice_channels`: `struct ChannelSlice { int64_t offset = 0; int64_t count = 0; static constexpr auto fields(); Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const; Shape output_shape(const Shape& input) const; };`. Acceptance: `{offset 32, count 32}` over `[H, W, 64]` reports `[H, W, 32]` and traces one `K_SLICE_CHANNELS`; a range past the channels throws.

**D04. `Mish` layer.** `layers/nn`. `haiku high`. Files: `nn/inc/mish.hpp`, `nn/source/mish.cpp`, `nn/tests/test_mish.cpp`, `tests/integration/test_mish_values.cpp`. Shaped as `SiLU` is: `struct Mish { static constexpr auto fields(); Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const; Shape output_shape(const Shape& input) const; };`, emitting `mul(x, tanh(softplus(x)))`. If the REPORT shows a cut-off in Darknet's softplus, the test names the largest difference it causes. Acceptance: the values at -20, -1, 0, 1, 20 equal `x * tanh(log1p(exp(x)))` computed in double, 1e-6 relative.

**D05. Rows `add`, `slice_channels`, `mish` and the key `trailing_padding`.** `layers/nn`. Waits on D01 to D04. `haiku high`. Files: `nn/source/layer_registry.cpp`, `nn/tests/test_node_registry.cpp`. `maxpool2d` accepts `trailing_padding`, an integer or `[h, w]`, default 0; `avgpool2d` refuses it. Acceptance: each of the three rows builds through `make_node` and refuses an unknown key, and `{"type": "maxpool2d", "kernel": 2, "stride": 1, "trailing_padding": 1}` over `[13, 13, 256]` reports `[13, 13, 256]`.

**D06. The anchor head decode.** `vision/detect_anchor`. Waits on the REPORT. `opus low`.
- Files: `da/inc/anchor_head_options.hpp`, `da/inc/anchor_head_shape.hpp`, `da/inc/anchor_head_decode.hpp`, the three `da/source/*.cpp` of the same names, `da/tests/test_anchor_head_decode.cpp`, `da/README.md`.

```cpp
struct AnchorSize { float width = 0.0f; float height = 0.0f; };
enum class HeadCoding : std::uint8_t { Logit, Activated };   // new_coords = 0, 1

struct AnchorHeadOptions
{
    std::vector<AnchorSize> anchors;   // the anchors of this head, mask applied, in network pixels
    std::int64_t num_classes = 0;
    HeadCoding coding = HeadCoding::Logit;
    float scale_xy = 1.0f;
};
[[nodiscard]] Status validate_anchor_head(const AnchorHeadOptions& options);

struct NetworkSize { std::int64_t height = 0; std::int64_t width = 0; };
struct AnchorHeadShape { std::int64_t batch = 0; std::int64_t height = 0; std::int64_t width = 0; };
[[nodiscard]] std::int64_t count_head_channels(const AnchorHeadOptions& options) noexcept;
[[nodiscard]] std::int64_t count_head_rows(const AnchorHeadShape& shape, const AnchorHeadOptions& options) noexcept;

void decode_anchor_head(SushiRuntime::span<const float> head, const AnchorHeadShape& shape,
                        const AnchorHeadOptions& options, NetworkSize network, float confidence_threshold,
                        SushiRuntime::span<float> boxes, SushiRuntime::span<float> scores);
```

- `head` is the input of the `[yolo]` layer, `[B, H, W, A * (5 + C)]`; `boxes` receives `[B, H * W * A, 4]` centre and size in network pixels and `scores` `[B, H * W * A, C]`, row `(h * W + w) * A + a`, by SPEC 4.4. Both are what `Detect::select_nms_detections` reads; a caller joins the heads on the row axis. The `Activated` formulae are copied from the REPORT's citation of `get_yolo_box`. Shaped as `vision/detect` is: free functions on spans, a status only where values come from a file.
- Tests: zero logits on a 1x1 grid give the cell centre, the anchor size and scores of 0.25; `scale_xy` 2.0 moves a saturated offset half a cell past the edge; the `Activated` size at inputs 0, 0.5 and 1; each gate; the row order on a 2x3 grid with two anchors; the refusals.
- Acceptance: every formula of both codings is pinned by a case whose expected value follows from inputs at zero or at saturation, to 1e-6 absolute.

**D07. Distance-IoU suppression.** `vision/detect`. Waits on the REPORT for the criterion. `haiku high`. Files: `de/inc/distance_suppression.hpp`, `de/source/distance_suppression.cpp`, `de/inc/detections_nms.hpp`, `de/source/detections_nms.cpp`, `de/tests/test_detect_distance_suppression.cpp`.

```cpp
enum class SuppressionCriterion : std::uint8_t { Iou, DistanceIou };
void suppress_overlaps_by_distance(SushiRuntime::span<const float> boxes, SushiRuntime::span<const float> scores,
                                   float threshold, float beta, std::vector<std::int64_t>& kept);
```

`NmsConfig` gains `SuppressionCriterion criterion = SuppressionCriterion::Iou` and `float distance_beta = 1.0f`; `select_nms_detections` picks the function by the field. Acceptance: with the default criterion every existing `detect` test is unchanged, and of two boxes with equal IoU the one with the farther centre is kept.

**D08. Inverse letterbox for a box.** `vision/imaging`. `haiku medium`. Files: `modules/vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp`, `source/letterbox.cpp`, `tests/test_imaging_letterbox_inverse.cpp`. `[[nodiscard]] PixelBox invert_letterbox(const PixelBox& box, const LetterboxPlan& plan) noexcept;`, nothing clipped. Acceptance: forward then inverse returns the source box within 1e-4 pixels for a wide, a tall and an unscaled plan.

### 4.2 Importer and reference

**D09. The `.cfg` reader.** `vision/importers`. `haiku high`. Files: `im/inc/darknet_config.hpp`, `im/source/darknet_config.cpp`, `im/tests/test_darknet_config.cpp`.

```cpp
struct DarknetOption { std::string key; std::string value; std::int64_t line = 0; };
struct DarknetSection { std::string type; std::int64_t line = 0; std::vector<DarknetOption> options; };
struct DarknetConfig { DarknetSection net; std::vector<DarknetSection> layers; };  // index = Darknet layer index
[[nodiscard]] Result<DarknetConfig> parse_darknet_config(std::string_view text, std::string_view origin);
[[nodiscard]] Result<DarknetConfig> load_darknet_config(const std::filesystem::path& path);
```

The grammar of SPEC 4.1 and nothing about what a section means. Tests: both comment forms, blank lines, spaces around `=` and inside lists, CRLF; a key before any section, a line without `=`, a repeated key and a first section other than `[net]` each fail with `InvalidDescription` naming origin and line; a missing file gives `FileNotFound`. Acceptance: a text with every lexical form of SPEC 4.1 parses to the sections, keys and lines the test states.

**D10. The `.weights` reader.** `vision/importers`. `haiku high`. Files: `im/inc/darknet_weight_slot.hpp`, `im/inc/darknet_weights.hpp`, `im/source/darknet_weights.cpp`, `im/tests/test_darknet_weights.cpp`, `tests/integration/test_darknet_weights_state.cpp`.

```cpp
enum class DarknetTensorLayout : std::uint8_t { Vector, ConvolutionWeight };
struct DarknetWeightSlot { std::string state_name; DarknetTensorLayout layout = DarknetTensorLayout::Vector; Shape source_shape; std::int64_t layer = 0; };
struct DarknetWeightsHeader { std::int32_t major = 0; std::int32_t minor = 0; std::int32_t revision = 0; std::uint64_t seen = 0; };

[[nodiscard]] Result<DarknetWeightsHeader> read_darknet_header(std::istream& stream);
void convert_darknet_tensor(SushiRuntime::span<const float> source, const DarknetWeightSlot& slot, SushiRuntime::span<float> converted);
[[nodiscard]] Result<DarknetWeightsHeader> load_darknet_weights(const std::filesystem::path& path,
    SushiRuntime::span<const DarknetWeightSlot> slots, NN::ModelState& state);
```

`Vector` `[n]` is copied; `ConvolutionWeight` `[out, in, kh, kw]` becomes `[out, kh, kw, in]`. The slots are in file order; the reader does not know what a layer is. Strict both ways (SPEC 4.2). Bytes are assembled field by field. Tests: both header forms; each layout on a tensor whose elements are their own indices; a file the test writes into the build tree fills a two-layer state; one float short, one float over and a slot of another shape each give `Incompatible` naming the layer. Acceptance: a weight written as `[out, in, kh, kw]` index values arrives in the state as `[out, kh, kw, in]` with every element at the permuted position.

**D11. The reference converter, forward.** `tools/reference`. Waits on the REPORT for the dump layout. `opus low`.
- Files: `tools/reference/export_darknet_reference.py`, `tools/tests/test_export_darknet_reference.py`, the Darknet section of `tools/reference/README.md`.
- Surface: `export_darknet_reference.py OUTPUT_DIR --model NAME --dump-dir DIR --data-dir DIR`; it reads a dump directory, checks the cfg and the weights under `--data-dir` against the sizes and hashes of SPEC 5.4, and writes `input.sacp`, `forward.sacp` and the manifest of SPEC 5.2. It runs nothing from Darknet and downloads nothing. Exit 1 with one line on stderr on failure. It imports the SACP encoder and the manifest writer of the YOLO26 exporter; if that needs a shared module under `tools/reference/`, the worker reports it and stops, since it moves code the YOLO26 exporter owns.
- Tests: the NCHW to NHWC rule, the manifest of a synthetic dump, the refusal above the 64 MiB ceiling, the hash check.
- Acceptance: on a synthetic dump directory the tool writes both files and a manifest whose hashes and shapes the test recomputes.

**D12. The translation.** `vision/importers`. Waits on D06, D07, D09, D10. `opus low`.
- Files: `im/inc/darknet_model.hpp`, `im/source/darknet_model.cpp`, `im/source/darknet_layer_table.{hpp,cpp}`, `im/source/darknet_layer_{convolutional,maxpool,upsample,route,yolo}.cpp`, `im/tests/test_darknet_model.cpp`, `im/README.md`.

```cpp
struct DarknetHead { std::string output; std::int64_t layer = 0; DetectAnchor::AnchorHeadOptions options;
                     Detect::SuppressionCriterion criterion = Detect::SuppressionCriterion::Iou; float distance_beta = 1.0f; };
struct DarknetModel { Config::ModelSpec spec; DetectAnchor::NetworkSize network; std::vector<DarknetHead> heads;
                      std::vector<DarknetWeightSlot> slots; std::vector<std::int64_t> row_of_layer; };
[[nodiscard]] Result<DarknetModel> translate_darknet_model(const DarknetConfig& config);
```

- `DarknetHead::output` is the published output's name, `"head.<k>"`; `spec` holds rows of `nn` types and publishes the heads; `heads` is in file order, `slots` in `.weights` order, and `row_of_layer[i]` is the row that holds Darknet layer `i`'s output.
- One private translator per section type, found through a table of `{type, function}`: a new section type is a new source file and one row. A `[convolutional]` becomes `conv2d`, then `batchnorm2d` with the epsilon of SPEC 4.3 when normalised, then the activation row from a table of names: `leaky` to `leaky_relu` with slope 0.1, `mish` to `mish`, `swish` to `silu`, `logistic` to `sigmoid`, `linear` to none. The `yolo` translator reads every forward key of SPEC 3.2 (`scale_x_y`, `new_coords`, `nms_kind`, `beta_nms`), accepts the training keys without reading them, and refuses any other. An ungrouped `route` only; D14 adds the grouped one.
- Tests: the v3-tiny cfg written out in the test (input data, not a reference) gives the row count, the `from` of each `route`, two outputs and their options; `build_model` over `register_layers` accepts the spec and reports heads `[13, 13, 255]` and `[26, 26, 255]`; every slot name is an entry of a `ModelState` created from the traced model; the slots' float count is 8 858 734; a v7 head translates to `Activated`, `scale_xy` 2.0 and `DistanceIou`; an unknown section and an unknown key fail naming the line.
- Acceptance: the translated v3-tiny spec builds, and every slot name is an entry of its state.

**D13. `shortcut` translator.** `vision/importers`. Waits on D12. `haiku medium`. Files: `im/source/darknet_layer_shortcut.cpp`, `im/source/darknet_layer_table.cpp`, `im/tests/test_darknet_shortcut.cpp`. An `add` row reading the layer before and layer `from`; unequal shapes and a non-linear `activation` refused. Acceptance: a three-layer cfg translates to an `add` row whose `from` names the two rows, for a negative and an absolute `from`.

**D14. Grouped `route`.** `vision/importers`. Waits on D12. `haiku medium`. Files: `im/source/darknet_layer_route.cpp`, `im/tests/test_darknet_route_groups.cpp`. With `groups > 1` each source gets a `slice_channels` row before the join. Acceptance: `layers=-1, groups=2, group_id=1` over 64 channels gives one row with offset 32 and count 32; a count `groups` does not divide is refused.

### 4.3 Forward parity

**D15. YOLOv3-tiny forward parity.** Root tests. Waits on D05, D11, D12 and the `yolov3_tiny` fixture. `opus low`.
- Files: `tests/integration/test_darknet_parity.hpp` (the fixture loader, the skip rule and the comparison every parity test shares), `tests/integration/test_darknet_yolov3_tiny.cpp`, `tests/fixtures/darknet/README.md`.
- Tests against `forward.sacp`, each tolerance stated: both head inputs; the detections after `select_nms_detections` at Darknet's thresholds, row for row. Skipped by name without the fixture or without the weights under `SA_REFERENCE_DATA_DIR`.
- Acceptance: the library's detections for the fixture image equal Darknet's.

**D16 to D20. Forward parity of the other five.** Root tests, one unit and one file each; all wait on D15's header and on their fixture. `haiku high`, since each is D15 with another model name.

| Unit | File | Also waits on | What it is first to exercise |
| --- | --- | --- | --- |
| D16 | `tests/integration/test_darknet_yolov3.cpp` | D13 | `shortcut`, three heads |
| D17 | `tests/integration/test_darknet_yolov4_tiny.cpp` | D14 | Grouped `route`, `scale_x_y`, the mask `1,2,3` |
| D18 | `tests/integration/test_darknet_yolov4.cpp` | D13 | `mish`, the SPP pools, 608 input, the v4 suppression kind |
| D19 | `tests/integration/test_darknet_yolov7_tiny.cpp` | none | `logistic` heads, the `Activated` coding, `diounms` |
| D20 | `tests/integration/test_darknet_yolov7.cpp` | none | `swish`, routes of six inputs, 640 input |

Acceptance of each: the library's detections for the fixture image equal Darknet's.

### 4.4 Training bricks

**T01. Training settings of a cfg.** `vision/importers`. Waits on D12, T02. `haiku high`. Files: `im/inc/darknet_training.hpp`, `im/source/darknet_training.cpp`, `im/tests/test_darknet_training.cpp`.

```cpp
struct DarknetSolver { double learning_rate = 0.001; double momentum = 0.9; double decay = 0.0005;
                       std::int64_t burn_in = 0; std::vector<std::int64_t> steps; std::vector<double> scales;
                       std::int64_t batch = 1; std::int64_t subdivisions = 1; std::int64_t max_batches = 0; };
struct DarknetTraining { DarknetSolver solver; std::vector<DetectAnchor::AnchorLossOptions> heads; };  // heads in file order
[[nodiscard]] Result<DarknetTraining> read_darknet_training(const DarknetConfig& config);
```

It reads `[net]` and the training keys of every `[yolo]` (SPEC 3.2) and leaves the translation of D12 alone. A `policy` other than `steps` is refused. Acceptance: each of the six key sets of SPEC 3.2, written out in the test, yields the options that table states.

**T02. The anchor assigner.** `vision/detect_anchor`. Waits on D06 and the REPORT. `opus low`.
- Files: `da/inc/anchor_loss_options.hpp`, `da/inc/anchor_targets.hpp`, `da/inc/anchor_assigner.hpp`, `da/source/anchor_loss_options.cpp`, `da/source/anchor_assigner.cpp`, `da/tests/test_anchor_assigner.cpp`.

```cpp
enum class BoxLoss : std::uint8_t { SquaredError, CompleteIou };
enum class ObjectnessTarget : std::uint8_t { One, Iou };   // Iou: objectness_smooth

struct AnchorLossOptions
{
    std::vector<AnchorSize> all_anchors;    // every anchor of the model, for the best-anchor match
    std::vector<std::int64_t> mask;         // the anchors of all_anchors this head owns
    float ignore_threshold = 0.7f;
    float match_threshold = 1.0f;           // iou_thresh; 1 assigns the best anchor only
    BoxLoss box_loss = BoxLoss::SquaredError;
    ObjectnessTarget objectness_target = ObjectnessTarget::One;
    float box_weight = 1.0f; float object_weight = 1.0f; float class_weight = 1.0f;
    float max_delta = 0.0f;                 // 0: no clip
};
[[nodiscard]] Status validate_anchor_loss(const AnchorLossOptions& loss, const AnchorHeadOptions& head);

struct AnchorTargets { std::vector<float> target; std::vector<float> logistic_weight; std::vector<float> square_weight; };

void assign_anchor_targets(SushiRuntime::span<const float> decoded_boxes, const Detect::GroundTruthBatch& truth,
                           const AnchorHeadShape& shape, const AnchorHeadOptions& head,
                           const AnchorLossOptions& loss, NetworkSize network, AnchorTargets& targets);
```

- `decoded_boxes` is what `decode_anchor_head` wrote for this head from the forward-only pass. The three leaves each have the head's shape; `logistic_weight` is zero where no cross-entropy term applies and `square_weight` where no squared-error term does, so the loss needs no slicing. This unit implements `SquaredError` with `One` (v3). The fields the later forms read exist from the start, so the structure does not change.
- Tests: one truth on a 2x2 grid with three anchors assigns the cell and anchor SPEC 4.5 names, with the stated targets and weight `2 - w * h`; a truth whose best anchor is outside the mask assigns nothing; a prediction above the ignore threshold has zero objectness weight; two truths in one cell; an empty truth list gives objectness terms only.
- Acceptance: for a hand-placed truth every non-zero element of the three leaves is the value SPEC 4.5 gives, as the REPORT corrects it.

**T03. The loss emitter, logit coding.** `vision/detect_anchor`. Waits on the REPORT. `opus low`. Files: `da/inc/anchor_loss.hpp`, `da/source/anchor_loss.cpp`, `da/tests/test_anchor_loss.cpp`, `tests/integration/test_anchor_loss_gradient.cpp`.

```cpp
struct AnchorLossLeaves { Graph::ValueId target; Graph::ValueId logistic_weight; Graph::ValueId square_weight; };
[[nodiscard]] AnchorLossLeaves declare_anchor_loss_leaves(const Graph::GraphBuilder& builder, Graph::ValueId head);
[[nodiscard]] Graph::ValueId emit_anchor_loss(const Graph::GraphBuilder& builder, Graph::ValueId head,
                                              const AnchorLossLeaves& leaves, double scale);
```

`sum(logistic_weight * (softplus(head) - head * target)) + 0.5 * sum(square_weight * (head - target)^2)`, times `scale` (1 / batch), reduced to one element by `reshape` to `[M, 1]` and `sum_rows`. No new operation. If the training graph refuses that value as its root (SPEC 2.6), the worker stops and reports; the fix is in the core tier and is the owner's decision. Acceptance: on an engine and a seeded input, the gradient at `head` equals `logistic_weight * (sigma(head) - target) + square_weight * (head - target)` to a stated tolerance and is exactly zero where both weights are zero.

**T04. The CIoU emitter.** `vision/detect`, for YOLO26 as well. `opus low`. Files: `de/inc/complete_iou_loss.hpp`, `de/source/complete_iou_loss.cpp`, `de/tests/test_detect_complete_iou_loss.cpp`, `tests/integration/test_complete_iou_gradient.cpp`.

```cpp
[[nodiscard]] Graph::ValueId emit_complete_iou(const Graph::GraphBuilder& builder, Graph::ValueId boxes,
                                               Graph::ValueId targets, Graph::ValueId row_mask);
```

`boxes` and `targets` are `[M, 4]` x1 y1 x2 y2; the result is `[M, 1]`. A row whose mask is zero is replaced by a unit box on both sides before the division, so it stays finite and its gradient is zero. Emitters: `sub`, `mul`, `div`, `minimum`, `maximum`, `atan`, `select`, `constant_like`. Acceptance: the values equal `Detect::elementwise_complete_iou` on seeded boxes to 1e-5, and the gradient at `boxes` equals a central finite difference on the unmasked rows and is zero on the masked ones.

**T05. `StepSchedule`.** `core/optim` (Q5). `haiku medium`. Files: `modules/core/optim/include/SushiAI/optim/step_schedule.hpp`, `modules/core/optim/source/step_schedule.cpp`, `modules/core/optim/tests/test_step_schedule.cpp`. `struct StepSchedule { double base_rate = 0.0; std::int64_t warmup_iterations = 0; double warmup_power = 4.0; std::vector<std::int64_t> steps; std::vector<double> scales; [[nodiscard]] double rate_at(std::int64_t iteration) const noexcept; };` and `[[nodiscard]] Status validate_step_schedule(const StepSchedule&)`. Acceptance: the rate at iterations 0, half the warm-up, the warm-up and one past each step equals SPEC 4.5's formula.

**T06. The reference converter, training.** `tools/reference`. Waits on D11 and Q2. `opus low`. Files: the three of D11. Adds `--training-dump DIR` and writes `training.sacp` (SPEC 5.3) with `learning_rate`, `momentum`, `decay`, `seen` and the batch size in its `context`. Acceptance: on a synthetic dump the tool writes `training.sacp` and its manifest entry.

**T07. The CIoU box term and the multi-anchor match.** `vision/detect_anchor`. Waits on T02, T03, T04. `opus low`. Files: `da/inc/anchor_box_loss.hpp`, `da/source/anchor_box_loss.cpp`, `da/source/anchor_assigner.cpp`, `da/tests/test_anchor_box_loss.cpp`, `da/tests/test_anchor_assigner_matches.cpp`. `[[nodiscard]] Graph::ValueId emit_anchor_box_loss(const Graph::GraphBuilder& builder, Graph::ValueId head, Graph::ValueId target_boxes, Graph::ValueId box_mask, const AnchorHeadShape& shape, const AnchorHeadOptions& head_options, NetworkSize network, double weight);` decodes the box entries in the graph by the head's coding (constants for the cell offsets and anchors) and calls `Detect::emit_complete_iou`. The assigner honours `match_threshold` and `max_delta` and, for `CompleteIou`, writes box targets as boxes and zero box weights in the other two leaves. Acceptance: the gradient at the head of a one-truth case equals the finite-difference gradient of `1 - CIoU`, in both codings.

**T08. The loss emitter, activated coding, and the IoU objectness target.** `vision/detect_anchor`. Waits on T07 and the REPORT. `opus low`. Files: `da/inc/anchor_activated_loss.hpp`, `da/source/anchor_activated_loss.cpp`, `da/source/anchor_assigner.cpp`, `da/tests/test_anchor_activated_loss.cpp`. `emit_anchor_activated_loss` has the signature of `emit_anchor_loss` and the same three leaves; its terms are the ones whose gradient at a logistic output is minus Darknet's delta, taken from the REPORT. The assigner gains `ObjectnessTarget::Iou`. Acceptance: on a seeded input the gradient at the head equals the expression the REPORT cites, and an assigned prediction's objectness target equals the REPORT's function of its IoU.

**T09. The anchor objective and its step.** `vision/training` (Q4). Waits on T05, T07, T08. `opus low`. Files: `tr/inc/anchor_objective.hpp`, `tr/inc/anchor_training_step.hpp`, their two sources, `tr/tests/test_anchor_objective.cpp`, `tests/integration/test_anchor_training_step.cpp`, `tr/README.md`.
- `class AnchorObjective final : public Train::Objective`, built from the `NetworkSize` and, per head, an `AnchorHeadOptions`, an `AnchorLossOptions` and the output name; it names no importer type. `declare` adds the leaves of every head; `loss` sums the emitter of each head's coding and, for `CompleteIou`, the box term.
- `class AnchorTrainingStep final`, owning a `Train::EvaluationStep` and a `Train::TrainingStep` over one `ModelState`: `run` does the forward-only pass, decodes each head, calls `assign_anchor_targets`, feeds the leaves and replays. Whether two steps can share one state was not checked; if they cannot, the worker stops and reports.
- Acceptance: a two-head model trains one step, and the leaves fed equal what `assign_anchor_targets` returns for the forward-only predictions.

### 4.5 Training parity

**T10 to T15.** Root tests, one unit and one file each, `tests/integration/test_darknet_<model>_training.cpp`, in the order of the models: T10 `yolov3_tiny`, T11 `yolov3`, T12 `yolov4_tiny`, T13 `yolov4`, T14 `yolov7_tiny`, T15 `yolov7`. All wait on T01, T06, T09, D15's header and the model's `training.sacp`. T10 is `opus low`; the rest are `haiku high`. Each compares, with stated tolerances, the assigner's leaves against the pattern and values of Darknet's `delta`, the loss where SPEC 5.3 allows it, and the gradient at each head's input against `-delta / batch`; then it runs 200 steps on the one batch. Acceptance of each: the gradient at every head equals Darknet's, and the loss on the one batch falls below a tenth of its first value.

## 5. Waves and builds

Files within a wave are disjoint. A build is the owner's.

| Wave | Units | Waits on | Closes with |
| --- | --- | --- | --- |
| 0 | O1 | Q4, Q5, Q7 | nothing |
| 1 | D01, D02, D03, D04, D06, D07, D08, D09, D10, D11, T04, T05 | wave 0; Q6; the REPORT for D06, D07, D11 | syntax checks |
| 2 | D05, D12, T02, T03, T06 | wave 1; Q2 for T06 | syntax checks |
| 3 | D13, D14, D15, T01, T07 | wave 2; the `yolov3_tiny` fixture | **B1**: both suites, then D15 with the fixture. YOLOv3-tiny runs |
| 4 | D16 to D20, T08, T09 | B1; the five other fixtures | **B2**. The six forward passes are done |
| 5 | T10 to T15 | B2; the six `training.sacp` | **B3**. The six models train |

Thirty-five units: twenty forward (D01 to D20) and fifteen training (T01 to T15). Five waves after the skeleton, three builds. B2 can be folded into B3 if the owner prefers two builds; the cost is that a forward mismatch in one of five models and a training mismatch surface together.

One editor per wave on a shared file: `darknet_layer_table.cpp` (D12 in wave 2, D13 in 3); `darknet_layer_route.cpp` (D12, D14); `anchor_assigner.cpp` (T02 in wave 2, T07 in 3, T08 in 4); the converter (D11 in wave 1, T06 in 2); `test_node_registry.cpp` and `layer_registry.cpp` (D05 only); `detections_nms.*` (D07 only).

## 6. Not planned here

- YOLOv1 and YOLOv2.
- A full training run on COCO and Darknet's data loader with its augmentation (Q3).
- A program under `applications/` that takes an image file and prints detections. The parity tests start from Darknet's input tensor; an image path needs Darknet's float resize or an accepted difference from it.
- A fused `K_MISH`. `yolov3-spp`, the scaled YOLOv4 cfgs and `yolov7x`.

## 7. Not verified

- Nothing was built, run or downloaded, and no Darknet source was read. Every **(REPORT)** mark of the SPEC is from memory until `REPORT.md` exists. The costliest if wrong: the `new_coords` decode, the maxpool offset rule, the two score gates, the epsilon form, what `greedynms` and `diounms` select in `detector test`, and all of SPEC 4.5, where the claim that the deltas are the gradient of a cross-entropy on logits carries the v3 loss.
- The SHA-256 values were copied from the ruling and not recomputed. The file sizes were read from the directory.
- Whether the training graph accepts a one-element value other than `NLL_LOSS` as its root (T03), whether an evaluation step and a training step can share one `ModelState` (T09), and whether `vision/detect` already depends on `graph` (T04).
- Whether `Tensor::data()` is host-writable on every device, and the exact state names `build_model` produces.
- Whether every 2/2 pool of the six cfgs meets an even extent.
- Line numbers in the SPEC that this revision did not reread are the ones of the first draft: sections 2.2, 2.3 and 2.5.
- The Darknet licence and the licence of the fixture image.
- Rule 6 of the dispatch block had nothing to check: this task revised two Markdown files.
