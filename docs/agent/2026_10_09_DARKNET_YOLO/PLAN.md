# Darknet-era YOLO: plan

**Status:** Draft, revised 2026-10-09 to the owner's rulings of that day and to `REPORT.md`; waiting on one question (section 1) and, for the units section 5 names, on the second Darknet pass. Nothing is written or built. Requirements, behaviour and formats are in `SPEC.md` beside this file; section numbers without a file refer to it.

## 1. Decisions

Ruled by the owner on 2026-10-09:

| # | Subject | Ruling |
| --- | --- | --- |
| R1 | Darknet's text and the references | No excerpt from AlexeyAB: no patch, code, comment or line-by-line port enters the repository. The implementation is written from the behaviour the REPORT describes. The reference tensors live in `D:\Projects\.sushisystems\models\darknet\references\<model>[_thresh_0p01]\` as raw `float32` `.bin` plus a `.json` per tensor, with the weights and cfg files beside them in `...\models\darknet\`. A second Darknet pass adds `yolov3`, `yolov4-tiny`, `yolov4`, every layer of `yolov3-tiny`, decoded boxes and scores before the threshold, the CIoU gradient in full, perhaps one batch-statistics training step |
| R2 | Home of the anchor code | Inside `modules/vision/detect`, one module and one target in three parts: the root `Vision::Detect`, `free/` `Vision::Detect::Free`, `anchor/` `Vision::Detect::Anchor`. No module `vision/detect_anchor`; `cmake/SushiAILayers.cmake` gains no row for it. `vision/importers` stays its own module |
| R3 | Home of training | A new module `vision/training`, depending on `train` and `vision/detect`, holds the detection objective, the host stage and the training step; YOLO26's objective lands beside the anchor one. `vision/detect` keeps depending on `core` only |
| R4 | Automatic anchoring | Its own SPEC section (5) and its own units (A01 to A07): derived anchors, a chosen count, a computed head assignment, a fit report with a proposal that replaces nothing, fractional storage, derived thresholds. Imported models keep the anchors of their cfg |
| R5 | Scope | v3, v4, v7 and their tiny variants, with training: assignment parity, cost parity, gradient parity, overfitting one batch; no COCO run. A parity test that needs outside files reads `SA_REFERENCE_DATA_DIR` and is reported skipped without it |

Earlier rulings of the same day that stand: YOLOv1 and YOLOv2 are out; the importer emits a `Config::ModelSpec`; the suppression criterion is a field of `NmsConfig`; a reference is produced by running Darknet and never typed.

Decided by the engineer in this revision. Each closes a question of the first draft; the owner can overturn any, at the cost in the last column.

| # | Decision | Alternative refused | Units it shapes |
| --- | --- | --- | --- |
| E1 | Forward parity is measured on a folded model: the importer folds batch norm into the convolution in double, as Darknet does before an inference forward. `Separate` stays the default and the training form (SPEC 4.3) | A tolerance over an unfolded `BatchNorm2d`, which grows with depth and would be chosen per model | F10, F11, F14 to F19 |
| E2 | The library computes Darknet's delta on the host and the graph's root is `-(1 / batch) * sum(delta * head)` (SPEC 4.6) | Closed-form IR losses. That reverses the first draft's "loss as IR" and drops its IR CIoU emitter: the clip, the step-3 division and the `objectness_smooth` rules are not the gradient of one expression | T01 to T05, T08, T09 |
| E3 | The parity tests read the `.bin` and `.json` dumps directly. No converter tool, no SACP fixture and no manifest under `tests/fixtures/darknet/` | Converting the dumps into fixtures the repository ignores, which adds a tool and a second copy | F14 |
| E4 | `SA_REFERENCE_DATA_DIR` is an environment variable read at run time | A compile definition, which bakes a machine's path into a binary | F14 |
| E5 | The schedule is `Optim::StepSchedule` in `core/optim` and names nothing of Darknet | A function in `vision/importers` no second family could reach | T07 |
| E6 | `MaxPool2d` gains `trailing_pad_h` and `trailing_pad_w`, row key `trailing_padding`; `ConvGeometry` is untouched | Bottom and right pads in `ConvGeometry`, which edits every emitter | F01 |
| E7 | `vision/importers` depends on `vision/detect`: a translated model carries an `AnchorLayout`, head options and the suppression criterion | The importer returns raw keys and a program builds the options | F11, T06 |
| E8 | The plan runs in the waves of section 5 | Inline, in unit order | Section 5 |

Open, one:

| # | Question | Recommendation | Alternative |
| --- | --- | --- | --- |
| Q1 | The parity tests read files outside the repository: weights, cfgs and reference tensors under `SA_REFERENCE_DATA_DIR`. That is an exception to rule 4 of the `testing` skill, and the dumps carry no `format_version` | Grant it for the Darknet parity tests alone, each skipped by name when a file is missing and none counted as passed | Commit the reference tensors: 14 MB for `yolov7`'s heads and input, the weights still outside |

## 2. Modules

| Module | Tier | Holds | Depends on |
| --- | --- | --- | --- |
| `vision/detect`, part `anchor/` (new part) | vision | Anchor sizes and layouts, the automatic anchoring, head decode, the host delta and cost | `core`; EXTERNAL_PUBLIC `SushiRuntime::SushiRuntime`. The anchor headers return `Status` and `Result`, so the edge to `core` becomes `PUBLIC_DEPENDS` |
| `vision/detect`, root (grown) | vision | The suppression criterion; the host CIoU gradient | same |
| `vision/importers` (new) | vision | The `.cfg` reader, the `.weights` reader, the fold, the translation, the training settings | PUBLIC `core`, `config`, `nn`, `vision/detect` |
| `vision/training` (new) | vision | The delta loss emitter, the anchor objective, its two-pass step | PUBLIC `train`, `vision/detect`, and `graph`, `nn`, `core` as `train` already names them |
| `layers/nn` (grown) | layers | Trailing padding on `MaxPool2d`; `Add`; `ChannelSlice`; `Mish`; their rows | unchanged |
| `core/optim` (grown) | core | `StepSchedule` | unchanged |

## 3. Orchestrator steps

Workers do not write build files (dispatch rule 3).

**O1, the skeleton, before wave 1.**

- Settle where `PredictionShape` lives: `detections_nms.hpp` and `detections_topk.hpp` in the root still include it by its old path (SPEC 2.4).
- `cmake/SushiAILayers.cmake` and `tools/layering/check_layering.py`: rows `vision/importers vision` and `vision/training vision`. No row for the anchor code.
- `modules/vision/CMakeLists.txt`: `add_subdirectory(importers)` and `add_subdirectory(training)` after `detect`, before `registration`.
- `modules/vision/importers/CMakeLists.txt` and `modules/vision/training/CMakeLists.txt`: one `sushiai_add_module(NAME ... TIER vision ...)` each with the dependencies of section 2, and a README stub the first worker replaces.
- `modules/vision/detect/CMakeLists.txt`: `PRIVATE_DEPENDS core` becomes `PUBLIC_DEPENDS core`. The folders `include/SushiAI/vision/detect/anchor/` and `source/anchor/` appear with their first files.
- `modules/vision/umbrella/CMakeLists.txt`: `vision/importers` and `vision/training` join `PUBLIC_DEPENDS`.

**After each wave.** What the orchestrator appends, per build file:

| File | Appended |
| --- | --- |
| `modules/layers/nn/CMakeLists.txt` | `source/{add,channel_slice,mish}.cpp`; `tests/test_{add,channel_slice,mish}.cpp` |
| `modules/core/optim/CMakeLists.txt` | `source/step_schedule.cpp`; `tests/test_step_schedule.cpp` |
| `modules/vision/detect/CMakeLists.txt` | Every `source/anchor/*.cpp` and `tests/test_detect_anchor_*.cpp` of the F06, A and T units; `source/suppression_criterion.cpp`, `source/iou_gradient.cpp` and their two tests |
| `modules/vision/importers/CMakeLists.txt` | The sources and tests of F08 to F13 and T06 |
| `modules/vision/training/CMakeLists.txt` | The sources and tests of T08, T09 |
| `tests/CMakeLists.txt` | To `sushiai_tests`: `integration/test_max_pool_trailing.cpp`, `integration/test_mish_values.cpp`. To `sushiai_vision_tests`: `common/darknet_reference.cpp` and every `integration/test_darknet_*.cpp`, `test_head_delta_loss.cpp`, `test_anchor_training_step.cpp` |
| `modules/vision/umbrella/include/SushiAI/vision/vision.h` | One include per new public header of `detect/`, `detect/anchor/`, `importers/`, `training/` |
| `layers` umbrella header | `add.hpp`, `channel_slice.hpp`, `mish.hpp`, `step_schedule.hpp` |

No `.gitignore` line and no test definition is added (E3, E4).

**At close.** The changelog lines, `PROGRAMME.md` stage 6, `REMAINING_WORK.md`, `docs/design/DARKNET_YOLO.md` drawn from `SPEC.md`, `DOMAIN_MODULES.md` for the two modules and the three parts of `detect`, and the manual pages the work makes false.

## 4. Units

Forty-four units: F01 to F21 forward, A01 to A07 automatic anchoring, T01 to T16 training. Paths are relative to the repository root. `nn/` is `modules/layers/nn/`, `de/` is `modules/vision/detect/`, `im/` is `modules/vision/importers/`, `tr/` is `modules/vision/training/`; `inc/` inside each is its `include/SushiAI/...` folder, so `de/inc/anchor/x.hpp` is `modules/vision/detect/include/SushiAI/vision/detect/anchor/x.hpp`. Anchor tests are `de/tests/test_detect_anchor_<subject>.cpp`. Dispatch: `haiku medium` (spec-complete, 1 or 2 files), `haiku high` (spec-complete, more files), `opus low` (integration or judgment). Every type below is a value type; a function that can be handed a wrong input returns `Status` or `Result`, and nothing throws across a public surface.

### 4.1 Layers

**F01. `MaxPool2d` trailing padding and its row key.** `layers/nn`. `haiku high`.
- Files: `nn/inc/max_pool2d.hpp`, `nn/source/max_pool2d.cpp`, `nn/source/layer_registry.cpp`, `nn/tests/test_max_pool2d.cpp`, `nn/tests/test_node_registry.cpp`, `tests/integration/test_max_pool_trailing.cpp`.
- Surface: two fields of `struct MaxPool2d`, `int64_t trailing_pad_h = 0;` and `int64_t trailing_pad_w = 0;`, rows below and columns right beyond the symmetric pad. `output_shape` adds them; `forward` calls `GraphBuilder::maxpool2d_with_extents`. Row `maxpool2d` accepts `trailing_padding`, an integer or `[h, w]`, default 0; `avgpool2d` refuses it. A negative value and an all-padding window fail as an illegal window does today.
- Tests: 13x13, 2/1, trailing 1 gives 13x13 and trailing 0 gives 12x12; 13x13, 2/2, trailing 1 gives 7x7; 416 at 2/2 gives 208; on an engine an all-negative input keeps the true maximum in the last row and column.
- Acceptance: the last row and column of a 2/1 pool over a known 3x3 input equal the maxima of the truncated windows.

**F02. `Add` node.** `layers/nn`. `haiku high`. Files: `nn/inc/add.hpp`, `nn/source/add.cpp`, `nn/tests/test_add.cpp`. Shaped as `Concat` is, over `GraphBuilder::add`:

```cpp
class Add final : public Node
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<Add>> create(ShapeSpan inputs);
        [[nodiscard]] Status forward(const Trace& trace, ValueSpan inputs, std::vector<Graph::ValueId>& outputs) override;
        [[nodiscard]] std::size_t input_count() const noexcept override;
        void walk_state(std::string_view prefix, const StateVisitor& visitor) override;
        [[nodiscard]] SushiRuntime::span<const Port> ports() const noexcept override;
};
[[nodiscard]] Result<std::unique_ptr<Node>> make_add_node(ShapeSpan inputs, const Config::Value& attributes, std::string_view context);
```

Tests: one, three and unequal inputs and any key are `InvalidDescription`. Acceptance: two `[H, W, C]` inputs trace to one `K_ADD` of that shape.

**F03. `ChannelSlice` layer.** `layers/nn`. `haiku high`. Files: `nn/inc/channel_slice.hpp`, `nn/source/channel_slice.cpp`, `nn/tests/test_channel_slice.cpp`. Shaped as `Upsample2d` is: `struct ChannelSlice { int64_t offset = 0; int64_t count = 0; static constexpr auto fields(); Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x) const; Shape output_shape(const Shape& input) const; };`. Acceptance: `{offset 32, count 32}` over `[H, W, 64]` reports `[H, W, 32]` and traces one `K_SLICE_CHANNELS`; a range past the channels fails as an illegal shape does in `Upsample2d`.

**F04. `Mish` layer.** `layers/nn`. `haiku high`. Files: `nn/inc/mish.hpp`, `nn/source/mish.cpp`, `nn/tests/test_mish.cpp`, `tests/integration/test_mish_values.cpp`. Shaped as `SiLU` is, emitting `mul(x, tanh(softplus(x)))`. The engine test also evaluates Darknet's form with the cut-off at 20 (SPEC 4.1) in double and states the largest difference over -30 to 30. Acceptance: the values at -20, -1, 0, 1, 20 equal `x * tanh(log1p(exp(x)))` computed in double, 1e-6 relative.

**F05. Rows `add`, `slice_channels`, `mish`.** `layers/nn`. Waits on F01 to F04. `haiku medium`. Files: `nn/source/layer_registry.cpp`, `nn/tests/test_node_registry.cpp`. Acceptance: each row builds through `make_node` and refuses an unknown key.

### 4.2 Detection bricks

**F06. Anchor types and the head decode.** `vision/detect`, `anchor/`. `opus low`.
- Files: `de/inc/anchor/{anchor_size,anchor_layout,head_options,head_decode}.hpp`, the four `de/source/anchor/*.cpp` of the same names, `de/tests/test_detect_anchor_decode.cpp`, `de/tests/test_detect_anchor_layout.cpp`, the anchor section of `de/README.md`.
- `AnchorSize`, `shape_iou`, `AnchorLayout`, `validate_anchor_layout` as SPEC 5.5 gives them, and:

```cpp
enum class HeadCoding : std::uint8_t { Logit, Activated };   // new_coords = 0, 1
struct HeadOptions { std::int64_t num_classes = 0; HeadCoding coding = HeadCoding::Logit; float scale_xy = 1.0f; };
struct HeadShape { std::int64_t batch = 0; std::int64_t height = 0; std::int64_t width = 0; };
struct NetworkSize { std::int64_t height = 0; std::int64_t width = 0; };
[[nodiscard]] Status validate_head(const HeadOptions& options, const AnchorLayout& layout, std::int64_t head_index);
[[nodiscard]] std::int64_t count_head_channels(const HeadOptions& options, const AnchorLayout& layout, std::int64_t head_index) noexcept;
[[nodiscard]] std::int64_t count_head_rows(const HeadShape& shape, const AnchorLayout& layout, std::int64_t head_index) noexcept;
void decode_head(SushiRuntime::span<const float> head, const HeadShape& shape, const HeadOptions& options,
                 const AnchorLayout& layout, std::int64_t head_index,
                 SushiRuntime::span<float> boxes, SushiRuntime::span<float> scores);
```

- `head` is the model's published output `[B, H, W, A * (5 + C)]`. `boxes` receives `[B, H * W * A, 4]` centre and size as fractions of the network input, `scores` `[B, H * W * A, C]`, row `(row * W + col) * A + a`, by SPEC 4.4. No pixel and no threshold enters.
- Tests: zero logits on a 1x1 grid give the cell centre, the anchor size and scores of 0.25; `scale_xy` 2.0 moves a saturated offset half a cell past the edge; the `Activated` size at inputs 0, 0.5 and 1 is 0, the anchor and four anchors; the row order on a 2x3 grid with two anchors; one layout on a 13 and a 19 grid gives the same fractional size (AR5); `shape_iou` of a size with itself, with a contained one and with a zero area; the layout refusals (an index out of range, an empty head).
- Acceptance: every formula of both codings is pinned by a case whose expected value follows from inputs at zero or at saturation, to 1e-6 absolute.

**F07. The suppression criterion.** `vision/detect`, root. `haiku high`. Files: `de/inc/suppression_criterion.hpp`, `de/source/suppression_criterion.cpp`, `de/inc/detections_nms.hpp`, `de/source/detections_nms.cpp`, `de/tests/test_detect_suppression_criterion.cpp`.

```cpp
enum class SuppressionCriterion : std::uint8_t { Iou, DistanceIou };
[[nodiscard]] float measure_suppression(SushiRuntime::span<const float> box1, SushiRuntime::span<const float> box2,
                                        SuppressionCriterion criterion, float distance_exponent) noexcept;
void suppress_overlaps(SushiRuntime::span<const float> boxes, SushiRuntime::span<const float> scores, float threshold,
                       SuppressionCriterion criterion, float distance_exponent, std::vector<std::int64_t>& kept);
```

`DistanceIou` is `IoU - (d / c)^distance_exponent`, and `IoU` when `c` is 0 (SPEC 4.5). `NmsConfig` gains `SuppressionCriterion criterion = SuppressionCriterion::Iou` and `float distance_exponent = 0.6f`; `select_nms_detections` passes them on; the existing four-argument `suppress_overlaps` stays. Acceptance: with the default criterion every existing `detect` test is unchanged, and of two boxes with equal IoU against a kept box the one with the farther centre survives.

### 4.3 Importer

**F08. The `.cfg` reader.** `vision/importers`. `haiku high`. Files: `im/inc/darknet_config.hpp`, `im/source/darknet_config.cpp`, `im/tests/test_darknet_config.cpp`.

```cpp
struct DarknetOption { std::string key; std::string value; std::int64_t line = 0; };
struct DarknetSection { std::string type; std::int64_t line = 0; std::vector<DarknetOption> options; };
struct DarknetConfig { DarknetSection net; std::vector<DarknetSection> layers; };   // index = Darknet layer index
[[nodiscard]] Result<DarknetConfig> parse_darknet_config(std::string_view text, std::string_view origin);
[[nodiscard]] Result<DarknetConfig> load_darknet_config(const std::filesystem::path& path);
```

The grammar of SPEC 4.1 and nothing about what a section means. Tests: both comment forms, blank lines, spaces around `=` and inside lists, CRLF; a key before any section, a line without `=`, a repeated key and a first section other than `[net]` or `[network]` each fail with `InvalidDescription` naming origin and line; a missing file gives `FileNotFound`. Acceptance: a text with every lexical form of SPEC 4.1 parses to the sections, keys and lines the test states.

**F09. The `.weights` reader.** `vision/importers`. `haiku high`. Files: `im/inc/darknet_weights.hpp`, `im/source/darknet_weights.cpp`, `im/tests/test_darknet_weights.cpp`.

```cpp
struct DarknetWeightsHeader { std::int32_t major = 0; std::int32_t minor = 0; std::int32_t revision = 0; std::uint64_t seen = 0; };
struct DarknetConvolutionShape { std::int64_t layer = 0; std::int64_t filters = 0; std::int64_t channels = 0; std::int64_t size = 0; bool normalised = false; };
struct DarknetConvolution { std::vector<float> biases; std::vector<float> scales; std::vector<float> rolling_mean;
                            std::vector<float> rolling_variance; std::vector<float> weights; };   // weights [out, in, kh, kw]
struct DarknetWeights { DarknetWeightsHeader header; std::vector<DarknetConvolution> convolutions; };
[[nodiscard]] Result<DarknetWeightsHeader> read_darknet_header(std::istream& stream);
[[nodiscard]] Result<DarknetWeights> read_darknet_weights(const std::filesystem::path& path,
                                                          SushiRuntime::span<const DarknetConvolutionShape> shapes);
```

The reader knows the file and not the model: the shapes arrive in file order. Bytes are assembled field by field, little-endian. Strict both ways (SPEC 4.2). Tests, on files the test writes into the build tree: both header forms, with `seen` as `uint32` below version 0.2; a `major` of 1001 is `Unsupported`; one float short and one float over are `InvalidFile` naming the layer and both byte counts; a normalised and a plain convolution each land in the right vectors. Acceptance: a two-convolution file whose floats are their own indices is read into vectors holding exactly those indices.

**F10. The fold and the store.** `vision/importers`. Waits on F09. `haiku high`. Files: `im/inc/darknet_fold.hpp`, `im/source/darknet_fold.cpp`, `im/inc/darknet_store.hpp`, `im/source/darknet_store.cpp`, `im/tests/test_darknet_fold.cpp`, `tests/integration/test_darknet_store.cpp`.

```cpp
void fold_darknet_normalisation(DarknetConvolution& convolution);
struct DarknetWeightTarget { std::string weight; std::string bias; std::string scale; std::string mean; std::string variance; };   // empty: absent
[[nodiscard]] Status store_darknet_weights(const DarknetWeights& weights, SushiRuntime::span<const DarknetConvolutionShape> shapes,
                                           SushiRuntime::span<const DarknetWeightTarget> targets, NN::ModelState& state);
```

The fold is SPEC 4.3's arithmetic in double, rounded once to `float32`; it leaves `scales`, `rolling_mean` and `rolling_variance` empty and the new bias in `biases`. The store permutes `[out, in, kh, kw]` to `[out, kh, kw, in]` and copies each vector to the state entry its target names; a name the state lacks or a shape that differs is `Incompatible` naming the layer. Tests: the fold of a one-filter convolution against the formula evaluated in `long double`; a folded and an unfolded convolution applied to one input on the host agree to 1e-5; the permutation on index-valued weights; the two refusals. Acceptance: a weight written as `[out, in, kh, kw]` index values arrives in the state as `[out, kh, kw, in]` with every element at the permuted position.

**F11. The translation.** `vision/importers`. Waits on F06, F07, F08, F10. `opus low`.
- Files: `im/inc/darknet_model.hpp`, `im/source/darknet_model.cpp`, `im/source/darknet_layer_table.{hpp,cpp}`, `im/source/darknet_layer_{convolutional,maxpool,upsample,route,yolo}.cpp`, `im/tests/test_darknet_model.cpp`, `im/README.md`.

```cpp
enum class DarknetNormalisation : std::uint8_t { Separate, Folded };
struct DarknetTranslation { DarknetNormalisation normalisation = DarknetNormalisation::Separate; };
struct DarknetHead { std::string output; std::int64_t layer = 0; Detect::Anchor::HeadOptions options; };
struct DarknetModel
{
    Config::ModelSpec spec;
    Detect::Anchor::NetworkSize network;
    Detect::Anchor::AnchorLayout layout;          // the cfg's anchors as fractions, its masks as head_anchors
    std::vector<DarknetHead> heads;               // file order; output is "head.<k>"
    Detect::SuppressionCriterion criterion = Detect::SuppressionCriterion::Iou;
    float distance_exponent = 0.6f;
    std::vector<DarknetConvolutionShape> convolutions;
    std::vector<DarknetWeightTarget> targets;
    std::vector<std::int64_t> row_of_layer;       // the row that holds Darknet layer i's output
};
[[nodiscard]] Result<DarknetModel> translate_darknet_model(const DarknetConfig& config, const DarknetTranslation& translation);
```

- One private translator per section type, found through a table of `{type, function}`: a new section type is a new source file and one row. A `[convolutional]` becomes `conv2d`, then `batchnorm2d` when normalised and `Separate`, then the activation row from a table of names (`leaky` to `leaky_relu` with slope 0.1, `mish`, `swish` to `silu`, `logistic` to `sigmoid`, `linear` to none). A `[maxpool]` becomes `maxpool2d` with `padding = (size - 1) / 2` and the `trailing_padding` SPEC 2.2 implies. The `yolo` translator divides the anchors by `width` and `height`, the only place a pixel anchor exists, copies the mask, reads `scale_x_y` and `new_coords`, takes the criterion from the last `[yolo]` (0.6 for `greedynms`, `beta_nms` for `diounms`), accepts the training keys without reading them and refuses any other. An ungrouped `route` only; F13 adds the grouped one.
- Tests, on the v3-tiny cfg written out in the test as input data: the row count, the `from` of each `route`, two outputs and their options; layer 11 has `trailing_padding` 1; `build_model` over `register_layers` accepts both forms and reports heads `[13, 13, 255]` and `[26, 26, 255]`; every target name is an entry of a `ModelState` created from the traced model, in both forms; the shapes' float count is 8 858 734; a v7 head translates to `Activated`, `scale_xy` 2.0, `DistanceIou` and exponent 0.6; a `greedynms` head with `beta_nms=0.3` still gives 0.6; an unknown section and an unknown key fail naming the line.
- Acceptance: the translated v3-tiny spec builds in both forms, and every target name is an entry of its state.

**F12. `shortcut` translator.** `vision/importers`. Waits on F11, F05. `haiku medium`. Files: `im/source/darknet_layer_shortcut.cpp`, `im/source/darknet_layer_table.cpp`, `im/tests/test_darknet_shortcut.cpp`. An `add` row reading the layer before and layer `from`; the refusals of SPEC 4.1. Acceptance: a three-layer cfg translates to an `add` row whose `from` names the two rows, for a negative and an absolute `from`.

**F13. Grouped `route`.** `vision/importers`. Waits on F11, F05. `haiku medium`. Files: `im/source/darknet_layer_route.cpp`, `im/tests/test_darknet_route_groups.cpp`. With `groups > 1` each source gets a `slice_channels` row before the join. Acceptance: `layers=-1, groups=2, group_id=1` over 64 channels gives one row with offset 32 and count 32; a count `groups` does not divide is refused.

### 4.4 Forward parity

**F14. The reference loader and YOLOv3-tiny forward parity.** Root tests. Waits on F01, F06, F11. `opus low`.
- Files: `tests/common/darknet_reference.hpp`, `tests/common/darknet_reference.cpp`, `tests/integration/test_darknet_yolov3_tiny_forward.cpp`.
- The loader: `find_darknet_reference(model, name)` returns the paths or the reason for a skip; `load_darknet_tensor` reads the `.json` through the `config` module's JSON reader, checks `dtype`, `byte_order`, the shape against the byte count, and permutes `CHW` to `HWC`. It is test code and no library module includes it.
- Tests, each with the tolerance of SPEC 6.3 written in it, against `yolov3_tiny` and `yolov3_tiny_thresh_0p01`: the two published heads of the `Folded` model against `layer_015_yolo_input` and `layer_022_yolo_input`; the host `[yolo]` output of 4.4 against `layer_016_yolo` and `layer_023_yolo`; the 16 detection rows at `thresh` 0.01 and 0.45, both sides sorted by class then score, with any pair within 1e-6 of a threshold reported by name; the largest difference between the `Folded` and the `Separate` model at each head, stated and bounded.
- Acceptance: with the files present, the library's detections for the reference input equal Darknet's 16 rows; without them every test is reported skipped.

**F15 to F19. Forward parity of the other five.** Root tests, one unit and one file each, `tests/integration/test_darknet_<model>_forward.cpp`; each is F14 with another model. `haiku high`.

| Unit | Model | Also waits on | First to exercise | References |
| --- | --- | --- | --- | --- |
| F15 | `yolov3` | F12, F14 | `shortcut`, three heads, 72 folded convolutions | second pass |
| F16 | `yolov4_tiny` | F13, F14 | Grouped `route`, `scale_x_y`, the mask `1,2,3`, `greedynms` | second pass |
| F17 | `yolov4` | F12, F14 | `mish`, the SPP pools, 608 input | second pass |
| F18 | `yolov7_tiny` | F07, F14 | `logistic` heads, the `Activated` coding, `diounms`, 11 rows | on disk |
| F19 | `yolov7` | F07, F14 | `swish`, routes of six inputs, 640 input, 7 rows | on disk (`_thresh_0p01` only) |

Acceptance of each: the library's detections for the reference input equal Darknet's.

**F20. YOLOv3-tiny layer by layer.** Root test. Waits on F14 and the second pass. `haiku high`. File: `tests/integration/test_darknet_yolov3_tiny_layers.cpp`. Publishes every row `row_of_layer` names and compares it with the second pass's per-layer dumps, reporting the first layer above tolerance. Acceptance: all 24 layers within the tolerance of SPEC 6.3.

**F21. Decode parity before the threshold.** Root test. Waits on F14 and the second pass. `haiku high`. File: `tests/integration/test_darknet_decode.cpp`, parameterised by model. Compares `decode_head` on each dumped head with the pre-threshold boxes and scores, under the names the revised REPORT gives. Acceptance: every box and score of every model with that dump within 1e-6 absolute.

### 4.5 Automatic anchoring (`vision/detect`, `anchor/`)

Signatures in full are in SPEC 5.5; the rules in SPEC 5.2 to 5.4; the distributions in SPEC 5.6. All wait on F06.

| Unit | Function | Files (`de/inc/anchor/`, `de/source/anchor/`, test subject) | Waits on | Dispatch | Tests pin | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| A01 | `measure_anchor_fit` | `anchor_fit.{hpp,cpp}`, `fit` | F06 | `haiku high` | One box per anchor gives 1.0 everywhere; boxes at half an anchor's sides give 0.25 and a recall of 1 at `recall_iou` 0.25, 0 at 0.3; the v4-tiny case of SPEC 5.6 gives `unowned_share` 1/6; an empty box list is `Incompatible` | The four numbers of a hand-built case equal the values the test computes box by box, 1e-9 |
| A02 | `assign_anchors_to_heads` | `head_assignment.{hpp,cpp}`, `head_assignment` | F06 | `haiku medium` | Nine anchors and strides 32, 16, 8 give `6,7,8`, `3,4,5`, `0,1,2`; six and 32, 16 give `3,4,5`, `0,1,2`; seven anchors over two heads; unsorted anchors; `per_head` that does not sum to the count is `Incompatible` | The masks of the five cfgs SPEC 5.3 names are reproduced from their anchors and strides |
| A03 | `cluster_anchors` | `anchor_clustering.{hpp,cpp}`, private `seeded_sequence.{hpp,cpp}`, `clustering` | A01 | `opus low` | Ladder: every centre within 5 %, `F > 0.95`, `mean_best_iou >= mean_best_iou_before_refinement`; Outliers; Degenerate; two runs bit-identical; another seed still within 5 %; the first eight outputs of the SplitMix64 against the published sequence | On Ladder with `K = 6` and the default seed every centre is recovered within 5 % |
| A04 | `derive_assignment_thresholds` | `assignment_thresholds.{hpp,cpp}`, `thresholds` | A01 | `haiku medium` | A fit with tenth percentile 0.35 and mean 0.62 gives 0.35 and 0.62; both clamps at each end | The formula of SPEC 5.4 on four fits |
| A05 | `choose_anchor_count` | `anchor_count.{hpp,cpp}`, `count` | A03 | `haiku high` | Ladder `K = 6`, two heads: 3 per head and four trials; Blob: 1; `max_per_head` 2 stops at 2; `min_per_head > max_per_head` is `InvalidDescription` | Ladder `K = 6` with two heads returns 3 |
| A06 | `review_anchors` | `anchor_review.{hpp,cpp}`, `review` | A01, A02, A03 | `haiku high` | A layout fitted to Ladder, on Ladder: `Keep`; on Shifted: `Propose`, with `current` and `proposed_fit` equal to `measure_anchor_fit` called directly; the proposal has the input's counts per head; the input layout compares equal before and after | Shifted yields `Propose` and leaves the input untouched |
| A07 | `plan_anchors` | `anchor_plan.{hpp,cpp}`, `plan`; the anchoring section of `de/README.md` | A02, A04, A05 | `opus low` | With `per_head` 0 the result equals the composition of A05, A02, A01, A04 called by hand; with `per_head` 3 no count is tried and `trials` is empty; strides in any order | On Ladder `K = 6` and strides 16, 32 the plan has two heads of three anchors, a recall of 1 and the thresholds of A04 |

### 4.6 Training bricks

**T01. Training options and the delta, squared-error boxes.** `vision/detect`, `anchor/`. Waits on F06. `opus low`.
- Files: `de/inc/anchor/head_training_options.hpp`, `de/inc/anchor/head_delta.hpp`, `de/source/anchor/head_training_options.cpp`, `de/source/anchor/head_delta.cpp`, private `de/source/anchor/box_delta.hpp` and `box_delta_squared.cpp`, `de/tests/test_detect_anchor_delta.cpp`.

```cpp
enum class BoxLoss : std::uint8_t { SquaredError, CompleteIou };
enum class ObjectnessTarget : std::uint8_t { One, BestIou };   // BestIou: objectness_smooth
struct HeadTrainingOptions
{
    AssignmentThresholds thresholds;      // match: iou_thresh; ignore: ignore_thresh
    BoxLoss box_loss = BoxLoss::SquaredError;
    ObjectnessTarget objectness_target = ObjectnessTarget::One;
    float box_weight = 0.75f;             // iou_normalizer
    float object_weight = 1.0f;           // obj_normalizer
    float class_weight = 1.0f;            // cls_normalizer
    float max_delta = 0.0f;               // 0: no clip
    float class_gate = 0.25f;             // the class probability best_match_iou asks for
};
[[nodiscard]] Status validate_head_training(const HeadTrainingOptions& options);
struct HeadDelta { std::vector<float> delta; std::vector<std::int64_t> assigned_row; std::vector<std::int64_t> assigned_class;
                   std::int64_t assignments = 0; double iou_sum = 0.0; };
void compute_head_delta(SushiRuntime::span<const float> head, const HeadShape& shape, const HeadOptions& options,
                        const AnchorLayout& layout, std::int64_t head_index, const HeadTrainingOptions& training,
                        const Detect::GroundTruthBatch& truth, NetworkSize network, HeadDelta& result);
```

- `delta` has the head's shape and is Darknet's delta of SPEC 4.6; `assigned_row` and `assigned_class` are `[B, H, W, A]` with -1 elsewhere, the best-anchor assignment the dumps hold. The box term is found through a private table of `{BoxLoss, function}`; this unit writes the `SquaredError` row with `ObjectnessTarget::One` and the `Logit` coding. `network` only turns the truth's pixel corners into fractions.
- Tests: one truth on a 2x2 grid with three anchors assigns the cell and slot SPEC 4.6 names, with `k = (2 - tw * th) * 0.75`; a truth whose best anchor another head owns assigns nothing; a prediction above the ignore threshold with a class above 0.25 has a zero objectness delta, and one with no class above 0.25 does not; two truths in one slot update one class only; an empty truth list gives objectness deltas alone.
- Acceptance: for a hand-placed truth every non-zero element of `delta` is the value SPEC 4.6 gives, to 1e-6.

**T02. The host CIoU gradient.** `vision/detect`, root. Waits on the second pass (the gradient in full). `opus low`. Files: `de/inc/iou_gradient.hpp`, `de/source/iou_gradient.cpp`, `de/tests/test_detect_iou_gradient.cpp`. `void complete_iou_gradient(SushiRuntime::span<const float> box, SushiRuntime::span<const float> target, SushiRuntime::span<float> gradient) noexcept;` over centre-and-size boxes, the four partial derivatives the second pass states. Acceptance: the gradient equals a central finite difference of SPEC 4.6's CIoU on 100 seeded pairs where the second pass says Darknet's is the true gradient, and equals the second pass's stated expression where it says otherwise; the test names which.

**T03. The delta, IoU boxes.** `vision/detect`, `anchor/`. Waits on T01, T02. `opus low`. Files: `de/source/anchor/box_delta_iou.cpp`, `de/source/anchor/box_delta.hpp`, `de/source/anchor/head_delta.cpp`, `de/tests/test_detect_anchor_delta_iou.cpp`. Adds the `CompleteIou` row (the gradient times `box_weight`, the `exp` factors of the `Logit` coding, NaN and infinity to 0, the clip), the extra assignments above `match_threshold` and the step-3 division. Acceptance: on a case with two anchors of one head above the threshold both slots are assigned, `assignments` is 2, and each box delta equals the gradient times `box_weight` divided by the class count of SPEC 4.6 step 3.

**T04. The delta, activated coding and `BestIou`.** `vision/detect`, `anchor/`. Waits on T03. `opus low`. Files: `de/source/anchor/head_delta.cpp`, `de/source/anchor/box_delta_iou.cpp`, `de/tests/test_detect_anchor_delta_activated.cpp`, the training section of `de/README.md`. The `Activated` coding has no `exp` factor on `w`, `h`; `BestIou` applies the larger-of rule in step 1 and writes the positive objectness delta only where the delta is still 0. Where `class_weight` multiplies is taken from the second pass. Acceptance: an ignored prediction with `best_match_iou` 0.8 and objectness 0.3 has an objectness delta of `object_weight * 0.5`.

**T05. The reported cost.** `vision/detect`, `anchor/`. Waits on T01. `haiku medium`. Files: `de/inc/anchor/head_cost.hpp`, `de/source/anchor/head_cost.cpp`, `de/tests/test_detect_anchor_cost.cpp`. `struct HeadCost { double total = 0.0; double box = 0.0; double non_box = 0.0; double stored = 0.0; };` and `[[nodiscard]] HeadCost measure_head_cost(const HeadDelta& delta, const HeadShape& shape, const HeadOptions& options, const HeadTrainingOptions& training) noexcept;` by SPEC 4.7. Acceptance: on a delta of known entries the four numbers equal the sums the test writes out.

**T06. Training settings of a cfg.** `vision/importers`. Waits on F08, F11, T01. `haiku high`. Files: `im/inc/darknet_training.hpp`, `im/source/darknet_training.cpp`, `im/tests/test_darknet_training.cpp`.

```cpp
struct DarknetSolver { double learning_rate = 0.001; double momentum = 0.9; double decay = 0.0005; std::int64_t burn_in = 0;
                       std::vector<std::int64_t> steps; std::vector<double> scales;
                       std::int64_t batch = 1; std::int64_t subdivisions = 1; std::int64_t max_batches = 0; };
struct DarknetTraining { DarknetSolver solver; std::vector<Detect::Anchor::HeadTrainingOptions> heads; };   // file order
[[nodiscard]] Result<DarknetTraining> read_darknet_training(const DarknetConfig& config);
```

It reads `[net]` and the training keys of every `[yolo]` with the defaults of SPEC section 3. A `policy` other than `steps`, a `truth_thresh` other than 1 and a non-zero `label_smooth_eps` are refused. Acceptance: each of the four key sets of SPEC section 3, written out in the test, yields the options that table states.

**T07. `StepSchedule`.** `core/optim`. `haiku medium`. Files: `modules/core/optim/include/SushiAI/optim/step_schedule.hpp`, `modules/core/optim/source/step_schedule.cpp`, `modules/core/optim/tests/test_step_schedule.cpp`. `struct StepSchedule { double base_rate = 0.0; std::int64_t warmup_iterations = 0; double warmup_power = 4.0; std::vector<std::int64_t> steps; std::vector<double> scales; [[nodiscard]] double rate_at(std::int64_t iteration) const noexcept; };` and `[[nodiscard]] Status validate_step_schedule(const StepSchedule& schedule);`. Acceptance: the rate at iterations 0, half the warm-up, the warm-up and one past each step equals SPEC 4.8's formula.

**T08. The delta loss emitter.** `vision/training`. `opus low`. Files: `tr/inc/head_delta_loss.hpp`, `tr/source/head_delta_loss.cpp`, `tr/tests/test_head_delta_loss.cpp`, `tests/integration/test_head_delta_loss.cpp`. `[[nodiscard]] Graph::ValueId emit_head_delta_loss(const Graph::GraphBuilder& builder, Graph::ValueId head, Graph::ValueId delta, double scale);` emits `-scale * sum(delta * head)` with `mul`, `reshape` to `[M, 1]` and `sum_rows`; `delta` is a leaf the objective declares. No new operation. If the training graph refuses that value as its root (SPEC 2.5), the worker stops and reports: the fix is in the core tier and is the owner's decision. Acceptance: on an engine and a seeded head and delta, the gradient at `head` equals `-scale * delta` to 1e-6.

**T09. The anchor objective and its step.** `vision/training`. Waits on T01, T05, T08. `opus low`. Files: `tr/inc/anchor_objective.hpp`, `tr/inc/anchor_training_step.hpp`, their two sources, `tr/tests/test_anchor_objective.cpp`, `tests/integration/test_anchor_training_step.cpp`, `tr/README.md`.
- `class AnchorObjective final : public Train::Objective`, built from a `NetworkSize`, an `AnchorLayout` and, per head, a `HeadOptions`, a `HeadTrainingOptions` and the output name. It names no importer type. `declare` adds one delta leaf per head; `loss` sums `emit_head_delta_loss` over the heads with `scale = 1 / batch`; `score` reports the summed `HeadCost`.
- `class AnchorTrainingStep final`, owning a `Train::EvaluationStep` and a `Train::TrainingStep` over one `ModelState`: `run` does the forward-only pass under batch statistics, calls `compute_head_delta` per head, feeds the leaves and replays. Whether the two steps can share one state, and whether the forward-only pass can run under batch statistics without updating the running ones twice, was not checked; if either fails the worker stops and reports.
- Acceptance: a two-head model trains one step, and the leaves fed equal what `compute_head_delta` returns for the forward-only heads.

### 4.7 Training parity

**T10 to T15.** Root tests, one unit and one file each, `tests/integration/test_darknet_<model>_training.cpp`. All wait on T05, T06, T09 and F14's loader. Each has two parts. The host part feeds the dumped `layer_<k-1>_yolo_input` and `truth` to `compute_head_delta` and compares `assigned_row`, `assigned_class`, `delta` (1e-6 absolute) and `stored` against the JSON's `cost`. The engine part trains the `Separate` model 200 steps on the reference input and truth and checks that the summed cost falls below a tenth of its first value; it has no Darknet counterpart.

| Unit | Model | Also waits on | Dispatch | References |
| --- | --- | --- | --- | --- |
| T10 | `yolov3_tiny` | none | `opus low` | on disk: costs 6.79358196 and 2.28762364 are read from the JSON, not typed |
| T11 | `yolov3` | F15 | `haiku high` | second pass |
| T12 | `yolov4_tiny` | T03, F16 | `haiku high` | second pass |
| T13 | `yolov4` | T03, F17 | `haiku high` | second pass |
| T14 | `yolov7_tiny` | T04, F18 | `haiku high` | on disk; waits on the second pass through T02 |
| T15 | `yolov7` | T04, F19 | `haiku high` | on disk; waits on the second pass through T02 |

Acceptance of each: the delta at every head equals Darknet's, and the cost on the one batch falls below a tenth of its first value.

**T16. One batch-statistics step.** Root test. Waits on T10 and on the second pass producing that dump; without the dump the unit is not dispatched. `opus low`. File: `tests/integration/test_darknet_yolov3_tiny_step.cpp`. Compares each head under batch statistics and whatever weight gradient or updated weight the dump holds, and says whether `Optim::SGD` is the update of SPEC 4.8. Acceptance: the heads under batch statistics equal the dump within the tolerance the test states.

## 5. Waves and builds

Files within a wave are disjoint. A build is the owner's; every other wave closes with syntax checks.

| Wave | Units | Waits on | Closes with |
| --- | --- | --- | --- |
| 0 | O1 | Q1 | nothing |
| 1 | F01, F02, F03, F04, F06, F07, F08, F09, T07 | wave 0 | syntax checks |
| 2 | F05, F10, A01, A02, T01, T08 | wave 1 | syntax checks |
| 3 | F11, A03, A04, T05, T06 | wave 2 | syntax checks |
| 4 | F12, F13, F14, A05, A06, T09 | wave 3 | **B1**: both suites, then F14 with `SA_REFERENCE_DATA_DIR` set. YOLOv3-tiny's forward pass matches |
| 5 | F18, F19, A07, T10 | B1 | **B2**: the two v7 forward passes, YOLOv3-tiny trains, the anchoring is whole |
| 6 | F15, F16, F17, F20, F21, T02, T11 | B2; the second pass | syntax checks |
| 7 | T03 | wave 6 | syntax checks |
| 8 | T04 | wave 7 | syntax checks |
| 9 | T12, T13, T14, T15, T16 | wave 8 | **B3**: six forward passes, six models train |

Nine waves after the skeleton, three builds. The first milestone needs seven units: F01, F06, F08, F09, F10, F11, F14. B2 can be folded into B3 for two builds; the cost is that the v7 forward passes and the first training result wait for the second Darknet pass.

Units that wait on the second pass: F15, F16, F17 (their directories), F20 (per-layer dumps), F21 (boxes and scores before the threshold), T02 (the CIoU gradient) and through it T03, T04, T12 to T15, T11 (the `yolov3` directory), T16 (the batch-statistics step). Everything of waves 1 to 5 runs on what is on disk.

One editor per wave on a shared file: `layer_registry.cpp` and `test_node_registry.cpp` (F01 in wave 1, F05 in 2); `darknet_layer_table.cpp` (F11 in 3, F12 in 4); `darknet_layer_route.cpp` (F11, F13); `head_delta.cpp` (T01 in 2, T03 in 7, T04 in 8); `box_delta.hpp` (T01, T03); `box_delta_iou.cpp` (T03, T04); `de/README.md` (F06 in 1, A07 in 5, T04 in 8); `detections_nms.*` (F07 only).

## 6. Not planned here

- YOLOv1 and YOLOv2; Darknet's data loader and augmentation; a COCO run.
- An IR CIoU emitter. YOLO26's training plan writes it in `vision/training` and can test it against `complete_iou_gradient` (E2).
- A program under `applications/` that takes an image and prints detections, and with it the inverse of a letterbox for a box, which the first draft had as a unit: parity starts from a tensor of the network's size.
- Storing an `AnchorPlan` in a file. It travels as a value; a persistent form would be a format with a `format_version` and an owner, and is a later decision.
- A fused `K_MISH`. `yolov3-spp`, the scaled YOLOv4 cfgs and `yolov7x`.

## 7. Not verified

- Nothing was built, run or downloaded, and no Darknet source was read for this revision. What it says of Darknet is the REPORT's, except the two recalled items SPEC marks: the warm-up power and `detector calc_anchors`.
- Open in the REPORT and carried as **(second pass)** marks: the CIoU gradient routine, where `cls_normalizer` multiplies, which IoU the stored cost averages, `mish` under a run, Darknet's own load of the three weights files it did not run, batch statistics and the update.
- The thresholds of SPEC 5.4 and the constants of SPEC 5.1 to 5.3 (`min_gain`, `recall_iou`, `min_recall`, the percentile) are engineering choices no experiment here supports. Whether the refinement of SPEC 5.2 gains anything measurable over the Lloyd result on real label sets is unknown until A03 runs on one.
- Whether the training graph accepts a one-element value other than `NLL_LOSS` as its root (T08); whether an evaluation step and a training step can share one `ModelState` (T09); whether `train`'s public edges give `vision/training` `graph` and `nn`; whether the `config` module exposes a JSON reader a test can call (F14); whether `sa test` passes the environment to the test binary (E4).
- Whether `Tensor::data()` is host-writable on every device, and the exact state names `build_model` produces.
- The starting tolerances of SPEC 6.3 are estimates; B1 measures them.
- Of the repository, this revision read `modules/vision/detect/CMakeLists.txt`, the detect include tree, `detections_nms.hpp`, `ground_truth.hpp`, `objective.hpp` (declarations only), `status.hpp`, the vision and umbrella `CMakeLists.txt`, the tier table and `DOMAIN_MODULES.md`. The `nn` and builder facts of SPEC 2.1 to 2.3 are the first draft's and were not reread; the name of the layers-tier umbrella header was not looked up.
- The SHA-256 values of SPEC 4.2 were copied from the REPORT and the ruling, not recomputed.
- Rule 6 of the dispatch block had nothing to check: this task revised two Markdown files.
