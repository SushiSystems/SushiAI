# Unit tests

Each file under `modules/*/*/tests/` that has a section stays on this page, under a heading
that gives the file's path and so names its module.

Most unit test files run without a device: they build a graph, a config document, a byte
stream, an image or a plain array on the host and inspect the result. A file whose fixture
derives from `EngineTest` needs an engine, because its cases allocate tensors, fill batches or
run a compiled step; the opening paragraph of its section says so.

Some cases touch the disk. These files write small files under the system's temporary
directory, read them back and remove them:

- `modules/core/config/tests/test_json_events.cpp`;
- the path-resolution cases of `modules/layers/nn/tests/test_config.cpp`;
- `modules/layers/io/tests/test_imported_model_writer.cpp`;
- `modules/language/text/tests/test_text_vocabulary_file.cpp`;
- under `modules/language/datasets/tests/`: `test_datasets_prompt_dataset.cpp`,
  `test_datasets_token_dataset.cpp`, `test_datasets_token_file.cpp` and
  `test_datasets_tokens_section.cpp`;
- `modules/vision/codecs/tests/test_codecs_decode_image.cpp`;
- under `modules/vision/datasets/tests/`: `test_datasets_cifar.cpp`,
  `test_datasets_coco_json.cpp`, `test_datasets_csv.cpp`, `test_datasets_fit_mask.cpp`,
  `test_datasets_idx.cpp`, `test_datasets_idx_file.cpp`, `test_datasets_image_folder.cpp`,
  `test_datasets_load_sample_image.cpp`, `test_datasets_mask_png.cpp`,
  `test_datasets_pipelines.cpp`, `test_datasets_reader_dataset.cpp`,
  `test_datasets_reader_section.cpp`, `test_datasets_sample_cache.cpp`,
  `test_datasets_stored_pixels.cpp`, `test_datasets_voc_xml.cpp` and
  `test_datasets_yolo_txt.cpp`;
- `modules/vision/imaging/tests/test_imaging_raw_cache.cpp`;
- `modules/vision/importers/tests/test_darknet_weights.cpp`.

Other cases read files the repository holds. The reference cases of `test_detect_anchors.cpp`,
`test_detect_assigner.cpp`, `test_detect_boxes.cpp` and `test_detect_postprocess.cpp`, and the
two reference cases of `test_batch_norm2d.cpp`, read tensors from `tests/fixtures/yolo26/` and
report themselves skipped when a file is absent. The YOLO26 cases of
`modules/vision/registration/tests/test_registration.cpp` read
`applications/examples/yolo26/model.json` and `tests/fixtures/yolo26/MANIFEST.json`, and the
two example cases of `test_model_writer.cpp` read the `model.json` of the examples `mnist_cnn`
and `yolo26`; both files report those cases skipped when the fixture folder is not defined for
the binary. One case of `modules/vision/registration/tests/test_datasets_mnist.cpp` reads the
MNIST IDX files and reports itself skipped when they are absent.

This page holds the reasoning behind the cases. A `//` line in a test states the invariant the
case relies on and cites this page. The section under the test's name says why the invariant
matters, which number it was measured at, or which failure it once let through. Where a test has
several such lines, each paragraph opens with the assertion or the statement it explains.
Sections follow the files in file-name order, whatever module a file is in, and inside a file
the headings follow the order of the tests.

## Files

| File | What it tests |
| --- | --- |
| [`test_add.cpp`](../../modules/layers/nn/tests/test_add.cpp) | That the Add node takes exactly two rank-3 inputs of one shape, emits one ADD reading them in order, returns a failure for values that do not add, and what its factory refuses. |
| [`test_anchor_objective.cpp`](../../modules/vision/training/tests/test_anchor_objective.cpp) | The leaves `AnchorObjective` declares per head, the delta losses its root sums scaled by the batch, its cost as the sum over the heads, and the heads, names and options `make` refuses. |
| [`test_anchor_objective_factory.cpp`](../../modules/vision/training/tests/test_anchor_objective_factory.cpp) | That the anchor objective takes its heads from the published shapes, its options from the `objective` section and its anchors from the task, and what it refuses. |
| [`test_anchor_training_loop.cpp`](../../modules/vision/training/tests/test_anchor_training_loop.cpp) | The rate each update takes from a step schedule and from a cosine schedule over two epochs, a run without a schedule, a start at a later epoch and batch, the run count of both passes restored once, the datasets, options and starts refused before any update, and a callback that throws. Needs an engine. |
| [`test_apply_step_schedule.cpp`](../../modules/core/optim/tests/test_apply_step_schedule.cpp) | That a schedule's rate reaches SGD and AdamW through `Optimizer::set_learning_rate`, and that an update moves the parameter by the rate that was set. Needs an engine. |
| [`test_avg_pool2d.cpp`](../../modules/layers/nn/tests/test_avg_pool2d.cpp) | The shapes the AvgPool2d layer traces and reports, and the inputs it refuses. |
| [`test_backward.cpp`](test_backward.cpp) | The graph that differentiation emits and the coverage of the backward rule table. |
| [`test_batch_norm2d.cpp`](../../modules/layers/nn/tests/test_batch_norm2d.cpp) | BatchNorm2d's two trace forms and the running-statistic update. |
| [`test_batched_matmul.cpp`](../../modules/core/graph/tests/test_batched_matmul.cpp) | The shape and transpose flags `GraphBuilder::matmul` gives a rank-2 and a rank-3 product, the operands it refuses, and the four nodes `GraphBuilder::softmax` emits. |
| [`test_binary_confusion.cpp`](../../modules/layers/train/tests/test_binary_confusion.cpp) | The four counts of a column, the ratios over them and over nothing, the threshold's cut and the threshold reader's refusals. Needs an engine. |
| [`test_block_attention.cpp`](../../modules/vision/blocks/tests/test_block_attention.cpp) | The Attention block's head and key widths, the nodes two heads trace and what each product and the add read, its fifteen state records against MANIFEST.json, the evaluation fold, and the widths it refuses. |
| [`test_block_bottleneck.cpp`](../../modules/vision/blocks/tests/test_block_bottleneck.cpp) | Bottleneck's trace, its state names against YOLO26n row 2, and its factory. |
| [`test_block_c2psa.cpp`](../../modules/vision/blocks/tests/test_block_c2psa.cpp) | The C2PSA block's trace at YOLO26n row 10, what its join and its repeats read, its 35 state records against MANIFEST.json, and what its factory defaults and refuses. |
| [`test_block_c3k.cpp`](../../modules/vision/blocks/tests/test_block_c3k.cpp) | C3k's trace, its state names against YOLO26n row 6, and its factory. |
| [`test_block_c3k2.cpp`](../../modules/vision/blocks/tests/test_block_c3k2.cpp) | C3k2's trace, its state names against YOLO26n rows 2, 6 and 22, its table of inner blocks, and its factories. |
| [`test_block_causal_self_attention.cpp`](../../modules/language/blocks/tests/test_block_causal_self_attention.cpp) | The attention's trace with and without the mask, its node counts, its state, its refusals and its factory; the rotary trace, its ring rows, its fed slot and angles, and the `positions` key. |
| [`test_block_conv_norm.cpp`](../../modules/vision/blocks/tests/test_block_conv_norm.cpp) | ConvNorm's trace, the depthwise `DWConv` and `DWConvLinear`, their state names against YOLO26n, and make_conv_node. |
| [`test_block_detect_head.cpp`](../../modules/vision/blocks/tests/test_block_detect_head.cpp) | The DetectHead's twelve ports and their names, its 204 state records against MANIFEST.json, the STOP_GRADIENT nodes of a training trace, the branch widths, and what `create` and the factory refuse. |
| [`test_block_expert_transformer_block.cpp`](../../modules/language/blocks/tests/test_block_expert_transformer_block.cpp) | The two branches of a block with experts, its fourteen state names, the `experts` and `top_k` keys of the `transformer_block` row, one position of the incremental form, and `positions` handed to the attention. |
| [`test_block_feed_forward.cpp`](../../modules/language/blocks/tests/test_block_feed_forward.cpp) | The feed-forward's trace, its state, its refusals and its factory. |
| [`test_block_language_head.cpp`](../../modules/language/blocks/tests/test_block_language_head.cpp) | The head's trace to vocabulary logits without a bias, its state, its shape rule and its factory. |
| [`test_block_moe_feed_forward.cpp`](../../modules/language/blocks/tests/test_block_moe_feed_forward.cpp) | The mixture's trace and its 31 nodes, the 7 + 4k nodes of `top_k`, the six state names, the refusals, the three-port node and its factory, and the incremental form. |
| [`test_block_position_embedding.cpp`](../../modules/language/blocks/tests/test_block_position_embedding.cpp) | The table added in one `BIAS_ADD`, the leading rows of a longer table, the refusal of a longer sequence, its state and its factory. |
| [`test_block_psa_block.cpp`](../../modules/vision/blocks/tests/test_block_psa_block.cpp) | The PSABlock's node counts in a training trace, what its two residual adds read, its 25 state records against MANIFEST.json, and the width it refuses. |
| [`test_block_rotary_embedding.cpp`](../../modules/language/blocks/tests/test_block_rotary_embedding.cpp) | The 18 nodes and two leaves of a turn, the tables against `rotary_angle`, float64 tables, no state, the two fed rows of the incremental form, and each refusal. |
| [`test_block_rotary_positions.cpp`](../../modules/language/blocks/tests/test_block_rotary_positions.cpp) | `rotary_angle` against hand-worked angles, position 0 as cosine 1 and sine 0, a row against the host's cosine and sine, a position of 2^40, and the fatal span mismatches. |
| [`test_block_segment_head.cpp`](../../modules/vision/blocks/tests/test_block_segment_head.cpp) | SegmentHead's shape, its trace ending at the upsample, its seven pieces of state, and its factory. |
| [`test_block_sppf.cpp`](../../modules/vision/blocks/tests/test_block_sppf.cpp) | SPPF's trace, its state names against YOLO26n row 9, and its factory. |
| [`test_block_transformer_block.cpp`](../../modules/language/blocks/tests/test_block_transformer_block.cpp) | The pre-norm order of the transformer block, its mask option, its state and its factory, which hands `positions` to its attention and refuses an odd head. |
| [`test_buffer.cpp`](../../modules/layers/nn/tests/test_buffer.cpp) | Buffer: its description, its one leaf per graph and its identity. |
| [`test_carrying_step.cpp`](../../modules/layers/train/tests/test_carrying_step.cpp) | An EvaluationStep fed two leaves by name and in order, a value carried across runs, its reset, the feeds refused, and an ordinary step that carries nothing. Its fixture cases need an engine. |
| [`test_causal_mask_backward.cpp`](../../modules/core/autograd/tests/test_causal_mask_backward.cpp) | That the causal mask differentiates into the same mask over the gradient with a fill of zero. |
| [`test_causal_mask_ops.cpp`](../../modules/core/graph/tests/test_causal_mask_ops.cpp) | The shape rules and IR of the causal mask emitter. |
| [`test_channel_slice.cpp`](../../modules/layers/nn/tests/test_channel_slice.cpp) | The ChannelSlice layer's shape inference, the one SLICE_CHANNELS node it emits with its offset, the offsets, counts and ranks it rejects, and its empty state. |
| [`test_checkpoint_format.cpp`](../../modules/layers/io/tests/test_checkpoint_format.cpp) | The checkpoint format's round trip, a version 1 file, the run-state section, and its refusal of each malformed byte pattern. |
| [`test_codecs_decode_image.cpp`](../../modules/vision/codecs/tests/test_codecs_decode_image.cpp) | The dispatch on an image file's leading bytes, the raw image path, and the load of a file from disk. |
| [`test_codecs_image_format.cpp`](../../modules/vision/codecs/tests/test_codecs_image_format.cpp) | The detection of JPEG, PNG and raw image files from their leading bytes. |
| [`test_codecs_jpeg.cpp`](../../modules/vision/codecs/tests/test_codecs_jpeg.cpp) | The JPEG decoder on files encoded in memory: colour spaces, layouts, cut files and the side limit. |
| [`test_codecs_png.cpp`](../../modules/vision/codecs/tests/test_codecs_png.cpp) | The PNG decoder on files encoded in memory: bit depths, alpha, palette indices, cut files and the side limit. |
| [`test_compiled_step.cpp`](../../modules/core/graph/tests/test_compiled_step.cpp) | That a compiled step restores its run count once and before its first run, and then draws the dropout mask of that run. Needs an engine. |
| [`test_concat.cpp`](../../modules/layers/nn/tests/test_concat.cpp) | The Concat node's shape check, its port and the node it traces. |
| [`test_config.cpp`](../../modules/layers/nn/tests/test_config.cpp) | How the config layer parses its JSON documents and which ones it refuses. |
| [`test_conv2d.cpp`](test_conv2d.cpp) | The Conv2d layer's trace, parameters, shape inference and rejections. |
| [`test_cosine_schedule.cpp`](../../modules/core/optim/tests/test_cosine_schedule.cpp) | The warm-up and the cosine decay at its quarters, the check, and the section's round trip and refusals. |
| [`test_darknet_config.cpp`](../../modules/vision/importers/tests/test_darknet_config.cpp) | The sections, keys, values and line numbers `parse_darknet_config` reads from a config text, the malformed texts it refuses with origin and line, and the code for a missing file. |
| [`test_darknet_fold.cpp`](../../modules/vision/importers/tests/test_darknet_fold.cpp) | That `fold_darknet_normalisation` gives the weights and bias of the formula in long double, that the folded and the unfolded convolution agree on the host, and the two misuses that stop the process. |
| [`test_darknet_import.cpp`](../../modules/vision/importers/tests/test_darknet_import.cpp) | The imported model of a cfg: rows and outputs kept, the task read back, tensor names, the recipe and its notes, and the files refused. |
| [`test_darknet_model.cpp`](../../modules/vision/importers/tests/test_darknet_model.cpp) | The model rows, sources, heads, anchors, convolution shapes and state targets a config text translates to in the Separate and the Folded form, and the sections, keys and values refused by line. Two cases need an engine. |
| [`test_darknet_recipe.cpp`](../../modules/vision/importers/tests/test_darknet_recipe.cpp) | The training recipe written from a solver and its heads, the notes that name a line and a value, and heads that differ refused. |
| [`test_darknet_route_groups.cpp`](../../modules/vision/importers/tests/test_darknet_route_groups.cpp) | The rows a grouped `[route]` section translates to: one `slice_channels` row per source, a `concat` over them, no row for an ungrouped source, and the group counts and indices refused by line. |
| [`test_darknet_shortcut.cpp`](../../modules/vision/importers/tests/test_darknet_shortcut.cpp) | That a `[shortcut]` section translates to an `add` row over the layer before it and the layer `from` names, and the sections refused by line. |
| [`test_darknet_tensors.cpp`](../../modules/vision/importers/tests/test_darknet_tensors.cpp) | One tensor per weight target in the kernel-last layout, batch-norm names, and the lengths, names and shapes refused. |
| [`test_darknet_training.cpp`](../../modules/vision/importers/tests/test_darknet_training.cpp) | The head training options and the solver settings `read_darknet_training` reads from a parsed config, with their defaults for absent keys, and the keys and values it refuses by line. |
| [`test_darknet_weights.cpp`](../../modules/vision/importers/tests/test_darknet_weights.cpp) | The header `read_darknet_header` reads for each version, the vectors `read_darknet_weights` fills per convolution in file order with their bits kept, and the files and shapes refused. |
| [`test_dataset_registry.cpp`](../../modules/core/data/tests/test_dataset_registry.cpp) | That `Data::DatasetRegistry` keeps its rows in insertion order, finds a row by type, refuses a type added twice, and that `Data::register_datasets` adds the "rings" row alone. |
| [`test_datasets_cifar.cpp`](../../modules/vision/datasets/tests/test_datasets_cifar.cpp) | The CIFAR-10 and CIFAR-100 binary reader: file order, the label taken, the planes read back, and the record a bad label names. |
| [`test_datasets_coco_json.cpp`](../../modules/vision/datasets/tests/test_datasets_coco_json.cpp) | The COCO annotation reader: dense class indices from sparse ids, boxes, dropped crowds, kept polygons, and an annotation naming an unknown image or category with its index. |
| [`test_datasets_collators.cpp`](../../modules/vision/datasets/tests/test_datasets_collators.cpp) | The scaled inputs and the roles the classification and detection collators write, and the batches they refuse. Needs an engine. |
| [`test_datasets_csv.cpp`](../../modules/vision/datasets/tests/test_datasets_csv.cpp) | The CSV reader: named feature and target columns, quoted fields and CRLF, and a cell that is no finite number with its line and column. |
| [`test_datasets_fit_mask.cpp`](../../modules/vision/datasets/tests/test_datasets_fit_mask.cpp) | A mask fitted by nearest neighbour with its image's geometry and padded with the void value, and a sample's mask loaded with its indices kept. |
| [`test_datasets_fit_sample.cpp`](../../modules/vision/datasets/tests/test_datasets_fit_sample.cpp) | A sample fitted to the input size by resize and by letterbox, with its boxes carried along. |
| [`test_datasets_idx.cpp`](../../modules/vision/datasets/tests/test_datasets_idx.cpp) | The IDX dataset reader: classes, pixel offsets, named files and the image or label files it refuses. |
| [`test_datasets_idx_file.cpp`](../../modules/vision/datasets/tests/test_datasets_idx_file.cpp) | The IDX header and payload readers and each malformed header they refuse. |
| [`test_datasets_image_folder.cpp`](../../modules/vision/datasets/tests/test_datasets_image_folder.cpp) | The folder-per-class reader: sorted classes and files, the extension table, and the splits it refuses. |
| [`test_datasets_load_sample_image.cpp`](../../modules/vision/datasets/tests/test_datasets_load_sample_image.cpp) | A sample's image read from stored pixels or decoded from its file, and the samples it refuses. |
| [`test_datasets_mask_png.cpp`](../../modules/vision/datasets/tests/test_datasets_mask_png.cpp) | The image and mask folder reader: pairs by relative path, and an image without a mask refused. |
| [`test_datasets_mnist.cpp`](../../modules/vision/registration/tests/test_datasets_mnist.cpp) | The "mnist" row of the vision module's dataset registry: the keys, splits and layouts it refuses, the files it names when they are absent, and the rank-4 batches of the "image" layout. Needs an engine. |
| [`test_datasets_next_token_collator.cpp`](../../modules/language/datasets/tests/test_datasets_next_token_collator.cpp) | The task description, labels as the next token of each position, and the ids and batches collate refuses. Needs an engine. |
| [`test_datasets_pipelines.cpp`](../../modules/vision/datasets/tests/test_datasets_pipelines.cpp) | The pipeline table, the fit the same in every epoch, a training sample identical alone, in a batch and reversed, the mosaic epochs, masks, and the refusals. |
| [`test_datasets_polygon_mask.cpp`](../../modules/vision/datasets/tests/test_datasets_polygon_mask.cpp) | Pixels a polygon paints, the later object winning, no object, an object without a polygon, a class above 254. |
| [`test_datasets_prompt_dataset.cpp`](../../modules/language/datasets/tests/test_datasets_prompt_dataset.cpp) | The one batch of a prompt, its last ids when it is longer than the window, its length role, and the `prompt` data block. Needs an engine. |
| [`test_datasets_reader_dataset.cpp`](../../modules/vision/datasets/tests/test_datasets_reader_dataset.cpp) | ReaderDataset's queries, a batch built on demand in list order, the tensors a second read reuses, and the refusals of make and read. Needs an engine. |
| [`test_datasets_reader_section.cpp`](../../modules/vision/datasets/tests/test_datasets_reader_section.cpp) | The reader and task tables, a detection and a classification section read into batches, and the sections refused: an unknown reader or task, a key of another task, a class count below the list's. Needs an engine. |
| [`test_datasets_sample_cache.cpp`](../../modules/vision/datasets/tests/test_datasets_sample_cache.cpp) | A first pass that writes and a second that reuses, pixels read back, the longest side, a changed source found stale, stored pixels and masks left alone. |
| [`test_datasets_stored_pixels.cpp`](../../modules/vision/datasets/tests/test_datasets_stored_pixels.cpp) | The copy of interleaved and planar pixels out of a file into an image. |
| [`test_datasets_table_dataset.cpp`](../../modules/vision/datasets/tests/test_datasets_table_dataset.cpp) | TableDataset: the rows of a value table read as feature and value batches. Needs an engine. |
| [`test_datasets_task_collators.cpp`](../../modules/vision/datasets/tests/test_datasets_task_collators.cpp) | The roles the binary, multi-label, regression and segmentation collators write, and the samples they refuse. Needs an engine. |
| [`test_datasets_token_dataset.cpp`](../../modules/language/datasets/tests/test_datasets_token_dataset.cpp) | The int32 inputs and labels of a window batch, the next-token property over overlapping windows, tensor reuse, and a bad id or a cut file named with its offset. Needs an engine. |
| [`test_datasets_token_file.cpp`](../../modules/language/datasets/tests/test_datasets_token_file.cpp) | The documented bytes, the header and ranges read back, the ids the writer refuses, truncated and oversized files, and an id outside the vocabulary with its byte offset. |
| [`test_datasets_tokens_section.cpp`](../../modules/language/datasets/tests/test_datasets_tokens_section.cpp) | A `tokens` data block built into a next-token dataset, its default batch size and the keys it refuses. Needs an engine. |
| [`test_datasets_voc_xml.cpp`](../../modules/vision/datasets/tests/test_datasets_voc_xml.cpp) | The Pascal VOC reader: boxes counted from one, difficult objects dropped or kept, class names stated or collected, and a file refused with its line. |
| [`test_datasets_window_layout.cpp`](../../modules/language/datasets/tests/test_datasets_window_layout.cpp) | The window count and starts of a stride, whole windows only, and the layouts refused. |
| [`test_datasets_xml_element.cpp`](../../modules/vision/datasets/tests/test_datasets_xml_element.cpp) | The XML subset parse_xml reads, its five entities, and each construct it refuses with a line. |
| [`test_datasets_yolo_txt.cpp`](../../modules/vision/datasets/tests/test_datasets_yolo_txt.cpp) | The image-and-label-text reader: the label path rule, both split forms, and the file and line of a bad label. |
| [`test_dense_losses.cpp`](../../modules/layers/train/tests/test_dense_losses.cpp) | The node sequence of the mean, of the logistic loss in its stable form and of the squared error. |
| [`test_dense_objectives.cpp`](../../modules/layers/train/tests/test_dense_objectives.cpp) | The leaf and role each of the binary, multi-label and squared-error objectives declares, what each refuses, their feeds and their metrics. Needs an engine. |
| [`test_dense_targets.cpp`](../../modules/layers/train/tests/test_dense_targets.cpp) | The fills of a real target leaf and of a 0 or 1 leaf, the counts and types refused with the leaf kept. Needs an engine. |
| [`test_depthwise_conv2d.cpp`](../../modules/layers/nn/tests/test_depthwise_conv2d.cpp) | The DepthwiseConv2d layer's one DEPTHWISE_CONV2D node, its [kernel_h, kernel_w, channels] weight with role and fans, shape inference against the trace, `same_padded`, and the inputs it rejects. |
| [`test_depthwise_ops.cpp`](../../modules/core/graph/tests/test_depthwise_ops.cpp) | The IR that `GraphBuilder::depthwise_conv2d` and its two backward emitters write: operand order, result shape, layout, element type, geometry slots, and each refusal. |
| [`test_detect_anchor_clustering.cpp`](../../modules/vision/detect/tests/test_detect_anchor_clustering.cpp) | `SeededSequence` against the published SplitMix64 numbers, and that `cluster_anchors` recovers a six-rung ladder of box sizes within 5 %, repeats bit for bit, leaves out degenerate boxes and refuses bad options. |
| [`test_detect_anchor_cost.cpp`](../../modules/vision/detect/tests/test_detect_anchor_cost.cpp) | The four numbers `measure_head_cost` reports for a head whose delta is known: total, box share, non-box share and stored cost, under the squared-error and the CompleteIoU box loss. |
| [`test_detect_anchor_count.cpp`](../../modules/vision/detect/tests/test_detect_anchor_count.cpp) | The anchors per head `choose_anchor_count` settles on for a ladder, a single blob and two sizes, the trials it returns, and the options, head counts and box lists it refuses. |
| [`test_detect_anchor_decode.cpp`](../../modules/vision/detect/tests/test_detect_anchor_decode.cpp) | The boxes and scores an anchor head decodes to under the Logit and the Activated coding and under `scale_xy`, the order of the decoded rows, the channel and row counts, and what `validate_head` refuses. |
| [`test_detect_anchor_delta.cpp`](../../modules/vision/detect/tests/test_detect_anchor_delta.cpp) | The delta an anchor head with Logit coding receives: the assigned cell and slot, the IoU sum, the ignore rule and its class gate, two truths in one slot, padding rows, and the training options refused. |
| [`test_detect_anchor_delta_activated.cpp`](../../modules/vision/detect/tests/test_detect_anchor_delta_activated.cpp) | The delta of an anchor head with the Activated coding: the objectness of an ignored and of an assigned slot under each `ObjectnessTarget`, the box delta without a size factor, and class deltas read as probabilities. |
| [`test_detect_anchor_delta_gradient.cpp`](../../modules/vision/detect/tests/test_detect_anchor_delta_gradient.cpp) | The box deltas an anchor head writes under `BoxGradient::Analytic` and `BoxGradient::DarknetCompatible`, against values worked by hand and against twelve dumped cells the file holds. |
| [`test_detect_anchor_delta_iou.cpp`](../../modules/vision/detect/tests/test_detect_anchor_delta_iou.cpp) | The delta of an anchor head under the IoU box loss: further anchors above the match threshold, the IoU sum, the division by positive class deltas, the clip at `max_delta`, and non-finite components. |
| [`test_detect_anchor_fit.cpp`](../../modules/vision/detect/tests/test_detect_anchor_fit.cpp) | The numbers `measure_anchor_fit` reports for a layout over a box list: mean best IoU, best possible recall, tenth percentile, unowned share and per-anchor share, and what it refuses. |
| [`test_detect_anchor_head_assignment.cpp`](../../modules/vision/detect/tests/test_detect_anchor_head_assignment.cpp) | The head masks `assign_anchors_to_heads` computes from anchors and strides: runs in ascending area from the smallest stride, the remainder, `per_head` counts, and the inputs it refuses. |
| [`test_detect_anchor_head_truth.cpp`](../../modules/vision/detect/tests/test_detect_anchor_head_truth.cpp) | The truth rows `validate_head_truth` accepts and refuses by centre, size, class, array shape and network size, and that `compute_head_delta` refuses the same truth and writes nothing. |
| [`test_detect_anchor_layout.cpp`](../../modules/vision/detect/tests/test_detect_anchor_layout.cpp) | The value of `shape_iou` for equal, contained, crossed and zero-area sizes, and the layouts `validate_anchor_layout` accepts and refuses. |
| [`test_detect_anchor_plan.cpp`](../../modules/vision/detect/tests/test_detect_anchor_plan.cpp) | That `plan_anchors` returns the layout, fit, thresholds and trials its four steps give one by one, for a chosen and a given count per head and any stride order, and what it refuses. |
| [`test_detect_anchor_review.cpp`](../../modules/vision/detect/tests/test_detect_anchor_review.cpp) | The verdict `review_anchors` gives a layout on its own boxes and on shifted ones, the proposal's anchors and counts per head, the two fits it reports, and what it refuses. |
| [`test_detect_anchor_task.cpp`](../../modules/vision/detect/tests/test_detect_anchor_task.cpp) | That an anchor task and the `task` section of a model file convert both ways without loss, and which sections and tasks are refused. |
| [`test_detect_anchor_thresholds.cpp`](../../modules/vision/detect/tests/test_detect_anchor_thresholds.cpp) | The default `AssignmentThresholds`, and the match and ignore thresholds `derive_assignment_thresholds` takes from a fit, with the clamp at each end of both ranges. |
| [`test_detect_anchor_training_section.cpp`](../../modules/vision/detect/tests/test_detect_anchor_training_section.cpp) | That `HeadTrainingOptions` and the `anchor` objective section of a training file convert both ways without loss, and which sections and options are refused. |
| [`test_detect_anchors.cpp`](../../modules/vision/detect/tests/test_detect_anchors.cpp) | The anchor grid, the per-image anchor leaves and the copies between a level and the joined anchor axis. |
| [`test_detect_assigner.cpp`](../../modules/vision/detect/tests/test_detect_assigner.cpp) | The task-aligned assigner, its presets, the padded ground truth and the target leaves. |
| [`test_detect_average_precision.cpp`](../../modules/vision/detect/tests/test_detect_average_precision.cpp) | Detection matching, the 101-point average precision and the validation table. |
| [`test_detect_boxes.cpp`](../../modules/vision/detect/tests/test_detect_boxes.cpp) | Box decoding and encoding, the two box formats and the IoU forms. |
| [`test_detect_iou_gradient.cpp`](../../modules/vision/detect/tests/test_detect_iou_gradient.cpp) | The IoU box gradient of `iou_gradient.hpp`: the analytic form against a central difference on fixed and seeded box pairs, and the `DarknetCompatible` form against hand values and twelve dumped cells. |
| [`test_detect_postprocess.cpp`](../../modules/vision/detect/tests/test_detect_postprocess.cpp) | The top-k selection, both postprocess paths and the overlap suppression. |
| [`test_detect_suppression_criterion.cpp`](../../modules/vision/detect/tests/test_detect_suppression_criterion.cpp) | The value of `measure_suppression` under the IoU and the DistanceIoU criterion, which rows `suppress_overlaps` keeps under each, the `NMSConfig` defaults, and the two fatal refusals. |
| [`test_diagnostics.cpp`](test_diagnostics.cpp) | That each host-lane precondition raises a message carrying the values that disagree. |
| [`test_dropout.cpp`](../../modules/layers/nn/tests/test_dropout.cpp) | That Dropout emits one DROPOUT node in training with its rate and site seed, returns its input in evaluation and at rate 0, gives each site its own seed, its rate check and its factory. |
| [`test_dropout_backward.cpp`](../../modules/core/autograd/tests/test_dropout_backward.cpp) | That the gradient of DROPOUT is a MUL reading the saved mask, that it draws no second mask, and that an input without a gradient adds no reader. |
| [`test_dropout_mask.cpp`](../../modules/core/graph/tests/test_dropout_mask.cpp) | The dropout generator against the published SplitMix64 sequence and golden bits and masks, the kept fraction of 2^20 draws within 0.002, and that a draw depends on seed, run and element alone. |
| [`test_dropout_ops.cpp`](../../modules/core/graph/tests/test_dropout_ops.cpp) | The IR `GraphBuilder::dropout` emits, the 64-bit seed through its two attribute slots, each refusal, and the row of `K_RANDOM_LOWERINGS`. |
| [`test_elementwise_ops.cpp`](../../modules/core/graph/tests/test_elementwise_ops.cpp) | The nodes the builder emits for the elementwise, comparison, selection and constant operations. |
| [`test_embedding.cpp`](../../modules/layers/nn/tests/test_embedding.cpp) | The Embedding layer's trace over an int32 index, its table parameter, its shape rule and its factory. |
| [`test_epoch_permutation.cpp`](../../modules/core/data/tests/test_epoch_permutation.cpp) | Every sample once for counts on both sides of a power of four, three golden orders by seed and epoch, and the count. |
| [`test_flatten.cpp`](../../modules/layers/nn/tests/test_flatten.cpp) | The Flatten layer's trace, shape rule, rejections and parameter walk. |
| [`test_fusion.cpp`](../../modules/core/graph/tests/test_fusion.cpp) | The matches and the refusals of the static fusion pass on synthetic graphs. |
| [`test_gather_backward.cpp`](../../modules/core/autograd/tests/test_gather_backward.cpp) | That a row gather differentiates into a scatter-add on its table, a scatter-add into a gather on its rows, and that neither index gets a gradient. |
| [`test_gather_ops.cpp`](../../modules/core/graph/tests/test_gather_ops.cpp) | The shape rules and IR of the row gather and scatter-add emitters, and an int32 input leaf under finalise and the cast pass. |
| [`test_generation_greedy_sampler.cpp`](../../modules/language/generation/tests/test_generation_greedy_sampler.cpp) | The largest logit, the first of equal ones, and a generator left as it was. |
| [`test_generation_nucleus_sampler.cpp`](../../modules/language/generation/tests/test_generation_nucleus_sampler.cpp) | The fewest tokens that reach a mass, their shares, a mass of one, and the values refused. |
| [`test_generation_position_feed.cpp`](../../modules/language/generation/tests/test_generation_position_feed.cpp) | `make_position_feed`: a step fed a position gets the table feed, one fed a slot gets the ring feed, and one fed both or neither is refused. |
| [`test_generation_ring_position_feed.cpp`](../../modules/language/generation/tests/test_generation_ring_position_feed.cpp) | `RingPositionFeed`: slot, cosines, sines in that order; the slot is the position modulo T while the angles keep the position; every row survives a slide; its refusals. |
| [`test_generation_sampler_section.cpp`](../../modules/language/generation/tests/test_generation_sampler_section.cpp) | The sampler of each type, the unknown type with the four listed, and the keys refused. |
| [`test_generation_table_position_feed.cpp`](../../modules/language/generation/tests/test_generation_table_position_feed.cpp) | `TablePositionFeed`: one `[1, 1]` int32 leaf, `place` writes the position, no row survives a slide, an absent or mistyped leaf is refused. |
| [`test_generation_task.cpp`](../../modules/language/generation/tests/test_generation_task.cpp) | The keys of a `next_token_generation` section, their defaults and their refusals. |
| [`test_generation_temperature_sampler.cpp`](../../modules/language/generation/tests/test_generation_temperature_sampler.cpp) | The shares of two hand-worked softmaxes, one sequence per seed, and the temperatures refused. |
| [`test_generation_top_k_sampler.cpp`](../../modules/language/generation/tests/test_generation_top_k_sampler.cpp) | The shares over the k largest logits, the tie at the edge, k of one, and the values refused. |
| [`test_global_avg_pool.cpp`](../../modules/layers/nn/tests/test_global_avg_pool.cpp) | The shapes GlobalAvgPool traces and reports, and the inputs it refuses. |
| [`test_gradcheck.cpp`](../../modules/core/autograd/tests/test_gradcheck.cpp) | The gradient checker against closed-form functions and a wrong gradient. |
| [`test_gradient_clip.cpp`](../../modules/core/optim/tests/test_gradient_clip.cpp) | The factor of a global norm over one and two slots, a norm set at run time, both SGD kernels and AdamW, the task count with and without a clip, and the key `clip_norm`. Needs an engine. |
| [`test_head_delta_loss.cpp`](../../modules/vision/training/tests/test_head_delta_loss.cpp) | The four nodes `emit_head_delta_loss` of the vision training module adds, the scale and element type of its one-element result, that autograd takes it as a root, and the deltas it refuses. |
| [`test_host_table.cpp`](../../modules/layers/nn/tests/test_host_table.cpp) | `Trace::host_table`: a table is a buffer leaf listed with its values, two declarations are two leaves, and a trace without a sink, a wrong type or a wrong count is refused. |
| [`test_identity.cpp`](../../modules/layers/nn/tests/test_identity.cpp) | That the Identity layer returns its input, emits nothing and holds no state. |
| [`test_imaging_affine.cpp`](../../modules/vision/imaging/tests/test_imaging_affine.cpp) | The plan and the application of the scale and translation warp to an image and its labels. |
| [`test_imaging_boxes.cpp`](../../modules/vision/imaging/tests/test_imaging_boxes.cpp) | The two box forms, their conversions and the candidate filter. |
| [`test_imaging_flip.cpp`](../../modules/vision/imaging/tests/test_imaging_flip.cpp) | The plan and the application of the horizontal flip to an image and its labels. |
| [`test_imaging_gray.cpp`](../../modules/vision/imaging/tests/test_imaging_gray.cpp) | The expansion of a one-channel image to three equal channels. |
| [`test_imaging_hsv.cpp`](../../modules/vision/imaging/tests/test_imaging_hsv.cpp) | The hue, saturation and value jitter: its plan, its tables and both colour conversions. |
| [`test_imaging_image.cpp`](../../modules/vision/imaging/tests/test_imaging_image.cpp) | The owned image, the view over foreign pixels and the checks both share. |
| [`test_imaging_label_set.cpp`](../../modules/vision/imaging/tests/test_imaging_label_set.cpp) | The label set that holds classes, boxes and polygons in step, and its conversions. |
| [`test_imaging_labels.cpp`](../../modules/vision/imaging/tests/test_imaging_labels.cpp) | The parser of a label file's text. |
| [`test_imaging_letterbox.cpp`](../../modules/vision/imaging/tests/test_imaging_letterbox.cpp) | The plan and the application of the letterbox to an image and its labels. |
| [`test_imaging_mask_pipeline.cpp`](../../modules/vision/imaging/tests/test_imaging_mask_pipeline.cpp) | The nearest warp and a mask replayed from a training trace: a flip mirrored exactly, a mosaic mask holding each box's class inside the box, and the refusals. |
| [`test_imaging_mosaic.cpp`](../../modules/vision/imaging/tests/test_imaging_mosaic.cpp) | The draw, the plan and the application of the four-image mosaic. |
| [`test_imaging_partners.cpp`](../../modules/vision/imaging/tests/test_imaging_partners.cpp) | The two sources of a mosaic's three partner images. |
| [`test_imaging_pipeline.cpp`](../../modules/vision/imaging/tests/test_imaging_pipeline.cpp) | The training and the validation pipeline that compose the imaging transforms into a sample. |
| [`test_imaging_polygon.cpp`](../../modules/vision/imaging/tests/test_imaging_polygon.cpp) | Polygon resampling, mapping, clamping, the point test and the visible box. |
| [`test_imaging_random.cpp`](../../modules/vision/imaging/tests/test_imaging_random.cpp) | The generator every augmentation draws from. |
| [`test_imaging_raw_cache.cpp`](../../modules/vision/imaging/tests/test_imaging_raw_cache.cpp) | The raw image file: its bytes, its header checks and its file reader and writer. |
| [`test_imaging_rectangle.cpp`](../../modules/vision/imaging/tests/test_imaging_rectangle.cpp) | The plan that orders a dataset by aspect ratio and gives each batch one shape. |
| [`test_imaging_resize.cpp`](../../modules/vision/imaging/tests/test_imaging_resize.cpp) | The bilinear and the nearest-neighbour resize and the fit of an image's long side. |
| [`test_imported_model_writer.cpp`](../../modules/layers/io/tests/test_imported_model_writer.cpp) | The three files of an imported model read back, the key order of the training file with `data` null, existing files refused or replaced, and a bad tensor writing nothing. |
| [`test_importer_registry.cpp`](../../modules/layers/io/tests/test_importer_registry.cpp) | The importer rows in insertion order, a repeated format, a row's failure passed on and an unknown format listing the rows. |
| [`test_input_match.cpp`](../../modules/layers/train/tests/test_input_match.cpp) | describe_input_leaf and match_input: the input a leaf reads, and the status that names both element types or both shapes when a dataset publishes another. |
| [`test_ir.cpp`](../../modules/core/graph/tests/test_ir.cpp) | The IR's single-assignment and ordering rules and the builder's shape inference. |
| [`test_json_events.cpp`](../../modules/core/config/tests/test_json_events.cpp) | The reader that streams a JSON file as events: their order, a document longer than the buffer, a handler that stops, and each refusal with its byte offset. |
| [`test_json_writer.cpp`](../../modules/core/config/tests/test_json_writer.cpp) | How a `Config::Value` is written as JSON text: layout, numbers that read back exactly, escapes and the refusals. |
| [`test_layer_norm.cpp`](../../modules/layers/nn/tests/test_layer_norm.cpp) | The LayerNorm layer's trace, its gain and bias, its refusals and its factory. |
| [`test_layer_norm_backward.cpp`](../../modules/core/autograd/tests/test_layer_norm_backward.cpp) | That one gradient node serves the input, the gain and the bias of a layer normalisation and reads its saved statistics. |
| [`test_layer_norm_ops.cpp`](../../modules/core/graph/tests/test_layer_norm_ops.cpp) | The shape rules and IR of the layer normalisation emitter and of its gradient emitter. |
| [`test_layer_registry.cpp`](test_layer_registry.cpp) | Each layer registry row's factory, the keys it reads and the model it builds. |
| [`test_layer_stack.cpp`](../../modules/layers/nn/tests/test_layer_stack.cpp) | The runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread. |
| [`test_layout.cpp`](../../modules/core/autograd/tests/test_layout.cpp) | That every value carries a tensor layout and that the rebuilding passes keep it. |
| [`test_leaky_relu.cpp`](../../modules/layers/nn/tests/test_leaky_relu.cpp) | The LeakyReLU layer's trace, shape inference and slope range. |
| [`test_linked_rows.cpp`](../../modules/layers/nn/tests/test_linked_rows.cpp) | The three forms of a row reference and its refusals, the linked rows of a registry, the rows `build_model` hands a linked factory, and `BuiltRows`. |
| [`test_log_softmax_ops.cpp`](../../modules/core/graph/tests/test_log_softmax_ops.cpp) | The shape rules and IR of the log-softmax gradient emitter. |
| [`test_loss_roots.cpp`](test_loss_roots.cpp) | Which values differentiation accepts as a loss, the seed it emits and the K_STOP_GRADIENT barrier. |
| [`test_loss_scaling.cpp`](test_loss_scaling.cpp) | The dynamic loss-scaling rule on LossScaleState, with no device involved. |
| [`test_lowering_table.cpp`](../../modules/core/graph/tests/test_lowering_table.cpp) | That every operation identity has one lowering row or is listed as unlowered, across the eight domain tables. |
| [`test_max_pool2d.cpp`](../../modules/layers/nn/tests/test_max_pool2d.cpp) | The MaxPool2d layer's trace, shape inference, trailing padding and rejections. |
| [`test_memory_plan.cpp`](../../modules/core/graph/tests/test_memory_plan.cpp) | That the memory planner never shares a byte between two values live together. |
| [`test_metrics.cpp`](../../modules/layers/train/tests/test_metrics.cpp) | The order and lookup of a step's named numbers, and the names and counts it refuses. |
| [`test_mish.cpp`](../../modules/layers/nn/tests/test_mish.cpp) | That the Mish layer emits SOFTPLUS, TANH and MUL over its input, keeps the input's shape, element type and layout, and holds no state. |
| [`test_model_graph.cpp`](../../modules/layers/nn/tests/test_model_graph.cpp) | ModelGraph: what add and output refuse, what a trace and an incremental trace emit, and the state names. |
| [`test_model_outputs.cpp`](../../modules/layers/nn/tests/test_model_outputs.cpp) | ModelOutputs: publishing named values and reading them back. |
| [`test_model_writer.cpp`](../../modules/core/config/tests/test_model_writer.cpp) | That a `ModelSpec` written by `write_model` parses back to the same spec, and which specs it refuses. |
| [`test_module.cpp`](../../modules/layers/nn/tests/test_module.cpp) | Parameter across graphs, the fill a parameter declares, and the state walk over layers, containers and fields. |
| [`test_module_list.cpp`](../../modules/layers/nn/tests/test_module_list.cpp) | ModuleList: its storage, its index names and its place in the state walk. |
| [`test_node.cpp`](../../modules/layers/nn/tests/test_node.cpp) | ModuleNode, LayerNode and the buffer check LayerAdapter makes at compile time. |
| [`test_node_registry.cpp`](../../modules/layers/nn/tests/test_node_registry.cpp) | NodeRegistry, register_layers, make_node and build_model over the rows of the nn module. |
| [`test_objective.cpp`](../../modules/layers/train/tests/test_objective.cpp) | The objective seam: the leaves it declares, the loss it emits, its feed and its metrics, and the defaults of the base. The feed and metric cases need an engine. |
| [`test_objective_context.cpp`](../../modules/layers/train/tests/test_objective_context.cpp) | The extents `list_output_shapes` lists for a traced model, from a probe trace with no engine. |
| [`test_objective_registry.cpp`](../../modules/layers/train/tests/test_objective_registry.cpp) | The objective rows a training file may name, the context a factory receives, and the refusal of an unknown type. |
| [`test_one_hot.cpp`](../../modules/layers/train/tests/test_one_hot.cpp) | fill_one_hot: the matrix it writes from class indices and the labels and leaves it refuses. Its fixture cases need an engine. |
| [`test_ops.cpp`](../../modules/core/core/tests/test_ops.cpp) | That the operation table is complete and that the reverse lookup resolves names. |
| [`test_parameter_decay.cpp`](../../modules/core/optim/tests/test_parameter_decay.cpp) | That decay reaches only the slots that take it in both optimizers and both kernels, and that `decay_all` and its key override it. Needs an engine. |
| [`test_pixel_objectives.cpp`](../../modules/vision/training/tests/test_pixel_objectives.cpp) | What the two per-pixel objectives keep and refuse, the INT32 bound at 54 and 55 images of 512 by 512 over 150 classes, their leaves, the nodes of both losses, and both factories. |
| [`test_pixel_targets.cpp`](../../modules/vision/training/tests/test_pixel_targets.cpp) | The label check, the indices and weights of the per-pixel cross entropy, the four Dice leaves, and that a refused fill writes nothing. Needs an engine. |
| [`test_precision.cpp`](../../modules/core/autograd/tests/test_precision.cpp) | The mixed-precision policy and where the cast insertion pass places its casts. |
| [`test_provenance.cpp`](../../modules/core/autograd/tests/test_provenance.cpp) | The forward origin that differentiation records for every node it appends. |
| [`test_rate_schedule.cpp`](../../modules/core/optim/tests/test_rate_schedule.cpp) | That a schedule section is read by the type it states, the default of `step`, the unknown type, and the step-only reader's refusal of a cosine file. |
| [`test_region_boundary.cpp`](../../modules/core/eager/tests/test_region_boundary.cpp) | Which values `find_boundaries` lists as the imports and exports of a region, and which values share a buffer. |
| [`test_region_emitter.cpp`](../../modules/core/eager/tests/test_region_emitter.cpp) | That `RegionEmitter` records a region's program on the recorder it is handed, and counts its calls. Needs an engine. |
| [`test_region_executor.cpp`](../../modules/core/eager/tests/test_region_executor.cpp) | That `RegionExecutor` builds, keeps, records again and drops regions as a step says, and runs trailing work behind them. Needs an engine. |
| [`test_region_program.cpp`](../../modules/core/eager/tests/test_region_program.cpp) | The graph, the memory plan and the fusion report `compile_region` makes for one region. |
| [`test_region_signature.cpp`](../../modules/core/eager/tests/test_region_signature.cpp) | The signature that `signature_of` computes for one region of a partition. |
| [`test_region_table.cpp`](../../modules/core/eager/tests/test_region_table.cpp) | What `RegionTable` keeps, records again, builds and drops from one step to the next. |
| [`test_registration.cpp`](../../modules/language/registration/tests/test_registration.cpp) | The rows the language module's `register_nodes`, `register_datasets` and `register_objectives` add, and that a decoder written as a model file builds from them and publishes what the balance objective reads. |
| [`test_registration.cpp`](../../modules/vision/registration/tests/test_registration.cpp) | The rows the vision module's `register_nodes`, `register_datasets`, `register_objectives` and `register_importers` add, that each block row builds from a minimal object, and that the YOLO26n model file builds to the state and shapes MANIFEST.json records. The dataset row case needs an engine. |
| [`test_rematerialise.cpp`](../../modules/core/autograd/tests/test_rematerialise.cpp) | The structure of the graph that the K_IM2COL rematerialisation pass rewrites. |
| [`test_result.cpp`](../../modules/core/core/tests/test_result.cpp) | Result, its reference form and SA_TRY_ASSIGN. |
| [`test_role_match.cpp`](../../modules/layers/train/tests/test_role_match.cpp) | match_roles: the roles a dataset must publish for a step, and the status that names a missing or differing one. |
| [`test_rule_table.cpp`](../../modules/core/autograd/tests/test_rule_table.cpp) | That every domain's backward rules reach the combined table exactly once. |
| [`test_run_progress.cpp`](../../modules/layers/io/tests/test_run_progress.cpp) | The default of a run's progress, the names of its four records, and the refusal of a resume under another seed. |
| [`test_run_schedule.cpp`](../../modules/core/optim/tests/test_run_schedule.cpp) | That a run reads its rate from one section: a schedule beside `optimizer.learning_rate` is refused, and so is a schedule without a `base_rate` above zero. |
| [`test_run_state.cpp`](../../modules/core/optim/tests/test_run_state.cpp) | The run state an optimizer lists and restores: the names of SGD and AdamW, the step count, and the refusal of each scalar that does not fit. Its fixture cases need an engine. |
| [`test_scope_partitioner.cpp`](../../modules/core/eager/tests/test_scope_partitioner.cpp) | The region keys ScopePartitioner gives the nodes of a differentiated graph. |
| [`test_scope_trace.cpp`](../../modules/core/eager/tests/test_scope_trace.cpp) | How ScopeTrace cuts a trace into segments and which calls it rejects. |
| [`test_seam.cpp`](test_seam.cpp) | The Status that core/seam.hpp gives each exception type a guard catches, what the guards pass through, and that `record_and_run` discards what a failed recording recorded. The `record_and_run` cases need an engine. |
| [`test_section_reader.cpp`](../../modules/core/config/tests/test_section_reader.cpp) | The Status-returning readers of one section and the full text of each refusal. |
| [`test_segment_confusion_matrix.cpp`](../../modules/vision/segment/tests/test_segment_confusion_matrix.cpp) | Pixel counts by true and predicted class, the ignore index, sums over batches, merge, and the refusals that count nothing. |
| [`test_segment_metrics.cpp`](../../modules/vision/segment/tests/test_segment_metrics.cpp) | IoU per class and its mean over the classes that occur, pixel accuracy, an empty matrix, and the class pick with its tie. |
| [`test_segment_rasterise.cpp`](../../modules/vision/segment/tests/test_segment_rasterise.cpp) | The even-odd fill by pixel centres, a shared edge, a hole, clipping, the later outline winning an overlap, and the refusals. |
| [`test_segmentation_metrics.cpp`](../../modules/vision/training/tests/test_segmentation_metrics.cpp) | Mean IoU and pixel accuracy of one step by hand, and the steps refused. Needs an engine. |
| [`test_session.cpp`](../../modules/core/eager/tests/test_session.cpp) | The steps an eager `Session` runs against a compiled step, over a settled loop and across a changed scope, and the points at which a `Step` runs. Needs an engine. |
| [`test_shape.cpp`](../../modules/core/core/tests/test_shape.cpp) | Shape's extents, indexing, validation, equality and hash. |
| [`test_silu.cpp`](../../modules/layers/nn/tests/test_silu.cpp) | The SiLU layer's trace, shape inference and empty state. |
| [`test_slot_ledger.cpp`](../../modules/core/eager/tests/test_slot_ledger.cpp) | Which slot `SlotLedger` gives a shared buffer from one step to the next. |
| [`test_spatial_shapes.cpp`](../../modules/core/autograd/tests/test_spatial_shapes.cpp) | The shapes the builder infers for spatial operations and the layouts it refuses. |
| [`test_status.cpp`](../../modules/core/core/tests/test_status.cpp) | Status, its macros, the fatal family and the assertions of test_status.hpp. |
| [`test_step_cost.cpp`](../../modules/core/eager/tests/test_step_cost.cpp) | Prints the host cost of an eager step whose structure held, for two and for fifty layers, and of a scope that appears and disappears. Needs an engine. |
| [`test_step_recorder.cpp`](../../modules/core/graph/tests/test_step_recorder.cpp) | That StepRecorderList drives its members as one recorder, in the order added. |
| [`test_step_schedule.cpp`](../../modules/core/optim/tests/test_step_schedule.cpp) | The rate `Optim::StepSchedule::rate_at` gives under a constant policy, a warm-up and listed steps, and the schedules `validate_step_schedule` refuses. |
| [`test_step_schedule_section.cpp`](../../modules/core/optim/tests/test_step_schedule_section.cpp) | That a `StepSchedule` and the `schedule` section of a training file convert both ways without loss, and which sections and schedules are refused. |
| [`test_steps.cpp`](../../modules/layers/train/tests/test_steps.cpp) | The training, probe, evaluation, carrying and incremental graph builders, the role of each parameter binding, and the batch shape function. |
| [`test_storage_bank.cpp`](../../modules/core/eager/tests/test_storage_bank.cpp) | That `StorageBank` holds, allocates and lets go of tensors by storage id. Needs an engine. |
| [`test_sum_objective.cpp`](../../modules/layers/train/tests/test_sum_objective.cpp) | The composite objective: its terms, the refusal of a step-fed term, the prefixed leaves, the emitted SCALE and ADD, the first-wins metric merge and the `sum` factory. |
| [`test_target.cpp`](../../modules/core/data/tests/test_target.cpp) | find_target over the targets of a batch and over the descriptions of a dataset. Its fixture cases need an engine. |
| [`test_text_bpe_trainer.cpp`](../../modules/language/text/tests/test_text_bpe_trainer.cpp) | Two hand-worked merge orders, the tie rule, the chunk edge, determinism over a seeded corpus, the stop rules and the size bounds. |
| [`test_text_codec.cpp`](../../modules/language/text/tests/test_text_codec.cpp) | The round trip of seeded random bytes and invalid UTF-8, merge order in encoding, a text of several chunks, special tokens in decoding and the id decode refuses. |
| [`test_text_stream_encoder.cpp`](../../modules/language/text/tests/test_text_stream_encoder.cpp) | The ids of a stream read in pieces of five sizes against those of the whole text. |
| [`test_text_vocabulary.cpp`](../../modules/language/text/tests/test_text_vocabulary.cpp) | The id ranges of bytes, special and merged tokens, the lookups, and the merges and names make refuses. |
| [`test_text_vocabulary_file.cpp`](../../modules/language/text/tests/test_text_vocabulary_file.cpp) | The documented bytes, the round trip, and the refusal of every truncated length, trailing bytes, another magic or version and fields out of bounds, each with its offset. |
| [`test_threaded_loader.cpp`](../../modules/core/data/tests/test_threaded_loader.cpp) | Index order on worker threads, no read further ahead than `prefetch`, a failed read as the status of its index alone, an index out of order, `begin_epoch` forwarded, the destructor with reads pending. No device, no clock. |
| [`test_throwing_bridge.cpp`](../../modules/layers/nn/tests/test_throwing_bridge.cpp) | bridge_throwing: a value passes through and a thrown Error becomes a Status. |
| [`test_trace.cpp`](../../modules/layers/nn/tests/test_trace.cpp) | Trace, the buffer-update sinks, the fed leaves and carried values of a carrying trace, the three layer forms and trace_layer. |
| [`test_traced.cpp`](../../modules/core/graph/tests/test_traced.cpp) | Traced and the refusal it carries through a chain of emitters. |
| [`test_train_config.cpp`](../../modules/core/config/tests/test_train_config.cpp) | That `TrainSpec` carries the `schedule` section unread, resolves `weights` and `resume` against the file and refuses the two together, reads `seed` as 0 when it is absent and refuses a negative one, and takes a data block with `reader` and no `type` whole. |
| [`test_training_expert_balance_objective.cpp`](../../modules/language/training/tests/test_training_expert_balance_objective.cpp) | The mean of the balance outputs, the factory's reading of outputs by name, its refusals, and `expert_usage_min` and `expert_usage_max`. The metrics suite needs an engine. |
| [`test_training_next_token_objective.cpp`](../../modules/language/training/tests/test_training_next_token_objective.cpp) | The int32 leaf and the labels role, the gather-built loss graph, the int32 limit, the factory's refusals and the registry row. |
| [`test_training_next_token_targets.cpp`](../../modules/language/training/tests/test_training_next_token_targets.cpp) | The index of each position's label, the labels and sizes the fill refuses, and the perplexity and metrics. Needs an engine. |
| [`test_truth_from_roles.cpp`](../../modules/vision/training/tests/test_truth_from_roles.cpp) | The three roles a detection step requires, their copy into a `GroundTruthBatch`, the storage kept between batches, and the refusals. Needs an engine. |
| [`test_unflatten.cpp`](../../modules/layers/nn/tests/test_unflatten.cpp) | The trace, the shape rule and the rejected inputs of the Unflatten layer. |
| [`test_upsample2d.cpp`](../../modules/layers/nn/tests/test_upsample2d.cpp) | The shape, layout and rejections of the Upsample2d layer. |
| [`test_version.cpp`](../../modules/core/version/tests/test_version.cpp) | That the runtime version agrees with the compile-time macros. |

## modules/layers/nn/tests/test_add.cpp

Suites: `Unit_Add`.

Tests `NN::Add` in `modules/layers/nn/include/SushiAI/nn/add.hpp` and its factory
`NN::make_add_node`. Unless a case says otherwise, the node is built over two [8, 6, 3] shapes
and traced over two [2, 8, 6, 3] inputs.

### CreateKeepsTheSharedShapeAsThePort

The node has one port, named "output", of the shape [8, 6, 3].

### CreateRejectsOneInput

`create` over one shape fails with `InvalidDescription`, and the message holds "takes 2 inputs"
and "was given 1".

### CreateRejectsThreeInputs

`create` over three shapes fails with "takes 2 inputs" and "was given 3".

### CreateRejectsNoInput

`create` over an empty list fails with "takes 2 inputs" and "was given 0".

### CreateRejectsUnequalChannels

[8, 6, 3] with [8, 6, 5] fails with `InvalidDescription` and "share one shape".

### CreateRejectsUnequalHeight

[8, 6, 3] with [4, 6, 3] fails with "share one shape".

### CreateRejectsUnequalWidth

[8, 6, 3] with [8, 12, 3] fails with "share one shape".

### CreateRejectsAnInputThatIsNotAFeatureMap

[8, 6, 3] with [144] fails with a message holding "add input 1" and "rank-3".

### CreateRejectsAFirstInputThatIsNotAFeatureMap

[144] with [144] fails with a message holding "add input 0" and "rank-3".

### TheInputCountIsTwo

`input_count()` is 2.

### ForwardEmitsOneAddReadingBothInputsInOrder

`forward` over the inputs b and a adds one node. Its operation is `Ops::K_ADD`, its inputs are
b then a, and its one output is the one value `forward` returned.

### ForwardResultIsThePortShapeWithABatchAxis

The traced value has the shape [2, 8, 6, 3], and the port's shape stays [8, 6, 3].

### ForwardRejectsAnInputCountItWasNotBuiltOver

`forward` over one input fails with `InvalidDescription`, "input count is 2" and "given is 1".
It returns no output and adds no node.

### ForwardReturnsTheFailureOfValuesThatDoNotAdd

`forward` over a [2, 8, 6, 3] value and a [2, 8, 6, 5] value returns an `InvalidDescription`
status, no output, and adds no node.

### TheFactoryBuildsANodeFromARowWithNoKey

`make_add_node` over two [8, 6, 3] shapes and an empty row returns an `Add` with two inputs and
one port of the shape [8, 6, 3].

### TheFactoryRefusesAKeyAndShapesThatDoNotMatch

A row with the key "alpha" fails with a message holding "layer 3 'add'", "'alpha'" and
"no key". An empty row over [8, 6, 3] and [8, 6, 5] fails with "layer 3 'add'" and
"share one shape".

### TheStateWalkVisitsNothing

After a trace, `walk_state` calls neither the parameter visitor nor the buffer visitor.

## modules/layers/nn/tests/test_avg_pool2d.cpp

Suites: `Unit_AvgPool2d`.

Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

This file needs no device. `NN::AvgPool2d` in `modules/layers/nn/include/SushiAI/nn/avg_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter, so a case either traces the layer into a host-side
graph or calls `output_shape()` on a per-sample shape.

### window

`window` returns a geometry with the given kernel and stride on each axis, dilation 1 and no
padding. A case that needs padding or dilation sets the field afterwards.

### traced_shape

`traced_shape` traces a layer over a batch of two images of the given per-sample shape and
returns the shape of the result, batch axis included.

### per_sample

`per_sample` drops the batch axis of a rank-4 shape.

### SatisfiesTheModuleTrait

Four `static_assert` lines hold the layer's place in the layer library: `AvgPool2d` is a
`Module`, `const AvgPool2d` is one too, which requires a const `forward()`, it is not a
`ModuleContainer`, and it has no parameter walk of its own, so the walk reaches it through
`fields()`.

### ForwardTracesThePooledShapeInNHWC

A 3x3 window with stride 3 over a [2, 6, 6, 3] input traces a [2, 2, 2, 3] value tagged NHWC.

### OutputShapeAgreesWithTheTracedShapeForASquareWindow

`output_shape([6, 6, 3])` is [2, 2, 3], and it equals the traced shape with the batch axis
dropped. The literal is asserted beside the comparison so the two cannot agree on a wrong
shape. The next two cases have the same form.

### OutputShapeAgreesWithTheTracedShapeForAPaddedUnevenWindow

A 3x2 window with stride 2 by 1 and one pixel of padding on each axis maps [7, 5, 4] to
[4, 6, 4].

### OutputShapeAgreesWithTheTracedShapeForADilatedWindow

A 3x3 unit-stride window dilated by 2 on the height axis maps [9, 8, 1] to [5, 6, 1]. The two
axes have different dilations, so an implementation that applied one axis's dilation to both
gives another shape.

### OutputShapeRejectsAFeatureVector

`output_shape([8])` throws, because a per-sample shape must be rank 3.

### OutputShapeRejectsABatchedShape

`output_shape([2, 4, 4, 3])` throws. A shape that still carries the batch axis is rank 4.

### OutputShapeRejectsAWindowLargerThanTheImage

A 5x5 window over a [4, 4, 3] sample throws.

### OutputShapeRejectsAZeroStride

A geometry with stride 0 on both axes throws.

### OutputShapeNamesAZeroStride

The message for the same geometry contains "AvgPool2d" and "stride 0,0". The case holds the
message to the stride, where a check ordered differently would report a window that does not
fit.

### OutputShapeRejectsANegativePadding

A top padding of -1 throws with a message containing "AvgPool2d" and "padding -1,0".

### ForwardRejectsARankTwoInput

`forward()` throws on a [2, 8] input.

### ForwardRejectsAnNCHWInput

A [1, 6, 6, 3] input is converted with `layout_convert` to NCHW, and `forward()` throws on the
converted value.

### ForwardNamesTheLayoutOfAnNCHWInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

### ForwardRejectsAZeroStride

`forward()` with stride 0 on both axes throws with a message containing "AvgPool2d" and
"stride 0,0".

### ForwardRejectsANegativePadding

`forward()` with a top padding of -1 throws with a message containing "AvgPool2d" and
"padding -1,0".

### ForwardRejectsAWindowLargerThanTheImage

`forward()` throws for a 5x5 window over a [2, 4, 4, 3] input.

### ATracedLayerHasNoParameter

After a trace the parameter walk visits nothing and `parameter_count` is 0. The graph holds
two values, the input and the pooled result.

## test_backward.cpp

Suites: `Unit_Backward`.

Tests the graph that differentiation emits and the coverage of the backward rule table.

This file needs no device. Differentiation is an IR-to-IR transform, so what it produces is a
graph that can be inspected node by node with no hardware involved. That is why
`modules/core/graph/include/SushiAI/graph/ir.hpp` insists that a backward rule emits IR and never device work.

The cases from DivReadsItsDivisorAndItsOwnQuotient on each put one operation under an K_NLL_LOSS
root and compare the appended nodes with a literal list of operation names. The first appended
node is always the K_SCALE that K_NLL_LOSS's rule emits, and its result is the gradient the tested
operation receives. The cases then check which operands each appended node reads, since a rule
that emits the right operations over the wrong operands gives a graph of the same shape.

### K_ATTRIBUTE_TOLERANCE

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule writes as an exact
constant.

### K_UNPROVEN_RULES

`K_UNPROVEN_RULES` is 8, the number of rules no finite-difference case names.
EveryRuleInTheTableIsNamedByANumericTest compares the size of its `unproven` list with it, so
adding a rule to that list takes an edit in two places.

### ForwardCase

`ForwardCase` holds a failure label and a function that emits one forward operation over two
parameters.

### Appended

`Appended` holds what one differentiation appended: the index of the first appended node, the
gradient the tested operation receives, and one gradient per requested value.

### differentiate_through

`differentiate_through` declares a `targets` input in the shape of a rank-2 result, puts
`nll_loss(result, targets)` over it and differentiates. It returns an `Appended` whose `seed`
is the first result of the first appended node.

### appended_node

`appended_node` returns the node a given number of places after the first appended one.

### EveryImplementedOperationHasARule

`has_backward_rule` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU.

`has_backward_rule` is false for K_LESS_EQUAL, K_CONSTANT and K_STOP_GRADIENT. Each is a gradient
barrier: the reverse walk steps over it, so no rule is ever asked of it.

`has_backward_rule` is false for K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD. An activation gradient
has no rule, so a second differentiation over one throws.

`has_backward_rule(K_BATCHNORM_UPDATE_STATS)` is false, and that is decision CV-7, not an
omission. A running mean is a buffer the training step writes beside the graph. No gradient
flows through it.

`has_backward_rule(K_BATCHNORM_FOLD)` is false too. The fold is the node an evaluation trace of
`NN::BatchNorm2d` emits, and DifferentiatingThroughBatchNormFoldFails holds what a gradient
request through it does.

`has_backward_rule` is false for K_LEAF and K_ZERO because both produce values from nothing. There
is nothing upstream to hand a gradient to, so there is no rule to write.

`has_backward_rule(K_CAST)` is false because of the order of the pipeline. Casts are inserted
after differentiation (`modules/core/graph/include/SushiAI/graph/cast_insertion.hpp` says why), so no graph this
transform sees can contain one. A graph that somehow did would fail loudly and name the
operation. It would not differentiate a conversion whose precision the pass had not established
yet.

### EveryRuleInTheTableIsNamedByANumericTest

The `ledger` vector records coverage; the test is not a tautology. "Every operation has a rule"
is the easy half. The half that rots is "every rule has been checked against a finite
difference". `differentiate()` picks up a rule added to a `modules/core/autograd/source/rules/<domain>.cpp`
table automatically, so a wrong rule produces a model that trains slightly worse and nothing
anywhere fails. Listing the proof beside the rule makes the omission visible: the test's second
loop, the one over `Ops::K_ALL`, fails when an operation gains a rule that nobody added a ledger
line for.

The named tests live in `tests/integration/test_autograd_numeric.cpp` and all run in
FLOAT64, for the reason `modules/core/autograd/include/SushiAI/autograd/gradcheck.hpp` derives.

The K_LOG_SOFTMAX entries name the cases of `tests/integration/test_attention_gradient.cpp`:
the unfused rule against its closed form, a softmax, and a causal attention block.

The `unproven` vector lists the eight rules no finite-difference case names: K_DIV, K_MINIMUM,
K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU. Its size must equal `K_UNPROVEN_RULES`, every
entry must have a rule, and no operation may appear in both the ledger and this list. The loop
over `Ops::K_ALL` accepts a rule that either list names. An entry leaves `unproven` when a
finite-difference case names its rule in the ledger.

### TwoLayerMLPProducesTheExpectedBackwardTopology

The `op_names` comparison expects the cross-entropy pair to be differentiated as one fused
rule: exp of the log-probabilities, minus the one-hot targets, scaled by 1/N. The chain then
walks back through the second linear layer, the activation and the first linear layer.

`count_op(graph, K_ACCUMULATE)` is 0 because every value in a plain feed-forward stack has
exactly one contributor, so the tiered accumulation emits nothing.

### TheInputIsNotDifferentiatedWhenItDoesNotRequireOne

The backward pass holds one K_MATMUL, the weight gradient, which with the forward one makes the
two the test counts. The input gradient would have been another K_MATMUL. It is skipped, not
computed and discarded.

### FanOutEmitsAnAccumulateChain

`x` feeds two branches, so its gradient has two contributors and the tiered accumulation drops
to its ordered K_ACCUMULATE form.

### ThreeContributorsChainTwoAccumulates

A parameter feeds a ReLU, a tanh and a sigmoid whose sum is the loss operand. After
`differentiate` the graph holds two ACCUMULATE nodes.

### ADisconnectedParameterGetsASymbolicZeroNotABuffer

A zero gradient must cost nothing: no K_ZERO node, no allocation, no wasted kernel. Only the live
path emitted anything, and what it emitted is the loss gradient and the activation gradient.

### ADeadLossPathEmitsNothingAtAll

Nothing in this graph requires a gradient, so the reverse walk visits every node and writes
none of them. A frozen model costs zero nodes. It does not cost a set of nodes computing
gradients nobody reads.

### AnOperationWithNoRuleThrowsAndNamesItself

The `mystery` value is produced by a synthetic identity that no rule covers. It stands in for a
future operation added to the lowering but not to the rule table.

### ANonScalarRootIsRejected

`differentiate` from a [4, 3] ReLU output throws `Error`.

### LogSoftmaxOutsideTheFusedPairAppendsItsGradientIdentity

A RELU between LOG_SOFTMAX and NLL_LOSS keeps the fusion away. The walk appends SCALE,
RELU_BACKWARD and one LOG_SOFTMAX_BACKWARD over the incoming gradient and the forward result.

### TheFusedPairAppendsNoLogSoftmaxGradient

NLL_LOSS directly over LOG_SOFTMAX appends EXP, SUB and SCALE, and no LOG_SOFTMAX_BACKWARD.

### ALogSoftmaxWhoseOperandNeedsNoGradientAddsNoNode

A log-softmax over an input, multiplied by a parameter, appends no gradient node of its own.

### ASoftmaxDifferentiatesThroughItsExponentialAndItsLogSoftmax

The LOG_SOFTMAX_BACKWARD of a softmax reads the MUL that the rule of EXP emitted and the
forward log-probabilities.

### DivReadsItsDivisorAndItsOwnQuotient

For `c = a / b` with both operands parameters, the appended nodes are K_SCALE, K_DIV, K_MUL, K_SCALE.
The K_DIV reads the seed and `b` and is the gradient of `a`. The K_MUL reads that quotient and the
forward result `c`, and the last K_SCALE negates it with the factor -1 and is the gradient of
`b`. The rule reuses `c` and does not divide by `b` a second time.

### DivEmitsOnlyTheQuotientWhenTheDivisorNeedsNoGradient

With `b` an input, the appended nodes are K_SCALE and K_DIV.

### DivEmitsTheDivisorGradientAloneWhenTheDividendNeedsNone

With `a` an input, the appended nodes are still K_SCALE, K_DIV, K_MUL, K_SCALE, because the divisor's
gradient is computed from the quotient. The one returned gradient is the result of the last
K_SCALE.

### MinimumRoutesByALessEqualBSoATieGoesToTheFirstOperand

For `minimum(a, b)` the appended nodes are K_SCALE, K_LESS_EQUAL, K_ZERO, K_SELECT, K_SELECT. The mask
reads `a` then `b`. The first K_SELECT reads the mask, the seed and the zero and is the gradient
of `a`; the second reads the mask, the zero and the seed and is the gradient of `b`. Where the
operands are equal the mask holds, so the first operand takes the gradient.

### MaximumRoutesByBLessEqualASoATieGoesToTheFirstOperand

For `maximum(a, b)` the node list is the same and the mask reads `b` then `a`. The two K_SELECT
nodes read the same operands as in the K_MINIMUM case, so a tie again goes to the first operand.

### MinimumEmitsOneSelectWhenOnlyTheSecondOperandNeedsAGradient

With `a` an input, one K_SELECT is appended, and it reads the mask, the zero and the seed.

### MaximumEmitsOneSelectWhenOnlyTheFirstOperandNeedsAGradient

With `b` an input, one K_SELECT is appended. The mask reads `b` then `a`, and the K_SELECT reads
the mask, the seed and the zero.

### AbsMultipliesByASignThatIsZeroAtZero

For `abs(x)` the appended nodes are K_SCALE, K_ZERO, K_LESS_EQUAL, K_LESS_EQUAL, K_SUB, K_MUL. The K_ZERO has
the shape of `x`. The first mask reads the zero then `x`, the second reads `x` then the zero,
the K_SUB reads the first mask then the second, and the K_MUL reads the seed and the difference.
The sign is `[0 <= x] - [x <= 0]`, which is 0 at x = 0.

### AtanDividesByOnePlusTheSquaredInput

For `atan(x)` the appended nodes are K_SCALE, K_MUL, K_CONSTANT, K_ADD, K_DIV. The K_MUL reads `x` twice.
The K_CONSTANT has no input, holds 1 and has the shape and element type of `x`. The K_ADD reads the
square and the one, and the K_DIV reads the seed and that sum.

### SoftplusBackwardReadsTheForwardInputNotTheOutput

For `softplus(x)` one K_SOFTPLUS_BACKWARD is appended after the K_SCALE, and it reads the seed and
`x`. The case asserts that `x` and the forward result are different values first, so the
operand check distinguishes the input from the output.

### SiLUBackwardReadsTheForwardInputNotTheOutput

For `silu(x)` one K_SILU_BACKWARD is appended after the K_SCALE, and it reads the seed and `x`.

### SelectRoutesItsGradientByItsOwnCondition

For `select(condition, a, b)` the appended nodes are K_SCALE, K_ZERO, K_SELECT, K_SELECT. The first
K_SELECT reads the condition, the seed and the zero and is the gradient of `a`. The second reads
the condition, the zero and the seed and is the gradient of `b`.

### SelectEmitsOneSelectWhenOnlyTheSecondBranchNeedsAGradient

With `a` an input, the appended nodes are K_SCALE, K_ZERO, K_SELECT, and the K_SELECT reads the
condition, the zero and the seed.

### SelectGivesItsConditionNoGradient

With the condition a parameter and both branches inputs, only the K_SCALE is appended and the
condition's gradient is `K_NO_VALUE`.

### AComparisonUnderASelectLeavesItsOperandsWithoutAGradient

The condition is `less_equal(p, q)` with `p` a parameter. The gradient of `p` is `K_NO_VALUE`,
the gradient of the branch `a` exists, and the appended nodes are K_SCALE, K_ZERO, K_SELECT.

### AComparisonMultipliedIntoAProductEmitsNothingForTheMask

For `mul(less_equal(p, q), a)` differentiation does not throw. The mask does not require a
gradient, the gradient of `p` is `K_NO_VALUE`, and the appended nodes are K_SCALE and one K_MUL that
reads the seed and the mask and is the gradient of `a`. K_MUL's rule emits no node for its mask
operand.

### AConstantAddedToAParameterPathEmitsNothingForTheConstant

For `(x + epsilon) * epsilon`, with `epsilon` a `constant_like` of `x`, differentiation does
not throw. The constant does not require a gradient, and the appended nodes are K_SCALE and one
K_MUL that reads the seed and the constant and is the gradient of `x`.

### MaximumOfXAndAZeroConstantRoutesATieToX

For `maximum(x, floor)` with `floor` a zero K_CONSTANT, the mask reads `floor` then `x`, and the
K_SELECT reads the mask, the seed and the zero. The mask `[floor <= x]` holds at x = 0, so `x`
takes the gradient there.

### MaximumOfAZeroConstantAndXGivesTheClampedOperandNothingAtATie

For `maximum(floor, x)` the mask reads `x` then `floor`, and the K_SELECT reads the mask, the
zero and the seed. The mask `[x <= floor]` holds at x = 0, so `x` takes the zero branch there.
The operand order of a clamp therefore decides its gradient at the bound. The case also checks
that no node of the graph, forward or appended, narrows under the mixed-precision policy, so
the clamp stays in the loss's precision.

### NoRuleOfTheElementwiseOperationsAppendsANodeThatNarrows

For each of `div`, `minimum`, `maximum`, `abs`, `atan`, `softplus`, `silu` and a `select` on a
`less_equal` mask, the case differentiates the operation over two parameters. The rule appends
a node beyond the K_SCALE, and `Graph::narrows_under_mixed_precision` is false for every appended
node.

### AGradientOnTheBatchVarianceOfBatchNormThrows

`GraphBuilder::batchnorm2d` returns four values, and `batch_variance` is output 3 of the node.
A loss built on a K_SCALE of that value is differentiated against gamma. `differentiate` throws
with a message containing "K_BATCHNORM2D" and "saved statistics received a gradient".

### BatchNormWithFourOutputsStillDifferentiatesThroughItsFirst

A loss built on `sum_channels` of output 0 is differentiated against gamma and beta. Both
gradients are values other than `K_NO_VALUE` and the graph holds one K_BATCHNORM2D_BACKWARD node.
Inputs 3 and 4 of that node are the saved mean and the saved inverse deviation. The batch
variance is not among its operands.

### DifferentiatingThroughBatchNormFoldFails

A loss built on a K_CHANNEL_AFFINE over the two results of `batchnorm_fold` is differentiated
against gamma, a parameter. `differentiate` throws with a message containing "No backward
rule" and "sushiai.batchnorm_fold".

### ABufferInWrtIsRefusedAndAppendsNoNode

A request that lists a weight and a buffer named "running_shift" throws with a message
containing "differentiate()", "running_shift" and "buffer", and the graph's node count is
unchanged. The same graph then differentiates against the weight alone: the one gradient has
the lifetime `GRADIENT`, and the buffer keeps the lifetime `BUFFER`.

## modules/layers/nn/tests/test_batch_norm2d.cpp

Suites: `Unit_BatchNorm2d`, `Unit_BatchNorm2dDeathTest`.

Tests BatchNorm2d's two trace forms and the running-statistic update.

This file needs no device. `NN::BatchNorm2d` in `modules/layers/nn/include/SushiAI/nn/batch_norm2d.hpp` is a
modal layer. `trace_training` emits one K_BATCHNORM2D node and requests two buffer updates
through the trace. `trace_evaluation` reads the two running statistics as buffer leaves and
emits a K_BATCHNORM_FOLD and a K_CHANNEL_AFFINE. Both forms record a refusal on the trace and
return their input.

The last two cases read `tests/fixtures/yolo26/batchnorm.sacp` through
`tests/common/reference_fixture.hpp` and report themselves skipped when the file is absent.
They compute the updated statistics on the host from the momentum and the source scale the
trace requested. No device runs the update.

Two `static_assert` lines after the helpers hold that `BatchNorm2d` is a `ModalModule` and
neither a `Module` nor a `TraceModule`, and that `Sequential<Conv2d, BatchNorm2d>` is a
`TraceModule` and not a `Module`.

### Constants

`K_REFERENCE_EPSILON` is 1e-3 and `K_REFERENCE_MOMENTUM` is 0.03, the two values the reference
layers ran with. `K_UNIT_CASE_TOLERANCE` is 5.3e-7 and `K_MODEL_CASE_TOLERANCE` is 7.0e-7. Each
is four float32 epsilons of the case's largest running variance, 1.1086 and 1.4578.
`K_EXACT_TOLERANCE` is 1e-15, for a double computed the same way on both sides.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### only_node

`only_node` returns the first node of a graph that carries one operation. Each caller has
asserted a count of one before it.

### count_buffer_leaves

`count_buffer_leaves` counts the values of a graph whose lifetime is `BUFFER`.

### ChannelStatistics

`ChannelStatistics` holds the per-channel mean and biased variance of one NHWC activation.

### batch_statistics

`batch_statistics` computes a `ChannelStatistics` over batch, height and width from a flat
NHWC array, in double.

### updated_vector

`updated_vector` applies `NN::updated_statistic` per channel and rounds each result to
float32.

### view

`view` returns a `SushiRuntime::span<const float>` over a vector, the argument
`Test::near_reference` takes.

### expect_reference_case

`expect_reference_case` reads one case of `batchnorm.sacp`: an input and the running mean and
variance before and after one torch training step. It traces a layer of the input's channel
count with the reference epsilon and momentum, and applies the two requested updates to the
"before" statistics on the host. The results must be within the tolerance of the "after"
statistics. The variance recomputed with a source scale of 1.0 must not be, so the fixture
tells the unbiased update from the biased one.

### refused_training

`refused_training` traces the training form, expects the input id back and returns the trace's
`status()`.

### refused_evaluation

`refused_evaluation` does the same for the evaluation form.

### train_under_an_evaluation_trace

`train_under_an_evaluation_trace` calls `trace_training` with an evaluation trace. It is the
statement of a death test.

### evaluate_over_a_value_the_graph_lacks

`evaluate_over_a_value_the_graph_lacks` calls `trace_evaluation` over value 7 in a graph of
one value. It is the statement of a death test.

### ATrainingTraceEmitsOneBatchNorm

A training trace over [2, 2, 2, 3] emits one K_BATCHNORM2D node and no K_BATCHNORM_FOLD or
K_CHANNEL_AFFINE. The graph gains three nodes: the two parameter leaves and the K_BATCHNORM2D. The
node reads the input, the weight's leaf and the bias's leaf, has four outputs of which the
first is the returned value, and carries the epsilon 1e-3 in attribute 0.

### ATrainingTraceReadsNoBuffer

After a training trace the graph holds no value of lifetime `BUFFER`, and neither running
statistic has a leaf in it.

### ATrainingTraceRequestsTheTwoRunningUpdates

The list holds two updates. The first reads output 1 of the node and targets the running
mean's state, and the second reads output 3 and targets the running variance's state. The two
states are distinct and neither is `K_NO_STATE`. Both momenta are 0.03 and the mean's source
scale is 1.0.

### TheVarianceUpdateCarriesTheUnbiasedFactor

The variance update's source scale is n / (n - 1), where n is the number of values per channel.
For [2, 2, 2, 3] it is 8/7, and for [1, 20, 20, 128] it is 400/399.

### AnEvaluationTraceFoldsTheRunningStatistics

An evaluation trace over [2, 4, 4, 3] emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE and no
K_BATCHNORM2D, and the graph holds two buffer leaves. It gains six nodes: two parameter leaves,
two buffer leaves, the fold and the affine. The fold reads the weight, the bias, the running
mean and the running variance in that order and carries the epsilon 1e-3. The affine reads the
input and the fold's two outputs, and its output is the returned value of shape [2, 4, 4, 3].

### BothFormsNameTheSameState

One layer is traced in training form into one graph and in evaluation form into another. In
the evaluation graph, `state_leaf` of each update's target is the leaf of the matching running
statistic. The weight's and the bias's states are unchanged by the second trace, and each has
a leaf in the evaluation graph.

### TheWalkYieldsTwoParametersThenTwoBuffers

`NN::walk_state(layer, "bn", visitor)` over a traced five-channel layer yields "bn.weight",
"bn.bias", "bn.running_mean" and "bn.running_var" in that order, each of shape [5]. Both
parameters have a constant fill. The starting values are 1.0 for the weight, 0.0 for the bias,
0.0 for the running mean and 1.0 for the running variance.

### TrainingOverOneValuePerChannelFails

A training trace over [1, 1, 1, 3] records `InvalidDescription` with a message containing
"BatchNorm2d(3)", "more than one value per channel" and "got 1". The graph gains no value and
no node, the list holds no update, the weight is not declared, and the graph is marked refused.

### AnIntegerInputFails

Over an INT32 input, both forms record `InvalidDescription` with a message containing
"BatchNorm2d(3)", "half, float32 or float64 input" and "int32". Nothing is added to the graph
or the list, and neither the weight nor the running mean is declared.

### EvaluationOverOneValuePerChannelTraces

An evaluation trace over [1, 1, 1, 3] returns a value of that shape. The limit that
TrainingOverOneValuePerChannelFails holds belongs to the training form.

### AnInputThatDoesNotFitFails

Over a [2, 2, 2, 4] input both forms fail with a message containing "BatchNorm2d(3)" and
"[N, H, W, 3]". Over a [2, 3] input both fail with "rank-4". The graph gains no value and the
list no update.

### AnNCHWInputFails

Over a value converted to NCHW both forms fail with a message containing "BatchNorm2d(3)" and
"NHWC", and the graph gains no value.

### ALayerDescribedWrongFails

A default-constructed layer fails with "positive channel count" and "got 0". A layer with
epsilon 0.0 fails with "positive epsilon". A layer with momentum 1.5 fails with "momentum in
[0, 1]" and "1.5". The graph gains no value.

### ASecondTraceUnderAnotherElementTypeFails

A layer traced over a FLOAT32 input and then, in a second graph, over a FLOAT64 input fails
the second time with a message containing "'weight'" and both type names. The running mean's
type stays FLOAT32.

### TheScaleStartsAtOneAndTheShiftAtZero

After a trace the weight has the role `ParameterRole::Scale` and a constant fill of 1.0, and
the bias has the role `BIAS` and a constant fill of 0.0.

### TheFactoryReadsTheChannelsAndTheTwoConstants

`NN::make_batch_norm2d_node` over the shape [8, 8, 6] with `bn_epsilon` 0.001 and `bn_momentum`
0.03 returns a `ModuleNode<BatchNorm2d>` whose layer has 6 channels and those two options. The
node has one port of the input's shape and an input count of 1.

### TheFactoryRefusesARowThatDoesNotFit

The key `channels` fails with a message containing the context, "'channels'" and "bn_epsilon".
The input shape [48] fails with "[48]" and "[H, W, C]". A `bn_epsilon` of -1.0 fails with
"'bn_epsilon'" and "positive".

### TheTrainingFormUnderAnEvaluationTraceIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_training` given an evaluation
trace stops the process with a report containing "trace_training" and "was given an evaluation
trace".

### ATraceOverAValueTheGraphLacksIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_evaluation` over value 7 stops the
process with a report containing "BatchNorm2d(3)", "value 7" and "holds 1 values".

### TraceLayerPicksTheFormOfTheMode

`NN::trace_layer` under a training trace leaves one K_BATCHNORM2D and no K_BATCHNORM_FOLD in its
graph. Under an evaluation trace it leaves one K_BATCHNORM_FOLD and no K_BATCHNORM2D.

### OutputShapeIsTheInputShape

`output_shape([7, 5, 3])` is [7, 5, 3]. A three-channel layer also returns [7, 5, 4] for
[7, 5, 4]: `output_shape` does not check the channel count.

### UpdatedStatisticBlendsTheScaledBatchValue

`NN::updated_statistic(running, batch, momentum, scale)` is
(1 - momentum) * running + momentum * scale * batch. (1.0, 3.0, 0.25, 2.0) gives 2.25 and
(4.0, 8.0, 0.5, 1.0) gives 6.0. A momentum of 0 returns the running value 4.0 and a momentum
of 1 returns the batch value 8.0. Every operand and result is exact in a double.

### TheUpdatesReproduceTheUnitReferenceCase

`expect_reference_case("unit", K_UNIT_CASE_TOLERANCE)` checks the [2, 2, 2, 3] case. The
test's own comment gives the miss of the unscaled variance on this case as 8.3e-3.

### TheUpdatesReproduceTheModelReferenceCase

`expect_reference_case("model.9.cv1.bn", K_MODEL_CASE_TOLERANCE)` checks the case recorded at
that layer of the reference model. The test's own comment gives the miss of the unscaled
variance on this case as 1.4e-5.

## modules/layers/nn/tests/test_buffer.cpp

Suites: `Unit_Buffer`, `Unit_BufferDeathTest`.

Tests Buffer: its description, its one leaf per graph and its identity.

This file needs no device. `NN::Buffer` in `modules/layers/nn/include/SushiAI/nn/buffer.hpp` is state a model
owns that takes no gradient. `define()` records its shape, starting value and element type and
returns a `Status`. `read()` declares a leaf of lifetime `Graph::Lifetime::Buffer` in the graph
of the builder it is given.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### read_undefined

`read_undefined` calls `read()` on a buffer under the name "running_mean" with a builder over
a fresh graph. It is the statement of the death test.

### ANewBufferIsUndefined

A default-constructed buffer reports `declared()` false and the state `Graph::K_NO_STATE`.

### DefineRecordsTheDescription

After `define([8], 1.0, FLOAT64)` the buffer is declared and returns that shape, that element
type and the starting value 1.0.

### DefineDefaultsToFloat32

`define([8], 0.0)` with no element type records `DType::Float32`.

### ADefinedBufferAddsNoLeafUntilItIsRead

After `define()` alone a graph holds no value and `id_in(graph)` is `Graph::K_NO_VALUE`.

### DefineDrawsTheIdentity

Two buffers that were only defined have states other than `K_NO_STATE`, and the two states
differ. A second identical `define()` and a `read()` leave the first buffer's state unchanged.

### ReadDeclaresABufferLeaf

`read()` adds one value to the graph. It has the lifetime `BUFFER`, `requires_grad` false and
the shape [8] and type FLOAT64 that were defined. `id_in(graph)` and
`graph.state_leaf(buffer.state())` both return it.

### ASecondReadInOneGraphReturnsTheFirstLeaf

A second `read()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

### ABufferReadInTwoGraphsHasALeafInEach

The first graph holds a padding input, so the buffer's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, and the buffer's state is
the same after both reads.

### ACopiedBufferHasANewStateID

A copy of a defined and read buffer has the original's shape and starting value and a state
that is neither `K_NO_STATE` nor the original's. `copy.id_in(graph)` is `K_NO_VALUE` until the
copy is read, and its leaf then differs from the original's, which keeps its own.

### AMovedBufferKeepsItsStateID

A buffer move-constructed from a read one has the same state and the same leaf in the graph.
Reading it again returns that leaf and adds no value.

### AMovedFromBufferIsUndefined

After a move construction and after a move assignment the source reports `declared()` false
and the state `K_NO_STATE`. The assigned target is declared and has a state.

### ACopyAssignedBufferHasANewStateID

A defined [2] buffer copy-assigned from a [8] FLOAT64 one takes that shape and type. Its state
differs from the state it had before and from the source's.

### ACopyOfAnUndefinedBufferHasNoIdentity

A copy of a buffer that was never defined is not declared and has the state `K_NO_STATE`.

### DefiningTwiceWithOneDescriptionIsAccepted

A second `define()` with the same shape, value and type returns ok and leaves the description
as it was.

### DefiningWithAnotherShapeFails

`define([4], 0.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as", "[8]" and "[4]". The shape stays [8].

### DefiningWithAnotherElementTypeFails

`define([8], 0.0, FLOAT64)` after a FLOAT32 definition returns `InvalidDescription` with a
message naming both types. The type stays FLOAT32.

### DefiningWithAnotherStartingValueFails

`define([8], 1.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as" and "is now defined as". The starting value stays 0.0.

### ReadingBeforeDefineIsFatal

The case is in the suite `Unit_BufferDeathTest`. `read()` on an undefined buffer stops the
process with a report containing "'running_mean'" and "before define()".

## modules/layers/nn/tests/test_channel_slice.cpp

Suites: `Unit_ChannelSlice`.

Tests `NN::ChannelSlice` in `modules/layers/nn/include/SushiAI/nn/channel_slice.hpp`, which
keeps `count` channels of an image batch from `offset` on.

Two `static_assert` lines at the top of the file hold that `ChannelSlice` is a `Module` and is
not a `ModuleContainer`.

### TheOffsetAndCountFieldsAreSetInOrder

`ChannelSlice{32, 16}` has `offset` 32 and `count` 16.

### OutputShapeKeepsHeightAndWidthAndTakesTheRange

With offset 32 and count 32, `output_shape([5, 3, 64])` is [5, 3, 32].

### OutputShapeAcceptsTheFirstLastAndWholeRanges

Over [4, 2, 4], the ranges {0, 1} and {3, 1} give [4, 2, 1], {1, 3} gives [4, 2, 3] and {0, 4}
gives [4, 2, 4].

### TheResultIsTaggedNHWC

The traced result over a [2, 5, 3, 64] input has rank 4 and the layout `TensorLayout::NHWC`.

### ForwardEmitsOneSliceNodeCarryingTheOffset

`forward` with offset 32 and count 32 adds one node. Its operation is `Ops::K_SLICE_CHANNELS`,
its one input is the value passed in, its attribute 0 is 32, and the result has the shape
[2, 5, 3, 32].

### TheWholeChannelRangeTracesToTheInputShape

`ChannelSlice{0, 64}` over a [2, 5, 3, 64] input traces to [2, 5, 3, 64].

### OutputShapeAgreesWithTheTracedShape

For {32, 32} over 64 channels and for {1, 2} over 6 channels, `output_shape` equals the traced
shape with its batch axis dropped.

### OutputShapeRejectsANegativeOffset

Offset -1 with count 4 throws an error naming "ChannelSlice", "offset=-1, count=4" and
"64 channels".

### OutputShapeRejectsAZeroCount

Offset 0 with count 0 throws an error naming "ChannelSlice", "offset=0, count=0" and
"64 channels".

### OutputShapeRejectsANegativeCount

Offset 8 with count -2 throws an error naming "ChannelSlice", "offset=8, count=-2" and
"64 channels".

### OutputShapeRejectsARangePastTheChannels

Offset 32 with count 40 over 64 channels throws an error naming "offset=32, count=40" and
"64 channels".

### OutputShapeRejectsAnOffsetAtTheChannelCount

Offset 64 with count 1 over 64 channels throws an error naming "offset=64, count=1".

### OutputShapeRejectsExtremeValuesWithoutOverflow

An offset of the largest `int64_t` with count 1, and offset 1 with a count of the largest
`int64_t`, each throw an error naming "ChannelSlice" and "64 channels".

### OutputShapeRejectsARankOnePerSampleShape

`output_shape([16])` throws an error naming "ChannelSlice", "[H, W, C]" and "[16]".

### OutputShapeRejectsARankFourShape

`output_shape([2, 3, 5, 7])` throws `Error`.

### ForwardRejectsARankTwoInput

`forward` over a [2, 12] input throws an error naming "ChannelSlice", "rank-4" and "[2, 12]".

### ForwardRejectsARangePastTheChannels

`forward` with offset 2 and count 2 over a three-channel input throws an error naming
"ChannelSlice", "offset=2, count=2" and "3 channels".

### ForwardRejectsAnNCHWInput

`forward` over a value converted to NCHW throws an error naming "slice_channels" and "NHWC".

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times and
`NN::parameter_count` is 0.

## modules/layers/io/tests/test_checkpoint_format.cpp

Suites: `Unit_CheckpointFormat`, `Unit_CheckpointFormatDeathTest`.

Tests the checkpoint format's round trip and its refusal of each malformed byte pattern.

This file holds the checkpoint format's whole malformed-file surface in the device-free tier.
That is the point of splitting `modules/layers/io/include/SushiAI/io/checkpoint_format.hpp` out of
`modules/layers/io/include/SushiAI/io/checkpoint.hpp`. Every case is a byte pattern, so none of them needs an
Engine, a TensorPool or a device to run.

Each corruption is written as a patch at a known offset into a good file, not as a hand-written
byte array. A case therefore says what is wrong with the file and does not restate the whole
format.

### Field offsets

The `OFFSET_` constants are byte offsets into `sample_stream()`, whose first record is
"0.weight", FLOAT32, [2, 3]. They are stated once so that a case names the field it breaks.

### RoundTripsNamesShapesAndPayloads

The sample stream opens as two records. The first is `0.weight`, `Float32`, [2, 3], 24
payload bytes, and its payload reads back as the six floats written. The second is `0.bias`
of [2] with the two floats written. `read_count` is then 2.

### DistinguishesTwoShapesWithTheSameElementCount

This case covers the defect the format exists to close: an element count alone lets a [2, 3]
weight load into a [3, 2] one. The shape is recorded, so the two records are not
interchangeable.

### RejectsAForeignMagic

With the magic bytes replaced by "JUNK" the read fails with `InvalidFile`, "SACP" and
"not a SushiAI checkpoint".

### RejectsAFutureVersion

With the version patched to 3 the read fails with `InvalidFile`, "version 3" and
"versions 1 to 2".

### RejectsACountOverTheBound

With the record count patched to 4097 the read fails with `InvalidFile`, "4097" and "4096".

### RejectsATruncatedHeader

The first six bytes of the stream fail with `InvalidFile`, "ended while reading" and
"version".

### RejectsATruncatedPayload

The stream without its last four bytes fails with `InvalidFile`, "ended while reading" and
"0.bias".

### RejectsANameLengthOverTheBound

With the first name length patched to 257 the read fails with `InvalidFile`, "257" and "256".

### RejectsARankOverTheBound

With the first rank patched to 7 the read fails with `InvalidFile`, "0.weight", "rank 7" and
"6".

### RejectsANonPositiveDim

With the first extent patched to 0 the read fails with `InvalidFile`, "0.weight" and
"must be positive".

### RejectsAnUnknownDType

With the first element type patched to 99 the read fails with `InvalidFile`, "0.weight", "99"
and "does not know".

### RejectsMoreRecordsThanDeclared

A writer created for 2 records writes two. A third `write` is fatal with "declared 2" and
"'c'".

### RejectsFewerRecordsThanDeclared

A writer created for 2 records writes one. `finish` is fatal with "declared 2" and
"1 were written".

### RejectsATensorCountOverTheBoundAtConstruction

`CheckpointWriter::create` for 4097 records fails with `InvalidDescription`, "4097" and
"4096".

### ReadingPastTheDeclaredCountIsAnError

After both records of the sample stream are read, a third `next` is fatal with "declared 2".

### WritesTheVersionTwoHeaderByteByByte

The sample stream opens with the sixteen bytes 'S', 'A', 'C', 'P', then 2, 2 and 0 as
little-endian 32-bit values. Its size is those sixteen bytes plus 60 bytes for the weight
record and 34 for the bias record.

### AVersionOneFileReadsAsModelStateAlone

A version 1 stream opens with `version` 1, two records and a run-state count of 0, and both
payloads read back as written.

### RejectsAVersionBelowTheFirst

With the version patched to 0 the read fails with `InvalidFile`, "version 0" and
"versions 1 to 2".

### RoundTripsARunStateSection

In the run-state stream, byte 50 opens the section with the owner length 3 and "sgd", and the
run-state count field holds 2. The reader reports version 2, one model record and two
run-state records. After the model record, `begin_run_state` returns "sgd". The next records
are `step_count`, `Int64` of [1] holding seven, and `0.bias.momentum` holding the two floats
written. `read_count` is 1 and `run_state_read_count` is 2.

### AReaderOfModelStateLeavesTheRunStateUnread

Reading the model records of the run-state stream alone succeeds.

### RejectsARunStateCountOverTheBound

With the run-state count patched to 16385 the read fails with `InvalidFile`, "16385",
"16384" and "run-state". `CheckpointWriter::create` with 16385 run-state records fails with
`InvalidDescription` and the same three texts.

### RejectsAMalformedRunStateOwner

The read fails with `InvalidFile` for an owner length of 0 ("names no owner"), an owner
length of 257 ("run-state section", "257", "256") and a stream cut inside the owner
("ended while reading", "run-state owner"). `begin_run_state("")` on a writer fails with
`InvalidDescription`, "owner" and "0 bytes".

### RejectsATruncatedRunStatePayload

The run-state stream without its last three bytes fails with `InvalidFile`,
"ended while reading" and "0.bias.momentum".

### TheRunStateOpensAfterTheModelRecords

On a writer that declared one model record and wrote none, `begin_run_state` is fatal with
"declared 1 records" and "0 were written". On a reader that has read none it is fatal with
"declared 1 records" and "0 were read".

### ARunStateThatWasNotDeclaredCannotOpen

On a writer created without run-state records, `begin_run_state` is fatal with
"declared no run-state record". On a reader of a version 1 stream, after both records are
read, it is fatal with "declares no run-state record".

### TheRunStateHoldsExactlyTheRecordsDeclared

On a writer that declared one run-state record, `finish` right after `begin_run_state` is
fatal with "declared 1 run-state records" and "0 were written". After one record is written,
a second `write` is fatal with "declared 1 run-state records" and "'b'".

## modules/layers/nn/tests/test_concat.cpp

Suites: `Unit_Concat`.

Tests the Concat node's shape check, its port and the node it traces.

This file needs no device. `NN::Concat` in `modules/layers/nn/include/SushiAI/nn/concat.hpp` is an `NN::Node`
that joins several feature maps along the channel axis. `Concat::create` takes one per-sample
shape per input and returns a `Result`, and `forward` returns a `Status`.

### create_over

`create_over` calls `Concat::create` over a vector of shapes viewed as an `NN::ShapeSpan`.

### span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

### CreateSumsTheChannelsOfItsInputs

Over [8, 6, 3], [8, 6, 5] and [8, 6, 1] the node has one port named "output" of shape
[8, 6, 9].

### CreateAcceptsOneInput

Over the one shape [8, 6, 3] the port's shape is [8, 6, 3].

### CreateRejectsUnequalHeight

[8, 6, 3] with [4, 6, 3] fails with `InvalidDescription` and a message containing "concat
input 1" and "H and W".

### CreateRejectsUnequalWidth

A third input of [8, 12, 3] fails with a message containing "concat input 2" and "H and W".

### CreateRejectsAnInputThatIsNotAFeatureMap

[8, 6, 3] with [144] fails with a message containing "concat input 1" and "rank-3".

### CreateRejectsAFirstInputThatIsNotAFeatureMap

[144] with [8, 6, 3] fails with a message containing "concat input 0" and "rank-3".

### CreateRejectsNoInput

No shape at all fails with a message containing "at least one input".

### ForwardEmitsOneConcatChannelsReadingEveryInputInOrder

`forward` over the values b and a adds one `Ops::K_CONCAT_CHANNELS` node whose inputs are b then
a. It appends one id to `outputs`, the node's one output.

### ForwardResultIsThePortShapeWithABatchAxis

Over [2, 8, 6, 3] and [2, 8, 6, 5] the traced value has the shape [2, 8, 6, 8], and the port
reports [8, 6, 8].

### ForwardRejectsAnInputCountItWasNotBuiltOver

A node built over two shapes and given one value fails with a message containing "input count
is 2" and "given is 1". `outputs` stays empty and the graph gains no node.

### TheInputCountIsTheNumberOfShapesItWasBuiltOver

`input_count()` is 1 for a node built over one shape and 3 for a node built over three.

### ForwardReturnsTheFailureOfValuesThatDoNotJoin

A node built over two [8, 6, 3] shapes is given a [2, 8, 6, 3] value and a [2, 4, 6, 3] value.
`forward` returns an `InvalidDescription` failure, appends nothing to `outputs` and adds no
node.

### TheFactoryBuildsANodeFromARowWithNoKey

`NN::make_concat_node` over [8, 6, 3] and [8, 6, 5] with the attributes `{}` returns a
`Concat` with an input count of 2 and one port of shape [8, 6, 8].

### TheFactoryRefusesAKeyAndShapesThatDoNotJoin

The attributes `{"axis": 3}` fail with a message containing the context "layer 2 'concat'",
"'axis'" and "no key". The shapes [8, 6, 3] and [4, 6, 5] with no attribute fail with the
context and "H and W".

### TheStateWalkVisitsNothing

After a trace, `walk_state("merge", visitor)` calls neither the parameter callback nor the
buffer callback.

## modules/layers/nn/tests/test_config.cpp

Suites: `Unit_Config`.

Tests how the config layer parses its JSON documents and which ones it refuses.

This file needs no device. The config layer is host-lane by construction: it turns text into a
typed tree and never touches an Engine.

Most of what the cases assert is a diagnostic, not a value. That is deliberate. A config
layer's job is to refuse a wrong file loudly, and a test that only checked "it threw" would keep
passing after a refactor collapsed five specific messages into one generic complaint.
`tests/common/test_throws.hpp` gives the argument in full.

The cases fall into two groups, in file order:

- the model document in both of its versions, and the layer registry it selects through
- the train and eval documents

### ParsesEveryScalarKind

An object with a whole number, a real, a string, a boolean, a null and an array of two parses
to `Kind::Object`, and each member reads back through `as_int`, `as_real`, `as_string`,
`as_bool`, `is_null` and `as_array`.

### KeepsTheKeyOrderTheFileSpelled

Key order is what lets a "this object carries ..." diagnostic read the way the file does, so
the test treats it as a property and not as an accident.

### AnIntegerReadsAsARealButARealIsNotAWholeNumber

JSON's grammar distinguishes 3 from 3.0 and the config layer keeps that, so `"units": 2.5` is
refused, not truncated. The widening goes one way only.

### AWrongKindNamesBothTheContextAndTheKindFound

`as_int("layer 0 'units'")` on a string raises an error holding "layer 0 'units'" and
"a string".

### AMissingKeyNamesTheKeyAndListsWhatIsPresent

`at("gamma", "model.json")` on an object of `alpha` and `beta` raises an error holding
"model.json", "'gamma'", "'alpha'" and "'beta'".

### ReadingAKeyOffANonObjectSaysWhatItActuallyIs

`at("units", "the model")` on an array raises an error holding "the model" and "an array".

### AnUnknownKeyIsAnErrorThatListsWhatIsAllowed

`reject_unknown_keys` with the allowed keys `type` and `units` over an object holding `unitss`
raises an error holding "layer 0 'linear'", "'unitss'", "'units'" and "'type'".

### AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything

A layer that takes no attributes, such as gelu, passes an empty list. Any key at all is then an
error and is not quietly dropped.

### WithoutRemovesTheDiscriminatorAndLeavesTheRest

Removing a key that is absent is not an error, because a layer spelled with no attributes still
goes through the same call.

### AtOrFallsBackWithoutThrowing

A non-object falls back and does not throw. A layer spelled `{"type": "gelu"}`, whose attributes
are an empty object, therefore reads its optional keys through the same call as any other layer.

### MalformedJSONNamesTheOriginAndTheComplaint

`parse_json("{ not json", "train.json")` raises an error holding "train.json" and
"not valid JSON".

### AnObjectBuiltFromMismatchedVectorsIsRefused

`Value::object` of two keys and one value raises an error holding "2 keys" and "1 value".

### ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes

"type" is gone from the attributes, so no factory has to allow the key that selected it.

### ChainsLayerWidthsFromTheInputFeatures

The activation preserves the width it was handed, which is why the file never states a width
for it.

### AConfiguredStackTracesToTheDeclaredOutputWidth

The stack `build_layer_stack` makes from `K_GOOD_MODEL` traces a [4, 6] input to a [4, 4]
value.

### AnUnknownLayerTypeListsEveryTypeThisBuildHas

The model names a layer type "hologram", which no row of `NN::K_LAYER_REGISTRY` carries. The
message contains "hologram", "layer type", "'linear'" and "'gelu'". The unknown type has to
be a name the registry does not hold, and "conv2d" is a row of the table.

### AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory

The `stray` document writes a width onto an activation. An activation takes no attributes, so
that width is an error and not a line that does nothing.

### The version 2 model document

The cases from AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite to
AMalformedOutputIsAnError cover the two versions of the model document that
`Config::parse_model` reads. `Config::K_MODEL_VERSION` is 2. A version 1 document carries the key
`sushiai_model` and an `input` of `features`. A version 2 document carries `format_version`,
an `input` of `shape`, an optional `from` on each row, and the optional top-level keys
`defaults` and `outputs`. Both parse to one `Config::ModelSpec`, in which every row holds its
sources as absolute row indices and the model input is `Config::K_MODEL_INPUT`, which is -1.

### K_FIRST_VERSION_MODEL

`K_FIRST_VERSION_MODEL` is a version 1 document of five rows over 784 features: `unflatten` to
28 by 28 by 1, `conv2d` of 8 filters and kernel 3 with a bias, `relu`, `flatten` and `linear`
of 10 units.

### K_REWRITTEN_MODEL

`K_REWRITTEN_MODEL` is the same model written as version 2, with the input shape [784] and
`"from": -1` on every row.

### scalar_text

`scalar_text` renders one scalar attribute as "int n", "bool true", "bool false" or "string
s", and any other kind as its kind name.

### spec_text

`spec_text` renders a `ModelSpec` as text: one line for the input shape and feature count, one
line per row with its type, its sources as `node:'port'` and its attributes, and one line per
output. Two specs compare equal as text when every one of those fields agrees.

### current_model_from

`current_model_from` wraps a `layers` array, and optionally more top-level keys, in a version
2 document whose input shape is [6].

### ALayerFactoryRefusesANonPositiveWidth

A `linear` row with `units` 0 makes `build_layer_stack` raise an error holding
"layer 0 'linear'" and "'units'".

### UseBiasDefaultsToTrueAndCanBeTurnedOff

After a trace, a `linear` row has two parameters, and one with `use_bias` false has one.

### AMissingModelVersionIsAnErrorThatSaysWhy

A model document without `sushiai_model` raises an error holding "sushiai_model" and
"misread".

### AFutureModelVersionIsAnError

`sushiai_model` 2 raises an error holding "version 2" and "version 1".

### AnUnknownTopLevelModelKeyIsAnError

A version 1 document with the key `optimizer` raises an error holding "'optimizer'" and
"'layers'".

### AModelWithNoLayersIsAnError

An empty `layers` array raises an error holding "no layers".

### ANonPositiveInputWidthIsAnError

`features` 0 raises an error holding "'features'" and "positive".

### ALayerThatIsNotAnObjectIsAnError

A layer given as the string "linear" raises an error holding "layer 0" and "a string".

### AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite

`spec_text` of the two documents is equal. The expected text is asserted as a literal too: the
input is [784] with 784 features, row 0 reads -1, each later row reads the row above it, and
the one output is row 4 with no port and no name.

### AVersionOneDocumentFillsTheInputShapeAndTheFeatureCount

`K_GOOD_MODEL`, a version 1 document of 6 features, parses to `input_shape` [6],
whose `num_elements()` is 6. The case also asserts that `Config::K_MODEL_VERSION` is 2.

### AVersionTwoInputShapeKeepsItsRankAndCountsItsElements

The input shape [28, 28, 1] is kept at rank 3 and its `num_elements()` is 784.

### AnEmptyOrNonPositiveInputShapeIsAnError

An empty `shape` throws with a message containing "'input' 'shape'" and "empty". The shape
[28, 0, 1] throws with "'shape' entry 1" and "positive". A version 2 input that states
`features` throws with "'features'" and "'shape'".

### TheInputElementTypeIsFloat32UnlessTheFileStatesInt32

A version 2 document without `dtype`, and one with `dtype` "float32", both parse to an input
of `Float32`. One with "int32" parses to `Int32`.

### AnInputElementTypeOutsideTheListIsAnError

The `dtype` "int64" raises an error holding "'input' 'dtype'", "'int64'" and
"'float32', 'int32'". The `dtype` 32 raises one holding "'input' 'dtype'".

### AnInputShapeAboveTheRankOrTheElementLimitIsAnError

A shape of `K_MAX_RANK` + 1 extents throws with "'shape'" and "at most". The shape
[4294967296, 4294967296], whose product is 2^64, throws with "'shape'", "entry 1" and "more
elements".

### AModelRootThatIsNotAnObjectIsAnError

The document `[1]` throws with a message containing "model.json", "'format_version'" and "an
array".

### AVersionAboveTheNewestIsRefusedWithBothNumbers

`"format_version": 3` throws with a message containing "model.json", "version 3" and "version
2".

### FormatVersionOneIsRefusedAndNamesTheVersionOneKey

`"format_version": 1` throws with a message containing "'format_version'" and
"'sushiai_model'". Version 1 files carry the second key and never the first.

### AVersionOneDocumentRefusesTheVersionTwoKeys

A version 1 document with a `from` on row 1 throws with "layer 1", "'from'" and "version 1".
One with a top-level `defaults` throws with "'defaults'" and "'layers'".

### EachFromFormResolvesToAnAbsoluteRow

Six rows use every form of `from`. An absent `from` on row 0 and the string "input" resolve to
-1. The number -1 resolves to the row above. An array `[-1, 0]` on row 3 resolves to rows 2
and 0. An object `{"node": 3, "port": "boxes"}` resolves to row 3 with that port. An array may
mix objects, "input" and a relative object. The whole spec is compared as text. Row 0's source
is `K_MODEL_INPUT`, and "from" is not among row 3's attributes.

### ANegativeFromOtherThanMinusOneNamesTheRow

`"from": -2` on row 2 throws with a message containing "layer 2 of 'model.json'", "'from'" and
"row -2". `{"node": -3}` inside an array throws with the layer and "row -3".

### AForwardReferenceNamesTheRow

`"from": 2` on row 1 throws with "layer 1 of 'model.json'", "row 2" and "above row 1".
`"from": [0, 1]` on row 1, which names the row itself, throws with the layer and "row 1".

### AnUnknownKeyOrFormInAFromEntryNamesTheRow

Each of five `from` entries on row 1 throws with "layer 1 of 'model.json'" and one more
needle: the key `prot` with "'prot'" and "'port'"; the string "previous" with "'previous'" and
"'input'"; an empty array with "empty"; the number 0.5 with "it is a number"; and an empty
`port` string with "'port'" and "empty string".

### AnUnknownTopLevelKeyOfAVersionTwoDocumentIsAnError

A top-level key `output` throws with a message containing "'output'", "'outputs'" and
"'defaults'".

### DefaultsFillAnOmittedKeyAndNeverOverrideAStatedOne

The `defaults` block gives `conv` rows `bn_momentum` 3 and `kernel` 5. Row 0 states only
`filters` and ends with the keys `filters`, `bn_momentum`, `kernel` and the values 3 and 5.
Row 1 states `bn_momentum` 7 and keeps it. Row 2 is a `relu` and receives no key.

### AMalformedDefaultsBlockIsAnError

A `defaults` that is an array throws with "'defaults'" and "an array". A type whose default is
the number 3 throws with "'defaults' for 'conv'" and "object". A default that sets `from`
throws with that context and "'from'", and one that sets `type` with "'type'".

### ADefaultsEntryForATypeNoLayerStatesIsAnError

A default for "covn" in a model of `conv` and `relu` rows throws with a message containing
"'defaults'", "'covn'" and "'conv', 'relu'".

### OutputsDefaultToTheLastRowAndKeepWhatTheFileStates

A document with no `outputs` gets one output: the last row, with an empty port and an empty
name. A document that states three outputs keeps each as written: row 0 alone, row 1 with the
port "boxes", and row 1 with the port "boxes" and the name "predictions".

### AMalformedOutputIsAnError

Seven `outputs` values each throw. Row 1 in a one-row model: "'outputs' entry 0", "row 1" and
"0 to 0". The key `nme`: "'nme'" and "'name'". A `name` without a `port`: "'name'" and
"'port'". An empty array: "'outputs'" and "empty". The number 3 as an entry: "a whole number".
An empty `port` string: "'outputs' entry 0 'port'" and "empty string". An empty `name` string:
"'outputs' entry 0 'name'" and "empty string".

### Path resolution in the train and eval documents

Path resolution is the one thing in this file that needs real files on disk. The property is
that a relative path resolves against the config file's own directory and not against the
process working directory, and that cannot be asserted without a config file that exists
somewhere.

### AModelWithoutATaskHoldsANullTask

A version 2 document without `task`, and `K_FIRST_VERSION_MODEL`, both parse to a null task.

### ATaskIsKeptWholeWithItsKeysInFileOrder

A task stating `classes`, `type` and `heads` in that order parses to an object with those
keys in that order, the type "anchor_detection", 80 classes and one head.

### AMalformedTaskIsAnError

The parse raises an error for a task given as an array ("'model.json's 'task'", "an array"),
as null ("'model.json's 'task'", "null"), without a type ("'model.json's 'task'",
"no 'type'", "'classes'") and with the type 3 ("'task' 'type'", "a whole number").

### AVersionOneDocumentRefusesATask

A version 1 document with a `task` raises an error holding "'task'" and "'layers'".

### ParsesATrainDocumentIntoItsFourTypedBlocks

`K_GOOD_TRAIN` parses to the objective type "cross_entropy", the data type "rings", the
optimizer type "adamw" and 3 epochs. The optimizer attributes hold no `type` and a
`learning_rate` of 0.003, and the data attributes hold a `batch_size` of 8.

### ResolvesRelativePathsAgainstTheConfigsOwnDirectory

`load_train` of a `train.json` written under the system's temporary directory gives a model
path of `model.json` and a checkpoint of `out/model.sushi`, both under that directory.

### LeavesAnAbsolutePathAlone

`generic_string()` spells separators as '/', which JSON does not need to escape. The test
therefore builds the document by concatenation and needs no second, test-only escaper that
nobody would maintain.

### EpochsDefaultsToOneAndMustBePositive

A train document without `epochs` parses to 1 epoch and an empty checkpoint path. `epochs` 0
raises an error holding "'epochs'" and "positive".

### AnUnknownTopLevelTrainKeyIsAnError

The key `epochz` raises an error holding "'epochz'" and "'epochs'".

### AMissingTrainBlockNamesIt

A train document without `data` raises an error holding "'data'".

### ATypedBlockThatIsNotAnObjectIsAnError

An `objective` given as a string raises an error holding "'objective'" and "a string".

### AFutureTrainVersionIsAnError

`sushiai_train` 2 raises an error holding "version 2" and "version 1".

### ParsesAnEvalDocument

`parse_data` of a `sushiai_data` 1 document gives the data type "rings", a `batch_size` of
512 and a `seed` of 3735928575.

### AnEvalDocumentCarriesNoModelBlock

The architecture is a separate file that inference is handed separately. Naming one in an eval
document is an error; it is not silently preferred over the one on the command line.

### LoadingAnAbsentFileNamesThePath

`load_train("no/such/train.json")` raises an error holding that path.

## modules/vision/datasets/tests/test_datasets_collators.cpp

Suites: `Unit_DatasetsCollators`.

Tests `write_batch_inputs`, `ClassificationCollator` and `DetectionCollator` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. The fixture is an `EngineTest`. It
holds two gray samples of one row and two columns, the first of level 51 from `first.png` and
the second of level 255 from `crowded.png`, and allocates batches of two such images.

### InputsAreBytesDividedBy255

`write_batch_inputs` writes 0.2, 0.2, 1.0, 1.0, each within 1e-6.

### AnImageOfAnotherSizeIsRefused

With the second image 2 by 2, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs", "image 1" and "2 by 2".

### ATensorOfAnotherImageCountIsRefused

With one sample and a tensor for two, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs" and "holds 2 images".

### ClassificationPublishesLabels

`ClassificationCollator::make(0)` fails with `InvalidDescription`, "ClassificationCollator" and
"0 classes". A collator of 7 classes reports the task `K_TASK_CLASSIFICATION` and one target:
the role `K_ROLE_LABELS`, an empty sample shape, `Int32` and 7 classes.

### ClassificationWritesTheClassOfEachSample

With the class indices 6 and 0, `collate` writes the labels 6 and 0, and the third input value
is 1.0.

### ClassificationNamesTheFileOfASampleWithoutAClass

With no class index on the second sample, `collate` fails with `InvalidDescription`,
"crowded.png", "ClassificationCollator" and "no class index". With the class index 7 over 7
classes it fails with "crowded.png", "class index 7" and "[0, 7)".

### ABatchWithoutTheRoleIsRefused

Over a batch allocated without targets, `collate` fails with `Incompatible`,
"ClassificationCollator" and "no role 'labels'".

### DetectionPublishesThreeRoles

`DetectionCollator::make` fails with `InvalidDescription` for 0 classes ("0 classes") and for
0 objects ("0 objects"). A collator of 3 classes and 4 objects reports the task
`K_TASK_DETECTION` and three targets: `K_ROLE_BOXES` of shape [4, 4] and `Float32`,
`K_ROLE_CLASSES` of shape [4], `Int32` and 3 classes, and `K_ROLE_VALID` of shape [4] and
`Float32`.

### DetectionWritesCornersAndPadsWithZero

Every target slot is first filled with 9. With one object of class 2 and the box (0.25, 0.5,
1.5, 1.0) on the second image and two slots per image, `collate` writes that box, the class 2
and the valid flag 1 into the third slot and zero into every other slot.

### DetectionNamesTheFileOfAnImageWithTooManyObjects

Three objects on the second image with two slots fail with `InvalidDescription`,
"crowded.png", "DetectionCollator", "3 objects" and "at most 2".

### DetectionRefusesAClassOutsideTheSet

An object of class 3 on the first image, over 3 classes, fails with `InvalidDescription`,
"first.png", "object 0", "class 3" and "[0, 3)".

## modules/vision/registration/tests/test_datasets_mnist.cpp

Suites: `Integration_VisionMnist`.

Tests the "mnist" dataset row the vision module registers, through `Data::make_dataset`, and
`Vision::Datasets::try_load_mnist`. The suite's fixture is an `EngineTest`, so every case
needs an engine. It keeps the prefix `Integration_` although the file is in a unit test
folder. The refusal cases point at a directory that does not exist. The one case that reads
images reports itself skipped when the MNIST IDX files are absent.

### TheDatasetRegistryListsMNIST

`find("mnist")` returns a row. `make_dataset` of the type "coco" fails with
`InvalidDescription` and a message that holds "'mnist'".

### MNISTRefusesADirectoryWithoutTheFiles

"mnist" over an absent directory fails with `FileNotFound`. The message holds "data", the
directory, "MNIST train split" and the names of the four IDX files.

### MNISTTestSplitLooksForTheT10kFiles

With "split" set to "test" the failure holds "MNIST test split" and
"t10k-images-idx3-ubyte, labels".

### MNISTRequiresADirectory

A section without "directory" throws an error naming "data", "has no" and "'directory'".

### AnUnknownMNISTKeyIsRejectedByItsOwnFactory

The key "batch_sizes" throws an error naming "data", "'batch_sizes'" and "'batch_size'".

### MNISTRefusesASplitItDoesNotKnow

The split "validation" fails with `InvalidDescription`, and the message holds "'split'",
"'validation'", "'train'" and "'test'".

### AnMNISTLayoutOtherThanFlatOrImageIsRefused

The layout "tiled" fails with `InvalidDescription`, and the message holds "'layout'", "'tiled'",
"'flat'" and "'image'".

### AnImageLayoutYieldsRankFourBatches

With `MNISTLayout::Image`, a batch size of 16 and 32 samples, the dataset's sample shape is
[28, 28, 1]. It has one target, of the role `Data::K_ROLE_LABELS`, a scalar sample shape, the
element type `Int32` and 10 classes. Batch 0 holds a rank-1 `Int32` label tensor of 16 entries
and a [16, 28, 28, 1] input tensor.

### MNISTFallsBackToSyntheticWhenTheFilesAreAbsent

For a directory that does not exist, `mnist_available` is false, `try_load_mnist` fails with
`FileNotFound` and names the directory, and `mnist_expected_paths` is not empty.

## modules/vision/datasets/tests/test_datasets_pipelines.cpp

Suites: `Unit_DatasetsPipelineTable`, `Unit_DatasetsFitPipeline`,
`Unit_DatasetsAugmentPipeline`.

Tests `pipeline_table` and `find_pipeline`, `FitPipeline` and `AugmentPipeline` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. The pipeline cases read four raw
RGB images of the sizes 20 by 16, 16 by 20, 16 by 16 and 10 by 20, written under the system's
temporary directory; sample i holds one object of class i. The training pipelines have a size
of 16 and the seed 20261011.

### ItListsNoneThenDefault

`pipeline_table` has two rows named "none" and "default". `find_pipeline` returns each row for
its name and null for "mixup".

### ARowBuildsThePipelineOfItsName

For a request of 16 by 16 and three channels, each row's `make` gives a pipeline whose `name`
is the row's name.

### DefaultRefusesAnInputItDoesNotMake

The "default" row fails with `InvalidDescription` and "'default'" for a size of 4 by 6
("square", "4 by 6") and for one channel ("3 channels", "has 1").

### FitsEachSampleTheSameInEveryEpoch

A `FitPipeline` that letterboxes to 8 by 8 is named "none". For the samples 3 and 0 it gives
images of 8 by 8 and three channels with one object each, of the classes 3 and 0. Epoch 7
gives the same image, mask, file, classes and boxes as epoch 0.

### RefusesWhatItCannotFit

`FitPipeline::make` with a size of 0 by 8 fails with `InvalidDescription`, "FitPipeline" and
"0 by 8". `produce` with two sample indices and one fitted slot fails with
`InvalidDescription`, "2 samples" and "1 fitted".

### MakesSquareSamplesOfThreeChannels

An `AugmentPipeline` with 1000 mosaic epochs is named "default". In epoch 0 each of the four
samples is 16 by 16 with three channels and an empty mask, and every box lies within
[0, 16] on both axes.

### ASampleDependsOnItsIndexAndItsEpochAlone

In epoch 2, each sample comes out the same when the four are produced twice, when they are
produced in reverse order and when sample 2 is produced alone. In epoch 3 the image of at
least one sample differs from its image in epoch 2.

### PastTheMosaicEpochsASampleHoldsItsOwnObjects

With 0 mosaic epochs, in the epochs 0, 1 and 2 every object of sample i has the class i.

### AMaskTakesTheSizeOfItsSample

Each sample is given a mask of its image's size that holds its index. In epoch 1 every
produced mask is 16 by 16 with one channel, and each byte is below 4 or is `K_MASK_VOID`.

### AnImageThatCannotBeReadIsTheFailure

With the file of sample 1 replaced by a path that does not exist, `produce` fails with
`FileNotFound` and "gone.saim".

### RefusesWhatItCannotMake

`AugmentPipeline::make` fails with `InvalidDescription` for a size of 0 ("AugmentPipeline",
"a training size of 0") and for -1 mosaic epochs ("-1 epochs of mosaic"). `produce` fails
with `InvalidDescription` for the epoch -1 ("epoch -1") and for an empty list
("a list of 0 samples").

## modules/vision/datasets/tests/test_datasets_task_collators.cpp

Suites: `Unit_DatasetsTaskCollators`.

Tests `BinaryCollator`, `MultiLabelCollator`, `RegressionCollator` and `SegmentationCollator`
in `modules/vision/datasets/include/SushiAI/vision/datasets/`. The fixture is an `EngineTest`.
It holds two gray samples of one row and two columns, from `first.png` and `second.png`, and
allocates batches of two samples.

### BinaryPublishesLabels

`BinaryCollator::make` fails with `InvalidDescription` for the positive class 2
("BinaryCollator", "the positive class is 2") and for -1 ("the positive class is -1"). A
collator of the positive class 1 reports the task `K_TASK_BINARY` and one target: the role
`K_ROLE_LABELS`, an empty sample shape, `Float32` and 0 classes.

### BinaryWritesOneForThePositiveClass

With the class indices 1 and 0, a collator of the positive class 1 writes the labels 1, 0 and
a collator of the positive class 0 writes 0, 1.

### BinaryNamesTheFileOfASampleItCannotLabel

With no class index on the second sample, `collate` fails with `InvalidDescription`,
"second.png", "BinaryCollator" and "no class index". With the class index 2 it fails with
"second.png", "class index 2" and "[0, 2)".

### MultiLabelWritesTheSetOfClassesPresent

`MultiLabelCollator::make(0)` fails with `InvalidDescription`, "MultiLabelCollator" and
"0 classes". A collator of 4 classes reports the task `K_TASK_MULTI_LABEL` and one target of
the role `K_ROLE_LABELS`, the sample shape [4] and `Float32`. The first sample has objects of
the classes 2, 0 and 2 and the class index 3; the second has neither. Over a target first
filled with 9, `collate` writes 1, 0, 1, 1 for the first sample and four zeros for the second.

### MultiLabelRefusesAClassOutsideTheSet

An object of class 4 on the second sample, over 4 classes, fails with `InvalidDescription`,
"second.png", "MultiLabelCollator", "the class 4" and "[0, 4)".

### RegressionCopiesFeaturesAndTargets

`RegressionCollator::make` fails with `InvalidDescription` for 0 features ("0 features") and
for 0 targets ("0 targets"), each with "RegressionCollator". A collator of 3 features and 2
targets reports the task `K_TASK_REGRESSION` and one target of the role `K_ROLE_VALUES`, the
sample shape [2] and `Float32`. For samples with the features 1, 2, 3 and 4, 5, 6 and the
targets 10, 20 and 40, 50, `collate` writes the inputs 1 to 6 and the values 10, 20, 40, 50.

### RegressionRefusesWhatHoldsNoRow

Over samples that hold images and no row, `collate` fails with `InvalidDescription`,
"first.png", "RegressionCollator" and "0 features and 0 targets". Over inputs of the sample
shape [2] it fails with `Incompatible`, "RegressionCollator" and "float32 of [2, 3]".

### SegmentationPublishesTheMask

`SegmentationCollator::make` fails with `InvalidDescription` for 0 classes
("SegmentationCollator", "0 classes"), for 256 classes ("256 classes", "1 to 255"), for the
ignore index 2 over 3 classes ("the ignore index 2 is one of the 3 classes") and for a size
of 0 by 2 ("0 by 2 pixels"). A collator of 3 classes reports the task `K_TASK_SEGMENTATION`
and one target of the role `K_ROLE_MASK`, the sample shape [1, 2], `Int32` and 3 classes.

### SegmentationWritesClassesAndTheIgnoreIndex

For the masks 2, `K_MASK_VOID` and 0, 1, `collate` writes 2, the ignore index, 0, 1, with the
ignore index `K_DEFAULT_IGNORE_INDEX` and with -100.

### SegmentationRefusesAByteThatIsNoClass

A mask byte of 3 over 3 classes fails with `InvalidDescription`, "second.png",
"SegmentationCollator", "holds 3 at row 0, column 1" and "below 3 and 255 names none".

### SegmentationRefusesASampleWithoutAMask

With no mask on the second sample, `collate` fails with `InvalidDescription`, "second.png",
"SegmentationCollator" and "the sample's mask is 0 by 0".

### SegmentationPaintsThePolygonsOfASample

With `MaskOrigin::Polygons` and the ignore index -1, the first sample has a polygon of class 2
over its second pixel and the second sample one of class 0 over its first pixel. `collate`
writes -1, 2, 0, -1. With `background` 1 it writes 1, 2, 0, 1.

### SegmentationRefusesAnObjectWithoutAPolygon

With `MaskOrigin::Polygons`, an object with a box and no polygon on the second sample fails
with `InvalidDescription`, "second.png", "paint_polygon_mask" and "no polygon".

### SegmentationRefusesABackgroundItCannotWrite

`SegmentationCollator::make` fails with `InvalidDescription` for `background` 1 without
`MaskOrigin::Polygons` ("a background of 1", "masks come from files") and for `background` 3
over 3 classes with it ("the background 3 is none of the 3 classes").

## modules/layers/train/tests/test_dense_objectives.cpp

Suites: `Unit_DenseObjectiveGraphs`, `Unit_DenseObjectives`.

Tests `Train::BinaryCrossEntropyObjective`, `Train::MultiLabelObjective` and
`Train::MeanSquaredErrorObjective`: the leaf and the role each declares, what `make` and the
factories refuse, what `feed` writes and what `measure` reports. The graph cases trace a
Linear, ReLU, Linear model of 6 features over a batch of 4 and need no device. The
`Unit_DenseObjectives` fixture needs an engine, and its `bind_targets` helper binds a tensor of
7s to the `targets` leaf before a feed.

### BinaryDeclaresAColumnLeafAndRealLabels

The objective is named "binary_cross_entropy" and its threshold is 0.5. In the built training
graph the `targets` leaf has the shape [4, 1] and is `spec.targets`, and the loss has one
element. The one required target has the role `K_ROLE_LABELS`, an empty sample shape, the type
float32 and 0 classes.

### MultiLabelDeclaresOneLabelPerClass

`MultiLabelObjective::make(4, 3)` is named "multi_label" and reports 3 classes. The `targets`
leaf has the shape [4, 3] and the loss has one element. The one required target has the role
`K_ROLE_LABELS`, the sample shape [3] and the type float32.

### MeanSquaredErrorDeclaresTheValuesRole

`MeanSquaredErrorObjective::make(4, 2)` is named "mean_squared_error" and reports the width 2.
The `targets` leaf has the shape [4, 2] and the loss has one element. The one required target
has the role `K_ROLE_VALUES`, the sample shape [2] and the type float32.

### MakeRefusesWhatNoLeafCanHold

Six calls fail with `InvalidDescription`. `BinaryCrossEntropyObjective::make` refuses a batch of
0 ("at least 1"), a threshold of 1 ("above 0 and below 1") and int32 ("float32", "int32").
`MultiLabelObjective::make` refuses 0 classes and a threshold of 0, and
`MeanSquaredErrorObjective::make` refuses a width of 0.

### TheFactoriesReadTheOneOutput

`make_binary_cross_entropy` over a [4, 1] output and `{"threshold": 0.9}` gives the threshold
0.9 and the batch 4. `make_multi_label` over a [4, 5] output and `{}` gives 5 classes and the
threshold 0.5. `make_mean_squared_error` over a [4, 2] output gives the width 2.

### TheFactoriesRefuseAnotherOutputAndAStrayKey

Each failure is `InvalidDescription` and names "objective". The binary factory refuses a [4, 2]
output ("one logit per sample") and a model that publishes two outputs ("publishes 2"). The
multi-label factory refuses a [4, 3, 5] output and the regression factory an [8, 2] output,
both with "[4, columns]". The key `huber` is refused by the regression factory and the key
`weights` by the multi-label factory, which also names `'threshold'`.

### BinaryFeedCopiesLabelsOfZeroAndOne

A batch whose labels are 1, 0.25, 0, 1 fails with `Incompatible`, "element 1" and "0 or 1",
and the leaf still holds four 7s. A batch of 1, 0, 0, 1 feeds ok and the leaf holds those
values.

### MeanSquaredErrorFeedReadsTheValuesRole

A batch that carries only the labels role fails with `InvalidDescription` and a message naming
`'values'`. A batch with the values 1.5, -2, 0.25, 8 under `K_ROLE_VALUES` feeds ok and the
leaf holds them.

### BinaryMeasuresAccuracyPrecisionAndRecall

Over five rows with the logits 2, 1, -1, -3, 4 and the labels 1, 0, 1, 0, 1, `measure` adds
three metrics in the order accuracy, precision, recall, with the values 3/5, 2/3 and 2/3. An
objective made with the threshold 0.95 reports the precision 1 and the recall 1/3.

### MultiLabelMeasuresTheMeanF1OfTheClasses

Over two rows of three classes `measure` adds one metric named "f1" that is a fraction and
equals 1/3.

### MeanSquaredErrorMeasuresTheMeanAbsoluteError

Over the predictions 1, 2, 3, 4 and the values 1.5, 2, 2, 6, `measure` adds one metric named
"mean_absolute_error" that is not a fraction and equals 3.5/4.

### AMetricRefusesAnOutputOfAnotherCount

A multi-label objective of 2 rows and 4 classes given a [2, 3] output fails with `Incompatible`
and a message containing "the logits", "8 values" and "holds 6". No metric is added.

## modules/layers/nn/tests/test_depthwise_conv2d.cpp

Suites: `Unit_DepthwiseConv2d`.

Tests `NN::DepthwiseConv2d`. Most cases use four channels, a 3x3 window and one pixel of
padding.

Two `static_assert` lines at the top of the file hold that `DepthwiseConv2d` is a `Module` and
is not a `ModuleContainer`.

### ForwardEmitsOneDepthwiseNodeAndNoBiasAdd

After `forward` over a [2, 8, 8, 4] input the graph holds one `Ops::K_DEPTHWISE_CONV2D` node
and no `Ops::K_BIAS_ADD` node.

### TheEmittedNodeReadsTheInputAndTheWeight

The depthwise node has two inputs: the value passed in, and a value of the shape [3, 3, 4]. Its
one output is the value `forward` returned.

### ForwardKeepsTheSpatialExtentsAndTagsNHWC

A padded 3x3 window at stride 1 over [2, 8, 8, 4] yields [2, 8, 8, 4], tagged
`TensorLayout::NHWC`.

### StrideTwoHalvesAnEvenExtent

At stride 2 a [2, 8, 6, 4] input yields [2, 4, 3, 4].

### TheWeightOfAnAnisotropicWindowIsHeightWidthChannels

With a window 2 high and 4 wide over three channels, the weight's shape after `forward` is
[2, 4, 3].

### TheParameterWalkNamesOnlyTheWeight

The parameter walk is empty before `forward`. After it the walk yields "weight:[3, 3, 4]"
alone, and `NN::parameter_elements` is 36.

### TheWeightRecordsTheWeightRole

After `forward`, `weight.role()` is `ParameterRole::Weight`.

### TheWeightRecordsItsWindowAsBothFans

After `forward` with a 3x3 window, `weight.fan_in()` and `weight.fan_out()` are both 9.

### TheWeightAndTheResultTakeTheInputElementType

Over a `Float64` input, the weight's element type and the result's element type are `Float64`.

### OutputShapeAtStrideOneKeepsTheExtents

`output_shape([8, 6, 4])` is [8, 6, 4] for a padded 3x3 window at stride 1.

### OutputShapeAtStrideTwoHalvesTheExtents

`output_shape([8, 6, 4])` is [4, 3, 4] for a padded 3x3 window at stride 2.

### OutputShapeAgreesWithForwardAtStrideOneAndTwo

For the sample [8, 6, 4] in a batch of 3, the traced shape equals `output_shape` with the batch
extent in front, at stride 1 and at stride 2.

### OutputShapeAgreesWithForwardOverAnOddExtent

At stride 2, `output_shape([7, 9, 4])` is [4, 5, 4], and the traced shape of a batch of 2 is
that shape with the batch extent in front.

### SamePaddedBuildsAPaddedLayerOfTheGivenStride

`same_padded(4, 4, 5, 2)` gives 4 channels, a 5x5 window, stride 2 on both axes, a top and left
padding of 2, and dilation 1 on both axes.

### SamePaddedRefusesUnequalChannelCounts

`same_padded(4, 8, 3, 1)` throws an error naming "keeps its channel count".

### ForwardRejectsAnInputThatIsNotRankFour

`forward` over a [2, 16] input throws an error naming "rank-4".

### ForwardRejectsAnNCHWInput

`forward` over a value converted to NCHW throws an error naming "NHWC".

### ForwardRejectsAChannelCountThatDiffersFromTheLayer

A four-channel layer over a [2, 8, 8, 3] input throws an error naming "[H, W, 4]".

### ForwardRejectsANonPositiveChannelCount

A layer with `channels` 0 throws an error naming "positive channel count".

### ForwardRejectsAWindowThatDoesNotFit

An unpadded 5x5 window over a [2, 4, 4, 4] input throws `Error`.

### OutputShapeRejectsAChannelCountThatDiffersFromTheLayer

`output_shape([8, 8, 5])` on a four-channel layer throws an error naming "[H, W, 4]".

### OutputShapeRejectsASampleOfTheWrongRank

`output_shape([8, 8])` throws an error naming "[H, W, 4]".

## modules/vision/detect/tests/test_detect_anchor_decode.cpp

Suites: `Unit_DetectAnchorDecode`.

Tests the decoding of one anchor head, with `count_head_channels`, `count_head_rows` and
`validate_head`. A head entry holds x, y, width, height, objectness and then the classes. A
saturated logit in this file is 100 or -100, and decoded values are compared to 1e-6.

### ZeroLogitsGiveTheCellCentreTheAnchorAndQuarterScores

A one-cell head of zero logits over a 0.25 x 0.5 anchor decodes to the centre (0.5, 0.5), the
size (0.25, 0.5) and a score of 0.25 for each of two classes.

### SaturatedLogitsReachTheCellEdgesAndTheScoreBounds

With x at +100, y at -100 and width at -100, the box has x 1, y 0, width 0 and height 0.5. With
objectness and the first class at +100 and the second class at -100, the scores are 1 and 0.

### ZeroObjectnessLogitHalvesASaturatedClass

A zero objectness logit with a saturated class logit gives the score 0.5.

### ScaleTwoMovesASaturatedOffsetHalfACellPastTheEdge

With `scale_xy` 2 on a grid of one cell, x at +100 decodes to 1.5 and y at -100 to -0.5.

### ScaleTwoKeepsAZeroLogitAtTheCellCentre

With `scale_xy` 2, zero logits decode to x 0.5 and y 0.5.

### ActivatedSizeAtZeroHalfAndOneIsZeroTheAnchorAndFourAnchors

Under the Activated coding with a 0.125 x 0.25 anchor, size inputs of 0, 0.5 and 1 decode to
the widths 0, 0.125 and 0.5 and the heights 0, 0.25 and 1.

### ActivatedOffsetsAndScoresTakeNoLogistic

Under the Activated coding, x 0.25 and y 1 decode to 0.25 and 1. Objectness 0.5 with the
classes 0.5 and 1 gives the scores 0.25 and 0.5.

### ActivatedScaleTwoMapsZeroAndOnePastTheCellEdges

Under the Activated coding with `scale_xy` 2, the offsets 0 and 1 decode to -0.5 and 1.5.

### RowsRunCellByCellThenByAnchorSlotInEveryImage

On a 2x3 grid with two slots and two images, decoded row `(row * 3 + col) * 2 + slot` holds the
centre of that cell and the size of that slot's anchor. The head's slots are anchors 2 and 0 of
the layout. Every score is 0.25 except row 11 of image 1, which the case saturated and which
is 1.

### OneLayoutGivesTheSameFractionalSizeOnTwoGrids

One 0.3 x 0.45 anchor decodes to that size on a 13x13 grid and to the identical floats on a
19x19 grid. The last cell's x is 12.5 / 13 on the first grid and 18.5 / 19 on the second.

### CountsChannelsAndRowsFromTheHeadsOwnAnchors

With 80 classes, a head of three anchors has 255 channels and a head of two has 170. A batch of
4 on a 13x13 grid with three anchors has 507 rows, and on a 26x26 grid with two anchors 1352.

### ValidatesAHeadTheLayoutHolds

`validate_head` is ok for 80 classes, the Activated coding, `scale_xy` 2 and head 0 of a
one-head layout.

### RefusesAHeadIndexOutsideTheLayout

Head index 1 and head index -1 of a one-head layout each fail with `InvalidDescription`.

### RefusesAHeadWithoutClasses

A class count of 0 fails with `InvalidDescription`.

### RefusesAScaleThatIsNotFiniteAndPositive

A `scale_xy` of 0 and a NaN `scale_xy` each fail with `InvalidDescription`.

### RefusesAHeadWhoseLayoutDoesNotValidate

A layout whose head names anchor 3 of one anchor fails with `InvalidDescription`.

## modules/vision/detect/tests/test_detect_anchor_delta.cpp

Suites: `Unit_DetectAnchorDelta`.

Tests the delta of one anchor head with the Logit coding, and `validate_head_training`. The
cases place truths in pixels of a 64-pixel network side and compare the whole delta with an
array built by hand, to 1e-6.

### DefaultTrainingOptionsValidate

`validate_head_training` is ok for a default `HeadTrainingOptions`.

### TrainingOptionsOutsideTheirRangeAreRefused

Nine option sets each fail with `InvalidDescription`: a match threshold of 0 and of 1.5, an
ignore threshold of -0.1, a class gate of 1.5, a box weight of -1 and of infinity, a NaN
object weight, a `max_delta` of -2, and a `box_gradient` that is no enumerator.

### OneTruthWritesTheCellAndSlotOfItsBestAnchor

One truth of the size (0.5, 0.5) centred at (0.75, 0.25) on a 2x2 grid with three anchors is
assigned to row 0, column 1, slot 1. That slot's box deltas are k times the differences in x, y,
log width and log height, with k = (2 - 0.5 * 0.5) * 0.75. Its objectness delta is 0.5 and its
class deltas are -0.5 and 0.5. Every other slot has an objectness delta of -0.5 and zeros
elsewhere. The result counts one assignment, and `assigned_row` and `assigned_class` are 0 and
1 at that slot and -1 at the others.

### AnAssignedSlotThatCoversItsTruthAddsOneToTheIoUSum

A one-cell head of zero logits whose anchor is the truth's box gives one assignment and an IoU
sum of 1 to 1e-6.

### ATruthWhoseBestAnchorAnotherHeadOwnsAssignsNothing

A truth whose size is the anchor head 1 owns leaves head 0 with no assignment, an IoU sum of 0,
`assigned_row` and `assigned_class` of -1 everywhere, and an objectness delta of -0.5 in each of
its four slots.

### AnOverlappingSlotIsIgnoredOnlyWithAClassAboveTheGate

Two images hold a slot that overlaps the truth with an IoU of 0.81 and is not assigned. Where
the class logit is -1.0, the objectness delta is exactly 0. Where it is -1.2, the objectness
delta is -0.5.

### AnOverlapBelowTheIgnoreThresholdIsNotIgnored

With an ignore threshold of 0.9, the same overlap of 0.81 leaves the objectness delta at -0.5.

### TwoTruthsInOneSlotUpdateOneClassOnly

Two truths with the same box and the classes 0 and 2 land in one slot. The box deltas are
twice those of one truth, the objectness delta is 0.5, and the class deltas are 0.5, -0.5 and
0.5. The result counts two assignments.

### AnEmptyTruthListGivesObjectnessDeltasAlone

Over random logits in [-3, 3] with an object weight of 2 and no truth, each objectness entry
holds 2 * (0 - objectness) and every other entry is 0. No slot is assigned.

### APaddingRowAssignsNothing

A truth row whose `valid` flag is 0 gives no assignment. The slot's objectness delta is -0.5
and its class delta is exactly 0.

## modules/vision/detect/tests/test_detect_boxes.cpp

Suites: `Unit_DetectBoxes`.

Tests box decoding and encoding, the two box formats and the IoU forms.

This file needs no device. `Unit_DetectBoxes` covers `modules/vision/detect/include/SushiAI/vision/detect/box_coding.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/box_format.hpp` and `modules/vision/detect/include/SushiAI/vision/detect/iou.hpp`. The hand cases
write the expected value as the arithmetic that produces it. Three cases read `forward.sacp`
and report themselves skipped when the file is absent.

### Tolerances

`K_EXACT` is 0, for values float32 holds exactly. `K_DECODE_TOLERANCE` is 1.3e-4, two float32
steps at 640, for a decoded box in pixels. `K_ROUND_TRIP_TOLERANCE` is 1.6e-5, two steps at 80,
for a decode followed by an encode in grid units. `K_IOU_TOLERANCE` is 1e-6, eight float32
steps at 1.

### K_REFERENCE_ANCHORS and K_REFERENCE_LEVELS

`K_REFERENCE_ANCHORS` is 8400 and `K_REFERENCE_LEVELS` holds the three levels of a 640 px
input, both from "levels" in `tests/fixtures/yolo26/MANIFEST.json`.

### ReferenceGrid

`ReferenceGrid` holds the `[8400, 2]` points and the `[8400]` strides of the reference levels
in grid units, and `make_reference_grid` fills one with `make_anchors`.

### join_raw_boxes

`join_raw_boxes` reads a branch's three per-level tensors, `<branch>.level<i>.boxes`, and
copies each into its place in an `[8400, 4]` array with `join_level`. It returns an empty
string, or the reason a level could not be read.

### DecodesDistancesToCorners

`decode_xyxy` of the distances (0.5, 1, 1.5, 2) at the point (1.5, 2.5) gives
(1, 1.5, 3, 4.5): left and top are subtracted from the point, right and bottom added.

### DecodesDistancesToCentreAndSize

`decode_xywh` of the same input gives the midpoint (2, 3) and the size (2, 3) of those corners.

### RepeatsTheAnchorPointsForEachImage

Two anchor points and four rows of distances decode as two images: rows 2 and 3 use points 0
and 1 again.

### EncodeInvertsDecode

`encode_distances` of the boxes `decode_xyxy` writes returns the three rows of distances at
tolerance 0.

### ScalesEachBoxByItsAnchorStride

`scale_boxes` multiplies all four values of a row by its anchor's stride: the row (1, 2, 3, 4)
becomes (8, 16, 24, 32) at stride 8 and (16, 32, 48, 64) at stride 16.

### ConvertsBetweenCornersAndCentreSize

`xyxy_to_xywh` maps (1, 2, 5, 8) to (3, 5, 4, 6), and `xywh_to_xyxy` maps that back to the
corners.

### RejectsRowsThatDoNotCoverTheAnchors

Three rows of distances over two anchor points throw with a message containing "2 anchors" and
"12".

### ReproducesTheReferenceOneToManyDecode

The raw one-to-many boxes, decoded with `decode_xywh` and scaled by the strides, equal
`one2many.decoded.boxes` within `K_DECODE_TOLERANCE`. The case is skipped when `forward.sacp`
is absent.

### ReproducesTheReferenceOneToOneDecode

The raw one-to-one boxes, decoded with `decode_xyxy` and scaled, equal `one2one.decoded.boxes`
within `K_DECODE_TOLERANCE`. The case is skipped when `forward.sacp` is absent.

### EncodeInvertsTheDecodeOfTheReferenceBoxes

Encoding the decode of the raw one-to-one boxes returns them within `K_ROUND_TRIP_TOLERANCE`.
The case is skipped when `forward.sacp` is absent.

### IdenticalBoxesScoreOne

A 2 by 2 box has IoU 1 and complete IoU 1 with itself.

### DisjointBoxesScoreZeroAndPayForDistance

Two unit squares two units apart have IoU 0. Their complete IoU is minus the squared centre
distance, 4, over the squared hull diagonal, 10.

### OverlappingSquaresLoseTheCentreTerm

Two 2 by 2 squares offset by 1 on both axes have IoU 1 / 7, and their complete IoU is that
minus 2 / 18. The aspect ratios are equal, so the aspect term is 0.

### DifferentAspectRatiosLoseTheAspectTerm

A 4 by 2 box and a 2 by 2 box at one corner have IoU 0.5. The expected complete IoU is computed
in double from the centre term, the angle gap `atan(1) - atan(2)` and the trade-off factor, and
compared within `K_IOU_TOLERANCE`.

### ADegenerateBoxStaysFinite

A box of zero size inside a 2 by 2 box has IoU 0. Its width over its height is 0 over eps, so
the angle gap is pi / 4, the aspect term is 0.25, and the complete IoU is -0.25 * 0.25 / 1.25,
which is -0.05. The complete IoU of the zero-size box with itself is finite and 0.

### ElementwiseFormsPairRows

`elementwise_iou` and `elementwise_complete_iou` over two rows give 1 / 7 and 0, and
1 / 7 - 2 / 18 and -4 / 10: row i of the first array is paired with row i of the second.

### PairwiseFormFillsEveryPair

`pairwise_iou` of two boxes against three writes six values, row n and column m holding the
IoU of box n with box m: 1, 1 / 7, 0, then 1 / 4, 0, 0.

### RejectsElementwiseInputsOfDifferentLength

Arrays of 8 and 4 values with an output of 2 throw with a message containing "got 8, 4 and 2".

## modules/vision/detect/tests/test_detect_postprocess.cpp

Suites: `Unit_DetectTopK`, `Unit_DetectPostprocess`.

Tests the top-k selection, both postprocess paths and the overlap suppression.

This file needs no device. `Unit_DetectTopK` covers `modules/vision/detect/include/SushiAI/vision/detect/top_k.hpp`, and
`Unit_DetectPostprocess` covers `modules/vision/detect/include/SushiAI/vision/detect/detections_topk.hpp` and
`modules/vision/detect/include/SushiAI/vision/detect/detections_nms.hpp`. The suppression path takes boxes as centre and
size, so the hand cases write them that way and expect corners. Four cases read `forward.sacp`
and report themselves skipped when the file is absent.

Two of the reference comparisons cannot compare row by row. The reference orders equal scores
in an order torch does not specify, and the fixture's scores are nearly constant.
`modules/vision/detect/README.md`, under `top_k.hpp` and "Reference rows the suppression test
reads", gives the counts.

### K_EXACT

`K_EXACT` is the tolerance 0. A postprocess copies its output values from its inputs.

### K_SUPPRESSION_CONFIDENCE

`K_SUPPRESSION_CONFIDENCE` is 1.8e-4, the `suppression.arguments.conf_thres` of `forward.sacp`
in `tests/fixtures/yolo26/MANIFEST.json`.

### Row

`Row` is an array of six floats, one detection.

### sorted_rows_above

`sorted_rows_above` returns the rows whose score exceeds a floor, sorted by their six values.
Two detection arrays that hold the same rows in different orders compare equal through it.

### is_anchor_class_pair

`is_anchor_class_pair` reports whether a row's box is some anchor's box and its score is that
anchor's score in the row's class.

### make_nms_config

`make_nms_config` returns an `NMSConfig` with the given confidence threshold, IoU threshold,
label mode and class mode, and the default limits.

### suppress_one_image

`suppress_one_image` runs `select_nms_detections` on one image of `[A, 4]` boxes and returns
that image's rows.

### OrdersByValueThenIndex

`select_top_k` of the values 0.5, 0.9, 0.5, 0.1, 0.9 with k = 4 returns the indices 1, 4, 0, 2:
larger values first, equal values by ascending index. Asking for 6 of 5 values throws with a
message containing "6" and "5 values".

### SelectsTheHighestScoringAnchorClassPairs

Three anchors with two classes and a limit of 2 give the row of anchor 1 with score 0.9 and
class 0, then the row of anchor 0 with score 0.7 and class 1.

### OneAnchorCanYieldTwoDetections

Anchor 1 scores 0.8 and 0.7 in its two classes, both above anchor 0's 0.6. Both output rows
carry anchor 1's box, with classes 0 and 1.

### LimitsRowsToTheAnchorCount

`count_top_detections` returns 3 for three anchors and a limit of 300, and 300 for 8400 anchors.

### SelectsEachImageFromItsOwnRows

A batch of two images with two anchors each and a limit of 1 gives one row per image: anchor 1
of image 0 at 0.6 in class 0, and anchor 1 of image 1 at 0.9 in class 1.

### RejectsADetectionArrayOfTheWrongSize

An output of 6 values where the shape needs `[1, 2, 6]` throws with a message containing
"[1, 2, 6] detections" and "6 values".

### ReproducesTheReferenceDetections

`select_top_detections` on `one2one.decoded.boxes` and `one2one.decoded.scores` with
`K_MAX_DETECTIONS` is compared with `one2one.detections` in three ways. The score column and
the class column are equal row by row. The rows above the last score are equal as a set,
through `sorted_rows_above`. Every output row passes `is_anchor_class_pair`. The case is
skipped when `forward.sacp` is absent.

### TheConfidenceFilterDropsAScoreAtTheThreshold

`keep_confident_detections` at 0.25 keeps the rows at 0.5 and 0.26, in their order, and drops
the rows at 0.25 and 0.1.

### TheConfidenceFilterKeepsEachImagesRows

Four rows split between two images at `K_VALIDATION_CONFIDENCE`: image 0 keeps its row at 0.9
and image 1 keeps none.

### TheConfidenceFilterRejectsRowsTheBatchDoesNotDivide

Three rows for a batch of 2 throw with a message containing "[2, k, 6] detections" and "18
values". A batch of 0 throws with "positive batch" and "got 0".

### TheConfidenceFilterKeepsTheRowsTheReferenceValidatorKeeps

At `K_VALIDATION_CONFIDENCE` the filter keeps none of the 300 rows of `one2one.detections`,
which is `validation.row_count.one2one` of the manifest. At threshold 0 it keeps all of them
unchanged, since every reference score is a sigmoid above zero. The case is skipped when
`forward.sacp` is absent.

### SuppressionReproducesTheReferenceRows

`select_nms_detections` at `K_SUPPRESSION_CONFIDENCE` on the one-to-many decode returns as many
values as `one2many.suppressed.detections`. The score column is equal row by row and the rows
above the last score are equal as a set. The case is skipped when `forward.sacp` is absent.

### NoReferenceScorePassesTheValidationConfidence

With a default `NMSConfig` the one-to-many decode yields no row, which is
`validation.row_count.one2many` of the manifest. The case is skipped when `forward.sacp` is
absent.

### SuppressionDefaultsMatchTheValidator

A default `NMSConfig` has confidence 0.001, IoU 0.7, `LabelMode::MultiLabel`,
`ClassMode::ClassAware`, 300 detections, 30000 candidates and a class offset of 7680.

### SuppressesTheLowerOfTwoOverlappingBoxes

Boxes 0 and 1 share 81 of 119 px, an IoU of 0.68, and box 2 is apart. `suppress_overlaps` at 0.5
keeps the indices 0 and 2.

### ListsSurvivorsByDescendingScore

Four boxes that do not overlap, with scores 0.1, 0.9, 0.5 and 0.5, are kept in the order
1, 2, 3, 0.

### KeepsABoxWhoseIoUEqualsTheThreshold

Two boxes share 2 of 4 px, an IoU of exactly 0.5 in float32. At threshold 0.5 both are kept; at
0.25 only the first is.

### OverlappingBoxesOfOneClassCollapse

Two anchors hold the same box and score 0.9 and 0.8 in class 0. One row remains, with corners
(0, 0, 10, 10), score 0.9 and class 0.

### OverlappingBoxesOfDifferentClassesBothSurvive

The same two boxes score in classes 0 and 1. `ClassMode::ClassAware` keeps both rows, and
`ClassMode::ClassAgnostic` keeps the one at 0.9.

### DropsAScoreEqualToTheConfidenceThreshold

An anchor at 0.25 with the threshold at 0.25 yields no row. The anchor at 0.5 yields the one
row.

### AnAnchorContributesEveryPassingClassOrItsBest

One anchor scores 0.6, 0.1 and 0.7 in three classes. `LabelMode::MultiLabel` yields two rows,
class 2 at 0.7 then class 0 at 0.6. `LabelMode::BestClass` yields the row of class 2 alone.

### LimitsDetectionsToTheHighestScores

Three boxes that are apart score 0.5, 0.9 and 0.7. With `max_detections` 2 the rows at 0.9 and
0.7 remain.

### LimitsCandidatesBeforeSuppression

With `max_candidates` 1 the same input yields the row at 0.9 alone.

### RejectsAThresholdOutsideTheUnitInterval

A confidence threshold of 1.5 throws with a message containing "[0, 1]" and "1.5".

### AnImageBelowTheThresholdYieldsNoRows

Scores of 0.25, 0.1, 0.2 and 0 at the threshold 0.25 yield an empty row vector.

### SuppressesEachImageFromItsOwnRows

A batch of two images with two anchors each. Image 0 holds one box twice, so the row at 0.9
remains. Image 1 holds two boxes that are apart, and its rows are 0.6 then 0.3.

### RejectsScoresThatDoNotFillTheShape

Six scores where the shape `[1, 2, 2]` needs four throw with a message containing "[1, 2, 2]
scores" and "8 and 6 values".

## test_diagnostics.cpp

Suites: `Unit_Diagnostics`.

Tests that each host-lane precondition raises a message carrying the values that disagree.

This file needs no device. It holds every host-lane precondition to the message it promises.

Other tests already prove that these preconditions throw. They do not prove that the messages
still say anything. A refactor that funnels five specific diagnostics into one generic "invalid
argument" leaves every EXPECT_THROW green and destroys the only thing that makes a shape
mismatch debuggable. So each case names the values the message must still carry (the two
extents that disagree, the operation that has no rule, the dtype that cannot be served) and
fails when one of them goes missing.

`tests/common/test_throws.hpp` defines SA_EXPECT_ERROR.

The cases fall into five groups, in file order:

- Shape
- GraphBuilder, the shape-inference boundary
- Graph: single assignment and operand existence
- layers
- differentiation

### ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit

K_MAX_RANK is SushiBLAS's, so the number in the message is the one a reader has to reconcile
against the tensor they were trying to build.

### ShapeNonPositiveExtentNamesTheOffendingExtent

A zero-element tensor is refused at the type that describes extents, which is why nothing
further down ever has to handle one.

### ShapeDimensionOutOfRangeNamesTheIndexAndTheRank

On a [4, 3] shape, `dim(2)` raises an error holding "Dimension 2" and "rank-2 shape", and
`dim(-3)` one holding "Dimension -3" and "rank-2 shape".

### MatmulContractionMismatchNamesBothInnerExtents

`matmul` of [4, 5] and [7, 3] raises an error holding "MATMUL contracts 5 against 7",
"[4, 5]" and "[7, 3]".

### MatmulRejectsARank1OperandAndPrintsWhatItGot

Rank 1 where rank 2 belongs is the commonest way a bias or a flattened sample reaches a gemm.
The message has to show both shapes or the reader cannot tell which side was wrong.

### MatmulNamesBothDtypesWhenTheyDisagree

`matmul` of a float32 and a float64 operand raises an error holding
"MATMUL needs operands of one element type", "float32" and "float64".

### BiasAddNamesTheAxisThatDoesNotLineUp

`bias_add` raises an error for a bias of 5 entries over [4, 3]
("BIAS_ADD bias has 5 entries", "[4, 3]", "3 columns"), for a [1, 3] bias
("BIAS_ADD needs a rank-1 bias", "[1, 3]") and for a rank-1 left operand
("BIAS_ADD needs a rank-2 left operand", "[3]").

### SumRowsAndLogSoftmaxRejectARank1Operand

Over a [6] operand, `sum_rows` raises an error holding "SUM_ROWS needs a rank-2 operand" and
"[6]", and `log_softmax` one holding "LOG_SOFTMAX needs a rank-2 operand" and "[6]".

### ElementwiseOperationsNameBothShapesAndBothDtypes

Over [4, 3] and [4, 2], `add` raises an error holding "ADD needs equally shaped operands" and
both shapes, and `sub`, `mul` and `accumulate` raise the same sentence under their own
operation names. `mul` of a float32 and a float64 operand raises
"MUL needs operands of one element type" with "float32" and "float64".

### NLLLossNamesTheShapeItCannotServe

`nll_loss` raises an error for [4, 3] against [4, 5]
("NLL_LOSS needs equally shaped operands", "[4, 3]", "[4, 5]") and for rank-1 operands
("NLL_LOSS needs rank-2 operands", "[4]").

### ActivationBackwardRefusesAnOperationThatIsNotOne

`activation_backward` with the identity of RELU raises an error holding "sushiai.relu" and
"is not an activation gradient".

### ASecondProducerNamesBothOperationsAndTheRule

The second write is what would make a differentiated graph silently wrong, so the message has
to say which two operations collided.

### AnUnknownOperandNamesTheOperationThatAskedForIt

`add_node` of a RELU that reads the value 9999 raises an error holding "sushiai.relu",
"reads value 9999" and "does not exist". One that writes the value 9999 raises
"writes value 9999" and "does not exist".

### LinearNamesItsOwnDimensionsWhenTheInputDoesNotFit

`Linear{6, 4}` over [8, 5] raises an error holding "Linear(6, 4)", "[8, 5]" and
"inner extent must be 6". `Linear{5, 4}` over a [5] input raises
"Linear needs a rank-2 input" with "[5]", and `Linear{0, 4}` raises
"Linear needs positive feature counts" with "in=0" and "out=4".

### ANonScalarRootNamesTheValueAndItsShape

Differentiating from a [4, 3] ReLU output raises an error holding
"needs a scalar to differentiate" and "[4, 3]".

### AOneElementRootOfADifferentiableOperationIsSeeded

The root is a [1, 1] K_MATMUL of a parameter `x` and an input `w`. Differentiation does not
throw and appends two nodes. The first is a K_CONSTANT in the root's shape, the seed. The second
is a K_MATMUL that reads the seed and `w`, and its result is the gradient of `x`, in the shape of
`x`. `test_loss_roots.cpp` covers the seed in full.

### ARootThatRequiresNoGradientNamesItsProducerAndTheCause

Two roots are refused. A K_MATMUL of two inputs is reached by no parameter, and the message
contains "value 3", "sushiai.matmul" and "requires no gradient". A K_MATMUL that reads the
parameter through `stop_gradient` is refused with a message containing "sushiai.matmul",
"requires no gradient" and "K_STOP_GRADIENT", so the message names the barrier as a cause the
caller can look for.

### ARootWhoseProducerHasNoRuleNamesTheProducer

The root is a K_DYNAMIC_SCALE of a parameter, an operation with no backward rule. The message
contains "sushiai.dynamic_scale", "has no backward rule" and "autograd/rules/", the folder a
rule is added in.

### ALossNoOperationProducesIsRejected

The `orphan` value is declared but never written. There is no operation to blame, so the
message has to say that and cannot name one.

### AnOperationWithNoBackwardRuleNamesItselfAndItsNode

"sushiai.unknown" is what `name_of()` reports for a foreign identity, and the node index is
what turns that into something findable.

### AGradientContributedToTheWrongShapeNamesBothShapes

This error is reachable only from inside a backward rule, so the test provokes it the way a
buggy rule would. It builds a graph by hand whose K_SUB claims a result of a different shape from
its operands, so the rule contributes a [4, 3] gradient to a [2, 6] value.

## modules/layers/nn/tests/test_dropout.cpp

Suites: `Unit_Dropout`.

Tests `NN::Dropout` in `modules/layers/nn/include/SushiAI/nn/dropout.hpp` and its factory
`NN::make_dropout_node`. A `static_assert` at the top of the file holds that `Dropout` is a
`ModalModule` and is neither a `Module` nor a `TraceModule`. The training traces state the
seed 5.

### TheTrainingFormEmitsOneDropoutNode

A layer of rate 0.25 over a [2, 3, 4] input adds one DROPOUT node that reads the input. The
node holds 0.25 in `K_DROPOUT_RATE_SLOT`, its seed is `dropout_site_seed(5, 0)`, and the result
has the input's shape. The layer has no parameter and no buffer.

### TheEvaluationFormReturnsItsInputAndEmitsNothing

Under `Trace::evaluation` the layer returns its input, the trace is ok, and the node count and
the value count do not change.

### ARateOfZeroEmitsNothingInTraining

A default `Dropout` under a training trace returns its input and adds no node. The next
`claim_random_site()` of the trace returns 0.

### EachSiteOfATraceGetsItsOwnSeed

Two layers traced one after the other get the seeds `dropout_site_seed(5, 0)` and
`dropout_site_seed(5, 1)`. A second graph traced the same way gets the same two seeds.

### ATraceWithoutAStatedSeedHasSeedZero

`Trace::training` without a seed and `Trace::evaluation` report the seed 0. `Trace::training`
given 5 reports 5.

### ARateOutsideItsIntervalFails

For the rates 1, -0.1 and NaN, `trace_training` and `trace_evaluation` return their input and
both traces fail with `InvalidDescription` and a message containing "rate in [0, 1)". The graph
has a refusal.

### OutputShapeIsTheInputShape

`output_shape([6, 4])` is [6, 4].

### TheFactoryReadsTheRate

`make_dropout_node` over one [6, 4] input and `{"rate": 0.1}` returns a `ModuleNode<Dropout>`
whose layer has the rate 0.1 and whose one port has the shape [6, 4].

### TheFactoryRefusesARowThatDoesNotFit

Four rows fail with `InvalidDescription` and a message naming "layer 3": no rate ("'rate'",
"absent"), a rate of 1 ("'rate'", "[0, 1)"), a `seed` key ("'seed'") and two inputs ("reads
one input", "given 2").

## modules/layers/nn/tests/test_embedding.cpp

Suites: `Unit_Embedding`, `Unit_EmbeddingDeathTest`.

Tests `NN::Embedding` in `modules/layers/nn/include/SushiAI/nn/embedding.hpp` and its factory
`NN::make_embedding_node`. Every layer has a table of 7 rows of length 3. A `static_assert`
holds that `Embedding` is a `TraceModule` and is neither a `Module` nor a `ModalModule`.

### ATraceDeclaresTheTableAndEmitsOneGather

Over a [2, 5] int32 input the trace is ok and adds two nodes, one of them GATHER_ROWS. That node
reads the weight and then the indices, and its result is a float32 value of shape [2, 5, 3].

### TheTableIsAnEmbeddingDrawnFromANormalOfFixedDeviation

After a trace of a float64 layer the weight is declared with the shape [7, 3], the type float64
and the role `ParameterRole::Embedding`. Its initialiser is `Normal` with the deviation
`K_EMBEDDING_DEVIATION`, which is 0.02, and its fan-in and fan-out are both 3. The weight's
value has the lifetime `Parameter` and requires a gradient, and the result is float64.

### TheWalkYieldsTheTable

`NN::walk_state` under the prefix `tokens` visits one name, `tokens.weight`, as a parameter of
shape [7, 3], and visits no buffer.

### TwoGraphsNameTheSameTable

After traces of one layer into two graphs, `weight.state()` is the same `StateID` and is not
`K_NO_STATE`, and `weight.id_in` returns a value in each graph.

### AnInputThatIsNotInt32Fails

A float32 input and an int64 input each fail with `InvalidDescription` and a message containing
"Embedding(7, 3)", "int32 input of indices" and the type given. `forward` returns its input,
the graph gains no value and no node, the weight stays undeclared and the graph has a refusal.

### AnInputOfTheHighestRankFails

A rank-6 int32 input fails with `InvalidDescription`, "rank 1 to 5" and
"[1, 1, 1, 1, 1, 1]", and the weight stays undeclared.

### ALayerDescribedWrongFails

`num_embeddings` 0 and `embedding_dim` -1 each fail with "positive extents" and the field with
its value. An int32 table fails with "half, float32 or float64 table" and "int32". The weights
of the first and the third layer stay undeclared.

### OutputShapeAppendsTheRowLength

`output_shape` maps [5] to [5, 3], [4, 2] to [4, 2, 3] and [1, 1, 1, 1] to [1, 1, 1, 1, 3].

### TheFactoryReadsTheTwoExtents

`make_embedding_node` over one [5] input with `num_embeddings` 11 and `embedding_dim` 4 returns
a `ModuleNode<Embedding>` whose layer holds 11, 4 and float32. The node reads one input and its
one port has the shape [5, 4].

### TheFactoryRefusesARowThatDoesNotFit

Five rows fail with `InvalidDescription` and a message naming "layer 3": a `padding` key, an
absent `embedding_dim`, a `num_embeddings` of 0 ("positive"), two inputs ("reads one input",
"given 2") and a rank-5 input ("[1, 1, 1, 1, 1]", "rank 1 to 4").

### ATraceOverAValueTheGraphLacksIsFatal

`forward` over value id 7 in a graph of one value stops the process with a report containing
"Embedding(7, 3)", "value 7" and "holds 1 values".

### OutputShapeOverARankFiveSampleIsFatal

`output_shape([1, 1, 1, 1, 1])` stops the process with a report containing
"Embedding::output_shape" and "at most 4".

## modules/layers/nn/tests/test_flatten.cpp

Suites: `Unit_Flatten`.

Tests the Flatten layer's trace, shape rule, rejections and parameter walk.

This file needs no device. `NN::Flatten` in `modules/layers/nn/include/SushiAI/nn/flatten.hpp` has no field and
no parameter. It folds every axis after the batch axis into one.

### traced_shape

`traced_shape` traces a `Flatten` over a new input of the given batched shape and returns the
result's shape.

### SatisfiesTheModuleTrait

Two `static_assert` lines require `Flatten` to be a `Module` and not a `ModuleContainer`.

### ARank4InputTracesToOneReshape

A [2, 4, 5, 3] input traces to a [2, 60] value. The trace adds exactly one node, a K_RESHAPE
with the input as its one operand and the result as its one output.

### ARank3InputFoldsItsTwoInnerAxes

A [2, 3, 4] input traces to [2, 12].

### ARank2InputPassesThroughWithNoNode

For a [2, 7] input `forward()` returns the id it was given. The node count and the value count
are unchanged, so a flatten placed after a layer that already yields feature vectors adds
nothing to the step.

### AnNHWCInputYieldsAnNHWCResult

The input is asserted to be NHWC first, and the flattened result is NHWC too.

### ForwardRejectsAnNCHWInput

The layout tag of a [2, 3, 4, 5] input is set to NCHW by hand, and `forward()` throws.

### OutputShapeAgreesWithTheTracedShape

`output_shape()` maps [4, 5, 3] to [60], [3, 4] to [12] and [7] to [7]. For each, the traced
shape of a batch equals the batch extent followed by that one extent.

### ForwardRejectsARank1Input

`forward()` throws on a [6] input, which has a batch axis and no per-sample axis.

### OutputShapeRejectsARank0Shape

`output_shape()` throws on an empty shape.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.

## modules/layers/nn/tests/test_global_avg_pool.cpp

Suites: `Unit_GlobalAvgPool`.

Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.

This file needs no device. `NN::GlobalAvgPool` in `modules/layers/nn/include/SushiAI/nn/global_avg_pool.hpp` has
no field and no parameter. It reduces the two spatial axes of an image to their mean.

### traced_shape

`traced_shape` traces the layer over an [N, H, W, C] input built from its four arguments and
returns the result's shape.

### SatisfiesTheModuleTrait

Three `static_assert` lines require `GlobalAvgPool` to be a `Module`, not to be a
`ModuleContainer`, and to have a `forward()` that is callable on a const object and returns
`Graph::ValueID`.

### ForwardReducesTheSpatialAxes

A [4, 7, 5, 16] input traces to [4, 16].

### ForwardEmitsOneSpatialMeanNode

The trace adds exactly one node, a K_SPATIAL_MEAN with the input as its one operand and the
result as its one output.

### ForwardKeepsTheNHWCLayoutTag

The result of an NHWC input is tagged NHWC, and `finalise()` accepts the graph.

### OutputShapeKeepsOnlyTheChannels

`output_shape([7, 5, 16])` is [16].

### OutputShapeAgreesWithForward

For a square [7, 7, 512] sample and an oblong [3, 9, 2] one, `output_shape()` is rank 1 and
the traced shape is the batch extent followed by its one extent.

### OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

### OutputShapeRejectsARankTwoShape

`output_shape([7, 16])` throws.

### OutputShapeRejectsAShapeWithTheBatchAxis

`output_shape([4, 7, 5, 16])` throws.

### ForwardRejectsAFeatureMatrix

`forward()` throws on a [4, 16] input.

### ForwardRejectsAnInputWithoutTheBatchAxis

`forward()` throws on a [7, 5, 16] input.

### ForwardRejectsAnNCHWInput

A [1, 4, 4, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace, and `parameter_elements` is 0 after it.

## modules/layers/nn/tests/test_host_table.cpp

Suites: `Unit_HostTable`.

Tests `NN::Trace::host_table` and the `NN::HostTableList` a trace collects into with
`collect_tables`.

### ATableIsALeafListedWithItsValues

A [2, 3] float32 table of six values adds one node. Its leaf has that shape and type, the
lifetime `Buffer` and no gradient, and `graph.state_leaves()` stays empty. The list holds one
table with that leaf and the six values, and the trace is ok.

### EveryDeclarationIsItsOwnLeaf

Two float64 tables declared on a training trace return two different leaves, and the list
holds them in the order they were declared.

### ATableThatNoStepCanFillIsRefused

`host_table` fails with `InvalidDescription` and a message naming `'angles'` three times: on a
trace without `collect_tables` ("collect_tables"), for the type int32 ("float32 or a float64")
and for two values under the shape [2, 3] ("6 elements", "2 values"). The graph gains no node
and the list stays empty.

## modules/layers/nn/tests/test_identity.cpp

Suites: `Unit_Identity`.

Tests that the Identity layer returns its input, emits nothing and holds no state.

This file needs no device. `NN::Identity` is in `modules/layers/nn/include/SushiAI/nn/identity.hpp`.

Three `static_assert` lines at the top of the file hold that `Identity` is a `Module`, is not
a `ModuleContainer`, and that `Identity::fields()` is an empty tuple.

### ForwardReturnsItsInput

`forward` over a [2, 8, 6, 3] input returns the id it was given.

### ForwardEmitsNothing

The graph's node count and value count are the same before and after `forward`.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## modules/vision/imaging/tests/test_imaging_partners.cpp

Suites: `Unit_ImagingPartners`.

Tests the two sources of a mosaic's three partner images.

This file needs no device. `Unit_ImagingPartners` covers `DatasetPartners` of
`modules/vision/imaging/include/SushiAI/vision/imaging/dataset_partners.hpp`, which draws partners from the whole dataset,
and `RecentPartners` of `modules/vision/imaging/include/SushiAI/vision/imaging/recent_partners.hpp`, which draws them from a
buffer of recently seen indices.

### buffer_after

`buffer_after` passes a list of indices to `RecentPartners::choose`, one by one with a
generator seeded 1, and returns the buffer, oldest first.

### DatasetSourceReturnsThreeIndexDrawsInOrder

`choose` on a dataset of 1000 returns three values equal to three `index(1000)` draws of a copy
of the generator, in draw order, and the two generators are equal afterwards.

### DatasetSourceCoversTheDatasetAndNeverLeavesIt

Over 300 seeds and a dataset of 7, every partner is below 7 and every index is a partner at
least once.

### DatasetSourceKeepsNoStateBetweenSamples

A source that already answered two calls and a new source return the same partners for the
same index and seed.

### DatasetSourceRejectsSizesAndIndicesOutOfRange

A dataset size of 0 and one above the largest `uint32` throw, and the largest `uint32` is
accepted. `choose(4)` on a dataset of 4 throws with "index 4" and makes no draw.

### RecentCapacityIsTheSmallestOfItsThreeBounds

`recent_capacity` is 5 for a dataset of 5, 128 for a batch of 16 and 1000 for a batch of 200:
the smallest of the dataset size, `K_RECENT_BATCHES` (8) batches and `K_RECENT_LIMIT` (1000).

### RecentSourceGivesTheFirstSampleItselfThreeTimes

The first call, for index 42, returns 42 three times, and the buffer holds 42.

### RecentSourceEvictsTheOldestIndexAtCapacity

With capacity 3 the buffer goes {0, 1}, {1, 2}, {2, 0}. With capacity 2 it holds one index at a
time. The reference removes the oldest entry when the buffer has more than one entry and has
reached its capacity, so a full buffer holds capacity minus one indices.

### RecentSourceDoesNotAddAnIndexItAlreadyHolds

After the indices 7, 9, 7, 7, 9 the buffer is {7, 9}.

### RecentSourceOfOneImageNeverEmptiesItsBuffer

With a dataset of one image, two calls both return 0 three times and the buffer holds 0.

### RecentSourceHoldsTheLastIndicesOfALongRun

With capacity 16, after the indices 0 to 19 the buffer holds 15 indices, from 5 to 19.

### RecentSourceDrawsThreeEntriesOfItsBuffer

After 40 earlier indices and the index 500, the buffer has 41 entries. Each partner is the
buffer entry that one `index(41)` draw of a copied generator names, in draw order, and the
source makes three draws.

### RecentSourceCoversItsBuffer

With the indices 10 to 14 buffered, over 200 seeds every partner is one of them and each is
drawn at least once.

### RecentSourceRejectsSizesAndIndicesOutOfRange

A dataset size of 0 and batch sizes of 0 and -3 throw. `choose(4)` on a dataset of 4 throws
with "index 4", makes no draw and leaves the buffer empty.

## modules/layers/io/tests/test_imported_model_writer.cpp

Suites: `Unit_ImportedModelWriter`.

Tests `IO::compose_training_file` and `IO::write_imported_model`. The imported model of every
case has one `linear` row over the input shape [4], a task of type `stub_task`, the float32
tensors `0.weight` of shape [2, 4] and `0.bias` of shape [2], a recipe with an `objective` and
an `optimizer` section, and one note. The cases that write files use a directory under the
system's temporary folder and remove it before and after.

### ComposesTheTrainingFileAroundTheRecipe

The composed file has the keys `sushiai_train`, `model`, `weights`, `data`, `checkpoint`,
`objective`, `optimizer`, in that order. `sushiai_train` is `Config::K_TRAIN_VERSION`, `model`
is "model.json", `weights` is "weights.sushi", `checkpoint` is "trained.sushi" and the
optimizer's `type` is "sgd".

### LeavesTheDataBlockNull

The composed file has a `data` key whose value is null.

### StatesNoWeightsKeyWithoutTensors

With no tensors the composed file has no `weights` key.

### ARecipeThatStatesAWriterKeyIsRefused

A recipe that states `model` fails with `InvalidDescription` and a message containing `'model'`
and "training recipe".

### ARecipeThatIsNoObjectIsRefused

A recipe that is the integer 3 fails with `InvalidDescription` and "must be an object".

### WritesTheThreeFiles

`write_imported_model` with `ExistingFiles::Refuse` returns three paths, `model.json`,
`weights.sushi` and `train.json` under the directory, and each is a regular file.

### TheModelFileReadsBack

`Config::load_model` over the written model file returns the input shape that was given, one
layer of type `linear` and a task of type `stub_task`.

### TheWeightFileHoldsEveryTensor

`IO::CheckpointReader` over the written weight file counts two records. Each record has the
name, the type and the shape of the tensor at its position, and its payload equals that
tensor's bytes.

### TheTrainingFileIsRefusedUntilItsDataIsFilled

`Config::load_train` over the written training file raises an error containing `'data'`, "must
be an object" and "null".

### WritesNoWeightFileWithoutTensors

With no tensors the write returns two paths and `weights.sushi` does not exist.

### AnExistingFileIsRefusedAndLeftAlone

With a `train.json` holding "kept" in the directory, the write under `ExistingFiles::Refuse`
fails with `Incompatible`, "train.json" and "exists already". The file still holds "kept" and
`model.json` does not exist.

### ReplaceOverwritesAnExistingFile

With a `train.json` holding "stale", the write under `ExistingFiles::Replace` is ok and the
file contains "sushiai_train".

### ATensorOfTheWrongSizeWritesNothing

With one byte removed from the payload of `0.bias`, the write fails with `InvalidDescription`,
`'0.bias'`, "7 bytes" and "needs 8", and `model.json` does not exist.

### ATensorNamedTwiceIsRefused

With both tensors named `0.weight`, the write fails with `InvalidDescription`, `'0.weight'` and
"twice".

## modules/layers/io/tests/test_importer_registry.cpp

Suites: `Unit_ImporterRegistry`.

Tests `IO::ImporterRegistry` and `IO::import_model`. The file registers two stub rows: `first`,
whose importer returns a model with the number of files it was given as its one note, and
`second`, whose importer fails with `InvalidFile` when given no file.

### AddKeepsInsertionOrder

After `add` of `first` and then `second`, `rows()` holds two rows in that order.

### FindReturnsNullForAnUnknownFormat

`find("first")` is null on an empty registry. After `first` is added, `find("onnx")` is null
and `find("first")` returns the row whose format is "first".

### AFormatAddedTwiceIsRefused

A second `add` of `first` fails with `InvalidDescription` and a message containing `'first'`
and "already registered".

### ImportModelCallsTheRowOfTheFormat

`import_model` for `first` with two files returns a model whose one note is "2".

### ImportModelReturnsTheFailureOfTheRow

`import_model` for `second` with no file fails with `InvalidFile` and "the stub reads no file".

### AnUnknownFormatListsTheFormatsTheRegistryHolds

`import_model` for `onnx` fails with `InvalidDescription` and a message containing `'onnx'` and
"'first', 'second'".

## modules/layers/nn/tests/test_layer_norm.cpp

Suites: `Unit_LayerNorm`, `Unit_LayerNormDeathTest`.

Tests `NN::LayerNorm` in `modules/layers/nn/include/SushiAI/nn/layer_norm.hpp` and its factory
`NN::make_layer_norm_node`. Every layer normalises over a last extent of 4. A `static_assert`
holds that `LayerNorm` is a `TraceModule` and is neither a `Module` nor a `ModalModule`.

### ATraceDeclaresTheGainAndTheBiasAndEmitsOneNode

Over a [2, 3, 4] input a training trace is ok and adds three nodes. The node that writes the
result is LAYER_NORM, reads the input, the weight and the bias in that order and holds
`K_LAYER_NORM_EPSILON` in attribute slot 0. The result is a float32 value of shape [2, 3, 4].

### TheGainIsAScaleAndTheBiasABias

After a trace over a [5, 4] float64 input, the weight and the bias are declared with the shape
[4] and the type float64, with the roles `ParameterRole::Scale` and `ParameterRole::Bias`. A
layer whose `epsilon` is 1e-3 puts 1e-3 in the node's attribute slot 0.

### TheWalkYieldsTheGainThenTheBias

`NN::walk_state` under the prefix `norm` visits `norm.weight` and then `norm.bias`.

### AnInputThatDoesNotFitFails

A [2, 5] input fails with `InvalidDescription`, "LayerNorm(4)", "[..., 4] input" and "[2, 5]".
An int32 input fails with "half, float32 or float64 input" and "int32". `forward` returns its
input, the graph gains no node, the weight stays undeclared and the graph has a refusal.

### ALayerDescribedWrongFails

A default layer, whose `dim` is 0, fails with "positive dim" and "got 0". An `epsilon` of 0
fails with "LayerNorm(4)" and "positive epsilon", and that layer's weight stays undeclared.

### OutputShapeIsTheInputShape

`output_shape([4])` is [4] and `output_shape([6, 4])` is [6, 4].

### TheFactoryReadsTheWidthAndTheEpsilon

`make_layer_norm_node` over one [6, 4] input and `{}` returns a `ModuleNode<LayerNorm>` whose
layer has the `dim` 4 and the epsilon `K_LAYER_NORM_EPSILON`, and whose one port has the shape
[6, 4]. With `{"ln_epsilon": 0.001}` the layer's epsilon is 0.001.

### TheFactoryRefusesARowThatDoesNotFit

Five rows fail with `InvalidDescription` and a message naming "layer 3": a `momentum` key, an
`ln_epsilon` of -1 ("positive"), an `ln_epsilon` of "small" ("a number"), two inputs ("reads
one input", "given 2") and a rank-6 input ("rank 1 to 5").

### ATraceOverAValueTheGraphLacksIsFatal

`forward` over value id 7 in a graph of one value stops the process with a report containing
"LayerNorm(4)", "value 7" and "holds 1 values".

## test_loss_roots.cpp

Suites: `Unit_LossRoots`.

Tests which values differentiation accepts as a loss, the seed it emits and the K_STOP_GRADIENT
barrier.

This file needs no device. Every case in `Unit_LossRoots` builds a host-side graph, calls
`Autograd::differentiate` and reads the appended nodes or the error. A root produced by
K_NLL_LOSS is seeded by that operation's own rule. Any other one-element root of a real floating
type whose producer has a rule gets a K_CONSTANT of 1 as its gradient. Most refusal cases also
check that the node count is unchanged, so a refused call leaves the graph as it was.

### Constants

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule copies from the
forward node. `K_TERM_WEIGHT` is 2.0, the factor of the K_SCALE term in `WeightedSum`.
`K_ROOT_WEIGHT` is 0.5, the factor of the K_SCALE that ends a loss in the cases that need one.

### op_names

`op_names` returns the operation names of the nodes from a given index to the end. A case
passes the forward node count, so the list covers the appended nodes.

### count_op

`count_op` counts the nodes of one operation in the whole graph.

### differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. The seed factor defaults to `K_NO_VALUE` and the provenance to null.

### WeightedSum

`WeightedSum` holds the loss `K_TERM_WEIGHT * a + b * c` over three one-element parameters:
a K_SCALE, a K_MUL and the K_ADD of the two. It keeps the ids of the parameters and of the three
results.

### ASumOfTwoTermsEndingInAddReachesEveryParameter

The root is the K_ADD. Four nodes are appended: K_CONSTANT, K_MUL, K_MUL, K_SCALE. The K_CONSTANT is the
seed. The two K_MUL nodes read the seed with `c` and with `b`, and produce the gradients of `b`
and `c`. The K_SCALE reads the seed with the factor `K_TERM_WEIGHT` and produces the gradient of
`a`. All three gradients have the lifetime GRADIENT.

### ASumOfTwoTermsEndingInScaleReachesEveryParameter

The root is a K_SCALE of the sum by `K_ROOT_WEIGHT`. The appended nodes are K_CONSTANT, K_SCALE, K_MUL,
K_MUL, K_SCALE. The first K_SCALE reads the seed with the factor `K_ROOT_WEIGHT`, and the three nodes
after it read that K_SCALE's result, so both terms under the root receive the scaled seed. No
gradient is `K_NO_VALUE`.

### TheSeedIsAConstantOneShapedAndTypedLikeTheRoot

The root is a FLOAT64 [1, 1, 1, 1] value tagged NCHW. The first appended node is a K_CONSTANT
with no input and the value 1 in attribute slot 0, and its result is [1, 1, 1, 1], FLOAT64 and
NCHW. The graph holds one K_CONSTANT, no K_ZERO and no K_EXP.

### TheSeedFactorMultipliesTheSeedOnce

With a seed factor, the appended nodes are K_CONSTANT, K_DYNAMIC_SCALE, K_MUL, K_MUL, K_SCALE. The
K_DYNAMIC_SCALE reads the K_CONSTANT and the factor, the graph holds one K_DYNAMIC_SCALE, and the
three rule nodes read its result.

### ANonScalarRootThrowsNamingItsShape

A [4, 3] root throws with a message containing "needs a scalar to differentiate" and "[4, 3]".

### ALeafRootThrowsNamingTheLeaf

A parameter passed as the root throws with a message containing "it is a leaf" and
"sushiai.leaf".

### ARootThatNeedsNoGradientThrowsNamingTheCause

The root is an K_ADD of two inputs and the requested parameter is not read by it. The message
contains "requires no gradient" and "sushiai.add".

### ARootWhoseProducerHasNoRuleThrowsNamingTheOperation

The root is a K_DYNAMIC_SCALE of a parameter. The message contains "no backward rule" and
"sushiai.dynamic_scale".

### ARootThatIsNotTheFirstResultThrowsNamingTheProducer

The root is the index result of a K_MAXPOOL2D, its second output. The message contains "not the
first result" and "sushiai.maxpool2d".

### ARootThatIsNotRealFloatingThrowsNamingItsType

The root is a K_SCALE of an INT32 parameter. The message contains "element type is int32" and
"sushiai.scale".

### AFusedCrossEntropyRootEmitsNoSeedNode

A two-layer `NN::Sequential` under `NN::CrossEntropyLoss` appends K_EXP, K_SUB, K_SCALE, K_SUM_ROWS,
K_MATMUL, K_MATMUL, K_RELU_BACKWARD, K_SUM_ROWS, K_MATMUL. The graph holds no K_ZERO and no K_CONSTANT: the
fused rule starts from the root's implicit gradient and needs no seed node.

### AnUnfusedNLLLossRootEmitsNoSeedNode

K_NLL_LOSS over a K_RELU result appends K_SCALE then K_RELU_BACKWARD. The K_SCALE reads the targets
alone, so no seed node precedes it.

### AnNLLLossRootAppliesTheSeedFactorInItsOwnRule

With a seed factor, a K_LOG_SOFTMAX and K_NLL_LOSS pair appends K_EXP, K_SUB, K_SCALE, K_DYNAMIC_SCALE, and
the result of the K_DYNAMIC_SCALE is the gradient of `x`.

### StopGradientEmitsAnIdentityShapedLikeItsOperand

`stop_gradient` of a FLOAT64 NCHW value writes one K_STOP_GRADIENT node that reads the operand.
The result has the operand's shape and is FLOAT64 and NCHW, and `finalise()` accepts the graph.

### StopGradientIsDeclaredAGradientBarrierAndNotDifferentiable

`Ops::K_STOP_GRADIENT` has `blocks_gradient` true and `differentiable` false.
`Ops::is_gradient_barrier` is true for it and false for K_RESHAPE, and its name is
"sushiai.stop_gradient".

### RequiresGradDoesNotCrossStopGradient

After `finalise()`, a parameter requires a gradient and so does a K_SCALE that reads it. The
K_STOP_GRADIENT result and a K_SCALE that reads that result do not.

### StopGradientBlocksAParameterReachableOnlyThroughIt

The loss is `stop_gradient(scale(blocked)) * open`. Differentiation does not throw and appends
K_CONSTANT and K_MUL. The gradient of `blocked` is `K_NO_VALUE`, and the gradient of `open` is the
K_MUL of the seed and the stopped value.

### AValueReadDirectlyAndThroughStopGradientKeepsTheDirectGradient

The loss is `stop_gradient(x) * x`. The appended nodes are K_CONSTANT and K_MUL, the K_MUL reads the
seed and the stopped value, and its result is the gradient of `x`. The graph holds no
K_ACCUMULATE, so the stopped path contributes nothing to the sum.

### ALossReachableOnlyThroughStopGradientThrowsNamingTheBarrier

The root is a K_SCALE of `stop_gradient(x)`. The message contains "requires no gradient" and
"sushiai.scale".

### AStopGradientRootThrowsForRequiringNoGradient

The root is the K_STOP_GRADIENT result. The message contains "requires no gradient" and
"sushiai.stop_gradient".

### AContributionToAStopGradientResultIsDroppedWithoutAskingForARule

The graph is that of StopGradientBlocksAParameterReachableOnlyThroughIt, with `requires_grad`
of the stopped value set by hand after the graph is built. `finalise()` only sets the flag, so
the hand-set value survives it and K_MUL's rule contributes a gradient to the stopped value.
Differentiation does not throw. It appends K_CONSTANT, K_MUL, K_MUL, the gradient of `blocked` is
`K_NO_VALUE`, and no appended node has the K_STOP_GRADIENT node as its origin in the `Provenance`.
The reverse walk steps over the barrier and asks no rule of it.

### TheSeedNodesHaveTheRootProducerAsOrigin

`WeightedSum` is differentiated with a seed factor and a `Provenance`. Five nodes are appended.
The origin of the K_CONSTANT and of the K_DYNAMIC_SCALE is the K_ADD that produces the root. The
origin of the two K_MUL nodes is the forward K_MUL, and the origin of the K_SCALE is the forward
K_SCALE.

### AConstantRootThrowsForRequiringNoGradient

The root is a `constant_like` of a parameter. The message contains "requires no gradient" and
"sushiai.constant".

### AComparisonRootThrowsForRequiringNoGradient

The root is `less_equal(x, bound)` over a parameter. The message contains "requires no
gradient" and "sushiai.less_equal".

### RequiresGradCrossesNoGradientBarrier

After `finalise()`, a K_LESS_EQUAL mask over a parameter, a K_CONSTANT, a K_ZERO and the K_MUL of the
mask with the K_CONSTANT require no gradient. The K_MUL of the mask with the parameter does,
through its second operand.

## modules/vision/training/tests/test_pixel_objectives.cpp

Suites: `Unit_VisionTrainingPixelObjectives`.

Tests `PixelCrossEntropyObjective` in
`modules/vision/training/include/SushiAI/vision/training/pixel_cross_entropy_objective.hpp`
and `DiceObjective` in
`modules/vision/training/include/SushiAI/vision/training/dice_objective.hpp`, with their
factories. The extents are a batch of 2, 4 by 4 pixels and 3 classes, so 32 pixels. The model
of the graph cases is a 1 by 1 `Conv2d` from 5 channels to 3.

### MakeKeepsItsExtentsAndRefusesNone

`PixelCrossEntropyObjective::make` with the ignore index 7 gives an objective of 32 pixels, 3
classes, the ignore index 7 and the name "pixel_cross_entropy". A height of 0 fails with
`InvalidDescription`, "pixel_cross_entropy" and "at least 1". `DiceObjective::make` with 0
classes fails with `InvalidDescription`, "dice" and "at least 1".

### MakeRefusesAnIgnoreIndexThatIsAClass

Over 3 classes the cross entropy fails with `InvalidDescription` for the ignore index 2
("ignore index 2", "3 classes") and is made for 3 and for -1. The Dice objective fails for the
ignore index 0 ("ignore index 0"). The cross entropy with `DType::Int32` fails with
`InvalidDescription` and "float32".

### MakeRefusesWhatInt32CannotIndex

Both objectives are made for the extents 54, 512, 512, 150. For a batch of 55 both fail with
`InvalidDescription`, "55 * 512 * 512 * 150" and "2147483647".

### TheCrossEntropyDeclaresIndicesAndWeights

`declare` adds two leaves: `targets` of shape [32] and `Int32`, and `weights` of shape [32, 1]
and `Float32`. It requires one role, `K_ROLE_MASK`, with the sample shape [4, 4], `Int32` and
0 classes.

### TheDiceDeclaresTwoIndexLeavesAndNoOneHot

`declare` adds five leaves and none named `one_hot`. `score_indices` and `class_indices` are
[32] and `Int32`, `kept` is [32, 1], and `class_pixels` and `class_weights` are [3]. No leaf
has more than 32 elements. The one required role is `K_ROLE_MASK`.

### TheCrossEntropyGathersOneLogProbabilityPerPixel

The training graph of the model under the cross entropy has a loss of one element and `Int32`
targets. It holds no NLL_LOSS node, one LOG_SOFTMAX and one GATHER_ROWS. Both of its
parameters have a gradient.

### TheDiceLossIsDifferentiated

The training graph under the Dice objective has a loss of one element and no targets value.
It holds two GATHER_ROWS nodes and two SCATTER_ADD_ROWS nodes, and both of its parameters
have a gradient.

### TheFactoriesReadTheModelsOutput

For a context that publishes one output of [2, 4, 4, 3], `make_pixel_cross_entropy_objective`
with an empty section gives a `PixelCrossEntropyObjective` named "pixel_cross_entropy" with 32
pixels and the ignore index 255. `make_dice_objective` with `ignore_index` 9 and a context of
0 classes gives a `DiceObjective` with the ignore index 9.

### TheFactoriesRefuseABadSection

The factories fail with `InvalidDescription` for the key `weight` ("the objective block",
"'weight'", "'ignore_index'"), for an `ignore_index` of 0.5 ("'ignore_index'",
"whole number") and for an `ignore_index` of 1 over 3 classes ("ignore index 1").

### TheFactoriesRefuseAModelThatDoesNotFit

The factories fail with `InvalidDescription` for a model with two outputs
("one [N, H, W, classes] output", "publishes 2"), an output of [2, 3] ("'output'", "[2, 3]"),
an output of [2, 2, 2, 3] under a [4, 4, 5] input ("[2, 2, 2, 3]", "[4, 4, 5]",
"input size") and data of 4 classes ("holds 4 classes", "scores 3").

## modules/layers/io/tests/test_run_progress.cpp

Suites: `Unit_RunProgress`.

Tests `IO::RunProgress` and `IO::check_run_seed` in
`modules/layers/io/include/SushiAI/io/run_progress.hpp`.

### ADefaultProgressIsTheStartOfARun

A default `RunProgress` has the seed 0, 0 step runs, the epoch 0 and the batch 0.

### TheRecordsHaveDistinctNames

`K_RUN_SEED_STATE` is "run.seed", `K_RUN_STEP_RUNS_STATE` is "run.step_runs",
`K_RUN_EPOCH_STATE` is "run.epoch" and `K_RUN_BATCH_STATE` is "run.batch".

### AResumeUnderAnotherSeedIsRefusedWithBothSeeds

For a progress saved under the seed 20261011, `check_run_seed` passes that seed. The seed 7
fails with `Incompatible`, "'last.sushi'", "seed 20261011" and "seed 7". A default progress
checked against 20261011 fails with `Incompatible`, "seed 0" and "seed 20261011".

## modules/vision/segment/tests/test_segment_metrics.cpp

Suites: `Unit_SegmentMetrics`.

Tests `class_iou`, `class_occurs` and `mean_iou` in
`modules/vision/segment/include/SushiAI/vision/segment/intersection_over_union.hpp`,
`pixel_accuracy` in `modules/vision/segment/include/SushiAI/vision/segment/pixel_accuracy.hpp`
and `argmax_classes` in
`modules/vision/segment/include/SushiAI/vision/segment/argmax_classes.hpp`. Each ratio is
compared within 1e-15, and the matrices count with the ignore index 255.

### ClassIoUIsIntersectionOverUnion

For the predictions 0, 1, 1, 2, 2, 2 against the truth 0, 0, 1, 2, 1, 2, `class_iou` is 1/2
for class 0, 1/3 for class 1 and 2/3 for class 2, and `mean_iou` is the mean of the three.

### TheMeanSkipsAClassThatDoesNotOccur

For the predictions 0, 0, 1, 2 against the truth 0, 0, 1, 1, `class_occurs` is true for class
0 and false for class 2, `class_iou` of class 2 is 0, and `mean_iou` is (1 + 1/2) / 2.

### PixelAccuracyIsTheDiagonalOverTheCountedPixels

For seven pixels, the last with the truth 255, `pixel_accuracy` is 4/6.

### AnEmptyMatrixGivesZero

A matrix whose one pixel was ignored gives 0 for `mean_iou` and for `pixel_accuracy`.

### ArgmaxPicksTheLargestScoreAndTheLowestClassOnATie

Over three pixels of three scores each, (0.1, 0.9, 0.0), (2.0, -1.0, 2.0) and
(-3.0, -2.0, -1.0), `argmax_classes` writes 1, 0, 2.

## modules/vision/segment/tests/test_segment_rasterise.cpp

Suites: `Unit_SegmentRasterise`.

Tests `fill_outline` in `modules/vision/segment/include/SushiAI/vision/segment/fill_outline.hpp`
and `rasterise_outlines` in
`modules/vision/segment/include/SushiAI/vision/segment/rasterise_outlines.hpp`. Every mask
starts filled with the background byte 9.

### ARectangleOnPixelEdgesFillsThePixelsItBounds

On a mask of 4 rows and 5 columns, the rectangle from (1, 1) to (4, 3) filled with 1 sets the
columns 1 to 3 of the rows 1 and 2 and leaves every other pixel 9.

### APixelIsFilledWhenItsCentreIsInside

On a mask of one row and three columns, a rectangle spanning x from 0.6 to 2.4 gives 9, 1, 9.

### ACentreOnASharedEdgeBelongsToOneSide

On a mask of one row and three columns, the rectangle from x = 0 to 1.5 filled with 1 gives
1, 9, 9. The rectangle from x = 1.5 to 3 filled with 2 then gives 1, 2, 2.

### ATriangleFillsTheCentresUnderItsSlope

On a 4 by 4 mask, the triangle (0, 0), (4, 0), (0, 4) fills the first 3, 2, 1 and 0 columns of
the rows 0 to 3.

### EveryPartOfAnOutlineIsFilled

On a mask of one row and five columns, an outline of the parts x from 0 to 1 and x from 3 to 5
gives 1, 9, 9, 1, 1.

### APartInsideAnotherCutsAHole

On a 3 by 3 mask, an outline of the rectangle from (0, 0) to (3, 3) and the rectangle from
(1, 1) to (2, 2) fills every pixel but the centre one.

### AnOutlineIsClippedToTheMask

On a 2 by 2 mask, the rectangle from (-5, -5) to (1, 50) fills column 0 of both rows.

### AnOutlineWithoutAreaFillsNothing

An outline of no part and an outline whose one part has two vertices both leave a 2 by 2 mask
at 9.

### FillRefusesABadMaskAndAVertexThatIsNotFinite

`fill_outline` fails with `InvalidDescription` for an unsized mask ("fill_outline",
"0 by 0"), for a mask of three channels ("3 channels") and for a second part whose second
vertex has a NaN coordinate ("vertex 1 of part 1"). After that last failure the mask still
holds 9 in every pixel.

### TheLaterOutlineWinsAnOverlap

`rasterise_outlines` of class 1 over x from 0 to 3 and class 2 over x from 2 to 4, at a size
of one row and five columns, gives a mask of that size and one channel holding 1, 1, 2, 2, 9.

### RasteriseRefusesABadSizeAndNamesTheOutline

`rasterise_outlines` fails with `InvalidDescription` for a size of 0 by 4
("rasterise_outlines", "0 by 4") and for a second outline with an infinite vertex
("outline 1", "vertex 0 of part 0").

