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
| [`test_region_signature.cpp`](../../modules/core/eager/tests/test_region_signature.cpp) | The signature that `signature_of` computes for one region of a partition. |
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
| [`test_shape.cpp`](../../modules/core/core/tests/test_shape.cpp) | Shape's extents, indexing, validation, equality and hash. |
| [`test_silu.cpp`](../../modules/layers/nn/tests/test_silu.cpp) | The SiLU layer's trace, shape inference and empty state. |
| [`test_spatial_shapes.cpp`](../../modules/core/autograd/tests/test_spatial_shapes.cpp) | The shapes the builder infers for spatial operations and the layouts it refuses. |
| [`test_status.cpp`](../../modules/core/core/tests/test_status.cpp) | Status, its macros, the fatal family and the assertions of test_status.hpp. |
| [`test_step_recorder.cpp`](../../modules/core/graph/tests/test_step_recorder.cpp) | That StepRecorderList drives its members as one recorder, in the order added. |
| [`test_step_schedule.cpp`](../../modules/core/optim/tests/test_step_schedule.cpp) | The rate `Optim::StepSchedule::rate_at` gives under a constant policy, a warm-up and listed steps, and the schedules `validate_step_schedule` refuses. |
| [`test_step_schedule_section.cpp`](../../modules/core/optim/tests/test_step_schedule_section.cpp) | That a `StepSchedule` and the `schedule` section of a training file convert both ways without loss, and which sections and schedules are refused. |
| [`test_steps.cpp`](../../modules/layers/train/tests/test_steps.cpp) | The training, probe, evaluation, carrying and incremental graph builders, the role of each parameter binding, and the batch shape function. |
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

## modules/vision/training/tests/test_anchor_objective.cpp

Suites: `Unit_AnchorObjective`.

Tests `AnchorObjective` of the vision training module over a 64-pixel network, a batch of 2
and two classes. Most cases use two heads, "head.0" on a 4x4 grid and "head.1" on a 2x2 grid,
each with two anchors and so 14 channels.

### DeclaresOneLeafPerHeadInTheShapeOfItsOutput

`declare` adds two leaves, "delta.head.0" of the shape [2, 4, 4, 14] and "delta.head.1" of
[2, 2, 2, 14]. Both are `Float32` and do not require a gradient, and `leaf_role(1)` is
"delta.head.1".

### TheRootSumsOneDeltaLossPerHeadScaledByTheBatch

`loss` adds MUL, RESHAPE, SUM_ROWS and SCALE once per head and then one ADD. Each MUL reads the
leaf of its head's name and then that head's output, although the model published "head.1"
first. Each SCALE carries -1/2 to 1e-12. The ADD reads the two SCALE results and writes the
root, whose shape is [1].

### AModelWithoutAHeadThrowsNamingTheHead

`loss` over outputs that hold "head.0" alone throws an error naming "head.1".

### AHeadOfAnotherShapeThrowsNamingBothShapes

`loss` over a "head.0" of the shape [2, 4, 4, 21] throws an error naming "[2, 4, 4, 14]" and
"[2, 4, 4, 21]".

### TheCostIsTheSumOverTheHeads

For two heads of one anchor and one class, the second with an object weight of 0.5, and deltas
set by hand, `measure_cost` returns a total of 7 + 2.5, a non-box share of 6.5 + 0.25, a box
share of 0.5 + 2.25 and a stored cost of 14 + 5, each to 1e-12.

### NamesItselfAndIsFedByItsOwnStep

`name()` is "anchor_delta" and `is_fed_by_batch()` is false. The objective holds two heads, four
anchors and a network width of 64.

### AHeadCountOtherThanTheLayoutsIsRefused

`make` with no head, and with one head, over a two-head layout fails with `InvalidDescription`.
The messages hold "0 heads" and "1 heads", each with "holds 2".

### HeadsOfDifferentBatchesAreRefused

A second head with a batch of 3 fails with `InvalidDescription`, and the message holds
"head.1", "batch of 3" and "head.0".

### AnEmptyOrARepeatedOutputNameIsRefused

A head with an empty output name fails with "head 1" and "names no model output". Two heads
that both name "head.0" fail with "both read" and "head.0". Both codes are
`InvalidDescription`.

### AnExtentOrANetworkSideBelowOneIsRefused

A head with a height of 0 fails with "head.0" and "[2, 0, 4]". A network size of 0 by 64 fails
with "0 by 64". Both codes are `InvalidDescription`.

### OptionsTheValidatorsRefuseAreRefused

A head with no class fails with `InvalidDescription` and "at least one class". A head with a
box weight of -1 fails with `InvalidDescription`.

## modules/vision/training/tests/test_anchor_objective_factory.cpp

Suites: `Unit_AnchorObjectiveFactory`.

Tests `build_anchor_objective` and `make_anchor_objective` in
`modules/vision/training/include/SushiAI/vision/training/anchor_objective_factory.hpp`. The
context has a batch of 2, 2 classes and the input sample [8, 6, 3]. Its task states four
anchors and two heads of two anchors each: "head.0" with the coding "logit" and "head.1" with
"activated" and a `scale_xy` of 1.5. It publishes the outputs "head.0" of [2, 4, 3, 14],
"stem" of [2, 4, 3, 8] and "head.1" of [2, 2, 1, 14].

### EachHeadTakesItsExtentsFromItsPublishedOutput

With an empty section the objective has a network of height 8 and width 6, four anchors and
two heads. The first reads "head.0" with a batch of 2, a height of 4, a width of 3, 2 classes
and `HeadCoding::Logit`. The second reads "head.1" with a height of 2, a width of 1,
`HeadCoding::Activated` and a `scale_xy` of 1.5.

### TheSectionSetsTheTrainingOptionsOfEveryHead

With the section `{"type": "anchor", "box_weight": 0.5}`, every head has a box weight of 0.5
and the default object weight and box loss of `HeadTrainingOptions`.

### TheClassCountOfTheDataMustBeTheTasks

A context with 0 classes builds. One with 80 classes fails with `Incompatible`, "80 classes"
and "task states 2".

### AModelWithoutATaskIsRefused

A context whose task is a null value fails with `InvalidDescription`, "'task'" and "'anchor'".

### AHeadTheModelDoesNotPublishIsRefused

Without the output "head.1" the build fails with `InvalidDescription`, "head 1", "'head.1'",
"'head.0'" and "'stem'".

### AnOutputOfOtherExtentsIsRefused

The build fails with `Incompatible` for "head.0" with 15 channels ("'head.0'",
"[2, 4, 3, 15]", "[2, H, W, 14]"), for "head.1" with a batch of 3 ("'head.1'",
"[2, H, W, 14]") and for "head.0" of rank 2 ("'head.0'", "[2, H, W, 14]").

### AnInputOfAnotherRankIsRefused

An input sample of [192] fails with `InvalidDescription`, "[192]" and "[H, W, C]".

### AKeyTheSectionDoesNotHaveIsRefused

A section with the key `label_smoothing` fails with `InvalidDescription`, "objective" and
"'label_smoothing'".

### TheRegistryFormPrefixesARefusalWithItsContext

`make_anchor_objective` with the context "the objective block" returns an objective whose
`name` is "anchor_delta". With a null task it fails with `InvalidDescription`,
"the objective block" and "'task'".

## modules/vision/training/tests/test_anchor_training_loop.cpp

Suites: `Unit_AnchorTrainingLoop`.

Tests `run_anchor_training` in
`modules/vision/training/include/SushiAI/vision/training/anchor_training_loop.hpp` and
`AnchorTrainingStep::restore_run_count`. The fixture is an `EngineTest`. It builds a model of
one convolution published as "head" over an [8, 8, 3] input, an `AnchorObjective` of one head
with two anchors and 2 classes on a 4 by 4 grid, an SGD made with the rate 0.2 and the
`AnchorTrainingStep` over them. Its dataset, named "truths", holds three batches of two seeded
images with the three truth roles of `describe_truth_roles(2, 2)`.

### EachUpdateTakesTheRateOfItsIteration

With a `StepSchedule` of base rate 0.1 that halves at the updates 2 and 5, a run of two epochs
calls `on_update` six times. Call i reports the epoch i / 3 + 1, the batch i modulo 3, the
iteration i + 1 and the rates 0.1, 0.05, 0.05, 0.05, 0.025, 0.025. The report states 6
updates and a last rate of 0.025, which is also the optimizer's rate, and its first and last
costs equal those of the first and last call. The optimizer's step count is 6 and the
training step's compile count is 1.

### ACosineScheduleSetsTheRateOfEachUpdate

With a `CosineSchedule` of base rate 0.1, 6 total iterations and a final rate of 0.01, each of
the six updates reports the rate `rate_at` gives its iteration, and the report's last rate is
`rate_at(6)`.

### ARunWithoutAScheduleKeepsTheOptimizersRate

A run without options reports 3 updates and a last rate of 0.2, and the optimizer's rate
stays 0.2.

### ADatasetTheStepCannotReadIsRefusedBeforeAnyUpdate

The run fails with `Incompatible` for batches of three images ("'truths'", "batch of 3",
"declared for 2"), and with `InvalidDescription` for a dataset of no batch ("'truths'",
"no batches") and for one whose only role is `labels` ("'truths'", "'boxes'"). The
optimizer's step count stays 0.

### OptionsARunCannotUseAreRefused

The run fails with `InvalidDescription` for 0 epochs ("at least one epoch", "0") and for a
schedule of two steps and one scale ("2 steps", "1 scales"). The optimizer's step count stays
0.

### ARunStartsAtTheEpochAndTheBatchItsOptionsState

With two epochs, `first_epoch` 3 and `first_batch` 2, over a dataset that records each
`begin_epoch`, the run makes 4 updates and begins the epochs 3 and 4. The first call reports
the epoch 1 and the batch 2, and the second the epoch 2 and the batch 0.

### AStartTheDatasetLacksIsRefused

`first_epoch` -1 fails with `InvalidDescription`, "epoch index" and "-1". `first_batch` 3
fails with `Incompatible`, "batch 3", "'truths'" and "3 batches". The optimizer's step count
stays 0.

### ARestoredRunCountIsTheCountOfBothPasses

The step's run count starts at 0. After `restore_run_count(4)` it is 4, and so is the run
count of the evaluation step. After a run of three batches both are 7. A second
`restore_run_count` then fails with `InvalidDescription`, "before its first run" and "run 7".

### ACallbackThatThrowsEndsTheRunAsAFailure

With an `on_update` that throws `std::runtime_error("stop here")`, the run returns a failure
whose message holds "stop here", and the optimizer's step count is 1.

## modules/core/optim/tests/test_apply_step_schedule.cpp

Suites: `Unit_ApplyStepSchedule`.

Tests `Optim::apply_step_schedule` and `Optimizer::set_learning_rate`. The fixture needs an
engine and gives each optimizer one float32 parameter of three zeros whose gradient is 2 in
every element. The schedule most cases use has the base rate 0.5 and halves at updates 2 and 3.

### TheRateSetIsTheScheduleAtTheNextUpdate

Over an SGD made with the rate 0.75, four calls of `apply_step_schedule`, each followed by a
step, return 0.5, 0.25, 0.125 and 0.125. `learning_rate()` equals the returned rate after each
call, and `step_count()` ends at 4.

### AnUpdateUsesTheRateThatWasSet

Over three updates, every element of the parameter equals the running sum of minus the applied
rate times the gradient, within 1e-6.

### AdamWTakesTheRateThroughTheSameSeam

An AdamW made with the rate 0.75 reports 0.75 through the `Optimizer` interface. After
`apply_step_schedule` returns 0.5, `options().learning_rate` is 0.5.

### ANegativeRateIsFatal

`set_learning_rate(-0.1)` stops the process with a report containing "learning rate" and
"negative".

### ACosineScheduleTakesTheSameSeam

A `CosineSchedule` with the base rate 0.5, two warm-up updates and four updates in total gives
0.25 at the first update. Held in a `RateSchedule` it gives 0.5 at the second update, and
`learning_rate()` is 0.5. At the third update a `RateSchedule` holding the halving schedule
gives 0.125 and the held cosine gives 0.25.

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

## modules/core/graph/tests/test_batched_matmul.cpp

Suites: `Unit_BatchedMatmul`.

Tests the IR that `GraphBuilder::matmul` emits for rank-2 and rank-3 operands and that
`GraphBuilder::softmax` emits over the last axis. Every refusal case also expects the graph's
node count to be what it was before the refused calls.

### Rank2ProductKeepsItsShapeAndFlags

`matmul` over [3, 4] and [5, 4] with the second operand transposed yields [3, 5]. The node's
operation is `Ops::K_MATMUL`, its flag 0 is false and its flag 1 is true.

### FourFlagCombinationsYieldBatchedProducts

For each of the four transpose pairs, with operands shaped to fit it, a rank-3 product yields
[2, 3, 5]. The node is a MATMUL that reads the two operands in order, writes one value and
carries the two flags it was given.

### BatchedProductTakesTheSharedBatchCount

`matmul` over [7, 2, 3] and [7, 3, 6] yields [7, 2, 6] with the first operand's layout and the
element type `Float32`.

### RefusesMixedRanksAndRanksOutsideTwoAndThree

`matmul` throws `Error` for a rank-2 operand with a rank-3 one in either order, for a rank-1
operand with a rank-3 one, and for two rank-4 operands.

### RefusesUnequalBatchesAndInnerExtents

Against a [2, 3, 4] operand, `matmul` throws `Error` for a [3, 4, 5] operand, whose batch
differs, and for a [2, 5, 3] operand, whose inner extent differs.

### SoftmaxEmitsReshapeLogSoftmaxExpReshape

`softmax` over a [2, 3, 3] input adds four nodes. Read back from the result they are RESHAPE,
EXP, LOG_SOFTMAX and RESHAPE. The first RESHAPE reads the input and yields [6, 3], LOG_SOFTMAX
yields [6, 3], and the result has the shape [2, 3, 3].

### SoftmaxOverRank2KeepsItsRowsAndLastAxis

`softmax` over a [4, 5] input yields [4, 5], and the EXP node behind the last RESHAPE reads a
[4, 5] value.

### SoftmaxRefusesRankBelowTwo

`softmax` over a [5] input throws `Error` and adds no node.

## modules/layers/train/tests/test_binary_confusion.cpp

Suites: `Unit_BinaryConfusion`, `Unit_BinaryConfusionRatios`, `Unit_DecisionThreshold`.

Tests `Train::count_confusion`, the four ratios over a `BinaryConfusion`, `Train::logit_of`,
`Train::read_threshold` and `Train::require_host_reals`. The `Unit_BinaryConfusion` fixture
needs an engine to allocate the logits and the truth. The other two suites run on the host.

### EachRowLandsInOneOfTheFourCounts

Five rows of one column with the logits 2, 1, -1, -3, 0 and the truth 1, 0, 1, 0, 1, cut at
`logit_of(0.5)`, give 2 true positives, 1 false positive, 1 false negative and 1 true negative.
The accuracy is 3/5, and the precision, the recall and the F1 are each 2/3.

### AColumnIsCountedAlone

Over a [2, 2] float64 matrix of logits and a [2, 2] truth, column 0 counts 2 true positives.
Column 1 counts 0 true positives, 1 false negative and 1 false positive.

### TheThresholdMovesTheCut

One row with the logit 1 and the truth 1 is a true positive at the threshold 0.5 and a false
negative at the threshold 0.9.

### ARatioOverNothingIsZero

For a `BinaryConfusion` of four zeros the accuracy, the precision, the recall and the F1 are
each 0. With 4 true negatives and nothing else the accuracy is 1 and the F1 is 0.

### LogitOfInvertsTheSigmoid

`logit_of(0.5)` is exactly 0. For 0.1, 0.731 and 0.9 the sigmoid of `logit_of(p)` is p within
1e-12.

### TheReaderDefaultsToOneHalf

`read_threshold` over `{}` returns `K_DEFAULT_THRESHOLD`, and over `{"threshold": 0.25}` it
returns 0.25.

### TheReaderRefusesWhatIsNoProbability

The key `cut` fails with `InvalidDescription` and a message naming "objective", `'cut'` and
`'threshold'`. The value `"high"` fails with a message containing "number". The thresholds 0,
1, -0.5 and 1.5 each fail with "above 0 and below 1".

### TheHostReadRefusesWhatItCannotRead

`require_host_reals` is ok for a float64 tensor of four elements asked for 4 values, and
`host_real(reals, 2)` is 2.5. An int32 tensor fails with `InvalidDescription` and a message
containing "host read of the logits" and "int32". Asking the float64 tensor for 6 values fails
with `Incompatible`, "6 values" and "holds 4".

## modules/vision/blocks/tests/test_block_attention.cpp

Suites: `Unit_BlockAttention`.

Tests the `Attention` block at the width of row 10 of YOLO26n: 128 channels, two heads, a
query and key width of 32 and a value width of 64 per head. The traced cases run the block in
training mode over a [2, 4, 4, 128] input, which is 16 positions per sample. The norm options
are the epsilon 1e-3 and the momentum 0.03 that `tests/fixtures/yolo26/MANIFEST.json` records.

### TwoHeadsOver128ChannelsGiveAHeadOf64AndAKeyOf32

`Attention(128, 2)` has `heads` 2, `head_dim` 64 and `key_dim` 32. Its `qkv` convolution has
256 output channels, and its `pe` convolution has a kernel height of 3 and a top padding of 1.

### OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

### TwoHeadsEmitSixSlicesTwoScalesFourProductsTwoSoftmaxesAndOneAdd

The training trace holds 6 SLICE_CHANNELS, 2 SCALE, 4 MATMUL over rank-3 operands,
2 LOG_SOFTMAX, 2 EXP, 1 ADD, 1 DEPTHWISE_CONV2D, 2 CONCAT_CHANNELS and 3 BATCHNORM2D nodes, and
no SILU node. The result has the shape [2, 4, 4, 128].

### TheScoreProductReadsScaledQueriesThenTransposedKeys

For each head, the first of its two products has transpose flags 0 and 1 and yields
[2, 16, 16]. Its first operand is a SCALE of the head's queries and its second is the head's
keys. Each is a RESHAPE to [2, 16, 32] of a slice of `qkv`'s normalised result, the queries
from channel `head * 128` and the keys 32 channels further on.

### TheValueProductReadsTheSoftmaxOfTheScoresThenTheValues

For each head, the second product has both transpose flags 0 and yields [2, 16, 64]. Its first
operand is the RESHAPE, EXP, LOG_SOFTMAX, RESHAPE chain over the head's scores, with
LOG_SOFTMAX reading a [32, 16] value. Its second operand is a RESHAPE to [2, 16, 64] of a
64-channel slice of `qkv`'s result that starts at channel `head * 128 + 64`.

### TheQueriesAreScaledByTheInverseRootOfTheKeyWidth

Both SCALE nodes carry 1 / sqrt(32) in attribute 0, which is 0.17677669529663687, to 1e-15
absolute.

### TheAddReadsTheAttendedHeadsThenThePositionalConvolution

The one ADD reads two values. The first is a CONCAT_CHANNELS of two [2, 4, 4, 64] maps, each a
RESHAPE of a head's value product. The second is the BATCHNORM2D of the DEPTHWISE_CONV2D over a
CONCAT_CHANNELS of both heads' value slices, a [2, 4, 4, 128] value. One RESHAPE reads the ADD,
and the block's result comes out of a BATCHNORM2D.

### StateMatchesManifestRecords175To189

Walked under the prefix "model.10.m.0.attn", the traced block yields fifteen records in order:
a convolution weight and four norm records each for `qkv` ([256, 1, 1, 128], width 256), `proj`
([128, 1, 1, 128], width 128) and `pe` ([3, 3, 128], width 128). In each group the norm's
weight and bias are parameters and its running mean and variance are buffers.

### AnEvaluationTraceFoldsTheThreeNorms

An evaluation trace holds no BATCHNORM2D node, three BATCHNORM_FOLD nodes and four rank-3
products, and the result has the shape [2, 4, 4, 128].

### AWidthTheHeadCountDoesNotDivideIsRefused

`Attention(100, 3)` throws an error naming "head count divides", "100 channels" and "3 heads".
`Attention(128, 0)` throws one naming "128 channels" and "0 heads".

### OneChannelPerHeadIsRefused

`Attention(2, 2)` throws an error naming "at least 2 channels per head", "2 channels" and
"2 heads".

## modules/vision/blocks/tests/test_block_bottleneck.cpp

Suites: `Unit_BlockBottleneck`.

Tests Bottleneck's trace, its state names against YOLO26n row 2, and its factory.

This file needs no device. `NN::Bottleneck` in `modules/vision/blocks/include/SushiAI/vision/blocks/bottleneck.hpp` is
two `Conv` blocks, `cv1` and `cv2`, and an optional K_ADD of the block's input to the second
one's result. `NN::make_bottleneck_node` builds the node of a `bottleneck` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 4 'bottleneck'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`Bottleneck(16, 24, 8, true)` maps [5, 7, 16] to [5, 7, 24].

### AShortcutOverEqualWidthsAddsTheInput

A training trace of `Bottleneck(16, 16, 8, true)` reports `adds_input` true and emits two
K_MATMUL, two K_BATCHNORM2D, two K_SILU and one K_ADD. The K_ADD's first operand is the block's input and
its second is produced by a K_SILU. Its output is the returned value, of shape [2, 4, 4, 16].

### NoShortcutTracesNoAdd

With the shortcut argument false, `adds_input` is false and the trace holds two K_MATMUL and no
K_ADD.

### AShortcutOverUnequalWidthsTracesNoAdd

`Bottleneck(16, 24, 8, true)` reports `adds_input` false and traces no K_ADD, although a shortcut
was asked for. The result has the shape [2, 4, 4, 24].

### StateMatchesManifestRow2Inner

The walk of a traced `Bottleneck(16, 16, 8)` under "model.2.m.0" yields ten records. The five
of `cv1` come first: "cv1.conv.weight" [8, 3, 3, 16], then "cv1.bn.weight", "cv1.bn.bias",
"cv1.bn.running_mean" and "cv1.bn.running_var", each [8]. The five of `cv2` follow in the same
order with the weight [16, 3, 3, 8] and the rest [16].

### EachKernelSetsItsOwnConvolution

`Bottleneck(16, 16, 8, false, K_YOLO_NORM, 1, 3)` traces a `cv1` weight of [8, 1, 1, 16] and a
`cv2` weight of [16, 3, 3, 8].

### TheFactoryBuildsTheBlockAndDefaultsTheShortcutToTrue

`make_bottleneck_node` over [4, 4, 16] with 16 filters and a hidden width of 8 has one port of
shape [4, 4, 16]. Its training trace holds one K_ADD, so a row without `shortcut` adds its input.
The walk under "model.2.m.0" yields ten records; record 0 is the `cv1` weight [8, 3, 3, 16]
and record 5 the `cv2` weight [16, 3, 3, 8].

### TheFactoryReadsTheShortcut

With `"shortcut": false` the trace holds no K_ADD.

### TheFactoryRefusesARowThatDoesNotFit

The key `kernel` fails with a message containing "layer 4 'bottleneck'" and "'kernel'". A row
without `hidden` fails with "'hidden'" and "absent". A `shortcut` given as the string "yes"
fails with "'shortcut'" and "true or false".

### TheFactoryRefusesAnInputThatIsNotAFeatureMap

The input shape [256] fails with "[256]" and "rank-3".

### TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

## modules/vision/blocks/tests/test_block_c2psa.cpp

Suites: `Unit_BlockC2psa`.

Tests the `C2PSA` block and its factory `Vision::Blocks::make_c2psa_node` at row 10 of YOLO26n:
256 channels in and out and a hidden width of 128. The traced cases use a [2, 4, 4, 256] input
and the norm options `tests/fixtures/yolo26/MANIFEST.json` records, epsilon 1e-3 and momentum
0.03.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`output_shape([20, 20, 256])` is [20, 20, 256].

### Row10EmitsEightSlicesThreeAddsAndThreeJoins

A training trace holds 8 SLICE_CHANNELS, 3 ADD, 3 CONCAT_CHANNELS, 3 SILU and 7 BATCHNORM2D
nodes, and the update list holds 14 updates. The result has the shape [2, 4, 4, 256]. Every
BATCHNORM2D node carries the epsilon 1e-3 and every update the momentum 0.03.

### TheJoinKeepsTheFirstHalfAndChainsTheSecondOffTheSecondSlice

The first two slices read the same value, at the offsets 0 and 128. The second ADD of the
graph reads the second slice as its first input. The third CONCAT_CHANNELS reads the first
slice and then the third ADD's result, and yields [2, 4, 4, 256].

### EachRepeatReadsTheResultOfThePreviousOne

With two repeats the block holds two inner blocks and an evaluation trace holds six ADD nodes.
The fifth ADD reads the third ADD's result as its first input.

### StateMatchesManifestRow10

Walked under the prefix "model.10" after a training trace, the block yields the 35 records the
file lists by name, shape, kind and order, from `model.10.cv1.conv.weight` on.

### TheFactoryBuildsRow10

`make_c2psa_node` over a [20, 20, 256] input, with the row's filters, hidden width, repeats,
heads and norm constants, returns a node with one port of the shape [20, 20, 256]. Its training
trace holds 7 BATCHNORM2D nodes with the row's epsilon and 14 updates with the row's momentum,
and its state walk yields the same 35 records.

### TheFactoryDefaultsToOneBlockAndHiddenOver64Heads

A row with only "filters" 256 and "hidden" 128 traces to two LOG_SOFTMAX nodes and three ADD
nodes, and its state walk yields 35 records.

### TheFactoryGivesOneHeadForAHiddenWidthUnder128

A row with "filters" 64 and "hidden" 64 over a 64-channel input traces to one LOG_SOFTMAX node.

### TheFactoryRefusesAFilterWidthThatIsNotTheInput

"filters" 128 over a 256-channel input fails with `InvalidDescription`, and the message holds
"'filters'", "128" and "256 channels".

### TheFactoryRefusesAHiddenWidthTheHeadsDoNotSplit

A hidden width of 128 with 3 heads, and with 128 heads, each fail with `InvalidDescription`,
"'hidden'" and "parts of at least 2 channels".

### TheFactoryRefusesARowThatDoesNotFit

An unknown key "inner" fails with a message holding "layer 10 'c2psa'" and "'inner'". A row
without "hidden" fails with "'hidden'" and "absent". A [512] input fails with "[512]" and
"rank-3".

## modules/vision/blocks/tests/test_block_c3k.cpp

Suites: `Unit_BlockC3k`.

Tests C3k's trace, its state names against YOLO26n row 6, and its factory.

This file needs no device. `NN::C3k` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3k.hpp` holds three `Conv`
blocks, `cv1`, `cv2` and `cv3`, and a `ModuleList` of `Bottleneck`s named `m`. The cases show
its trace: a chain of bottlenecks on one branch, `cv2` on the other, one K_CONCAT_CHANNELS of the
two, and `cv3` after it. `NN::make_c3k_node` builds the node of a `c3k` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 5 'c3k'", the row name handed
to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`C3k(64, 96, 2, true)` maps [5, 7, 64] to [5, 7, 96].

### TheChainHoldsOneBottleneckPerRepeat

`C3k(64, 64, 3, true)` has `m.size()` 3.

### ATrainingTraceEmitsSevenConvBlocksTwoAddsAndOneConcat

A training trace of `C3k(64, 64, 2, true)` over [2, 4, 4, 64] emits seven each of K_MATMUL,
K_BATCHNORM2D and K_SILU, two K_ADD, one K_CONCAT_CHANNELS and no K_SLICE_CHANNELS. The list holds 14
updates and the result has the shape [2, 4, 4, 64]. Seven is three `Conv` blocks of the C3k
plus two in each of the two bottlenecks.

### TheJoinReadsTheChainThenTheSecondBranch

The K_CONCAT_CHANNELS node has two inputs. The first is produced by an K_ADD, the end of the last
bottleneck, and the second by a K_SILU, the end of `cv2`. Each is [2, 4, 4, 32] and the joined
value is [2, 4, 4, 64].

### NoShortcutTracesNoAdd

With the shortcut argument false, an evaluation trace holds seven K_MATMUL and no K_ADD.

### TheHiddenWidthIsHalfTheOutputWidthRoundedDown

`C3k(8, 7, 1, true)` has a hidden width of 3. The traced weights are [3, 1, 1, 8] for `cv1`
and `cv2`, [7, 1, 1, 6] for `cv3` and [3, 3, 3, 3] for the bottleneck's `cv1`. The result is
[2, 4, 4, 7], and `output_shape([4, 4, 8])` is [4, 4, 7].

### StateMatchesManifestRow6Inner

The walk of a traced `C3k(64, 64, 2)` under "model.6.m.0" yields 35 records: five each for
`cv1`, `cv2`, `cv3`, `m.0.cv1`, `m.0.cv2`, `m.1.cv1` and `m.1.cv2`, in that order. Each group
is the convolution weight, the norm's weight and bias as parameters, then its running mean and
variance as buffers. The weights are [32, 1, 1, 64] for `cv1` and `cv2`, [64, 1, 1, 64] for
`cv3` and [32, 3, 3, 32] for the four bottleneck convolutions.

### TheFactoryDefaultsToOneBottleneckWithAShortcut

`make_c3k_node` over [4, 4, 64] with only 32 filters has one port of shape [4, 4, 32]. Its
evaluation trace appends one [2, 4, 4, 32] value and holds five K_MATMUL and one K_ADD: one
bottleneck that adds its input.

### TheFactoryReadsRepeatsAndShortcut

With 64 filters, `repeats` 2 and `shortcut` false, the trace holds seven K_MATMUL and no K_ADD. The
walk under "model.6.m.0" yields 35 records, and record 30 is "model.6.m.0.m.1.cv2.conv.weight"
[32, 3, 3, 32].

### TheFactoryRefusesAWidthThatLeavesNoHiddenChannel

`filters` 1 fails with `InvalidDescription` and a message containing "layer 5 'c3k'",
"'filters'" and "at least 2".

### TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `repeats` 0 fails with "'repeats'" and "positive".

### TheFactoryRefusesTwoInputs

Two input shapes fail with a message containing "one input" and "2".

### TheFactoryHandsTheNormOptionsToEveryNorm

A row with `repeats` 2, `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces seven K_BATCHNORM2D
nodes and 14 updates, and `expect_norm_options` finds those two values on all of them.

## modules/vision/blocks/tests/test_block_c3k2.cpp

Suites: `Unit_BlockC3k2`.

Tests C3k2's trace, its state names against YOLO26n rows 2 and 6, and its factories.

This file needs no device. `NN::C3k2<Inner>` in `modules/vision/blocks/include/SushiAI/vision/blocks/c3k2.hpp` holds a
stem `cv1`, a `ModuleList<Inner>` named `m` and a closing `cv2`. The cases show its trace: the
stem's result is cut into two halves by two K_SLICE_CHANNELS, the inner blocks are chained off
the second half, and one K_CONCAT_CHANNELS joins both halves and every inner block's result
before `cv2`. `Inner` is `Bottleneck` or `C3k`. `NN::make_c3k2_node` reads the row's `inner`
key and selects a factory from `NN::K_C3K2_INNER`.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 2 'c3k2'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### row2_records

`row2_records` lists the 20 records `MANIFEST.json` holds under "model.2": five for `cv1`
(weight [32, 1, 1, 32]), five for `cv2` (weight [64, 1, 1, 48]), then five each for `m.0.cv1`
(weight [8, 3, 3, 16]) and `m.0.cv2` (weight [16, 3, 3, 8]).

### row6_records

`row6_records` lists the 45 records `MANIFEST.json` holds under "model.6": five for `cv1`
(weight [128, 1, 1, 128]), five for `cv2` (weight [128, 1, 1, 192]), then the 35 records of
the inner C3k under "model.6.m.0" that `test_block_c3k.cpp` lists.

### make_row2

`make_row2` builds `C3k2<Bottleneck>(32, 64, 16)` over one `Bottleneck(16, 16, 8, true)`.

### make_row6

`make_row6` builds `C3k2<C3k>(128, 128, 64)` over one `C3k(64, 64, 2, true)`.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

Row 2 maps [160, 160, 32] to [160, 160, 64].

### Row2EmitsFourConvBlocksTwoSlicesAndOneConcat

A training trace of row 2 over [2, 4, 4, 32] emits four each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, one K_ADD and one K_CONCAT_CHANNELS. The list holds eight updates, the result
is [2, 4, 4, 64], and every norm carries `K_YOLO_NORM`.

### TheJoinReadsBothHalvesThenTheChainOffTheSecondHalf

The K_CONCAT_CHANNELS of row 2 has three inputs. The first two are produced by K_SLICE_CHANNELS
nodes that read the same value, with the offsets 0 and 16 in attribute 0, and each is
[2, 4, 4, 16]. The third is produced by the bottleneck's K_ADD, whose first operand is the second
half. The joined value is [2, 4, 4, 48].

### TwoInnerBlocksAreChainedAndBothJoined

With two bottlenecks the join has four inputs. The producer of input 2 reads input 1, and the
producer of input 3 reads input 2. The `cv2` weight is [64, 1, 1, 64], four parts of 16
channels, and the second bottleneck's `cv1` weight is [8, 3, 3, 16].

### Row6EmitsNineConvBlocksAndTwoConcats

A training trace of row 6 over [2, 4, 4, 128] emits nine each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, two K_ADD and two K_CONCAT_CHANNELS. The list holds 18 updates, the result is
[2, 4, 4, 128], every norm carries `K_YOLO_NORM`, and `output_shape([40, 40, 128])` is
[40, 40, 128].

### StateMatchesManifestRow2

The walk of traced row 2 under "model.2" equals `row2_records()`.

### StateMatchesManifestRow6

The walk of traced row 6 under "model.6" equals `row6_records()`.

### TheInnerTableListsEachInnerBlockWithItsFactory

`K_C3K2_INNER` has three entries: "bottleneck" with `&make_c3k2_of_bottleneck`, "c3k" with
`&make_c3k2_of_c3k` and "bottleneck_psa" with `&make_c3k2_of_bottleneck_psa`.

### TheFactoryBuildsRow22

`make_c3k2_node` over the input [20, 20, 384], with `filters` 256, `hidden` 128, `repeats` 1,
`inner` "bottleneck_psa" and `shortcut` true, gives one port of shape [20, 20, 256]. Its
training trace holds 9 BATCHNORM2D, 5 SILU, 4 ADD, 8 SLICE_CHANNELS and 3 CONCAT_CHANNELS
nodes and 18 buffer updates, every norm carries the row's epsilon and momentum, and the state
walk under "model.22" equals `row22_records()`.

### TheFactoryRefusesAPSAHeadCountThatDoesNotDivideHidden

With `hidden` 200 and `inner` "bottleneck_psa", `make_c3k2_node` fails with
`InvalidDescription`, "'hidden'" and "must split into 3 parts".

### TheFactoryBuildsRow2

`make_c3k2_node` over [160, 160, 32] with 64 filters, hidden 16, one repeat, inner
"bottleneck", shortcut true and the two norm keys has one port of shape [160, 160, 64]. Its
training trace holds one K_ADD, four K_BATCHNORM2D and eight updates carrying `K_YOLO_NORM`, and
its walk under "model.2" equals `row2_records()`.

### TheFactoryBuildsRow6

`make_c3k2_node` over [40, 40, 128] with 128 filters, hidden 64, one repeat, inner "c3k",
shortcut true and the two norm keys has a port of shape [40, 40, 128]. Its training trace
holds two K_CONCAT_CHANNELS, two K_ADD, nine K_BATCHNORM2D and 18 updates carrying `K_YOLO_NORM`,
and its walk under "model.6" equals `row6_records()`. The inner C3k therefore has two
bottlenecks that add their input.

### TheFactoryReadsTheShortcutOfAC3kInnerBlock

With inner "c3k" and shortcut false the evaluation trace holds nine K_MATMUL and no K_ADD.

### TheFactoryDefaultsToOneBottleneckWithAShortcut

A row stating only 64 filters and hidden 16 traces four K_MATMUL, one K_ADD and one
K_CONCAT_CHANNELS.

### TheFactoryReadsRepeatsAndShortcut

With `repeats` 3 and `shortcut` false the trace holds eight K_MATMUL and no K_ADD. The walk yields
40 records: record 5 is "cv2.conv.weight" [64, 1, 1, 80] and record 30 is "m.2.cv1.conv.weight"
[8, 3, 3, 16].

### TheFactoryRefusesAnInnerBlockItDoesNotList

`"inner": "bottleneck_psa"` fails with `InvalidDescription` and a message containing "layer 2
'c3k2'", "'bottleneck_psa'" and "bottleneck, c3k". `"inner": 2` fails with "'inner'" and
"string".

### TheFactoryRefusesAHiddenWidthOfOne

`hidden` 1 fails with "'hidden'" and "at least 2" for the default inner block and for inner
"c3k".

### TheFactoryRefusesARowThatDoesNotFit

The key `heads` fails with the context and "'heads'". A row without `hidden` fails with
"'hidden'" and "absent". The input shape [512] fails with "[512]" and "rank-3".

### AnInnerFactoryRefusesARowThatNamesAnotherInnerBlock

`make_c3k2_of_bottleneck` given `"inner": "c3k"` fails with a message containing the context,
"'inner' is 'c3k'" and "builds 'bottleneck'". `make_c3k2_of_c3k` given `"inner": "bottleneck"`
fails with "'inner' is 'bottleneck'" and "builds 'c3k'". `make_c3k2_of_bottleneck` given
`"inner": 2` fails with "'inner'" and "string". Each factory accepts a row that names its own
block.

### TheBottleneckPSAFactoryRefusesARowThatNamesAnotherInnerBlock

`make_c3k2_of_bottleneck_psa` fails for a row whose `inner` is "c3k" with
`InvalidDescription`, "'inner' is 'c3k'" and "builds 'bottleneck_psa'". It builds a row whose
`inner` is "bottleneck_psa".

## modules/language/blocks/tests/test_block_causal_self_attention.cpp

Suites: `Unit_BlockCausalSelfAttention`.

Tests `Language::Blocks::CausalSelfAttention` in
`modules/language/blocks/include/SushiAI/language/blocks/causal_self_attention.hpp` and its node
factory `make_causal_self_attention_node`: the ordinary trace, the incremental trace of one
position, the rotary form of both, the state walk and the refusals. Every case uses a width of
8 and two heads. Most ordinary traces read a [2, 5, 8] input, and the incremental ones read a
[1, 1, 8] input over a per-sample shape of [5, 8].

### ACausalTraceMasksEveryHead

A causal block traced over [2, 5, 8] returns a [2, 5, 8] value, records no failure and no
refusal, and adds 45 nodes: 15 around the heads and 15 per head. The graph holds two
CAUSAL_MASK, two LOG_SOFTMAX, six SLICE_CHANNELS, one CONCAT_CHANNELS, six MATMUL and two SCALE
nodes.

### EachMaskReadsTheScaledScoresOfItsHead

The first input of every CAUSAL_MASK node has the shape [2, 5, 5] and comes from a SCALE node
whose attribute is 0.5 and whose input comes from a MATMUL node. The mask's own attribute is
below -1e8.

### ATraceWithoutAMaskEmitsNone

A block built with `AttentionMask::None` adds no CAUSAL_MASK node and 43 nodes in total, 15
around the heads and 14 per head.

### TheWalkYieldsTheTwoProjections

After a trace, `NN::walk_state` under the prefix "attn" yields "attn.qkv.weight",
"attn.qkv.bias", "attn.proj.weight" and "attn.proj.bias" in that order. The `qkv` weight has
the shape [24, 8] and the `proj` weight [8, 8].

### AnInputThatDoesNotFitFails

`forward` returns its input and the trace fails with `InvalidDescription` for a [5, 8] input
("CausalSelfAttention", "rank-3 [N, T, 8] input", "[5, 8]"), a [2, 5, 10] input
("CausalSelfAttention", "[2, 5, 10]") and an `Int32` input ("half, float32 or float64 input",
"int32"). The three traces add no node, the `qkv` weight stays undeclared, and the graph holds
a refusal.

### ABlockDescribedWrongFails

A block of width 8 with three heads fails with `InvalidDescription` and a message containing
"head count divides" and "dim=8 heads=3". A default-constructed block fails with "dim=0
heads=0". Each `forward` returns its input.

### TheFactoryReadsTheHeadsAndTheMask

Over a [5, 8] sample, the row `{"heads": 4}` builds a `ModuleNode<CausalSelfAttention>` whose
layer has `dim` 8, `heads` 4 and the mask `Causal`, with one port of the shape [5, 8]. The row
`{"heads": 2, "causal": false}` builds a layer whose mask is `None`.

### TheFactoryRefusesARowThatDoesNotFit

The factory fails with `InvalidDescription` and a message containing "layer 3" for a `dropout`
key ("'dropout'"), a row without `heads` ("'heads'", "absent"), three heads over a width of 8
("dim=8 heads=3") and a [4, 4, 8] sample ("rank-2 [T, C] input", "[4, 4, 8]").

### TheIncrementalFormWritesOneRowPerCacheAndReadsEveryRow

`forward_incremental` over a carrying trace returns a [1, 1, 8] value with no failure and no
refusal. The graph holds two SCATTER_ADD_ROWS, no CAUSAL_MASK, two LOG_SOFTMAX, nine
SLICE_CHANNELS, one CONCAT_CHANNELS, six MATMUL, two SCALE and four ADD nodes.

### TheIncrementalFormDeclaresTwoFedLeavesAndTwoCarriedValues

The trace declares two fed leaves: `K_POSITION_LEAF`, an `Int32` value of the shape [1, 1],
then `K_ATTENTION_MASK_LEAF` of the shape [1, 5]. It carries two values. The leaf and the next
value of each have the shape [5, 8], the leaf has the lifetime `Input`, and an ADD node
produces the next value.

### TwoBlocksOfOneTraceShareTheirFedLeaves

Two blocks traced incrementally one after the other in one carrying trace leave two fed leaves
and four carried values.

### TheIncrementalFormDeclaresTheParametersOfTheOrdinaryForm

After an incremental trace, `NN::parameter_count` of the block is 4 and `NN::buffer_count` is 0.

### TheIncrementalFormRefusesWhatItCannotCache

`forward_incremental` returns its input and the trace fails with `InvalidDescription` in four
cases: a block without a mask ("no mask", "no incremental form"), a [1, 3, 8] input ("one
position", "[1, 3, 8]"), an evaluation trace in place of a carrying one ("'position'",
"carrying trace") and a per-sample shape of [5] ("per-sample", "[5]").

### ARotaryTraceTurnsTheQueriesAndTheKeysOnce

A causal block with `AttentionPositions::Rotary`, traced with a `HostTableList` collecting
tables, returns a [2, 5, 8] value and adds 68 nodes, 23 more than the learned form. The graph
holds two CHANNEL_AFFINE, nine SLICE_CHANNELS, four CONCAT_CHANNELS and two CAUSAL_MASK nodes.
The list holds two tables, each a [5, 2] leaf of the lifetime `Buffer` that requires no
gradient and has ten values. The first value of the first table is 1 and of the second is 0.

### ARotaryBlockHoldsTheStateOfALearnedOne

After a rotary trace, `NN::walk_state` under "attn" yields the same four names as the learned
block: the weight and the bias of `qkv`, then of `proj`.

### ARotaryTraceRefusesWhatItCannotTurn

A rotary block traced without `collect_tables` fails with `InvalidDescription` and a message
containing "'rotary_cosine'" and "collect_tables". A rotary block of width 6 with two heads
fails with "rotary positions" and "3 channels". Each `forward` returns its input.

### TheFactoryReadsThePositions

The row `{"heads": 2}` builds a layer whose `positions` is `Learned`, and `"positions":
"rotary"` builds one whose `positions` is `Rotary`. The factory fails with `InvalidDescription`
and "layer 3" for `"positions": "sinusoidal"` ("'positions'", "'sinusoidal'", "'learned'",
"'rotary'"), for `"positions": 1` ("'positions'", "must be a string") and for a rotary row
over a [5, 6] sample ("rotary positions", "3 channels").

### TheRotaryIncrementalFormReplacesARowOfEachRing

The incremental trace of a rotary block returns a [1, 1, 8] value with no failure and no
refusal. The graph holds three SCATTER_ADD_ROWS, two SELECT, three ADD, two CHANNEL_AFFINE and
no CAUSAL_MASK nodes. Two values are carried; each leaf has the shape [5, 8], a SELECT node
produces each next value, and the third input of that node is the leaf.

### TheRotaryIncrementalFormIsFedASlotAndTheAngles

Two rotary blocks traced incrementally in one carrying trace declare four fed leaves in the
order `K_CACHE_SLOT_LEAF` (`Int32`, [1, 1]), `K_ATTENTION_MASK_LEAF` ([1, 5]),
`K_ROTARY_COSINE_LEAF` ([1, 2]) and `K_ROTARY_SINE_LEAF` ([1, 2]), and carry four values.

## modules/vision/blocks/tests/test_block_conv_norm.cpp

Suites: `Unit_BlockConvNorm`.

Tests ConvNorm's trace, its state names against YOLO26n row 0, and make_conv_node.

This file needs no device. `NN::ConvNorm` in `modules/vision/blocks/include/SushiAI/vision/blocks/conv_norm.hpp` is a
bias-free same-padded `Conv2d`, a `BatchNorm2d` and an activation, traced in that order.
`NN::Conv` is `ConvNorm<Conv2d, SiLU>` and `NN::ConvLinear` is `ConvNorm<Conv2d, Identity>`.
`NN::make_conv_node` builds the node of a `conv` row of a model file.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`, the epsilon and momentum `MANIFEST.json`
records for YOLO26n. `K_CONTEXT` is "layer 0 'conv'", the row name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### OutputShapeHalvesAnEvenExtentAtStrideTwo

`Conv(3, 16, 3, 2)` maps [640, 640, 3] to [320, 320, 16].

### OutputShapeOfAPointwiseBlockKeepsTheExtents

`Conv(32, 64, 1, 1)` maps [7, 9, 32] to [7, 9, 64].

### ATrainingTraceEmitsConvolutionNormThenSiLU

A training trace of `Conv(3, 16, 3, 2)` over [2, 8, 8, 3] emits one K_MATMUL, one K_BATCHNORM2D
and one K_SILU, in that order, and no K_BIAS_ADD. The K_SILU's output is the returned value, of
shape [2, 4, 4, 16].

### ATrainingTraceCarriesTheNormOptions

The K_BATCHNORM2D node carries the epsilon 1e-3 in attribute 0, and both requested updates carry
the momentum 0.03.

### AnEvaluationTraceFoldsTheNorm

An evaluation trace emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE, one K_SILU and no K_BATCHNORM2D,
and returns a [2, 4, 4, 16] value.

### ConvLinearEmitsNoActivation

A training trace of `ConvLinear(3, 16, 1, 1)` emits one K_BATCHNORM2D and no K_SILU. The returned
value is output 0 of the K_BATCHNORM2D node.

### StateMatchesManifestRow0

The walk of a traced `Conv(3, 16, 3, 2)` under "model.0" yields five records in this order:
"model.0.conv.weight" [16, 3, 3, 3], "model.0.bn.weight" [16] and "model.0.bn.bias" [16] as
parameters, then "model.0.bn.running_mean" [16] and "model.0.bn.running_var" [16] as buffers.

### ADepthwiseTraceEmitsDepthwiseConvolutionNormThenSiLU

A training trace of `DWConv(16, 16, 3, 1)` over [2, 8, 8, 16] holds one DEPTHWISE_CONV2D, one
BATCHNORM2D and one SILU in that order and no MATMUL. The convolution reads the input, the
SILU writes the result, and the result has the input's shape.

### DWConvLinearEmitsNoActivation

A training trace of `DWConvLinear(16, 16, 3, 1)` holds one DEPTHWISE_CONV2D, one BATCHNORM2D
and no SILU, and the BATCHNORM2D writes the result.

### ADepthwiseOutputShapeKeepsTheWidth

For the input [80, 80, 64], a 3x3 `DWConv` reports [80, 80, 64] at stride 1 and [40, 40, 64]
at stride 2.

### DepthwiseStateMatchesTheManifestPositionalConvolution

After a training trace of `DWConvLinear(128, 128, 3, 1)`, the state walk under
"model.10.m.0.attn.pe" gives five records in order: the parameters `conv.weight` [3, 3, 128],
`bn.weight` [128] and `bn.bias` [128], then the buffers `bn.running_mean` [128] and
`bn.running_var` [128].

### ADepthwiseBlockRefusesUnequalWidths

Constructing `DWConv(16, 32, 3, 1)` raises an error holding "keeps its channel count",
"in=16" and "out=32".

### AnUntracedBlockYieldsNoState

The walk of a block that was never traced is empty.

### TheFactoryBuildsRow0

`make_conv_node` over [640, 640, 3] with 16 filters, kernel 3, stride 2, `bn_epsilon` 0.001 and
`bn_momentum` 0.03 returns a node with one port named `NN::K_SINGLE_PORT_NAME` of shape
[320, 320, 16]. A training trace over [1, 640, 640, 3] appends one value of shape
[1, 320, 320, 16], the K_BATCHNORM2D carries 1e-3, and the first update carries 0.03. The walk
under "model.0" yields five records, the first "model.0.conv.weight" [16, 3, 3, 3] and the
last "model.0.bn.running_var" [16].

### TheFactoryDefaultsKernelAndStrideToOne

A row stating only 8 filters over [6, 10, 3] has a port of shape [6, 10, 8]. After a trace its
first state record is "conv.weight" [8, 1, 1, 3].

### TheFactoryRefusesAnUnknownKey

The key `padding` fails with `InvalidDescription` and a message containing "layer 0 'conv'",
"'padding'" and "filters".

### TheFactoryRefusesARowWithoutFilters

A row without `filters` fails with a message containing the context, "'filters'" and "absent".

### TheFactoryRefusesAWindowThatIsNotAPositiveWholeNumber

A `stride` of 0 fails with "'stride'", "positive" and "0". A `kernel` given as the string "3"
fails with "'kernel'" and "whole number".

### TheFactoryRefusesNormOptionsOutOfRange

A `bn_epsilon` of 0.0 fails with "'bn_epsilon'" and "positive", and one given as a string
fails with "'bn_epsilon'" and "number". A `bn_momentum` of 1.5 fails with "'bn_momentum'" and
"[0, 1]", and one of -0.1 fails with those and "-0.1".

### TheFactoryRefusesInputsThatAreNotOneFeatureMap

Two input shapes fail with a message containing the context, "one input" and "2". The one
shape [192] fails with "[192]" and "rank-3".

## modules/vision/blocks/tests/test_block_detect_head.cpp

Suites: `Unit_BlockDetectHead`.

Tests `DetectHead` and its factory `Vision::Blocks::make_detect_node` at row 23 of YOLO26n:
three feature levels of 64, 128 and 256 channels with the sides 80, 40 and 20, and 80 classes.
The traced cases feed a batch of 2 with the sides 8, 4 and 2.

### ThePortsCarryTheFixtureNamesAndShapes

The head takes three inputs and has twelve ports. They are named
`<copy>.level<n>.<role>`, with the copy "one2many" before "one2one", the role "boxes" before
"scores", and the levels in order inside a role. A boxes port has 4 channels and a scores port
80, at its level's side.

### StateMatchesManifestRecords390To593

Walked under the prefix "model.23" after a training trace, the head yields 204 records in
order: the box branches under `cv2`, the class branches under `cv3`, then the same two groups
under `one2one_cv2` and `one2one_cv3`.

### ATrainingTraceDetachesEachFeatureOnce

A training trace holds three STOP_GRADIENT nodes, whose inputs are the three features in level
order. `forward` appends twelve values, each of its port's channel count at the traced side of
its level, with the batch extent 2 in front.

### AnEvaluationTraceEmitsNoStopGradient

An evaluation trace holds no STOP_GRADIENT and no BATCHNORM2D node, 36 BATCHNORM_FOLD nodes and
12 DEPTHWISE_CONV2D nodes, and appends twelve values.

### TheFirstLevelSetsTheBranchWidths

A head over one [4, 4, 128] level and 3 classes has 68 state records after a trace. Among them
are `cv2.0.0.conv.weight` of [32, 3, 3, 128], `cv2.0.2.weight` of [4, 1, 1, 32],
`cv3.0.0.1.conv.weight` of [128, 1, 1, 128] and `cv3.0.2.weight` of [3, 1, 1, 128]. Port 1 has
the shape [4, 4, 3].

### ForwardRefusesAWrongInputCount

`forward` given two values for three levels fails with `InvalidDescription`, "input count is 3"
and "given is 2", and appends nothing.

### CreateRefusesLevelsOrClassesThatDoNotFit

`create` fails with `InvalidDescription` for no level ("at least one feature level"), for a
rank-2 second level ("level 1", "[40, 128]", "rank-3") and for a class count of zero
("at least one class", "0").

### TheFactoryBuildsRow23

`make_detect_node` over the three levels, with "classes" 80 and the norm constants, returns a
node with three inputs and twelve ports. Port 0 is "one2many.level0.boxes" of [80, 80, 4] and
port 11 is "one2one.level2.scores" of [20, 20, 80]. A training trace holds 36 BATCHNORM2D nodes
that carry the row's epsilon, every update carries the row's momentum, and the state walk
yields 204 records.

### TheFactoryRefusesARowThatDoesNotFit

An unknown key "filters" fails with a message holding "layer 23 'detect'" and "'filters'". A
row without "classes" fails with "'classes'" and "absent". An empty level list fails with
"layer 23 'detect'" and "at least one feature level".

## modules/language/blocks/tests/test_block_expert_transformer_block.cpp

Suites: `Unit_BlockExpertTransformerBlock`.

Tests `Language::Blocks::ExpertTransformerBlock` in
`modules/language/blocks/include/SushiAI/language/blocks/expert_transformer_block.hpp`, and the
`transformer_block` row of `make_transformer_block_node` when it states `experts`. The block of
every case is causal, has a width of 8, two heads and four experts of hidden width 16.

### ATraceAddsANormalisedAttentionThenANormalisedMixture

`trace_experts` over a [2, 5, 8] input returns an output of the shape [2, 5, 8], a balance of
[1, 1] and a usage of [1, 4], with no failure and no refusal. The trace adds 84 nodes, among
them two LAYER_NORM, two CAUSAL_MASK and one UPSAMPLE2D. An ADD node produces the output; its
first input comes from another ADD node whose first input is the block's input.

### TheWalkYieldsTheFourChildren

After a trace, `NN::walk_state` under "h" yields fourteen names: the weight and the bias of
`norm1`, of `attn.qkv`, of `attn.proj`, of `norm2`, of `ff.router`, of `ff.expand` and of
`ff.project`, in that order.

### AnInputThatDoesNotFitFails

Over a [2, 5, 9] input `forward` returns the input and the trace fails with
`InvalidDescription` and a message containing "LayerNorm(8)" and "[2, 5, 9]". The graph holds
no LAYER_NORM node and the router's weight stays undeclared.

### TheTransformerRowWithExpertsHasThreePorts

Over a [5, 8] sample, the rows `{"heads": 2}` and `{"heads": 2, "experts": 0}` build nodes of
one port. The row with `"experts": 4, "top_k": 1` builds a node of three ports:
`K_SINGLE_PORT_NAME` of the shape [5, 8], `K_BALANCE_PORT`, and `K_USAGE_PORT` of the shape [4].
Traced over [2, 5, 8], that node writes three outputs, the third of the shape [1, 4]; the graph
holds one MAXPOOL2D node and a GELU node whose result has the shape [10, 128].

### TheTransformerRowRefusesCountsThatDoNotFit

The row fails with `InvalidDescription` and "layer 3" for a `top_k` without `experts`
("'top_k'", "no 'experts'"), for `"experts": 2, "top_k": 3` ("top_k=3") and for `"experts": -1`
("'experts'", "zero or more").

### TheIncrementalFormTracesEachChildOverOnePosition

`trace_experts_incremental` of a block that keeps one expert, over a [1, 1, 8] input and a
per-sample shape of [5, 8], returns an output of [1, 1, 8] and a usage of [1, 4]. The graph
holds two LAYER_NORM, two SCATTER_ADD_ROWS, no CAUSAL_MASK and one MAXPOOL2D node, and two
values are carried. The node of the row `{"heads": 2, "experts": 4}` writes three outputs
through `forward_incremental`, the first of the shape [1, 1, 8], and carries two values.

### TheBlockAndItsRowPassThePositionsToTheAttention

A block built without positions has `attn.positions` equal to `Learned`, and one built with
`AttentionPositions::Rotary` has `Rotary`. The node of a row with four experts and `"positions":
"rotary"`, traced with tables collected, leaves two host tables and two CHANNEL_AFFINE nodes.
The same row with eight heads fails with `InvalidDescription` and a message containing "layer
3", "rotary positions" and "1 channels".

## modules/language/blocks/tests/test_block_feed_forward.cpp

Suites: `Unit_BlockFeedForward`.

Tests `Language::Blocks::FeedForward` in
`modules/language/blocks/include/SushiAI/language/blocks/feed_forward.hpp` and its node factory
`make_feed_forward_node`. Every case uses a width of 6 and a hidden width of 24.

### ATraceMapsTheRowsThroughBothLayers

A trace over a [2, 5, 6] input returns a [2, 5, 6] value and adds 11 nodes. A RESHAPE node
produces the result; walking back through first inputs gives BIAS_ADD, MATMUL and GELU, and
the GELU result has the shape [10, 24].

### TheWalkYieldsTheTwoLinearLayers

After a trace, `NN::walk_state` under "ff" yields "ff.expand.weight", "ff.expand.bias",
"ff.project.weight" and "ff.project.bias" in that order. The `expand` weight has the shape
[24, 6] and the `project` weight [6, 24].

### AnInputOrADescriptionThatDoesNotFitFails

A [5, 6] input fails the trace with `InvalidDescription` and a message containing
"FeedForward", "rank-3 [N, T, 6] input" and "[5, 6]". A default-constructed block fails with
"positive widths" and "dim=0 hidden=0". Each `forward` returns its input, no node is added and
the `expand` weight stays undeclared.

### TheFactoryReadsTheHiddenWidth

Over a [5, 6] sample, the row `{"hidden": 24}` builds a `ModuleNode<FeedForward>` whose layer
has `dim` 6 and `hidden` 24, with one port of the shape [5, 6]. The factory fails with
`InvalidDescription` and "layer 3" for a row without `hidden` ("'hidden'", "absent") and for a
`bias` key ("'bias'").

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

`forward_incremental` over a [1, 1, 6] input in a carrying trace returns a [1, 1, 6] value. Its
graph has as many nodes as the graph of `forward` over a [1, 1, 6] input, with the same
operation at every index. The trace declares no fed leaf and carries no value.

## modules/language/blocks/tests/test_block_language_head.cpp

Suites: `Unit_BlockLanguageHead`, `Unit_BlockLanguageHeadDeathTest`.

Tests `Language::Blocks::LanguageHead` in
`modules/language/blocks/include/SushiAI/language/blocks/language_head.hpp` and its node factory
`make_language_head_node`, with and without a weight tied to an embedding table. Every case
uses a width of 6 and a vocabulary of 11.

### ATraceNormalisesThenMapsToTheVocabulary

A trace over a [2, 5, 6] input returns a [2, 5, 11] value and adds 7 nodes. Walking back from
the result through first inputs gives RESHAPE, MATMUL, RESHAPE and LAYER_NORM, and the
LAYER_NORM node reads the head's input. The bias of `linear` stays undeclared.

### TheWalkYieldsTheNormalisationAndTheWeight

After a trace, `NN::walk_state` under "head" yields "head.norm.weight", "head.norm.bias" and
"head.linear.weight". The `linear` weight has the shape [11, 6].

### AnInputOrADescriptionThatDoesNotFitFails

A [2, 5, 7] input fails the trace with `InvalidDescription` and a message containing
"LanguageHead", "rank-3 [N, T, 6] input" and "[2, 5, 7]". A default-constructed head fails
with "positive widths" and "dim=0 vocabulary=0". Each `forward` returns its input and no node
is added.

### ATiedHeadScoresWithTheEmbeddingsTable

A head tied with `tie_to` to the weight of an `NN::Embedding` of 11 rows and 6 columns, traced
over that embedding's [2, 5, 6] result, returns a [2, 5, 11] value and adds 6 nodes, one fewer
than the untied head. The second input of its MATMUL node is the embedding's weight. The
head's own `linear` weight stays undeclared, `NN::parameter_count` of the head is 2 and of the
embedding is 1.

### ATiedTableThatDoesNotFitFails

A head tied to a parameter that was never declared fails the trace with `InvalidDescription`
and "tied to a table no layer has declared". A head tied to a declared [11, 5] table fails with
"table of [11, 6]" and "tied to a float32 table of [11, 5]". Each `forward` returns its input.

### OutputShapeReplacesTheWidthByTheVocabulary

`output_shape([5, 6])` is [5, 11].

### TheFactoryReadsTheVocabulary

Over a [5, 6] sample, the row `{"vocabulary": 11, "ln_epsilon": 0.001}` builds a
`ModuleNode<LanguageHead>` whose layer has `dim` 6, `vocabulary` 11 and a normalisation epsilon
of 0.001, with one port of the shape [5, 11]. The factory fails with `InvalidDescription` and
"layer 3" for an empty row ("'vocabulary'", "absent") and for a [6] sample ("rank-2 [T, C]
input").

### TheFactoryTiesTheHeadToAnEmbeddingRow

Above the head stands a `ModelGraph` whose row 0 is an 11 by 6 embedding and whose row 1 is a
layer normalisation. With `NN::BuiltRows` over that model, the row `{"tie": 0}` builds a head
of vocabulary 11 whose `tied_weight` is the address of the embedding's weight, with a port of
the shape [5, 11]. The factory fails with `InvalidDescription` for `"tie": -1` ("layer 2",
"'tie'", "row 1", "no 'embedding' row"), for `"tie": 0, "vocabulary": 12` ("layer 2", "12
tokens", "[11, 6] table"), for `"tie": 2` ("'tie'", "row 2", "not above") and for a tied row
built without the rows above it ("layer 0", "not above").

### OutputShapeOverARankOneSampleIsFatal

`output_shape([6])` stops the process with a report containing "LanguageHead::output_shape"
and "[6]". The case is in the suite `Unit_BlockLanguageHeadDeathTest`.

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

`forward_incremental` over a [1, 1, 6] input in a carrying trace returns a [1, 1, 11] value.
Its graph has as many nodes as the graph of `forward` over a [1, 1, 6] input, with the same
operation at every index. The trace declares no fed leaf and carries no value.

## modules/language/blocks/tests/test_block_moe_feed_forward.cpp

Suites: `Unit_BlockMoeFeedForward`.

Tests `Language::Blocks::MoEFeedForward` in
`modules/language/blocks/include/SushiAI/language/blocks/moe_feed_forward.hpp` and its node
factory `make_moe_feed_forward_node`. Every case uses a width of 6 and four experts of hidden
width 8. A `top_k` of 0 keeps every expert.

### ATraceMixesEveryExpertByItsGate

`trace_experts` of a block with `top_k` 0 over a [2, 5, 6] input returns an output of the
shape [2, 5, 6], a balance of [1, 1] and a usage of [1, 4], and adds 31 nodes. The graph holds
one LOG_SOFTMAX, no MAXPOOL2D, one UPSAMPLE2D, one GELU and five MATMUL nodes. Walking back
from the output through first inputs gives RESHAPE, ADD, MATMUL and MUL, and the MUL result
has the shape [10, 32].

### KeepingTheTopScoresAddsOneWindowPerLevel

For `top_k` 1, 2 and 4, the trace returns a [2, 5, 6] output and adds 31 + 7 + 4 * `top_k`
nodes. The graph holds `top_k` MAXPOOL2D, `top_k` LESS_EQUAL and `top_k` SELECT nodes and two
LOG_SOFTMAX nodes, and a RESHAPE node produces the usage.

### TheWalkYieldsTheRouterAndTheStackedLayers

After a trace, the state walk under "ff" yields the weight and the bias of `router`, of
`expand` and of `project`, in that order. The shapes are [4, 6] for the router's weight,
[32, 6] for the `expand` weight, [6, 32] for the `project` weight and [4, 6] for the `project`
bias, whose role is `ParameterRole::Bias`.

### AnInputOrADescriptionThatDoesNotFitFails

`forward` over a [5, 6] input returns the input and fails the trace with `InvalidDescription`
("MoEFeedForward", "rank-3 [N, T, 6] input", "[5, 6]"). A default-constructed block fails with
"at least one expert" and "dim=0 hidden=0 experts=0". `trace_experts` of a block with `top_k`
5 returns a failure containing "1 to 4 of them" and "top_k=5" and leaves the trace's own
status ok. No node is added and the router's weight stays undeclared.

### TheFactoryBuildsANodeOfThreePorts

Over a [5, 6] sample, the row `{"experts": 4, "hidden": 8, "top_k": 2}` builds a node of one
input and three ports: `K_SINGLE_PORT_NAME` of [5, 6], `K_BALANCE_PORT` of [1] and
`K_USAGE_PORT` of [4]. Traced over [2, 5, 6], the node writes three outputs of the shapes
[2, 5, 6], [1, 1] and [1, 4], the graph holds two MAXPOOL2D nodes, and the state walk yields
six names. `forward` over a [5, 6] input fails with `InvalidDescription` ("MoEFeedForward",
"[5, 6]") and leaves the three outputs in the vector.

### TheFactoryRefusesARowThatDoesNotFit

The factory fails with `InvalidDescription` and "layer 3" for a row without `experts`
("'experts'", "absent"), a row without `hidden` ("'hidden'", "absent"), a `noise` key
("'noise'"), a `top_k` of 5 ("top_k=5"), a `top_k` of -1 ("'top_k'", "zero or more") and a
`top_k` of 0.5 ("'top_k'", "whole number").

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

`forward_incremental` of a block with `top_k` 1 over a [1, 1, 6] input in a carrying trace
returns a [1, 1, 6] value. Its graph has as many nodes as the graph of `forward` over a
[1, 1, 6] input, with the same operation at every index. The trace declares no fed leaf and
carries no value.

## modules/language/blocks/tests/test_block_position_embedding.cpp

Suites: `Unit_BlockPositionEmbedding`.

Tests `Language::Blocks::PositionEmbedding` in
`modules/language/blocks/include/SushiAI/language/blocks/position_embedding.hpp` and its node
factory `make_position_embedding_node`. The sequence of every case has five positions and a
width of 6.

### ATraceAddsTheTableToEveryRow

A block of five rows traced over a [2, 5, 6] input returns a [2, 5, 6] value and adds 5 nodes.
Its weight has the shape [5, 6] and the role `ParameterRole::Embedding`. A RESHAPE node
produces the result from a BIAS_ADD node whose inputs have the shapes [2, 30] and [30].

### AShorterSequenceReadsTheLeadingRows

A block of eight rows traced over a [2, 5, 6] input adds 7 nodes and declares a [8, 6] weight.
The second input of the BIAS_ADD node comes from a RESHAPE node that reads a SLICE_CHANNELS
node; the slice's result has the shape [1, 1, 1, 30] and its first attribute is 0.

### ASequenceLongerThanTheTableIsRefused

Over a [2, 5, 6] input, `forward` returns the input and the trace fails with
`InvalidDescription` for a block of four rows ("5 positions", "4 rows"), whose weight stays
undeclared, for a block of width 7 ("PositionEmbedding", "[N, T, 7]") and for a
default-constructed block ("positive extents").

### TheWalkYieldsTheTable

After a trace, `NN::for_each_parameter` under "positions" yields the one name
"positions.weight".

### TheFactoryReadsTheLength

Over a [5, 6] sample, an empty row builds a `ModuleNode<PositionEmbedding>` whose layer has
`length` 5 and `dim` 6, with one port of the sample's shape. The row `{"length": 9}` builds a
layer of `length` 9. The factory fails with `InvalidDescription` and "layer 1" for `"length":
4` ("'length' is 4", "5 positions"), for a `rows` key ("'rows'") and for a [6] sample ("rank-2
[T, C] input").

### TheIncrementalFormAddsTheRowOfTheFedPosition

`forward_incremental` over a [1, 1, 6] input in a carrying trace returns a [1, 1, 6] value.
The graph holds one GATHER_ROWS, one ADD and no BIAS_ADD node, and the weight's value has the
shape [5, 6]. The trace declares one fed leaf, `K_POSITION_LEAF`, an `Int32` value of the
shape [1, 1], and carries no value.

### TheIncrementalFormRefusesAWindowItsTableCannotServe

`forward_incremental` returns its input and the trace fails with `InvalidDescription` for a
per-sample shape of [6, 6] over a table of five rows ("6 positions", "5 rows"), which declares
no fed leaf, and for a [1, 2, 6] input ("one position", "[1, 2, 6]").

## modules/vision/blocks/tests/test_block_psa_block.cpp

Suites: `Unit_BlockPsaBlock`.

Tests `PSABlock` at the width of row 10 of YOLO26n, 128 channels over two heads, traced in
training mode over a [2, 4, 4, 128] input.

### OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

### EmitsThreeAddsOneSiLUFiveNormsAndTenUpdates

The trace holds three ADD nodes, one SILU node and five BATCHNORM2D nodes, and the buffer
update list holds ten updates. The result has the shape [2, 4, 4, 128].

### TheResidualAddsReadTheInputThenAttentionThenTheFeedForward

The second ADD of the graph reads the block's input and a BATCHNORM2D result. The third ADD
reads the second ADD's result and a BATCHNORM2D result, and its output is the value `forward`
returned.

### StateMatchesManifestRow10Block

Walked under the prefix "model.10.m.0", the traced block yields 25 records in order: five each
for `attn.qkv`, `attn.proj`, `attn.pe`, `ffn.0` and `ffn.1`. `ffn.0` has a [256, 1, 1, 128]
weight and `ffn.1` a [128, 1, 1, 256] weight.

### AWidthTheHeadCountDoesNotDivideThrows

`PSABlock(100, 3)` throws an error naming "head count divides", "100 channels" and "3 heads".

## modules/language/blocks/tests/test_block_rotary_embedding.cpp

Suites: `Unit_BlockRotaryEmbedding`.

Tests `Language::Blocks::RotaryEmbedding` in
`modules/language/blocks/include/SushiAI/language/blocks/rotary_embedding.hpp` and
`Language::Blocks::check_rotary_widths`. Every case uses a width of 8 cut into two groups, so
each group has two pairs. The ordinary traces read five positions.

### ATraceWeighsTheValueAndItsSwappedHalves

A trace over a [2, 5, 8] input with tables collected returns a [2, 5, 8] value, with no
failure and no refusal, and adds 20 nodes: two table leaves and 18 others. The graph holds two
CHANNEL_AFFINE, two SLICE_CHANNELS, three CONCAT_CHANNELS, one SCALE and one ADD node. Each
CHANNEL_AFFINE node reads a [2, 1, 1, 40] value and a [40] value that requires no gradient.

### TheTablesHoldTheAnglesOfEveryPosition

The trace leaves two host tables, cosines then sines. Each leaf has the shape [5, 2], the
cosine leaf is `Float32`, and each table has ten values. For every row and pair the two values
are within 1e-15 of the cosine and the sine of `rotary_angle(row, pair, 2)`.

### ADoubleValueReadsDoubleTables

Over a `Float64` input of the shape [1, 5, 8] the result is `Float64`, the first of the two
table leaves is `Float64` and the graph holds no CAST node.

### TheBlockHoldsNoState

After a trace, `NN::walk_state` yields no parameter and no buffer, and the graph's
`state_leaves()` is empty. `output_shape([5, 8])` is [5, 8].

### TheIncrementalFormIsFedTheAnglesOfItsPosition

`forward_incremental` over a [1, 1, 8] input in a carrying trace returns a [1, 1, 8] value and
adds 20 nodes, with no failure and no refusal. The trace carries no value and declares two
fed leaves, `K_ROTARY_COSINE_LEAF` then `K_ROTARY_SINE_LEAF`, each a `Float32` value of the
shape [1, 2].

### ABlockOrAnInputThatDoesNotFitIsRefused

`check_rotary_widths(8, 2)` is ok. It fails with `InvalidDescription` for (6, 2) ("pairs",
"width=6 groups=2"), (8, 3) ("width=8 groups=3") and (0, 0) ("width=0 groups=0"). `forward`
over a [2, 5, 10] input returns the input, fails with "RotaryEmbedding" and "[2, 5, 10]" and
collects no table. `forward` in a trace that collects no tables fails with "'rotary_cosine'"
and "collect_tables", and `forward_incremental` in an evaluation trace fails with
"'rotary_cosine'" and "carrying trace".

## modules/language/blocks/tests/test_block_rotary_positions.cpp

Suites: `Unit_BlockRotaryPositions`, `Unit_BlockRotaryPositionsDeathTest`.

Tests `Language::Blocks::rotary_angle` and `Language::Blocks::write_rotary_row` in
`modules/language/blocks/include/SushiAI/language/blocks/rotary_positions.hpp`. Angles are
compared within 1e-12 and `float` rows within 1e-6.

### AnAngleIsThePositionTimesTheFrequencyOfThePair

`K_ROTARY_BASE` is 10000. `rotary_angle(position, pair, pairs)` is 0 for (0, 1, 4), 7 for
(7, 0, 4), 0.03 for (3, 1, 2), 0.5 for (5, 1, 4), 0.05 for (5, 2, 4) and 0.005 for (5, 3, 4).

### PositionZeroTurnsNothing

`write_rotary_row` for position 0 over four pairs writes a cosine of exactly 1 and a sine of
exactly 0 for every pair.

### ARowHoldsTheCosineAndTheSineOfEachAngle

The row of position 11 over four pairs holds, for every pair, the cosine and the sine of
`rotary_angle(11, pair, 4)` converted to `float`. The cosine of pair 0 is 0.0044257 and its
sine is -0.9999902.

### ALargePositionWritesAPointOfTheUnitCircle

For position 2^40 over four pairs, the squared cosine plus the squared sine of every pair is 1
within 1e-6.

### ARowWithoutMatchingSpansIsFatal

`write_rotary_row` stops the process for two empty spans, with a report containing "0
cosines", and for a span of two cosines beside a span of three sines, with "2 cosines" and "3
sines". The case is in the suite `Unit_BlockRotaryPositionsDeathTest`.

## modules/vision/blocks/tests/test_block_segment_head.cpp

Suites: `Unit_BlockSegmentHead`.

Tests `Vision::Blocks::SegmentHead` and `make_segment_head_node` in
`modules/vision/blocks/include/SushiAI/vision/blocks/segment_head.hpp`. Every head takes 16
input channels and five classes, with an epsilon of 1e-3 and a momentum of 0.03 for its
normalisation.

### OutputShapeScalesTheExtentsAndSetsTheClasses

For the input [6, 10, 16], a head of scale 4 reports [24, 40, 5] and a head of scale 1 reports
[6, 10, 5].

### TheTraceEndsAtTheUpsample

A training trace of a head with 8 hidden channels and scale 2 over [2, 4, 4, 16] holds one
BATCHNORM2D, one SILU and one UPSAMPLE2D. The result has the shape [2, 8, 8, 5] and the
UPSAMPLE2D node produces it.

### StateIsListedUnderCv1AndCv2

After a training trace the state walk gives seven records in this order: `cv1.conv.weight`
[8, 3, 3, 16], `cv1.bn.weight`, `cv1.bn.bias`, `cv1.bn.running_mean` and `cv1.bn.running_var`,
each [8], then `cv2.weight` [5, 1, 1, 8] and `cv2.bias` [5].

### TheFactoryBuildsTheRow

Over the input [4, 4, 16], the row with `classes` 5, `hidden` 8 and `scale` 2 gives a node with
one port of shape [8, 8, 5]. The row with only `classes` 5 gives one port of shape [4, 4, 5];
after an evaluation trace it has seven pieces of state, the first being `cv1.conv.weight`
[16, 3, 3, 16].

### TheFactoryRefusesARowThatDoesNotFit

`make_segment_head_node` fails with `InvalidDescription` for a row with the key `mode`
("layer 3 'segment_head'", "'mode'"), for a row without `classes` ("'classes'", "absent") and
for `scale` 0 ("'scale'", "positive").

## modules/vision/blocks/tests/test_block_sppf.cpp

Suites: `Unit_BlockSppf`.

Tests SPPF's trace, its state names against YOLO26n row 9, and its factory.

This file needs no device. `NN::SPPF` in `modules/vision/blocks/include/SushiAI/vision/blocks/sppf.hpp` holds a
`ConvLinear` named `cv1`, a `MaxPool2d` named `pool` and a `Conv` named `cv2`. The cases show
its trace: `cv1` narrows the input, the pool is applied a number of times in a chain, one
K_CONCAT_CHANNELS joins `cv1`'s result and every pooled value, `cv2` follows, and an optional K_ADD
closes the block. `NN::make_sppf_node` builds the node of an `sppf` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 9 'sppf'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`SPPF(256, 256, 5, 3, true)` maps [20, 20, 256] to [20, 20, 256], and `SPPF(64, 96, 5, 3,
true)` maps [5, 7, 64] to [5, 7, 96].

### ThePoolKeepsTheExtents

The pool of `SPPF(256, 256, 5, 3, true)` has a 5x5 kernel, stride 1 and a pad of 2 on top and
left, and its `output_shape([20, 20, 128])` is [20, 20, 128].

### Row9EmitsThreePoolsOneConcatAndOneAdd

A training trace over [2, 4, 4, 256] emits two K_MATMUL, two K_BATCHNORM2D, one K_SILU, three
K_MAXPOOL2D, one K_CONCAT_CHANNELS and one K_ADD. The list holds four updates and the result is
[2, 4, 4, 256]. The one K_SILU is `cv2`'s.

### TheJoinReadsTheNarrowedValueThenEachPoolInAChain

The K_CONCAT_CHANNELS has four inputs. The first is produced by a K_BATCHNORM2D, because `cv1` is
a `ConvLinear` and emits no activation. Each of the other three is produced by a K_MAXPOOL2D
that reads the input before it, and each is [2, 4, 4, 128]. The joined value is
[2, 4, 4, 512].

### TheClosingAddReadsTheResultThenTheInput

`adds_input` is true. The K_ADD's first operand is produced by a K_SILU and its second is the
block's input, and its output is the returned value. The operand order is the reverse of
`Bottleneck`'s.

### NoAddWithoutAShortcutOrOverUnequalWidths

`SPPF(64, 64, 5, 3, false)` and `SPPF(64, 96, 5, 3, true)` both report `adds_input` false and
trace no K_ADD.

### ThePoolCountSetsThePoolsAndTheJoinedWidth

`SPPF(64, 64, 3, 1, false)` traces one K_MAXPOOL2D. Its `cv1` weight is [32, 1, 1, 64] and its
`cv2` weight is [64, 1, 1, 64], two parts of 32 channels.

### StateMatchesManifestRow9

The walk of a traced `SPPF(256, 256, 5, 3)` under "model.9" yields ten records. The five of
`cv1` come first, with the weight [128, 1, 1, 256] and the rest [128]. The five of `cv2`
follow, with the weight [256, 1, 1, 512] and the rest [256]. The pool has no state.

### TheFactoryBuildsRow9

`make_sppf_node` over [20, 20, 256] with 256 filters, kernel 5, 3 pools and shortcut true has
one port of shape [20, 20, 256]. Its training trace holds three K_MAXPOOL2D and one K_ADD. The walk
under "model.9" yields ten records; record 0 is the `cv1` weight and record 5 the `cv2`
weight.

### TheFactoryDefaultsToThreePoolsOfFiveAndNoShortcut

A row stating only 64 filters traces three K_MAXPOOL2D and no K_ADD, and its `cv2` weight is
[64, 1, 1, 128].

### TheFactoryRefusesAnEvenKernel

`kernel` 4 fails with `InvalidDescription` and a message containing "layer 9 'sppf'",
"'kernel'", "odd" and "4".

### TheFactoryRefusesAnInputOfOneChannel

The input shape [4, 4, 1] fails with "[4, 4, 1]" and "at least 2 channels".

### TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `pools` 0 fails with "'pools'" and "positive".

### TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

### TheFactoryReadsKernelAndPools

With `kernel` 3 and `pools` 1 the trace holds one K_MAXPOOL2D whose attributes 0, 1 and 6, which
are kernel_h, kernel_w and pad_top, are 3, 3 and 1. The `cv2` weight is [64, 1, 1, 64].

## modules/language/blocks/tests/test_block_transformer_block.cpp

Suites: `Unit_BlockTransformerBlock`.

Tests `Language::Blocks::TransformerBlock` in
`modules/language/blocks/include/SushiAI/language/blocks/transformer_block.hpp` and its node
factory `make_transformer_block_node`. Every block has a width of 8, two heads and a
feed-forward of hidden width 16, and is causal unless the case says otherwise.

### ATraceAddsANormalisedAttentionThenANormalisedFeedForward

A trace over a [2, 5, 8] input returns a [2, 5, 8] value and adds 64 nodes, among them two
LAYER_NORM, two ADD, two CAUSAL_MASK and one GELU. An ADD node produces the result; its first
input comes from the other ADD node, whose first input is the block's input. Each LAYER_NORM
node reads either the block's input or the result of the first ADD node.

### ABlockWithoutAMaskEmitsNone

A block built with `AttentionMask::None` traces without failure into a graph of no CAUSAL_MASK
node and two LOG_SOFTMAX nodes.

### TheWalkYieldsTheFourChildren

After a trace, `NN::walk_state` under "h" yields twelve names: the weight and the bias of
`norm1`, of `attn.qkv`, of `attn.proj`, of `norm2`, of `ff.expand` and of `ff.project`, in that
order.

### AnInputThatDoesNotFitFails

Over a [2, 5, 9] input `forward` returns the input and the trace fails with
`InvalidDescription` and a message containing "LayerNorm(8)" and "[2, 5, 9]". The graph holds
no LAYER_NORM node and holds a refusal.

### TheFactoryReadsItsKeysOverTheirDefaults

Over a [5, 8] sample, the row `{"heads": 2}` builds a `ModuleNode<TransformerBlock>` with two
heads, a hidden width of 32, the mask `Causal` and `K_LAYER_NORM_EPSILON` in `norm1`, and a
port of the shape [5, 8]. The row `{"heads": 4, "hidden": 16, "causal": false, "ln_epsilon":
0.001}` builds one with four heads, a hidden width of 16, the mask `None` and an epsilon of
0.001 in `norm2`.

### TheFactoryRefusesARowThatDoesNotFit

The factory fails with `InvalidDescription` and "layer 3" for a `depth` key ("'depth'"), a row
without `heads` ("'heads'", "absent") and five heads over a width of 8 ("dim=8 heads=5").

### TheIncrementalFormTracesEachChildOverOnePosition

`forward_incremental` over a [1, 1, 8] input and a per-sample shape of [5, 8] returns a
[1, 1, 8] value with no failure and no refusal. The graph holds two LAYER_NORM, two
SCATTER_ADD_ROWS, no CAUSAL_MASK and one GELU node. The trace declares two fed leaves and
carries two values.

### BothFormsDeclareTheSameParameters

A block traced with `forward` over [1, 5, 8] and a second block traced with
`forward_incremental` over [1, 1, 8] have the same `NN::parameter_count`, and the second has a
`NN::buffer_count` of 0.

### AnUnmaskedBlockHasNoIncrementalForm

`forward_incremental` of a block built with `AttentionMask::None` returns its input, and the
trace fails with `InvalidDescription` and a message containing "no mask".

### TheFactoryPassesThePositionsToItsAttention

The row `{"heads": 2}` builds a layer whose `attn.positions` is `Learned`, and `"positions":
"rotary"` builds one whose `attn.positions` is `Rotary`. A rotary row over a [5, 6] sample
fails with `InvalidDescription` and a message containing "layer 3", "rotary positions" and "3
channels".

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

## modules/layers/train/tests/test_carrying_step.cpp

Suites: `Unit_CarryingStep`, `Unit_PlainEvaluationStep`.

Tests a `Train::EvaluationStep` that is fed two leaves and carries a value between runs. Both
fixtures need an engine. The model of `Unit_CarryingStep` is a local `Accumulator` whose
`forward` feeds a leaf named `gain`, declares a carried leaf `sum` with the initial value 1 and
returns sum + x * gain. The fixture builds the step with `Train::build_carrying_graph` over a
[1, 3] row and fills x with 1, 2, 3 and the gain with 2, 2, 2. A `static_assert` holds that
`Accumulator` is a `TraceModule`.

### TheStepNamesTheModelInputThenTheFedLeaves

`input_names()` is `K_MODEL_INPUT_NAME` then `gain`, and both `carries_state()` and
`step().has_trailing_work()` are true. `input("gain")` returns a tensor whose second extent is
3. `input("absent")` fails with `InvalidDescription` and a message naming `'absent'`,
`'inputs'` and `'gain'`.

### ARunReadsWhatTheRunBeforeCarried

After one, two and three runs by name, element i of the output is 1 + (i + 1) * 2 * runs within
1e-6. The step compiled once and ran three times.

### AResetRestoresTheInitialValue

After two runs, `reset()` and one more run, the output is what one run gives. The compile count
stays 1.

### ARunInOrderFeedsTheLeavesAsTheStepListsThem

`run_in_order` with x then the gain gives the output of one run by name.

### ARunOfOneTensorIsRefusedByAStepFedTwoLeaves

`run(x)` fails with `Incompatible` and a message containing "1 tensors", "fed 2" and `'gain'`.
The run count stays 0.

### NamedFeedsThatDoNotCoverTheLeavesAreRefused

`run_by_name` fails with `InvalidDescription` three times: for a feed named `scale` ("'scale'",
"no input leaf"), for `gain` given twice ("'gain' twice") and for one feed alone ("1 named
tensors", "fed 2"). The run count stays 0.

### AFeedOfAnotherShapeIsRefusedUnderItsName

A [1, 4] tensor fed as `gain` fails with `Incompatible` and a message containing "gain",
"[1, 4]" and "[1, 3]".

### AnOrdinaryStepFeedsOneLeafAndCarriesNothing

A step built with `Train::build_evaluation_graph` over `NN::Linear{3, 3}` lists one input,
`K_MODEL_INPUT_NAME`. `carries_state()` and `has_trailing_work()` are false, `reset()` is ok and
the run count is 0.

## modules/core/autograd/tests/test_causal_mask_backward.cpp

Suites: `Unit_CausalMaskBackward`.

Tests the nodes `Autograd::differentiate` appends for CAUSAL_MASK. The two graph cases mask
[4, 4] scores with the fill -1e9 and take the `nll_loss` of the `log_softmax` of the masked
value against a `targets` input.

### TheMaskHasARule

`Ops::K_CAUSAL_MASK` is differentiable, does not block the gradient, and
`Autograd::has_backward_rule` is true for its id.

### TheScoreGradientIsTheGradientMaskedWithZero

With the scores as a parameter, the graph holds two CAUSAL_MASK nodes after differentiation.
The first has the fill -1e9 in attribute slot 0 and the second has 0. The one output of the
second is the returned gradient, whose shape is [4, 4].

### ScoresThatNeedNoGradientAddNoNode

With the scores as an input, the returned gradient is `K_NO_VALUE` and the graph holds one
CAUSAL_MASK node.

## modules/core/graph/tests/test_causal_mask_ops.cpp

Suites: `Unit_CausalMaskOps`.

Tests `GraphBuilder::causal_mask`: the node it emits, the shapes it accepts and the operands it
refuses. Every case uses the fill -1e9.

### TheMaskWritesItsIdentityAndKeepsTheShape

Over [2, 5, 5] float64 scores the call is ok and emits one CAUSAL_MASK node that reads the
scores, writes the returned value and holds the fill in attribute slot 0. The result has the
shape [2, 5, 5], the type float64 and the lifetime `Activation`, and the graph has no refusal.

### TheMaskAcceptsAnyNumberOfLeadingAxes

A [3, 3] matrix and a [2, 4, 3, 3] value both return a value of their own shape.

### TheMaskRefusesOperandsThatDoNotFit

Four calls fail with `InvalidDescription`: a [5] vector ("[..., T, T] scores"), a [2, 4, 5]
value ("[2, 4, 5]"), int32 scores ("half, float32 or float64 scores") and a fill of minus
infinity ("finite fill"). After each the node count is unchanged and `has_refusal()` is true.

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

## modules/vision/codecs/tests/test_codecs_decode_image.cpp

Suites: `Unit_CodecsDecodeImage`.

Tests `decode_image` and `load_image` in
`modules/vision/codecs/include/SushiAI/vision/codecs/decode_image.hpp`. The inputs are a 2 by 2
RGB PNG that libpng encodes, an 8 by 8 gray JPEG of level 140 that libjpeg encodes at quality
100, and raw image files that `write_raw_image` writes. A level decoded from the JPEG may
differ from 140 by 3. The `Load` cases write their files into a directory under the system's
temporary directory and remove it.

### PNGBytesReachThePNGDecoder

The PNG decodes to a 2 by 2 image of three channels whose bytes are the encoded samples.

### JPEGBytesReachTheJPEGDecoder

The JPEG decodes to an 8 by 8 image of one channel, every byte within 3 of 140.

### OptionsReachTheSelectedDecoder

With `ChannelLayout::Color` the gray JPEG decodes to three channels. With
`ChannelLayout::Gray` the RGB PNG decodes to one.

### RawRGBFileDecodesToItsStoredSamples

A raw RGB file decodes to a 2 by 2 image of three channels holding the samples written, with
the default options and with `ChannelLayout::Color`.

### RawBGRFileDecodesToRGB

A raw file written in BGR order decodes with the first and third byte of every pixel swapped:
0, 0, 255, 0, 255, 0, 255, 0, 0, 30, 20, 10.

### RawGrayFileDecodesToOneChannelOrToThreeEqualOnes

A raw gray file of 3, 60, 190, 255 decodes to those four bytes in one channel, by default and
with `ChannelLayout::Gray`. With `ChannelLayout::Color` it decodes to three channels, each
level repeated three times.

### RawRGBFileIsRefusedAsGray

A raw RGB file read with `ChannelLayout::Gray` fails with `Unsupported`, "decode_image: ",
"3 channels" and "gray".

### RawFileCutShortIsInvalid

A raw file one byte short fails with `InvalidFile`, "holds 35 bytes" and "declares 36". Its
first 10 bytes fail with `InvalidFile`, "decode_image: " and "10 of 24".

### RawFileWithTrailingBytesIsInvalid

A raw file with one byte appended fails with `InvalidFile`, "holds 37 bytes" and "declares 36".

### RawFileWithASideAboveTheLimitIsInvalidBeforeAnyAllocation

A raw file whose height bytes are set to 16385 fails with `InvalidFile`, "height 16385" and
"16384".

### BytesOfNoKnownFormatAreUnsupported

`decode_image` fails with `Unsupported` and "decode_image: " for 64 bytes of 0x5A ("64 bytes"),
for a GIF signature and for an empty span ("0 bytes").

### SignatureWithNothingBehindItFailsInItsDecoder

Each signature alone fails with `InvalidFile`: the three JPEG bytes with "decode_jpeg: ", the
eight PNG bytes with "decode_png: ", and the letters SAIM with "decode_image: " and "4 of 24".

### LoadReportsAMissingPathAndNamesIt

`load_image` of a file that does not exist fails with `FileNotFound`, "absent.png" and
"load_image: ".

### LoadDecodesAPNGFile

The PNG written to `image.png` loads as three channels holding the encoded samples.

### LoadDecodesAJPEGFileWhateverItsExtension

The JPEG written to a file named `image.png` loads with `ChannelLayout::Color` as 8 by 8 by 3
bytes, each within 3 of 140.

### LoadDecodesARawImageFile

The raw gray file written to `image.saim` loads as one channel holding 3, 60, 190, 255.

### LoadNamesThePathOfAFileItsDecoderRejects

The first 40 bytes of the PNG, written to `cut.png`, fail with `InvalidFile`, "cut.png" and
"decode_png: ".

### LoadNamesThePathOfAFileOfNoKnownFormat

A file `notes.txt` holding "hello" fails with `Unsupported`, "notes.txt" and "decode_image: ".
An empty `empty.png` fails with `Unsupported`, "empty.png" and "0 bytes".

### LoadRefusesADirectory

`load_image` of the scratch directory itself fails with `InvalidFile`, the directory's name
and "regular file".

## modules/vision/codecs/tests/test_codecs_image_format.cpp

Suites: `Unit_CodecsImageFormat`.

Tests `detect_image_format` in
`modules/vision/codecs/include/SushiAI/vision/codecs/image_format.hpp`.

### JPEGIsDetectedFromItsThreeLeadingBytes

The bytes FF D8 FF give `ImageFormat::JPEG`.

### JPEGIsDetectedWhateverFollowsItsSignature

FF D8 FF followed by E0 00 10, and followed by E1 12 34, both give `ImageFormat::JPEG`.

### PNGIsDetectedFromItsEightByteSignature

The eight signature bytes give `ImageFormat::PNG`, and so do the same bytes with one zero
byte appended.

### RawImageIsDetectedFromTheCacheMagic

The four bytes of `K_RAW_IMAGE_MAGIC`, least significant first, give `ImageFormat::Raw`.

### RawImageIsDetectedFromTheLettersSAIM

The bytes 'S', 'A', 'I', 'M', 1, 0, 0, 0 give `ImageFormat::Raw`.

### NoBytesAreUnknown

An empty span gives `ImageFormat::Unknown`.

### GarbageIsUnknown

The text "not an img", sixteen zero bytes and the GIF signature each give
`ImageFormat::Unknown`.

### SignatureCutShortIsUnknown

The first two bytes of the JPEG signature, the first seven of the PNG signature and the first
three of SAIM each give `ImageFormat::Unknown`.

### SignatureWithOneByteChangedIsUnknown

The PNG signature with its last byte set to 0x0B, the bytes FF D8 00 and the letters SAIN each
give `ImageFormat::Unknown`.

## modules/vision/codecs/tests/test_codecs_jpeg.cpp

Suites: `Unit_CodecsJpeg`.

Tests `decode_jpeg` in `modules/vision/codecs/include/SushiAI/vision/codecs/jpeg_decoder.hpp`
on files the cases encode in memory with libjpeg at quality 100 and without subsampling. A
quadrant image is 16 by 16 pixels with one colour per 8 by 8 quadrant, and a decoded channel
may differ from its colour by 3 levels.

### GrayFileDecodesToOneChannel

The gray quadrant file of levels 10, 100, 180 and 250 decodes to one channel holding those
levels.

### GrayFileDecodesToThreeEqualChannelsWhenColorIsAsked

With `ChannelLayout::Color` the gray file decodes to three channels holding the gray levels,
and the three bytes of every pixel are equal.

### ColorFileDecodesToRGB

A YCbCr file of four RGB colours decodes with the default options to three channels holding
those colours.

### ColorFileDecodesToRGBWhenColorIsAsked

The same file decodes to the same colours with `ChannelLayout::Color`.

### ColorFileOfNeutralPixelsDecodesToTheirLevelWhenGrayIsAsked

A YCbCr file whose pixels have three equal channels of 10, 100, 180 and 250 decodes with
`ChannelLayout::Gray` to one channel holding those levels.

### RGBFileWithoutAColorTransformDecodesToRGB

A file stored in the RGB colour space decodes to three channels holding its colours.

### CMYKFileDecodesToRGB

A file stored as CMYK, where 255 means no ink, decodes to three channels: (255, 255, 255, 255)
to white, (0, 255, 255, 255) to (0, 255, 255), (255, 255, 255, 0) to black and
(128, 255, 64, 255) to (128, 255, 64).

### YCCKFileDecodesToRGB

The same CMYK pixels stored as YCCK decode with `ChannelLayout::Color` to the same four RGB
colours.

### CMYKFileIsRefusedAsGrayAndTheColorSpaceIsNamed

With `ChannelLayout::Gray` the CMYK file fails with `Unsupported`, "decode_jpeg: ", "CMYK" and
"gray", and the YCCK file fails with `Unsupported`, "decode_jpeg: ", "YCCK" and "gray".

### FileCutInHalfIsInvalid

The first half of the colour file fails with `InvalidFile` and "decode_jpeg: ".

### FileWithoutItsEndMarkerIsInvalidAndCarriesTheLibraryMessage

The gray file without its last two bytes fails with `InvalidFile`,
"decode_jpeg: the data is corrupt: " and "JPEG".

### EveryProperPrefixOfAFileFails

`decode_jpeg` returns no image for every prefix of the gray file, from zero bytes to one byte
short of the whole.

### NoBytesAreInvalid

An empty span fails with `InvalidFile` and "decode_jpeg: ".

### BytesThatAreNotAJPEGAreInvalid

64 bytes of 0x5A fail with `InvalidFile` and "decode_jpeg: ".

### DecoderIsUsableAfterAFailure

After 64 bytes of 0x5A fail with `InvalidFile`, the gray file decodes to its four levels in
one channel.

### SideAtTheLimitDecodes

A gray file `K_MAX_IMAGE_SIDE` pixels wide and one high decodes to an image of that width, a
height of 1 and one channel.

### WidthAboveTheLimitIsRefused

A file 16385 pixels wide and one high fails with `Unsupported`, "1x16385" and "16384".

### HeightAboveTheLimitIsRefused

A file one pixel wide and 16385 high fails with `Unsupported`, "16385x1" and "16384".

## modules/vision/codecs/tests/test_codecs_png.cpp

Suites: `Unit_CodecsPng`.

Tests `decode_png` in `modules/vision/codecs/include/SushiAI/vision/codecs/png_decoder.hpp` on
files the cases encode in memory with libpng's simplified writer. The gray file is 2 rows by 3
columns of 0, 17, 128, 200, 254, 255. The palette file is 2 by 4 with the indices 3, 0, 1, 2,
2, 2, 0, 3 over the palette red, green, blue and (40, 40, 40). The pattern file is a 16 by 16
RGB image of generated bytes.

### GrayFileDecodesToItsStoredSamples

The gray file decodes to 2 rows, 3 columns and one channel holding its samples.

### GrayFileDecodesToThreeEqualChannelsWhenColorIsAsked

With `ChannelLayout::Color` the gray file decodes to 18 bytes in three channels, each pixel
holding its gray sample three times.

### ColorFileDecodesToItsStoredSamplesInRGBOrder

A 2 by 2 RGB file decodes to three channels whose bytes are the samples encoded.

### ColorFileOfGrayPixelsDecodesToThoseLevelsWhenGrayIsAsked

An RGB file whose pixels have three equal channels of 0, 90, 200 and 255 decodes with
`ChannelLayout::Gray` to one channel holding 0, 90, 200, 255.

### AlphaIsDroppedAndTheColorsKeepTheirStoredValues

An RGBA file with the alphas 0, 128, 255 and 1 decodes to three channels holding the stored
red, green and blue of each pixel.

### GrayAlphaFileDecodesToItsStoredGrayLevels

A gray-alpha file of the levels 5, 100, 180 and 250 with the alphas 0, 60, 255 and 128
decodes to one channel holding 5, 100, 180, 250.

### GrayAlphaFileDecodesToThreeEqualChannelsWhenColorIsAsked

A gray-alpha file of the levels 5 and 250 decodes with `ChannelLayout::Color` to 5, 5, 5, 250,
250, 250.

### PaletteFileYieldsItsStoredIndicesWhenIndicesAreAsked

With `PaletteRead::Indices` and `ChannelLayout::Stored` the palette file decodes to 2 rows, 4
columns and one channel holding its indices.

### PaletteIndicesComeOutInOneChannelWhateverTheLayout

With `PaletteRead::Indices` the palette file decodes to one channel holding its indices under
`ChannelLayout::Color` and under `ChannelLayout::Gray`.

### PaletteFileDecodesToItsPaletteColorsByDefault

With the default options the palette file decodes to three channels, each pixel holding the
palette entry its index names.

### PaletteFileReadAsGrayIsAConversionAndNotItsIndices

With `ChannelLayout::Gray` and palette colours the palette file decodes to one channel of
eight bytes. The first and the last pixel, both of index 3, hold 40.

### IndicesOptionLeavesAFileWithoutAPaletteToItsLayout

The gray file read with `PaletteRead::Indices` decodes to one channel holding its samples.

### SixteenBitGrayFileDecodesToEightBits

A 16-bit gray file of 0, 65535, 65535, 0, 65535, 0 decodes to 2 rows, 3 columns and one
channel holding 0, 255, 255, 0, 255, 0.

### SixteenBitColorFileDecodesToEightBitRGB

A 16-bit RGB file of a red, a green, a blue and a black pixel at 65535 decodes to three
channels holding 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0.

### FileCutInHalfIsInvalid

The first half of the pattern file fails with `InvalidFile` and "decode_png: ".

### FileCutInsideItsHeaderIsInvalid

The first 0, 7, 8 and 20 bytes of the pattern file each fail with `InvalidFile` and
"decode_png: ".

### EveryPrefixOfAFileFailsOrDecodesToTheFullSize

Every prefix of the pattern file shorter than the file either fails with `InvalidFile` or
decodes to an image of 16 by 16 by 3 bytes.

### BytesThatAreNotAPNGAreInvalid

64 bytes of 0x5A fail with `InvalidFile` and "decode_png: ".

### FileWithADamagedChunkIsInvalid

The pattern file with one bit of byte 19 flipped fails with `InvalidFile` and "decode_png: ".

### SideAtTheLimitDecodes

A gray file `K_MAX_IMAGE_SIDE` pixels wide and one high decodes to an image of that width and
a height of 1 holding its samples.

### WidthAboveTheLimitIsRefused

A file 16385 pixels wide and one high fails with `Unsupported`, "1x16385" and "16384".

### HeightAboveTheLimitIsRefused

A file one pixel wide and 16385 high fails with `Unsupported`, "16385x1" and "16384".

## modules/core/graph/tests/test_compiled_step.cpp

Suites: `Unit_CompiledStep`.

Tests `CompiledStep::restore_run_count` in
`modules/core/graph/include/SushiAI/graph/compiled_step.hpp`. The fixture is an `EngineTest`.
Each case compiles one dropout of rate 0.25 over a leaf of 256 ones with a fixed site seed, so
the step's output is the mask: 4/3 where `dropout_keeps` keeps an element and 0 elsewhere,
compared within 1e-6.

### ARestoredRunCountIsTheIndexOfTheNextRun

One step runs three times. A second step, with a run count of 0, is restored to 2 and runs
once. Its run count is then 3, its output equals the first step's third output, and that
output is the mask `dropout_keeps` gives for run 2.

### ARestoreAfterARunIsRefusedWithTheCount

After one run, `restore_run_count(4)` fails with `InvalidDescription`, "before its first run"
and "run 1", and the run count stays 1.

### ASecondRestoreIsRefused

`restore_run_count(0)` and then `restore_run_count(5)` succeed. A third call with 7 fails with
`InvalidDescription` and "run 5", and the run count stays 5.

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

## test_conv2d.cpp

Suites: `Unit_Conv2d`.

Tests the 2-D convolution layer's trace, parameters, shape inference and rejections.

This file needs no device. `NN::Conv2d` in `modules/layers/nn/include/SushiAI/nn/conv2d.hpp` declares a weight
and a bias on its first trace and hands the node sequence to `GraphBuilder::conv2d`, so every
case reads a host-side graph, the layer's two parameters or the result of `output_shape()`.
`test_spatial_shapes.cpp` covers the nodes the builder emits; this file covers the layer.

Two `static_assert` lines at file scope require `Conv2d` to be a `Module` and not a
`ModuleContainer`.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### make_conv

`make_conv` builds a layer with a square window and one stride and one padding for both axes.
The stride defaults to 1 and the padding to 0. A case that needs an anisotropic window or a
dilation sets the geometry field afterwards.

### with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

### traced_shape

`traced_shape` traces a copy of a layer over a batch of the given per-sample shape and returns
the result's shape. It takes the layer by value, so a const layer can be traced although
`forward()` declares parameters.

### walk

`walk` returns the parameter walk's output as "name:extents" strings.

### count_op

`count_op` counts the nodes of one operation in the whole graph.

### ForwardOfAPaddedThreeByThreeKeepsTheSpatialExtents

A 3x3 window with padding 1, one input channel and eight filters traces a [4, 28, 28, 1]
input to a [4, 28, 28, 8] value tagged NHWC.

### TheParametersAndTheResultTakeTheInputElementType

With a FLOAT64 input the weight, the bias and the result are FLOAT64.

### TheParameterWalkNamesTheWeightAndTheBias

The walk is empty before the trace. After it the walk yields "weight:[8, 3, 3, 1]" then
"bias:[8]", the names `Linear` uses, and `parameter_elements` is 8 * 3 * 3 * 1 + 8.

### TheWeightOfAnAnisotropicWindowIsOutHeightWidthIn

A 2x4 window with three input channels and five filters gives a weight of shape [5, 2, 4, 3].
The four extents differ, so a weight with two axes swapped fails the comparison.

### ALayerWithABiasTracesOneBiasAdd

The traced graph holds exactly one K_BIAS_ADD node.

### ABiasFreeLayerHasNoBiasEntryAndNoBiasAdd

With `use_bias` false the walk yields the weight alone, the bias is not declared, the graph
holds no K_BIAS_ADD node, and the result shape is still [4, 28, 28, 8].

### TheWeightAndTheBiasRecordTheirFans

For a 3x5 window with four input channels and six filters, both parameters record a fan-in of
3 * 5 * 4 and a fan-out of 3 * 5 * 6. The bias carries the same pair as the weight.

### TheParametersRecordTheirRoles

The weight has role WEIGHT and the bias has role BIAS.

### TracingTwiceDoesNotDuplicateParameters

After a second trace into the same graph the weight and the bias keep their value ids and
`parameter_count` is 2.

### OutputShapeAgreesWithForwardForAStridedWindow

A 3x3 stride-2 unpadded window with sixteen filters maps [9, 12, 3] to [4, 5, 16], and the
traced shape of a batch of two equals that shape with the batch extent in front. The four
cases from here assert the literal and the agreement together.

### OutputShapeAgreesWithForwardForAPaddedWindow

A 5x5 window with padding 2 maps [7, 10, 2] to [7, 10, 4].

### OutputShapeAgreesWithForwardForADilatedStridedWindow

A 3x3 window with stride 2, padding 1 and dilations 2 by 3 maps [11, 13, 2] to [5, 5, 3].

### OutputShapeAgreesWithForwardForAPointwiseWindow

A 1x1 window with six filters maps [5, 7, 4] to [5, 7, 6]. This is the geometry for which
`GraphBuilder::conv2d` emits no K_IM2COL.

### SamePaddedPadsByHalfTheKernelAndDropsTheBias

`Conv2d::same_padded(3, 16, 5, 2)` has 3 input and 16 output channels, a 5x5 kernel, stride 2
on both axes, a pad of 2 on top and left, dilation 1 and `use_bias` false.

### SamePaddedAtStrideOneKeepsTheSpatialExtents

`same_padded(4, 4, 3, 1)` maps [7, 10, 4] to [7, 10, 4] in `output_shape()` and traces
[2, 7, 10, 4] over a batch of two.

### SamePaddedAtStrideOneKeepsTheExtentsForAPointwiseAndAFiveByFiveKernel

Over [9, 6, 2], `same_padded(2, 8, 1, 1)` and `same_padded(2, 8, 5, 1)` both report [9, 6, 8].

### SamePaddedAtStrideTwoHalvesAnEvenExtent

`same_padded(3, 16, 3, 2)` maps [640, 320, 3] to [320, 160, 16] in `output_shape()` and in the
trace.

### SamePaddedTracesNoBias

A traced `same_padded(2, 8, 3, 1)` layer walks as "weight:[8, 3, 3, 2]" alone. Its bias is not
declared and the graph holds no K_BIAS_ADD node.

### ANonZeroBiasInitReachesTheBiasParameter

A layer with `bias_init` set to -4.5 traces a bias whose `init()` is a constant of -4.5 and
whose role is `ParameterRole::Bias`. -4.5 is exact in binary.

### TheDefaultBiasInitIsTheConstantZero

A layer left at the default traces a bias whose `init()` is a constant of 0.0.

### BiasInitLeavesTheWeightFanScaled

With `bias_init` set to 1.25 the weight's `init()` is still `ParameterInit::Kind::FanScaled`
and its role is `WEIGHT`.

### OutputShapeRejectsANonPositiveInputChannelCount

A layer with `in_channels` 0 throws with a message containing "Conv2d", "in=0" and "out=8".

### OutputShapeRejectsANonPositiveOutputChannelCount

A layer with `out_channels` -1 throws with a message containing "Conv2d", "in=3" and "out=-1".

### OutputShapeRejectsAWrongChannelCount

A three-channel layer given [8, 8, 4] throws. The message contains "Conv2d(3, 8)", the shape
"[8, 8, 4]" and the expected form "[H, W, 3]".

### OutputShapeRejectsAFeatureVector

`output_shape([3])` throws with a message containing "Conv2d(3, 8)", "[3]" and "[H, W, 3]".

### OutputShapeRejectsABatchedShape

`output_shape([2, 8, 3, 3])` throws with a message containing the shape and "[H, W, 3]". The
third extent equals `in_channels`, so only the rank check can refuse this shape.

### OutputShapeRejectsAWindowThatDoesNotFit

A 5x5 window over [4, 4, 3] throws with a message containing "Conv2d(3, 8)", "[4, 4, 3]" and
"5x5".

### OutputShapeRejectsAZeroStride

A stride of 0 throws with a message containing "Conv2d(3, 8)" and "stride 0,0".

### OutputShapeRejectsANegativePadding

A padding of -1 throws with a message containing "Conv2d(3, 8)" and "padding -1,-1".

### ForwardRejectsAWrongChannelCount

`forward()` on a [2, 8, 8, 4] input throws with a message containing "Conv2d(3, 8)",
"[8, 8, 4]" and "[H, W, 3]", the three strings OutputShapeRejectsAWrongChannelCount expects.
The weight is not declared afterwards.

### ForwardRejectsAnInputThatIsNotRankFour

`forward()` on a [2, 3] input throws with a message containing "Conv2d(3, 8)", "rank-4" and
"[2, 3]", and the weight is not declared afterwards.

### ForwardRejectsNonPositiveChannelCounts

`forward()` on a layer with `out_channels` 0 throws with a message containing "Conv2d",
"in=3" and "out=0".

### ForwardRejectsAWindowThatDoesNotFit

`forward()` with a 5x5 window over a [2, 4, 4, 3] input throws with a message containing
"Conv2d(3, 8)", "[4, 4, 3]" and "5x5".

### ForwardRejectsAnNCHWInput

The input is [2, 3, 3, 3] converted to NCHW. Every extent is 3, so the permuted shape passes
every check but the one on the layout tag. The message contains "Conv2d" and "NHWC". The node
count and the value count are unchanged and neither parameter is declared, so the layer
refuses before it writes anything to the graph.

### DifferentiatingALossYieldsAGradientForTheWeightAndTheBias

The layer's result is reduced with `spatial_mean` and passed to `NN::CrossEntropyLoss`.
`Autograd::differentiate` returns two gradients, neither is `K_NO_VALUE`, and each has the shape
of its parameter. The case checks that a gradient exists and has the right shape; it computes
no number.

## modules/core/optim/tests/test_cosine_schedule.cpp

Suites: `Unit_CosineSchedule`.

Tests `Optim::CosineSchedule::rate_at`, `Optim::validate_cosine_schedule` and the section
reader and writer of `cosine_schedule_section.hpp`. The schedule most cases start from has the
base rate 1, 10 warm-up updates, 110 updates in total and the final rate 0.1. Rates are compared
within 1e-15.

### TheWarmupIsLinear

The rate at iterations 0, 1, 5 and 10 is 0, 0.1, 0.5 and 1.

### TheDecayFollowsHalfACosine

At iterations 35, 60 and 85 the rate is 0.1 + 0.9 * 0.5 * (1 + c), where c is the square root
of 2 over 2, then 0, then minus the square root of 2 over 2. The middle value is 0.55.

### TheRateStaysAtTheFinalOne

The rate at iterations 110 and 1000000 is 0.1.

### WithoutAWarmupTheFirstRateIsTheBaseRate

With `warmup_iterations` 0 the rate at iteration 0 is 1 and the rate at iteration 1 is below 1.

### ADecayOfNoLengthDividesByNothing

With `total_iterations` equal to `warmup_iterations`, `validate_cosine_schedule` is ok, the rate
at iteration 9 is 0.9 and the rate at iteration 10 is 0.1.

### ANegativeIterationIsFatal

`rate_at(-1)` stops the process with a report containing "iteration 0 or later".

### TheCheckRefusesWhatRateAtCannotUse

`validate_cosine_schedule` fails with `InvalidDescription` for a warm-up of -1 ("warm-up",
"negative"), a total of 5 under a warm-up of 10 ("ends at iteration 5", "warm-up of 10"), a
final rate of -0.1 ("final rate", "negative") and a final rate of 2 ("above the base rate").

### ASectionRoundTripsEveryField

A schedule with the base rate 0.0006 and the final rate 0.00006 goes through
`write_cosine_schedule_section`, `Config::write_json`, `parse_json` and
`read_cosine_schedule_section`. The section's `type` is "cosine" and the four fields read back
equal to the ones written.

### AnEmptySectionReadsAsTheDefaults

`read_cosine_schedule_section` over `{}` returns a base rate of 0, a warm-up of 0, a total of 0
and a final rate of 0.

### TheReaderRefusesAndNamesTheKey

The reader fails with `InvalidDescription` for the type `step` ("schedule 'type'", "'step'",
"a cosine schedule states 'cosine'"), the key `steps` ("schedule states the key 'steps'",
"'total_iterations'"), a `total_iterations` of 1.5 ("schedule 'total_iterations'", "whole
number"), a final rate above the base rate ("schedule: ", "above the base rate") and an array
in place of the section ("schedule must be an object").

### TheWriterRefusesARateThatIsNotFinite

`write_cosine_schedule_section` over a base rate of infinity fails with `InvalidDescription`
and a message containing "schedule 'base_rate'" and "finite".

## modules/vision/importers/tests/test_darknet_config.cpp

Suites: `Unit_DarknetConfig`.

Tests `parse_darknet_config`, which reads a config text into a `DarknetConfig` of one network
section and a list of layer sections, each with its options and line numbers, and
`load_darknet_config`. Every refusal of the parser has the code `InvalidDescription`.

### ParsesEveryLexicalFormOfTheGrammar

A text with `#` and `;` comment lines, a blank line, indented lines, tabs and spaces around
`=`, and spaces inside comma lists parses to a "network" section on line 4 with four options
and to three layers, "convolutional", "route" and "yolo". Each option has its key, its value
and its line. A list value loses its inner spaces ("10,14,23,27"), an empty value stays empty,
and a value keeps a second `=` ("a=b").

### IgnoresCarriageReturnsAtLineEnds

A text with CRLF line ends parses to the same keys and values, with "width" on line 2, the
layer on line 3 and "filters" on line 4.

### AcceptsOneKeyInTwoSections

"width" in the network section and again in a layer section parses, and the layer's value is
"8" on line 4.

### RefusesAKeyBeforeAnySection

A key on line 1 before any section fails with a message holding "test.cfg:1" and
"key 'width'".

### RefusesALineThatIsNeitherSectionNorOption

A line without `=` fails with a message holding "test.cfg:2".

### RefusesAnEmptyKey

A line that starts with `=` fails with a message holding "test.cfg:2".

### RefusesAKeyRepeatedWithinASection

"width" twice in one section fails with a message holding "test.cfg:3", "key 'width'" and
"on line 2".

### RefusesAFirstSectionThatIsNotNetOrNetwork

A text whose first section is `[convolutional]` fails with a message holding "test.cfg:1" and
"[convolutional]".

### RefusesAMalformedSectionLine

A section line without its closing bracket fails with a message holding "test.cfg:1" and
"']'". An empty section name and text after the closing bracket each fail on "test.cfg:1".

### RefusesATextWithoutASection

A text of one comment line fails with a message holding "test.cfg" and "[net]".

### MissingFileGivesFileNotFound

`load_darknet_config` of a path that does not exist fails with `FileNotFound` and names the
path.

## modules/vision/importers/tests/test_darknet_fold.cpp

Suites: `Unit_DarknetFold`.

Tests `fold_darknet_normalisation`, which folds the scales, rolling means and rolling
variances of a `DarknetConvolution` into its weights and biases with the epsilon 0.00001.

### OneFilterFoldMatchesTheFormulaInLongDouble

For one filter of eight weights with the bias 0.75, the scale 1.5, the mean 0.25 and the
variance 0.5, each folded weight equals the weight times 1.5 / sqrt(0.5 + epsilon), and the
bias equals 0.75 - 1.5 * 0.25 / sqrt(0.5 + epsilon). Both are computed in long double and
compared within 1e-6 relative plus 1e-9 absolute. After the fold the scales, means and
variances are empty.

### FoldedAndUnfoldedConvolutionsAgreeOnTheHost

For three filters over a two-channel 5x5 input with values drawn from the seed 20261009, the
folded convolution plus its bias and the unfolded convolution followed by the normalisation
differ by at most 1e-5 at every output pixel.

### FoldingTwiceStopsTheProcess

A second fold of the same convolution stops the process with a message holding
"not normalised".

### NonPositiveVarianceStopsTheProcess

A rolling variance of -1 stops the process with a message holding "positive variance".

## modules/vision/importers/tests/test_darknet_import.cpp

Suites: `Unit_DarknetImport`.

Tests `import_darknet_model` and `import_darknet_files` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_import.hpp`. The sample
`.cfg` text describes a 32 by 32 network of three channels: a normalised convolution of 4
filters, a plain convolution of 18 filters and one `[yolo]` head of one class with the mask
0, 1, 2 over the anchors 4,8, 8,16 and 16,24. The cases parse it with `parse_darknet_config`
and translate it with `translate_darknet_model`.

### KeepsTheRowsAndTheOutputsOfTheTranslation

Imported without weights, the model has the translation's input shape and the same type in
every row. It has one output, named "head.0", at the node the translation's first output
names.

### StatesTheHeadsAsAnAnchorDetectionTask

The task section has the type "anchor_detection" and reads through `read_anchor_task` as one
class and one head on the output "head.0" with `HeadCoding::Logit` and the mask {0, 1, 2}.
It has three anchors; the first has the width 4/32 and the third the height 24/32, within
1e-6.

### HoldsNoTensorWithoutWeights

An import with a null weights pointer has an empty `weights` list.

### NamesEveryTensorAsItsTargetDoes

With weights of the lengths the translation's shapes state, the names of the imported tensors
are the set of the non-empty names in the translation's targets, and there are seven tensors.

### CarriesTheRecipeAndItsNotes

The imported `training` value has the keys `objective`, `optimizer`, `schedule`, and the
import carries five notes.

### AnotherCountOfFilesIsRefused

`import_darknet_files` fails with `InvalidDescription` for no file ("'darknet'", "0 files")
and for three files (".cfg", "3 files").

### AMissingCfgIsRefusedByName

`import_darknet_files` of a `.cfg` path that does not exist fails with `FileNotFound` and the
path's name.

## modules/vision/importers/tests/test_darknet_model.cpp

Suites: `Unit_DarknetModel`, `Unit_DarknetModelState`.

Tests the translation of a config text into a `DarknetModel`: the rows of its model spec, the
row each layer ends on, its heads, its anchor layout, the shape of each convolution and the
state names its weights go to. Most cases translate a 24-layer text with two heads that the
file holds, under `DarknetNormalisation::Separate` or `DarknetNormalisation::Folded`. The
refusal cases use a small 32x32 network. `Unit_DarknetModelState` is an engine fixture.

### SeparateFormHasTheCountedRowsAndSources

The Separate form has 43 rows, the input shape [416, 416, 3] and the `row_of_layer` list the
case states. Rows 0 to 2 are `conv2d`, `batchnorm2d` and `leaky_relu`, with a slope of 0.1, a
`bn_epsilon` of 1e-5 and no convolution bias; row 33 has a bias. Row 34 is a `conv2d` that reads
row 29, and row 38 is a `concat` of rows 37 and 18. Row 23 is a `maxpool2d` of kernel 2, stride 1
and padding 0 with a trailing padding of 1 and 1, and row 3 has a trailing padding of 0 and 0.

### FoldedFormHasNoNormalisationRow

The Folded form has 32 rows, none of the type `batchnorm2d`, and its own `row_of_layer` list.
Row 0 has a bias, row 25 reads row 21 and row 28 reads rows 27 and 13. There are 13 targets,
and the first names "0.weight" and "0.bias" and leaves scale, mean and variance empty.

### HeadsCarryTheirOutputsOptionsAndAnchors

The model has two heads and two outputs, "head.0" on layer 16 and row 33 and "head.1" on layer
23 and row 42, each with 80 classes, the Logit coding and a `scale_xy` of 1. The criterion is
`SuppressionCriterion::IoU` with the exponent 0.6. The six anchors are their pixel sizes
divided by 416, and the masks are {3, 4, 5} and {0, 1, 2}.

### ShapesCountTheFloatsOfTheWeightsFile

The Folded form lists 13 convolution shapes on the layers 0, 2, 4, 6, 8, 10, 12, 13, 14, 15,
18, 21 and 22. A shape is normalised unless it has 255 filters. Summed over the shapes, the
biases, normalisation vectors and weights come to 8858734 floats, and the twelfth convolution
reads 384 channels.

### SeparateFormBuildsAndItsTargetsAreStateEntries

In the Separate form the first target names "0.weight", "1.bias", "1.weight", "1.running_mean"
and "1.running_var", and there are 11 * 5 + 2 * 2 target names. The model builds, and every
target names an entry of its state.

### FoldedFormBuildsAndItsTargetsAreStateEntries

In the Folded form there are 13 * 2 target names. The model builds, and every target names an
entry of its state.

### AnActivatedHeadReadsItsStretchAndItsSuppression

A head with `new_coords=1`, `scale_x_y = 2.0`, `nms_kind=diounms` and `beta_nms=0.6` among its
keys gives one head with the Activated coding, a `scale_xy` of 2 and one class. The criterion is
`SuppressionCriterion::DistanceIoU` with the exponent 0.6. The model has two rows, the second a
`sigmoid`, and the output is published from row 1.

### OnlyDiounmsReadsBetaNMS

With `nms_kind=greedynms` and `beta_nms=0.3` the criterion is `DistanceIoU` and the exponent
stays 0.6. With `nms_kind=diounms` and `beta_nms=0.3` the exponent is 0.3.

### TheLastHeadDecidesTheCriterion

A text whose first head sets `nms_kind=diounms` and whose second head sets nothing has two
heads, the criterion `IoU` and the exponent 0.6.

### AnUnknownSectionIsRefusedNamingItsLine

A `[reorg]` section fails with `Unsupported`, "line 11", "layer 1" and "[reorg]".

### AnUnknownKeyIsRefusedNamingItsLine

Three unknown keys fail with `Unsupported`: `groups` in a convolution ("line 7",
"[convolutional]", "'groups'"), `label_smooth_eps` in a head ("line 16", "[yolo]",
"'label_smooth_eps'") and `crop` in the first section ("line 5", "'crop'").

### UntranslatedValuesAreRefusedNamingTheirLine

Three values fail with `Unsupported`: `letter_box=1` ("line 5"), an upsample `scale=2`
("line 11", "scale 2") and `activation=relu` ("line 6", "'relu'").

### ValuesThatDoNotFitAreRefusedNamingTheirLine

Five texts fail with `InvalidDescription`: `filters=many` ("line 6", "'filters'", "'many'"), a
route to a later layer ("line 12", "names layer 1"), a head over 12 channels that needs 18
("line 8", "reads 18 channels", "has 12"), a mask that names anchor 3 of three ("line 12",
"names anchor 3"), and a text with no head ("no [yolo] section").

## modules/vision/importers/tests/test_darknet_recipe.cpp

Suites: `Unit_DarknetRecipe`.

Tests `write_darknet_recipe` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_recipe.hpp`. Each case
parses `.cfg` text, reads it with `read_darknet_training` and writes the recipe. The sample
`[net]` section states `batch=8`, `subdivisions=2`, `momentum=0.9`, `decay=0.0005`,
`learning_rate=0.001`, `burn_in=100`, `max_batches=2000`, `policy=steps`, `steps=1600,1800`
and `scales=.1,.1`. A `[yolo]` section states a mask, one class, `jitter=.3` and an
`ignore_thresh`.

### WritesTheSolverAsSGD

The `optimizer` section has three keys: the type "sgd", a `momentum` of 0.9 and a
`weight_decay` of 0.0005, each within 1e-12. It has no `learning_rate`.

### WritesTheRateAsAStepSchedule

The `schedule` section reads through `read_step_schedule_section` as a base rate of 0.001, 100
warm-up iterations, the steps 1600 and 1800 and the scales 0.1 and 0.1.

### WritesTheSharedHeadOptionsAsTheAnchorObjective

With two heads of `ignore_thresh=.7`, the `objective` section has the type "anchor" and reads
through `read_head_training_section` with an ignore threshold of 0.7 and
`BoxGradient::DarknetCompatible`.

### HoldsTheObjectiveTheOptimizerAndTheSchedule

The recipe's `training` value has the keys `objective`, `optimizer`, `schedule`.

### NotesEveryKeyWithoutAPlace

The recipe has four notes. One holds "line 2", "batch=8", "[net]" and "batch_size"; one
"line 3", "subdivisions=2" and "[net]"; one "line 8", "max_batches=2000" and "epochs"; one
"line 15", "jitter=.3" and "[yolo]".

### LeavesNoNoteWhenEveryKeyHasAPlace

A `.cfg` of `learning_rate=0.01` and a head of `mask=0` and `classes=1` gives a recipe with no
note.

### HeadsThatDifferAreRefused

Heads with `ignore_thresh` .7 and .5 fail with `InvalidDescription`, "head 0 (line 12)",
"head 1 (line 17)", "'ignore_threshold'" and "one set".

### AFileWithoutAHeadIsRefused

The `[net]` section alone fails with `InvalidDescription`, "[yolo]" and "no objective".

## modules/vision/importers/tests/test_darknet_route_groups.cpp

Suites: `Unit_DarknetRouteGroups`.

Tests how the translation of a config text turns a `[route]` section with `groups` and
`group_id` into model rows. Each case starts from a network whose first convolution has 64
filters and ends with a head, and reads the rows, their sources and the channels of the
convolutions of the `DarknetModel`.

### OneSourceGroupIsOneSliceRow

A route of the layer before it with `groups=2` and `group_id=1` gives three rows: a `conv2d`, a
`slice_channels` that reads row 0 with the offset 32 and the count 32, and a row that reads the
slice. The second convolution reads 32 channels.

### EverySourceIsSlicedBeforeTheJoin

A grouped route of two layers gives six rows. Rows 2 and 3 are `slice_channels` rows over rows
1 and 0, each with the offset 32 and the count 32. Row 4 is a `concat` of rows 2 and 3, and row
5 reads it. The third convolution reads 64 channels.

### UngroupedSourceAddsNoRow

A route of one layer without groups gives two rows, the second a `conv2d` that reads row 0, and
the second convolution reads 64 channels.

### UnequalSplitsAndGroupsOutOfRangeAreRefused

Four routes fail with `InvalidDescription`: `groups=3` over 64 channels ("line 12", "groups 3",
"64 channels"), `group_id=2` with two groups ("line 11", "group_id 2"), `groups=0` ("line 11",
"groups 0"), and a `group_id` of 1 without `groups` ("line 11", "group_id 1").

## modules/vision/importers/tests/test_darknet_shortcut.cpp

Suites: `Unit_DarknetShortcut`.

Tests how the translation of a config text turns a `[shortcut]` section into a model row. Each
case starts from an 8x8 three-channel network with two convolutions.

### NegativeFromAddsTheLayerBeforeAndTheNamedLayer

`from=-2` gives four rows. Row 2 has the type "add", no attribute key, and the sources 1 and 0.

### AbsoluteFromAddsTheSameTwoRows

`from=0` with `activation=linear` gives the same "add" row with the sources 1 and 0.

### FromAboveTheShortcutIsRefusedNamingItsLine

`from=2` fails with `InvalidDescription`, "line 18" and "only layers 0 to 1".

### DifferentChannelCountsAreRefusedNamingItsLine

A shortcut over convolutions of 4 and 5 filters fails with `InvalidDescription`, "line 18" and
"share their shape".

### NonLinearActivationIsRefusedNamingItsLine

`activation=leaky` fails with `Unsupported`, "line 19" and "'leaky'".

### WeightsTypeIsRefusedNamingItsLine

`weights_type=per_feature` fails with `Unsupported`, "line 19" and "weights_type".

### MissingFromIsRefusedNamingTheSection

A shortcut without `from` fails with `InvalidDescription` and "no 'from'".

### MoreThanOneFromIsRefused

`from=-1,-2` fails with `InvalidDescription` and "names one layer".

## modules/vision/importers/tests/test_darknet_tensors.cpp

Suites: `Unit_DarknetTensors`.

Tests `list_darknet_tensors` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_tensors.hpp`. The weights
are a plain convolution of 2 filters over 3 channels with a 2 by 2 kernel, whose weights are
0 to 23, and a normalised convolution of 2 filters over 1 channel with a 1 by 1 kernel. The
targets name `0.weight` and `0.bias` for the first, and `1.weight`, `2.bias`, `2.weight`,
`2.running_mean` and `2.running_var` for the second.

### ListsOneTensorPerNamedTarget

Seven tensors are listed. Each is `Float32` and its payload has the byte count its record
states.

### WeightsMoveToKernelLast

The tensor `0.weight` has the shape [2, 2, 2, 3]. Its 24 values open with 0, 4, 8, 1 and end
with 23.

### EveryVectorLandsUnderItsTarget

`2.weight` has the shape [2] and holds the scales 2, 3. `2.bias` holds 1.5, -1.5,
`2.running_mean` 0.25, 0.75 and `2.running_var` 4, 9. No tensor is named `0.running_mean`.

### AVectorOfTheWrongLengthIsRefused

With a third scale on the second convolution, the listing fails with `Incompatible`,
"layer 1", "'2.weight'" and "3 values".

### AnEntryNamedTwiceIsRefused

With the second target's bias named `0.bias`, the listing fails with `Incompatible`,
"layer 1", "'0.bias'" and "already fills".

### AShapeThatIsNotPositiveIsRefused

With 0 channels in the first shape, the listing fails with `InvalidDescription`, "layer 0"
and "channels 0".

### MismatchedCountsAreRefused

With one target for two convolutions, the listing fails with `Incompatible`,
"2 convolutions" and "the targets 1".

## modules/vision/importers/tests/test_darknet_training.cpp

Suites: `Unit_DarknetTraining`.

Tests `read_darknet_training` over texts parsed with `parse_darknet_config`. A head's options
are checked as seven values: box loss, objectness target, box weight, object weight,
`max_delta`, match threshold and ignore threshold. Floats are compared to 1e-6 and doubles to
1e-12.

### ReadsTheV3HeadSettingsWhenTheTrainingKeysAreAbsent

A head with only `ignore_thresh=0.7` and `truth_thresh=1` among the training keys reads as
`BoxLoss::SquaredError`, `ObjectnessTarget::One`, a box weight of 0.75, an object weight of 1,
a `max_delta` of 0, a match threshold of 1 and an ignore threshold of 0.7.

### ReadsTheV4TinyHeadSettings

With `iou_loss=ciou` and `iou_normalizer=0.07`, the head reads as `BoxLoss::CompleteIoU` with a
box weight of 0.07. The other five values are as in the case above.

### ReadsTheV4HeadSettings

With `iou_thresh=0.213` and `max_delta=5` added, the head reads a match threshold of 0.213 and
a `max_delta` of 5.

### ReadsTheV7HeadSettings

With `objectness_smooth=1`, `iou_normalizer=0.05`, `iou_thresh=0.2` and `max_delta=2`, the head
reads `ObjectnessTarget::BestIoU`, a box weight of 0.05, a match threshold of 0.2 and a
`max_delta` of 2. `cls_normalizer=0.5` is accepted.

### KeepsTheSolverDefaultsWhenTheKeysAreAbsent

With no solver key, the solver has a learning rate of 0.001, a momentum of 0.9, a decay of
0.0005, a `burn_in` of 0, no steps and no scales, a batch of 1, 1 subdivision and a
`max_batches` of 0.

### ReadsEverySolverKeyTheFileSpells

Batch 64, subdivisions 16, momentum 0.949, decay 0.0005, learning rate 0.00261, `burn_in` 1000,
`max_batches` 500200, the steps 400000 and 450000 and the scales 0.1 and 0.1 are each read to
the value the text states.

### KeepsTheHeadsInFileOrderAndSkipsOtherSections

A text with a convolution, a head, a route and a second head gives two heads, with the box
weights 0.07 and 0.05 in that order.

### ReadsMSEAndObjectnessSmoothZeroAsTheDefaults

`iou_loss=mse` with `objectness_smooth=0` gives the seven values of a head without training
keys.

### SelectsTheDarknetCompatibleBoxGradientForEveryHead

A head with `iou_loss=ciou` and a head with `iou_loss=mse` both read
`BoxGradient::DarknetCompatible`. A default `HeadTrainingOptions` has `BoxGradient::Analytic`.

### AcceptsTruthThreshOneAndLabelSmoothZero

`truth_thresh=1.0` with `label_smooth_eps=0` reads as one head.

### RefusesAPolicyOtherThanSteps

`policy=exp` fails with `Unsupported`, "line 2" and "policy".

### RefusesATruthThreshOtherThanOne

`truth_thresh=0.9` fails with `Unsupported`, "line 3" and "truth_thresh".

### RefusesANonZeroLabelSmooth

`label_smooth_eps=0.1` fails with `Unsupported`, "line 3" and "label_smooth_eps".

### RefusesAnIoULossOtherThanMSEAndCIoU

`iou_loss=giou` fails with `Unsupported`, "line 3" and "iou_loss".

### RefusesAClsNormalizerThatIsNotANumber

`cls_normalizer=half` fails with `InvalidDescription`, "line 3" and "cls_normalizer".

### RefusesAnObjectnessSmoothOtherThanZeroOrOne

`objectness_smooth=2` fails with `InvalidDescription`, "line 3" and "objectness_smooth".

### RefusesAHeadWhoseMatchThresholdIsZero

`iou_thresh=0` fails with `InvalidDescription`, "line 2", the head section's line, and
"match threshold".

### RefusesAHeadWhoseIgnoreThresholdIsAboveOne

`ignore_thresh=1.5` fails with `InvalidDescription`, "line 2" and "ignore threshold".

### RefusesALearningRateThatIsNotANumber

`learning_rate=abc` fails with `InvalidDescription`, "line 2", "learning_rate" and "abc".

### RefusesANonFiniteLearningRate

`learning_rate=nan` fails with `InvalidDescription`, "line 2" and "nan".

### RefusesABatchBelowOne

`batch=0` fails with `InvalidDescription` and a message holding "batch".

### RefusesAnEmptyEntryInAList

`steps=1,,2` fails with `InvalidDescription`, "line 2" and "steps".

### RefusesStepsAndScalesOfDifferentLengths

Two steps with one scale fail with `InvalidDescription`, "line 1", "'steps' holds 2" and
"'scales' holds 1".

## modules/vision/importers/tests/test_darknet_weights.cpp

Suites: `Unit_DarknetWeights`.

Tests `read_darknet_header` over byte streams built in memory and `read_darknet_weights` over
files the cases write under the system's temporary directory. The sample file holds a 20-byte
header and index-valued floats for two convolutions, a plain one on layer 0 and a normalised
one on layer 2.

### ReadsAVersionZeroTwoHeaderWithA64BitSeen

A header of major 0, minor 2 and revision 5 reads an eight-byte seen count, and the stream
stands at byte 20.

### ReadsAVersionZeroOneHeaderWithA32BitSeen

A header of major 0 and minor 1 reads a four-byte seen count, and the stream stands at byte
16.

### ReadsAMajorOneHeaderWithA64BitSeen

A header of major 1 and minor 0 reads an eight-byte seen count, and the stream stands at byte
20.

### AcceptsAMajorAtTheLimitOfOneThousand

A header of major 1000 is read.

### RefusesAMajorAboveOneThousandAsUnsupported

A major of 1001 fails with `Unsupported` and a message holding "1001".

### RefusesAMinorAboveOneThousandAsUnsupported

A minor of 1001 fails with `Unsupported` and a message holding "1001".

### RefusesAHeaderThatEndsEarly

A stream that ends two bytes into the revision fails with `InvalidFile` and a message holding
"revision".

### ReadsEveryVectorInFileOrder

With floats 0 to 21 in the file, the plain convolution's biases start at 0 and its weights at
2. The normalised one's biases start at 4, its scales at 7, its rolling means at 10, its
rolling variances at 13 and its weights at 16. The header reads major 0 and minor 2.

### KeepsTheBatchNormVectorsOnlyForNormalisedLayers

The plain convolution has empty scales, means and variances and two weights. The normalised
one has three of each vector and six weights.

### DecodesEveryBitPatternAsItIs

Biases of negative zero and the smallest subnormal, and weights of a NaN with payload 1 and of
1.0, come back with the bits 0x80000000, 0x00000001, 0x7FC00001 and 0x3F800000.

### RefusesOneFloatShortWithTheLayerAndBothByteCounts

A file one float short fails with `InvalidFile`, "104 bytes" and
"convolution layer 2 needs bytes up to 108".

### RefusesOneFloatOverWithTheLayerAndBothByteCounts

A file one float over fails with `InvalidFile`, "112 bytes", "108 bytes" and
"4 bytes are left over after convolution layer 2".

### RefusesBytesAfterAHeaderWithNoConvolution

A file with one float after its header and an empty shape list fails with `InvalidFile`,
"no convolution is listed" and "4 bytes are left over".

### RefusesAShapeThatIsNotPositive

A shape with zero filters fails with `InvalidDescription`, "convolution layer 0" and
"filters 0".

### RefusesAShapeWhoseCountOverflows

A shape of 2^40 filters over 2^40 channels fails with `InvalidDescription`,
"convolution layer 0" and "too large to read".

### ReportsAMissingFileAsFileNotFound

A path that does not exist fails with `FileNotFound` and "could not open".

## modules/core/data/tests/test_dataset_registry.cpp

Suites: `Unit_DatasetRegistry`.

Tests `Data::DatasetRegistry` and `Data::register_datasets`. The file uses two rows: "rings",
as `register_datasets` adds it, and "moons", a type no module registers.

### AddKeepsInsertionOrder

After `add` of the "rings" row and then the "moons" row, `rows()` has two entries, "rings"
first and "moons" second.

### FindReturnsNullForAnUnknownType

`find("rings")` is null on an empty registry. After the "rings" row is added, `find("coco")` is
null and `find("rings")` returns a row whose type is "rings".

### ATypeAddedTwiceIsRefused

A second `add` of the "rings" row fails with `InvalidDescription`, and the message holds
"'rings'" and "already registered".

### RegisterDatasetsAddsRingsOnly

After `Data::register_datasets`, `rows()` has one entry of type "rings" and `find("mnist")` is
null.

## modules/vision/datasets/tests/test_datasets_cifar.cpp

Suites: `Unit_DatasetsCifar`.

Tests `read_cifar` in `modules/vision/datasets/include/SushiAI/vision/datasets/cifar.hpp` over
batch files the cases write under the system's temporary directory. A record is its label
bytes followed by three planes of 32 by 32 bytes, each plane filled with one level.

### TheCIFAR10TrainingSplitListsItsFiveFilesInOrder

With `data_batch_1.bin` to `data_batch_5.bin` each holding one record labelled with its batch
number, the "train" split of CIFAR-10 has 10 classes, five files and five samples. Sample i
names file i, the file is `data_batch_<i + 1>.bin` and the class is i + 1.

### ASampleRefersToThePlanesAfterItsLabel

A `test_batch.bin` of two records gives the classes 3 and 9. The second sample's stored pixels
start at byte 3074, are `K_CIFAR_SIDE` square with three channels and are `PlaneLayout::Planar`.
`read_stored_pixels` over them gives 3072 bytes whose sampled pixels hold 40, 50, 60.

### CIFAR100TakesTheFineOrTheCoarseLabel

A `train.bin` record with the label bytes 19 and 87 reads as class 87 of 100 under
`CIFARVariant::CIFAR100Fine`, with its pixels at offset 2, and as class 19 of 20 under
`CIFARVariant::CIFAR100Coarse`.

### ALabelOutsideTheClassesNamesItsRecord

A CIFAR-10 file whose second record has the label 10 fails with `InvalidFile`,
"test_batch.bin", "record 2 of 2" and "label 10". A CIFAR-100 record with the coarse label 20
fails with `InvalidFile`, "test.bin", "record 1 of 1" and "label 20".

### AFileThatIsNotWholeRecordsIsRefused

A record one byte short fails with `InvalidFile`, "test_batch.bin", "3072 bytes" and
"3073 bytes". An empty file fails with `InvalidFile`, "test_batch.bin" and "0 bytes".

### AnAbsentFileIsNotFound

With only `data_batch_1.bin` present, the "train" split fails with `FileNotFound` and
"data_batch_2.bin".

### AnotherSplitIsRefused

The split "val" fails with `InvalidDescription` and "'val'".

## modules/vision/datasets/tests/test_datasets_coco_json.cpp

Suites: `Unit_DatasetsCocoJson`.

Tests `read_coco_json` in
`modules/vision/datasets/include/SushiAI/vision/datasets/coco_json.hpp` over files the cases
write under the system's temporary directory. Unless a case says otherwise the annotation file
is `annotations/instances_train.json` beside a folder `train`. It lists `b.jpg` with id 9 and
`sub/a.jpg` with id 4, both 200 by 100 pixels, and the categories brush (90), person (1) and
cat (17).

### BoxesAndDenseClassesAreRead

The class names come out as person, cat, brush, and the samples as `train/b.jpg` with no object
and `train/sub/a.jpg`. Of that image's four annotations the crowd one and the one of width 0
are left out; the other two have the classes 2 and 0 and no polygon. The box [10, 20, 50, 20]
reads as the centre (35/200, 30/100) and the size (50/200, 20/100), and the sample has no class
index.

### TheFileAndTheFolderMayBeNamed

With `annotation_file` "labels.json" and `image_folder` "pictures", the first sample's file is
`pictures/b.jpg`.

### PolygonsAreKeptWhenAskedFor

By default an annotation with two polygon parts keeps no polygon. With
`COCOSegmentation::Polygons` the image has one object, whose polygon is the part of four
vertices, and its third vertex is (100/200, 50/100). The crowd annotation with a run-length
mask is left out.

### ARunLengthMaskIsRefusedWhenPolygonsAreAskedFor

A file whose second annotation holds a run-length mask reads as two objects by default. With
`COCOSegmentation::Polygons` it fails with `InvalidFile`, "instances_train.json",
"read_coco_json" and "annotation 1 holds a run-length mask".

### AMalformedPolygonIsRefused

A polygon part of seven numbers fails with `InvalidFile`, "annotation 0",
"'segmentation' part" and "at least 3 pairs".

### AnUnknownImageOrCategoryNamesTheAnnotation

An annotation of image 5 fails with `InvalidFile`, "instances_train.json", "read_coco_json"
and "annotation 1 has the 'image_id' 5, which 'images' lacks". A crowd annotation of category
2 fails with "annotation 0 has the 'category_id' 2, which 'categories' lacks".

### AMalformedBoxNamesTheAnnotation

A `bbox` of three numbers, and one whose fourth entry is a string, each fail with
`InvalidFile` and "annotation 0 has no 'bbox' of 4 numbers". An `image_id` given as a string
fails with "annotation 0 has no whole number 'image_id'".

### TheImagesAndTheCategoriesAreChecked

Each of these fails with `InvalidFile`: two categories of id 3
("category 0 and category 2 share the id 3"), an image of width 0
("image 1 is 2 by 0 pixels") and an image without `file_name`
("image 0 has no string 'file_name'").

### TheFileMustExistAndHoldItsThreeLists

Without the folder `train` the read fails with `FileNotFound`, "train", "read_coco_json" and
"no such image folder". Without the annotation file it fails with `FileNotFound`,
"annotations/instances_train.json" and "no such file". A file cut after `"images": [` fails
with `InvalidFile` and "not valid JSON", and a file without `categories` with
"the file has no list 'categories'".

### TheListsMayComeInAnyOrder

One file states `images`, `categories` and `annotations` in that order. A second states an
`info` object, then `annotations`, `categories` and `images`. With `COCOSegmentation::Polygons`
both give the same class names, files, classes, box centre x, box height and polygon vertices.
The second sample has the class 2 and a polygon of four vertices, and the first has no polygon.

### AListStatedTwiceIsRefused

A file that states `images` a second time fails with `InvalidFile`, "instances_train.json",
"read_coco_json" and "states 'images' twice".

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

## modules/vision/datasets/tests/test_datasets_csv.cpp

Suites: `Unit_DatasetsCsv`.

Tests `read_csv` in `modules/vision/datasets/include/SushiAI/vision/datasets/csv.hpp` over
files the cases write under the system's temporary directory. Unless a case names another
file, the reader takes `train.csv` with the default columns.

### TheLastColumnIsTheTargetByDefault

A file with the header `a,b,y` and two records gives one file, two samples and a class count
of 0. The table has two features and one target, the columns a, b, y and the values 1, 2, 3,
-0.5, 4, 10; `features(1)[1]` is 4 and `targets(1)[0]` is 10.

### NamedColumnsAreTakenInTheOrderNamed

Over `data/set.csv` with the header `id,x,y,z`, the features z, x and the targets y, z fail
with `InvalidDescription`, "set.csv", "read_csv" and
"the column 'z' is named as a feature and as a target". With the target y alone the columns
are z, x, y and the values 3, 1, 2, 6, 4, 5.

### QuotedFieldsAndCarriageReturnsAreRead

The file opens with a byte order mark, ends its lines with CRLF and has an empty line. Its
header holds a quoted name with a doubled quote and a comma, a quoted name with a line break,
and ` y ` between blanks. With the first column as the feature, the columns are
`the "x", one` and `y`, and the values are 1.5, 2, 3, 4.

### ANonNumericCellNamesItsLineAndColumn

The cell `five` fails with `InvalidFile`, "train.csv", "line 3", "column 2 ('b')", "read_csv"
and "the cell 'five'". An empty target cell fails with "line 2", "column 3 ('y')" and
"the cell ''". After a record whose quoted field spans lines 2 and 3, the cell `inf` fails
with "line 4", "column 3 ('y')" and "the cell 'inf'".

### ARecordOfAnotherFieldCountIsRefused

A record of two fields on line 4, after an empty line, fails with `InvalidFile`, "train.csv",
"line 4", "read_csv" and "the record has 2 fields; the header has 3".

### MalformedQuotesAreRefused

Each of these fails with `InvalidFile` and "line 2": a quote left open ("column 2",
"the quoted field is not closed"), a letter after a closing quote ("column 1",
"text follows the closing quote") and a quote at the end of an unquoted field ("column 2",
"a quote stands inside a field").

### AColumnTheHeaderLacksIsRefused

Over the header `a,b,y`, the target z fails with `InvalidDescription`, "train.csv",
"no target column 'z'" and "'a', 'b', 'y'". The feature c fails with `InvalidDescription` and
"no feature column 'c'".

### AHeaderMustNameItsColumnsOnceAndLeaveAFeature

The header `a, b ,a` on line 2 fails with `InvalidFile`, "line 2", "column 3" and
"the name 'a' is also the name of column 1". A file of the single column y fails with
`InvalidDescription` and "no column is left as a feature".

### AFileWithoutRecordsIsRefused

With the split "val" and no file, the read fails with `FileNotFound`, "val.csv" and
"read_csv". A `val.csv` of two empty lines fails with `InvalidFile` and
"holds no header row", and one with a header alone with "holds no record below its header".

## modules/vision/datasets/tests/test_datasets_fit_mask.cpp

Suites: `Unit_DatasetsFitMask`, `Unit_DatasetsLoadSampleMask`.

Tests `fit_mask` in `modules/vision/datasets/include/SushiAI/vision/datasets/fit_mask.hpp` and
`load_sample_mask` in
`modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_mask.hpp`. The
`load_sample_mask` cases that read a file write it under the system's temporary directory.

### AResizeRepeatsTheNearestClass

A 2 by 2 mask of 0, 1, 2, 3 resized to 4 by 4 has one channel and holds each class in a 2 by 2
block: the rows 0 0 1 1, 0 0 1 1, 2 2 3 3, 2 2 3 3.

### AReductionTakesThePixelUnderEachCentre

A 4 by 4 mask of the bytes 0 to 15 resized to 2 by 2 holds 5, 7, 13, 15.

### ALetterboxPadsWithTheVoidByte

A mask of 2 rows and 4 columns letterboxed to 4 by 4 keeps its two rows in the middle, with a
row of `K_MASK_VOID` above and below.

### ALetterboxThatReducesSamplesTheNearestPixel

A mask of 4 rows and 8 columns holding the bytes 0 to 31, letterboxed to 4 by 4, has the rows
9, 11, 13, 15 and 25, 27, 29, 31 between two rows of `K_MASK_VOID`.

### AMaskMustMatchItsImage

`fit_mask` fails with `Incompatible` and "fit_mask" for a mask of three channels
("the mask has 3 channels") and for a 2 by 2 mask whose image is 2 by 4
("the mask is 2 by 2 pixels; its image is 2 by 4"). A target size of 0 by 4 fails with
`InvalidDescription`, "fit_mask" and "0 by 4 pixels".

### TheMaskFileIsDecoded

The sample names file 1 of the list as its mask, a raw image file of 2 rows and 3 columns.
`load_sample_mask` gives an image of that size and one channel holding the bytes written: 0,
3, 255, 1, 2, 0.

### ASampleMustNameAMaskOfTheList

Over a list of one file and one sample, `load_sample_mask` fails with `InvalidDescription`
for sample 1 ("load_sample_mask", "sample 1 is outside a list of 1 samples"), for a sample
without a mask ("sample 0 has no mask") and for the mask index 4
("sample 0 names the mask file 4 of a list of 1 files").

### AnAbsentMaskFileIsReported

A mask file that does not exist fails with `FileNotFound` and "absent.png".

## modules/vision/datasets/tests/test_datasets_fit_sample.cpp

Suites: `Unit_DatasetsFitSample`.

Tests `fit_sample` in `modules/vision/datasets/include/SushiAI/vision/datasets/fit_sample.hpp`.
A box corner may differ from its expected value by 1e-4 pixels.

### ResizeScalesTheLabelsWithEachAxis

A gray image of 2 rows and 4 columns at level 100, resized to 4 by 4, gives 16 bytes of 100
in one channel. The fitted sample keeps the file index 6 and the class index 3. Its one
object keeps the class 1, and the box of centre (0.5, 0.5) and size (0.5, 0.5) becomes the
corners (1, 1) and (3, 3).

### LetterboxPadsAndMovesTheLabels

The same shape at level 200, letterboxed to 4 by 4, has rows 1 and 2 at 200 and rows 0 and 3
at 114. The sample has no class index, and the box of centre (0.5, 0.5), width 0.5 and height
1.0 becomes the corners (1, 1) and (3, 3).

### AnObjectWithoutAreaIsRemoved

Of two objects, the one of class 2 with a width of 0 is removed and the one of class 5 is
kept.

### AnInputSizeOutsideTheBoundsIsRefused

A target size of 0 by 4 fails with `InvalidDescription`, "fit_sample" and "0 by 4".

### ASourceOfNoPixelsIsAFailure

A default `ImageView` as the source fails with `InvalidFile` and "fit_sample".

## modules/vision/datasets/tests/test_datasets_idx.cpp

Suites: `Unit_DatasetsIdx`.

Tests `read_idx` in `modules/vision/datasets/include/SushiAI/vision/datasets/idx.hpp` over IDX
files the cases write under the system's temporary directory. The training split of the first
two cases is three images of 2 rows and 3 columns holding the bytes 100 to 117, with the
labels 4, 0 and 2.

### ASampleHasItsLabelAndThePlaceOfItsPixels

The list has one file, `train-images-idx3-ubyte`, no class names, a class count of 5 and three
samples. Sample i names file 0, has the class of its label and no object, and its stored
pixels start at byte 16 + 6i, are 2 by 3 with one channel and are `PlaneLayout::Interleaved`.

### ASamplesPixelsReadBackFromTheImageFile

`read_stored_pixels` over the third sample gives the bytes 112 to 117.

### TheTestSplitReadsTheT10kFiles

With the split "test", `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` are read: one
sample of class 7 and a class count of 8.

### NamedFilesAreReadForAnySplit

With the split "balanced" and the files `images.idx` and `labels.idx` named, two samples are
read. The image file has four dimensions, the last being 3; the second sample has three
channels and its pixels start at byte 23.

### ASplitWithoutAFileNameIsRefused

The split "validation" with no file named fails with `InvalidDescription` and "validation".

### AnAbsentFileIsNotFound

With the image file alone present, the read fails with `FileNotFound` and
"train-labels-idx1-ubyte".

### CountsThatDifferAreRefused

Two images with three labels fail with `InvalidFile`, "train-images-idx3-ubyte", "2 images",
"train-labels-idx1-ubyte" and "3 labels".

### AnImageFileOfAnotherFormIsRefused

An image file of two dimensions fails with `InvalidFile` and "2 dimensions", and one of
element code 0x0C with `InvalidFile` and "code 12". One whose fourth dimension is 5 fails with
`Unsupported` and "5 channels". Each message names "train-images-idx3-ubyte".

### AnImageFileWithoutAnImageIsRefused

An image file whose first extent is 0 fails with `InvalidFile`, "train-images-idx3-ubyte" and
"no image".

## modules/vision/datasets/tests/test_datasets_idx_file.cpp

Suites: `Unit_DatasetsIdxFile`.

Tests `idx_element_bytes`, `read_idx_header` and `read_idx_payload` in
`modules/vision/datasets/include/SushiAI/vision/datasets/idx_file.hpp` over files the cases
write under the system's temporary directory.

### ElementBytesFollowTheFormat

`idx_element_bytes` is 1 for `UInt8` and `Int8`, 2 for `Int16`, 4 for `Int32` and `Float32`,
and 8 for `Float64`.

### AHeaderGivesTheElementTheExtentsAndThePayloadSize

A file of element code 0x08 with the extents 2, 3, 4 and 24 payload bytes reads as `UInt8`,
those extents, a header of 16 bytes and a payload of 24 bytes.

### AnExtentIsReadMostSignificantByteFirst

A file with the single extent 258 reads that extent as 258.

### ThePayloadSizeCountsTheElementBytes

A file of element code 0x0C with the extent 3 reads as `Int32` with a payload of 12 bytes.

### ThePayloadIsEveryByteAfterTheHeader

`read_idx_payload` over a file of five bytes gives 7, 0, 9, 3, 1.

### AnAbsentFileIsNotFound

`read_idx_header` of a path that does not exist fails with `FileNotFound` and "absent.idx".

### AMalformedHeaderIsRefused

`read_idx_header` fails with `InvalidFile` and the file's name for a file of three bytes
("leading bytes"), a non-zero second byte ("two zero bytes"), the element code 10
("element code 10"), a rank of 0 ("no dimension") and a second extent cut after one byte
("dimension 1 of 2").

### AFileOfAnotherSizeThanItsHeaderDeclaresIsRefused

A file with the extents 2, 2 and three payload bytes fails with `InvalidFile`, "short.idx",
"declares 4 bytes" and "holds 15 bytes". A file with the extent 2 and three payload bytes
fails with "long.idx", "declares 2 bytes" and "holds 11 bytes".

### ExtentsThatOverflowAreRefused

A file of element code 0x0E with three extents of 0xFFFFFFFF fails with `InvalidFile`,
"huge.idx" and "overflow".

## modules/vision/datasets/tests/test_datasets_image_folder.cpp

Suites: `Unit_DatasetsImageFolder`.

Tests `read_image_folder` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_folder.hpp` and
`has_image_extension` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_extensions.hpp`. The folder
cases write one-byte files under the system's temporary directory.

### ImageExtensionsIgnoreCase

`has_image_extension` is true for `a.jpg`, `a.JPEG`, `dir/a.PNG` and `a.saim`, and false for
`a.txt`, `a.jpg.bak` and `jpg`.

### ClassesAreTheSortedFolderNames

The folders zebra, ant and moth give the class names ant, moth, zebra and a class count of 3.
Sample i of the three has the class i, no stored pixels and no object.

### AClassListsItsImageFilesSortedAtAnyDepth

The list holds four files in the order `cat/a.JPG`, `cat/b.png`, `cat/more/c.saim`,
`dog/a.png`, and leaves `cat/notes.txt` out. The third sample has the class 0 and the fourth
the class 1.

### ASplitNamesAFolderOfTheRoot

The split "val" gives the classes no and yes and two samples. The file `val/stray.png` and the
folder `train` add nothing.

### AFolderWithoutAnImageIsStillAClass

With an empty folder `a` and one image in `b`, the class count is 2 and the one sample has the
class 1.

### AnAbsentSplitIsNotFound

The split "train" in an empty root fails with `FileNotFound` and "train".

### ASplitWithoutAClassOrAnImageIsRefused

A split holding only a file fails with `InvalidFile`, "flat" and "no class folder". A split
whose one class folder holds only a text file fails with `InvalidFile`, "text" and
"no image file".

## modules/vision/datasets/tests/test_datasets_load_sample_image.cpp

Suites: `Unit_DatasetsLoadSampleImage`.

Tests `load_sample_image` in
`modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_image.hpp`. The cases
that read a file write it under the system's temporary directory.

### StoredPixelsAreReadFromTheirOffset

Over a file of the bytes 0 to 9, a sample whose stored pixels are 2 by 2 with one channel at
offset 2 loads as a 2 by 2 image of one channel holding 2, 3, 4, 5.

### AnImageFileIsDecoded

A sample without stored pixels, whose file is a raw RGB image of 2 rows and 3 columns written
by `write_raw_image_file`, loads as an image of that size and three channels holding the bytes
written.

### ASampleOutsideTheListIsRefused

Sample 3 of a list of one sample fails with `InvalidDescription`, "load_sample_image",
"sample 3" and "1 samples".

### AFileIndexOutsideTheListIsRefused

A sample naming file 5 of a list of one file fails with `InvalidDescription`,
"load_sample_image" and "file 5".

### AnAbsentFileIsNamed

An image file that does not exist fails with `FileNotFound` and "absent.saim".

## modules/vision/datasets/tests/test_datasets_mask_png.cpp

Suites: `Unit_DatasetsMaskPng`.

Tests `read_mask_png` in `modules/vision/datasets/include/SushiAI/vision/datasets/mask_png.hpp`
over one-byte files the cases write under the system's temporary directory. Unless a case says
otherwise the options name four classes and the split "train", so the images are under
`images/train` and the masks under `masks/train`.

### EachImageRefersToTheMaskOfItsStem

The list has a class count of 4, no class names, two samples and four files. The first sample
is `images/train/a/c.jpeg` with the mask `masks/train/a/c.png`, and has no class index and no
object. The second is `images/train/b.jpg` with `masks/train/b.png`. The files `notes.md` and
`index.txt` are not listed.

### TheFoldersMayBeNamedAndTheSplitLeftOut

With an empty split, `image_folder` "leftImg8bit" and `mask_folder` "gtFine", the one sample
is `leftImg8bit/a.png` with the mask `gtFine/a.png`.

### AnImageWithoutAMaskIsRefused

An image `b.jpg` without a mask fails with `FileNotFound`, "images/train/b.jpg",
"read_mask_png", "has no mask" and "masks/train/b.png".

### AMaskWithoutAnImageIsRefused

A mask `z.png` without an image fails with `InvalidFile`, "masks/train/z.png", "read_mask_png"
and "the mask has no image".

### TwoImagesOfOneStemAreRefused

The images `a.jpg` and `a.png` with one mask `a.png` fail with `InvalidFile`,
"images/train/a.png", "shares the mask" and "masks/train/a.png".

### TheOptionsAndTheFoldersAreChecked

A class count of 0 fails with `InvalidDescription`, "read_mask_png" and
"the class count is 0". An absent image folder fails with `FileNotFound`, "images/train" and
"no such directory", and an absent mask folder with `FileNotFound`, "masks/train" and
"no such directory". Two empty folders fail with `InvalidFile`, "images/train" and
"holds no image".

### TheSuffixesPairNamesThatDiffer

With `image_suffix` "_leftImg8bit" and `mask_suffix` "_gtFine_labelIds" under the folders
`leftImg8bit` and `gtFine`, the two images pair with their `_gtFine_labelIds.png` masks, and
the list has two samples and four files. The mask `aachen_000000_gtFine_color.png` is not
listed.

### AnImageWithoutTheImageSuffixIsRefused

With `image_suffix` "_left", the image `b.png` fails with `InvalidFile`, "images/train/b.png",
"read_mask_png" and "does not end in '_left'".

### AMaskOfTheSuffixWithoutAnImageIsRefused

With `mask_suffix` "_ids", the mask `b_ids.png` without an image fails with `InvalidFile`,
"masks/train/b_ids.png" and "the mask has no image".

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

## modules/language/datasets/tests/test_datasets_next_token_collator.cpp

Suites: `Unit_LanguageDatasetsNextTokenCollator`.

Tests `Language::Datasets::NextTokenCollator` in
`modules/language/datasets/include/SushiAI/language/datasets/next_token_collator.hpp`. The
fixture provides an engine and allocates a batch of an input tensor and an `Int32` labels
tensor. Every collator has a window of three positions over a vocabulary of 50.

### ItDescribesTheTask

`task()` is `K_TASK_NEXT_TOKEN`, `window_length()` is 3, `tokens_per_window()` is 4 and
`vocabulary_size()` is 50. `targets()` holds one description: the role `K_ROLE_LABELS`, the
sample shape [3], the type `Int32` and 50 classes.

### LabelsHoldTheNextTokenOfEachPosition

`collate` over the ids 10, 11, 12, 13, 20, 21, 22, 23 writes the inputs 10, 11, 12, 20, 21, 22
and the labels 11, 12, 13, 21, 22, 23 into a batch of two rows.

### MakeRefusesExtentsOutOfBounds

`NextTokenCollator::make` fails with `InvalidDescription` for a window of 0
("NextTokenCollator", "window of 0 positions") and for a vocabulary of 0 ("vocabulary of 0
tokens").

### CollateRefusesWhatDoesNotFit

`collate` into a batch of two rows fails with `Incompatible` for seven ids
("NextTokenCollator", "7 ids", "windows of 4") and for twelve ids ("inputs", "6 elements",
"writes 9"). It also fails for a batch whose inputs are `Float32` ("inputs", "int32") and for
a batch without tensors ("one role 'labels'").

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

## modules/vision/datasets/tests/test_datasets_polygon_mask.cpp

Suites: `Unit_DatasetsPolygonMask`.

Tests `paint_polygon_mask` in
`modules/vision/datasets/include/SushiAI/vision/datasets/polygon_mask.hpp` over masks of 4 by
4 pixels. Each object is a rectangle given as a polygon and as its box.

### APolygonPaintsThePixelsWhoseCentresItHolds

A rectangle of class 2 from (1, 1) to (3, 3) over the background `K_MASK_VOID` gives a mask of
one channel whose four middle pixels are 2 and whose other twelve are 255.

### TheLaterOfTwoObjectsWinsAnOverlap

A rectangle of class 0 from (0, 0) to (3, 2), then one of class 1 from (2, 0) to (4, 4), over
the background 7 give the rows 0 0 1 1, 0 0 1 1, 7 7 1 1, 7 7 1 1.

### NoObjectIsTheBackgroundEverywhere

An empty label set over the background 3 gives sixteen bytes of 3.

### RefusesAnObjectItCannotPaint

An object with a box and no polygon fails with `InvalidDescription`, "paint_polygon_mask",
"object 0" and "no polygon". An object of class 255 fails with `InvalidDescription`,
"the class 255" and "0 to 254".

## modules/language/datasets/tests/test_datasets_prompt_dataset.cpp

Suites: `Unit_LanguageDatasetsPromptDataset`.

Tests `Language::Datasets::PromptDataset` in
`modules/language/datasets/include/SushiAI/language/datasets/prompt_dataset.hpp` and
`make_prompt_dataset`, which builds one from a data block. The fixture provides an engine and
reads the row and the stated prompt length of batch 0.

### AShortPromptIsFollowedByZeros

A dataset of the ids 3, 1, 4, 1, 5 over a window of 8 has a batch size of 1, one batch, the
sample shape [8], the type `Int32`, the name `K_PROMPT_DATA_TYPE` and one target of the role
`K_ROLE_PROMPT_LENGTH`. Its row is 3, 1, 4, 1, 5, 0, 0, 0 and its stated length is 5.

### ALongPromptKeepsItsLastIDs

A dataset of the ids 0 to 9 over a window of 4 has the row 6, 7, 8, 9 and the stated length 4.

### MakeRefusesAPromptOrAWindowThatDoesNotFit

`PromptDataset::make` fails with `InvalidDescription` for no id ("no id"), for the ids 1 and
-2 ("-2") and for a window of 0 ("window of 0").

### TheDataBlockEncodesItsText

A vocabulary made of no merges is written to a file in a scratch directory. A block with the
text "hi", a window of 4 and that file as `vocabulary` builds a dataset whose row is 'h', 'i',
0, 0 and whose stated length is 2. `make_prompt_dataset` fails with `InvalidDescription` for a
block without `text` ("the data block", "has no 'text'") and for a block with a `tokens` key
("'tokens'").

## modules/vision/datasets/tests/test_datasets_reader_dataset.cpp

Suites: `Unit_DatasetsReaderDataset`.

Tests `ReaderDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/reader_dataset.hpp`. The fixture is an
`EngineTest`. Its list stores gray samples of 2 by 2 pixels in one file `pixels.bin` of rising
bytes, written under the system's temporary directory, so sample i opens with the byte 4i and
has the class i modulo 2. The default options are batches of two such inputs under the name
"tiny", with a classification collator over two classes.

### ItDescribesItsBatches

Over five samples the dataset reports a batch size of 2, the sample shape [2, 2, 1], two
batches, the name "tiny" and one target of the role `K_ROLE_LABELS`.

### ABatchHoldsItsSamplesInListOrder

Over four samples, batch 1 holds the inputs (8 + i) / 255 for i from 0 to 7, each within 1e-6,
and the labels 0 and 1.

### ADetectionBatchLetterboxesImagesAndBoxes

Two raw RGB files of 2 rows and 4 columns at level 255 are letterboxed to 4 by 4, with a
detection collator of 3 classes and 2 slots. The first input value is 114 / 255 and value 12
is 1.0. The second image's object fills slot 2: valid is 0, 1, 0 at the slots 0, 2 and 3, the
class is 2 and the box is (1, 1, 3, 3) within 1e-4.

### MakeRefusesOptionsOutsideTheirBounds

`ReaderDataset::make` fails with `InvalidDescription` for a null collator ("ReaderDataset",
"tiny", "no collator"), a size of 0 by 2 ("0 by 2"), five channels ("5 channels"), a batch
size of 0 ("batch size of 0") and a batch size of 3 over two samples ("2 samples",
"one batch of 3").

### ABatchOutsideTheEpochStopsTheProcess

Over two samples, `read(1, batch)` is fatal with "ReaderDataset", "1 batches" and "batch 1".

### AReadReusesTheTensorsItFilledBefore

After `read(0, batch)` and `read(1, batch)` into one batch, the inputs and the labels have the
data pointers of the first read, and the first input value is 8 / 255.

### AReadDoesNotWriteATensorSomethingElseHolds

With a copy of the input tensor kept after `read(0, batch)`, `read(1, batch)` gives inputs at
another data pointer. The kept tensor still opens with 0 and the batch opens with 8 / 255.

### ItIsADatasetATrainerCanRead

Through a `Data::Dataset` reference, `read(1, batch)` gives as many targets as `targets()`
lists, the first with the same role, and inputs whose first extent is `batch_size()`.

### AnImageOfAnotherChannelCountNamesItsFile

With the options asking for three channels over the gray list, `read(0, batch)` fails with
`Incompatible`, "pixels.bin", "ReaderDataset", "1 channels" and "reads 3".

### AFileAbsentAtReadTimeIsNamed

With the list's file replaced by a path that does not exist, `make` succeeds and
`read(0, batch)` fails with `InvalidFile` and "gone.bin".

### AShuffledEpochReadsEverySampleOnceInTheOrderOfTheSeed

With `shuffle` on and the seed 42 over eight samples, the rows read across the four batches
are the samples in the order `Data::EpochPermutation(42, 0, 8)` gives, and each of the eight
is read once.

### BeginEpochSelectsTheOrderOfTheEpoch

After `Data::begin_epoch(dataset, 3)` the samples are read in the order of
`EpochPermutation(42, 3, 8)`, which differs from the order of epoch 0. After
`begin_epoch(dataset, 0)` they are read in the order of epoch 0.

### WithoutShuffleEveryEpochKeepsTheOrderOfTheList

Without `shuffle`, six samples are read as 0, 1, 2, 3, 4, 5, and again in that order after
`begin_epoch(dataset, 5)`.

### MakeRefusesANullPipeline

The form of `make` that takes a pipeline fails for a null one with `InvalidDescription`,
"ReaderDataset", "tiny" and "no pipeline".

## modules/vision/datasets/tests/test_datasets_reader_section.cpp

Suites: `Unit_DatasetsReaderTable`, `Unit_DatasetsTaskTable`, `Unit_DatasetsReaderSection`.

Tests `reader_table` and `find_reader`, `task_table` and `find_task`, and
`make_reader_dataset` in `modules/vision/datasets/include/SushiAI/vision/datasets/`. The
`Unit_DatasetsReaderSection` fixture is an `EngineTest` that builds a dataset from a section
called "data" whose `root` is a directory under the system's temporary directory. Its YOLO
split is two white raw images of 2 rows and 4 columns under `images/train`, the second with
one object of class 2 in `labels/train/b.txt`. Its folder split is one file under each of
`train/cat` and `train/dog`.

### ItListsTheEightReaders

`reader_table` lists yolo_txt, image_folder, idx, cifar, coco_json, voc_xml, mask_png and csv
in that order. Every row has a `read` function and is the row `find_reader` returns for its
name, and `find_reader("tfrecord")` is null.

### ARowListsTheKeysItReads

The rows yolo_txt and cifar list no key. coco_json lists `annotations`, `images`, `polygons`;
voc_xml lists `class_names`, `keep_difficult`; mask_png lists `images`, `masks`,
`image_suffix`, `mask_suffix`; csv lists `file`, `features`, `targets`.

### ARowReadsItsKeysOfTheSection

The csv row, given `file`, the feature c and the targets a, b over a file with the header
`a,b,c`, reads the values 3, 1, 2. It raises an error holding "data 'features'" and "array"
when `features` is a string. The coco_json row raises one holding "data 'polygons'" and
"boolean" when `polygons` is 3.

### TheMaskPNGRowPassesTheClassCountAndTheFolders

With `images` "pictures" and `masks` "labels" and no class count, the mask_png row fails with
`InvalidDescription`, "read_mask_png" and "the class count is 0". With a class count of 5 it
reads one sample and reports 5 classes.

### TheVOCXMLRowPassesTheClassListAndTheFlag

With `class_names` cat, dog over an annotation whose one object is a difficult dog, the
voc_xml row reports 2 classes and no object. With `keep_difficult` true the sample has one
object of class 1.

### TheCIFARRowRefusesAnotherClassCount

The cifar row with a class count of 7 fails with `InvalidDescription`, "7 classes",
"CIFAR-10" and "coarse".

### EachTaskStatesItsFitAndItsKeys

Detection has `SampleFit::Letterbox` and the key `max_objects`. Classification has
`SampleFit::Resize` and no key. Binary has `SampleFit::Resize` and the key `positive`.
Segmentation has `SampleFit::Letterbox` and the keys `ignore_index`, `background`.
`task_table` lists detection, classification, binary, multi_label, regression and
segmentation in that order, and `find_task("pose")` is null.

### ARowBuildsTheCollatorOfItsTask

For a list of two named classes with a table of two features and one target, and a section
with `max_objects` 5, every row has a `build` function and its `make_collator` gives a
collator whose `task` is the row's name.

### TheBinaryRowReadsThePositiveClass

Over the classes cat, dog the binary row builds a collator for an empty section, for
`positive` "cat" and for `positive` 0. It fails with `InvalidDescription` for `positive`
"bird" ("data 'positive' is 'bird'", "'cat', 'dog'"), for `positive` 2
("the positive class is 2") and for a class count of 3 ("data 'task' is 'binary'",
"the samples have 3").

### TheSegmentationRowReadsTheIgnoreIndex

For three classes and a size of 4 by 4, `ignore_index` -1 builds a collator whose first target
has the sample shape [4, 4]. `ignore_index` 1 fails with `InvalidDescription` and
"the ignore index 1 is one of the 3", and 4294967296 with "data 'ignore_index' is 4294967296"
and "32 bits".

### ADetectionSectionBuildsLetterboxedBatches

A detection section over the YOLO split with `input` [4, 4], `max_objects` 2 and `classes` 3
gives a dataset named "yolo_txt" with a batch size of 2, one batch and the sample shape
[4, 4, 3]. Its boxes target has the sample shape [2, 4] and its classes target 3 classes. In
batch 0 the first input value is 114 / 255, the valid flags of the slots 0 and 2 are 0 and 1,
and the class of slot 2 is 2.

### AClassificationSectionTakesTheClassesOfTheList

A classification section over the folder split with `input` [2, 3] and `channels` 1 gives the
sample shape [2, 3, 1] and one target of the role `K_ROLE_LABELS` with 2 classes.

### TheClassCountMayExceedTheListAndNotFallBelowIt

Over the folder split, `classes` 5 gives a target of 5 classes. `classes` 1 fails with
`InvalidDescription`, "data 'classes' is 1" and "name 2 classes".

### AnUnknownReaderListsTheKnownOnes

The reader "tfrecord" fails with `InvalidDescription`, "data 'reader'", "'tfrecord'" and the
quoted name of each of the eight readers.

### AnUnknownTaskListsTheKnownOnes

The task "pose" fails with `InvalidDescription`, "data 'task'", "'pose'" and the quoted name
of each of the six tasks.

### AKeyTheTaskDoesNotReadIsRefused

The key `batch_sizes` raises an error holding "data", "'batch_sizes'" and "'batch_size'". The
key `max_objects` on a classification section raises one holding "data" and "'max_objects'".

### ADetectionSectionNeedsMaxObjects

A detection section without `max_objects` raises an error holding "data" and "'max_objects'".

### ValuesOutsideTheirBoundsAreRefused

The build fails with `InvalidDescription` for an `input` of three entries ("data 'input'",
"3 entries", "[height, width]"), for `input` [4, 0] ("data 'input[1]' is 0") and for
`channels` 0 ("data 'channels' is 0").

### TheFailureOfTheReaderIsReturned

A yolo_txt section over an empty root fails with `FileNotFound`, "train.txt" and
"images/train".

### ARegressionSectionReadsTheRowsOfACSVFile

A regression section over a `train.csv` of three records with the target y and a batch size
of 2 gives a dataset named "csv" with one batch, the sample shape [2] and one target of the
role `K_ROLE_VALUES` and the sample shape [1]. Batch 0 holds the inputs 1, 2, 4, 5 and the
values 3, 6.

### ASegmentationSectionFitsTheMaskOfEachImage

A segmentation section over one image of 2 rows and 4 columns and its mask, with `input`
[4, 4], `classes` 3 and `ignore_index` -1, gives the sample shape [4, 4, 3] and one target of
the role `K_ROLE_MASK`, the sample shape [4, 4] and 3 classes. In batch 0 the mask's two rows
0, 1, 2, 0 and 1, 1, 2, 2 lie between two rows of -1.

### AMaskOfAnotherSizeThanItsImageIsRefused

With a 2 by 2 mask for an image of 2 rows and 4 columns, the build succeeds and `read(0,
batch)` fails with `Incompatible`, "masks/train/a.png", "fit_mask" and
"the mask is 2 by 2 pixels; its image is 2 by 4".

### ABinarySectionWritesOneForThePositiveFolder

A binary section over the folders cat and dog with `positive` "cat" has one target of
`Float32`. Batch 0 holds the labels 1 and 0.

### AMultiLabelSectionFlagsTheClassesOfTheObjects

A multi_label section over the YOLO split with `classes` 3 has a target of the sample shape
[3]. Batch 0 holds 0, 0, 0 for the first image and 0, 0, 1 for the second.

### ATaskAndAReaderOfAnotherSampleKindAreRefused

The task classification over the reader csv fails with `InvalidDescription`,
"data 'task' is 'classification'", "reads images" and
"the reader 'csv' lists rows of numbers". The task regression over image_folder fails with
"data 'task' is 'regression'", "reads rows of numbers" and
"the reader 'image_folder' lists none".

### AKeyOfAnotherReaderIsRefused

The key `features` on a yolo_txt detection section raises an error holding "data",
"'features'" and "'max_objects'".

### ACacheIsWrittenAndRead

With `cache` naming a directory, the build writes two `.saim` files into it. After the folder
`images` is removed, `read(0, batch)` succeeds and input value 12 is 1.0.

### AShuffledSectionWithThePipelineReadsABatch

A detection section with `input` [16, 16], `shuffle` true, `seed` 7 and `pipeline` "default"
builds. After `begin_epoch(dataset, 1)`, `read(0, batch)` succeeds, the sample shape is
[16, 16, 3] and the inputs hold 2 * 16 * 16 * 3 elements.

### APipelineTheTableLacksIsRefused

The pipeline "mixup" fails with `InvalidDescription`, "data 'pipeline' is 'mixup'", "'none'"
and "'default'". The pipeline "default" with `input` [4, 6] fails with `InvalidDescription`,
"square" and "4 by 6".

### WorkersReadTheSameBatchesAhead

With `workers` 2 and `prefetch` 2 the dataset is a `Data::ThreadedLoader`; without them it is
not. Both have the same batch count, and every input element of every batch is equal between
the two.

### KeysThatDoNotApplyAreRefused

The build fails with `InvalidDescription` for `prefetch` without workers ("data 'prefetch'",
"data 'workers' is 0"), for `cache_side` without a cache ("data 'cache_side'",
"data 'cache'") and for `shuffle` on a csv regression section ("data 'shuffle'",
"'regression'", "rows of numbers").

### ASegmentationSectionPaintsThePolygonsOfALabelFile

A segmentation section over the reader yolo_txt, with a label file whose one polygon of class
1 covers the top left quarter of a 4 by 4 image and with `background` 2, gives a mask in
batch 0 whose top left 2 by 2 pixels are 1 and whose other pixels are 2.

## modules/vision/datasets/tests/test_datasets_sample_cache.cpp

Suites: `Unit_DatasetsSampleCache`.

Tests `write_sample_cache`, `use_sample_cache` and `sample_cache_path` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample_cache.hpp`. The source images
are raw image files the cases write under the system's temporary directory, and the cache is
the folder `cache` beside them.

### TheFirstPassWritesAndTheSecondReuses

Over two RGB images of 2 by 3 and 4 by 2 pixels, the first `write_sample_cache` reports 2
written, 0 reused and the byte count of the two raw image files. The second reports 0 written,
2 reused and 0 bytes.

### AListPointedAtTheCacheLoadsTheSamePixels

After a write, `use_sample_cache` gives a list whose one file lies in the cache directory, has
the extension `.saim` and a name that starts with "a-". `load_sample_image` over it gives the
size, the three channels and every byte of the source image.

### ALongestSideReducesOnlyWhatExceedsIt

With `longest_side` 4, an image of 4 rows and 8 columns at level 200 loads from the cache as 2
rows and 4 columns with a first byte of 200. A 2 by 2 image loads as 2 by 2.

### AnImageThatChangedIsStale

After the source is rewritten with another size, `use_sample_cache` fails with `FileNotFound`,
"images/a.jpg", "sample cache" and "changed since". A second write reports 1 written and 0
reused, and the image then loads from the cache with a width of 3 and a first byte of 2.

### StoredPixelsAndMasksAreLeftAlone

For a list of one sample with stored pixels and one sample with an image and a mask, the write
reports 1 written. In the list `use_sample_cache` returns, the file of stored pixels and the
mask keep their paths and the image's path differs.

### ANameFollowsTheSourceAndTheOptions

`sample_cache_path` returns the same path twice for one source and one set of options, and its
file name is 23 characters long. With `longest_side` 320 the path differs. A source that does
not exist fails with `FileNotFound` and "gone.jpg".

### RefusesOptionsItCannotServe

`write_sample_cache` fails with `InvalidDescription` for options without a directory
("no directory"), for `longest_side` -1 ("a longest side of -1") and for `longest_side` 1 over
a list whose sample has a mask ("a longest side of 1", "masks").

## modules/vision/datasets/tests/test_datasets_stored_pixels.cpp

Suites: `Unit_DatasetsStoredPixels`.

Tests `StoredPixels` and `read_stored_pixels` in
`modules/vision/datasets/include/SushiAI/vision/datasets/stored_pixels.hpp`. Three cases read
a file written under the system's temporary directory and the others read a string stream.

### ByteCountIsTheProductOfTheExtents

`byte_count` of 4 by 5 pixels with three channels is 60.

### InterleavedBytesAreCopiedFromTheOffset

Over a file of the bytes 0 to 19, interleaved pixels of 2 rows, 3 columns and two channels at
offset 5 read as an image of that shape holding the bytes 5 to 16.

### PlanarBytesAreInterleaved

Planar pixels of 2 by 2 with three channels at offset 1, stored as the planes 1, 2, 3, 4, then
11, 12, 13, 14, then 21, 22, 23, 24, read as 1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24.

### APlaneLongerThanOneChunkIsCopiedWhole

Planar pixels of 100 by 100 with two channels read so that, at every 997th pixel, channel 0
holds the byte of the first plane and channel 1 the byte of the second.

### AFileThatEndsEarlyIsInvalid

Twelve bytes asked at offset 4 of a file of ten fail with `InvalidFile`, "short.bin" and
"6 of 12".

### AFileThatCannotBeOpenedIsInvalid

A path that does not exist fails with `InvalidFile` and "absent.bin".

### ExtentsOutsideAnImageAreRefused

A size of 0 by 2 fails with `InvalidDescription` and "0 by 2", and five channels with
`InvalidDescription` and "5 channels".

## modules/vision/datasets/tests/test_datasets_table_dataset.cpp

Suites: `Unit_DatasetsTableDataset`.

Tests `TableDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/table_dataset.hpp`. The fixture is an
`EngineTest`. Its list names the file `rows.csv` and holds rows of two features and one
target, row i being 10i, 10i + 1 and 10i + 2. The default options are batches of two rows
under the name "rows", with a `RegressionCollator` of two features and one target.

### ItDescribesItsBatches

Over five rows the dataset reports the name "rows", a batch size of 2, two batches, the sample
shape [2] and one target of the role `K_ROLE_VALUES` and the sample shape [1].

### ABatchHoldsItsRowsInListOrder

Through a `Data::Dataset` reference, batch 1 holds the inputs 20, 21, 30, 31 and the values
22, 32. A following `read(0, batch)` writes the same input tensor, whose first value is then 0.

### MakeRefusesWhatItCannotBatch

`TableDataset::make` fails with `InvalidDescription` for a null collator ("TableDataset",
"'rows' has no collator"), a list without a table ("a table of 0 features"), a table one
number short ("4 samples and 11 numbers", "a row holds 3"), a batch size of 0
("a batch size of 0") and one row for batches of two ("1 samples", "one batch of 2").

### ABatchOutsideTheEpochStopsTheProcess

Over two rows, `read(1, batch)` is fatal with "TableDataset", "1 batches" and "batch 1".

### TheFailureOfTheCollatorIsReturned

With a collator of two features and two targets over the list of one target, `make` succeeds
and `read(0, batch)` fails with `InvalidDescription`, "rows.csv", "RegressionCollator" and
"2 features and 1 targets".

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

## modules/language/datasets/tests/test_datasets_token_dataset.cpp

Suites: `Unit_LanguageDatasetsTokenDataset`.

Tests `Language::Datasets::TokenDataset` in
`modules/language/datasets/include/SushiAI/language/datasets/token_dataset.hpp`. The fixture
provides an engine and writes `rising.satk` to a scratch directory: the ids 0 to 19 over a
vocabulary of 20, so each id equals its position. Unless a case says otherwise the dataset
has a next-token collator of four positions, a batch size of 2 and the name "tiny".

### ItDescribesItsBatches

The dataset has a batch size of 2, the sample shape [4], the type `Int32`, two batches, the
name "tiny" and a vocabulary size of 20. Its one target has the role `K_ROLE_LABELS`, the
sample shape [4] and 20 classes.

### ABatchHoldsItsWindowsAsInt32

`read(1)` fills `Int32` inputs of the shape [2, 4] and one `Int32` target of eight elements.
The inputs are the ids 8 to 15 and each label is its input plus 1.

### EveryLabelIsTheTokenAfterItsInput

With a stride of 3 the dataset has three batches. In every batch, row `r` of batch `b` starts
at the id `(2b + r) * 3`: input `t` is that id plus `t`, label `t` is that id plus `t + 1`, and
each label but the last equals the next input of its row.

### MakeRefusesOptionsOutsideTheirBounds

`TokenDataset::make` fails with `InvalidDescription` for a null collator ("TokenDataset",
"tiny", "no collator"), a batch size of 0 ("batch size of 0"), a stride of 0 ("TokenDataset",
"tiny", "stride of 0") and a batch size of 5 ("rising.satk", "4 windows of 5", "one batch of
5"). It fails with `Incompatible` for a collator over 19 tokens ("rising.satk", "vocabulary of
20", "one of 19") and with `FileNotFound` for a path that does not exist ("absent.satk").

### AnIDOutsideTheVocabularyNamesItsFileOffset

Byte 8 of the file is overwritten with 5, which makes its vocabulary size 5. The dataset is
still made, and `read(0)` fails with `InvalidFile` and a message containing "rising.satk", "id
5", "byte 44" and "[0, 5)".

### AFileCutAfterMakeNamesWhereItEnds

The file is resized to 64 bytes after the dataset is made. `read(0)` is ok and `read(1)` fails
with `InvalidFile` and a message containing "rising.satk", "ends at byte 64" and "end at byte
76".

### ABatchOutsideTheEpochStopsTheProcess

`read(2)` on a dataset of two batches stops the process with a report containing
"TokenDataset", "2 batches" and "batch 2".

### AReadReusesTheTensorsItFilledBefore

After `read(0)` and `read(1)` into the same batch, the inputs and the labels have the data
pointers they had after the first read, and the first input is 8.

### AReadDoesNotWriteATensorSomethingElseHolds

A copy of the inputs tensor is kept after `read(0)`. After `read(1)` the batch's inputs have
another data pointer, the first id of the kept tensor is 0 and the first id of the batch is 8.

### ItIsADatasetATrainerCanRead

Read through a `Data::Dataset` reference, `read(1)` fills as many targets as `targets()`
lists, the first target has the role the description states, and the first extent of the
inputs equals `batch_size()`.

## modules/language/datasets/tests/test_datasets_token_file.cpp

Suites: `Unit_LanguageDatasetsTokenFile`.

Tests `write_token_file`, `read_token_file_header` and `read_token_range` in
`modules/language/datasets/include/SushiAI/language/datasets/token_file.hpp`. Most cases start
from a hand-written file of 36 bytes: a header of 24 bytes (the magic "SATK", version 1, a
vocabulary of 300, an id width of 4 and a count of 3) and the ids 1, 258 and 0. Files are
written to a scratch directory.

### TheWriterProducesTheDocumentedBytes

`write_token_file` of the ids 1, 258 and 0 over 300 tokens writes exactly those 36 bytes.

### TheHeaderAndARangeOfIDsSurviveTheFile

A file of the ids 0 to 39999 over 40000 tokens has a header with a vocabulary size of 40000
and a token count of 40000. `read_token_range` of five ids from 16382 returns 16382 to 16386,
the range from 39995 ends at 39999, and an empty range at 40000 is ok.

### TheWriterRefusesAnIDOutsideTheVocabulary

`write_token_file` fails with `InvalidDescription` for the id 300 over 300 tokens
("write_token_file", "id 300", "index 1", "[0, 300)"), for the id -2 ("id -2", "index 0") and
for a vocabulary of 0 ("vocabulary of 0 tokens"). No file exists afterwards.

### ATruncatedFileIsRefusedAtEveryLength

For every length from 0 to 35 bytes, `read_token_file_header` over the file cut to that length
fails with `InvalidFile` and a message containing "cut.satk", "read_token_file_header" and
"ends at byte". A cut at 10 bytes names "ends at byte 10" and "header of 24 bytes"; a cut at
30 names "ends at byte 30" and "declares 36 bytes".

### BytesAfterTheDeclaredEndAreRefused

With two bytes appended, `read_token_file_header` fails with `InvalidFile` and a message
containing "long.satk", "2 bytes follow" and "byte 36".

### AHeaderFieldOutOfBoundsNamesItsOffset

`read_token_file_header` fails with `InvalidFile` for a first byte of 'X' ("'SATK'", "no token
file"), a version of 2 ("byte 4", "format version 2", "versions 1 to 1"), a vocabulary of 0
("byte 8", "vocabulary of 0 tokens"), an id width of 8 ("byte 12", "ids of 8 bytes") and a
count whose last byte is 0x7F ("byte 16", "ids").

### AnIDOutsideTheVocabularyNamesItsOffset

With the vocabulary size lowered to 258, the header reads and `read_token_range` of three ids
fails with `InvalidFile` ("above.satk", "read_token_range", "id 258", "byte 28", "[0, 258)"),
while a range of the first id alone returns 1. In a file whose third id is the bytes FF FF FF
FF, the range at id 2 fails with "id -1" and "byte 32".

### ARangeOutsideTheFileAndAnAbsentFileAreRefused

`read_token_range` of two ids from id 2 fails with `InvalidDescription` and a message
containing "2 ids from id 2" and "file of 3". Over a path that does not exist,
`read_token_file_header` fails with `FileNotFound` ("absent.satk") and `read_token_range` with
`InvalidFile` ("absent.satk", "cannot be opened").

## modules/language/datasets/tests/test_datasets_tokens_section.cpp

Suites: `Unit_LanguageDatasetsTokensSection`.

Tests `Language::Datasets::make_tokens_dataset` in
`modules/language/datasets/include/SushiAI/language/datasets/tokens_section.hpp`, which builds
a dataset from a `tokens` data block. The fixture provides an engine and writes a token file
of the ids 0 to 39 over a vocabulary of 40.

### TheKeysDescribeTheDataset

A block with `window` 4, `stride` 5 and `batch_size` 2 builds a dataset with a batch size of
2, the sample shape [4], the type `Int32` and four batches. Its `K_ROLE_LABELS` target has 40
classes. In batch 1 the first input is 10 and the fifth is 15.

### AnAbsentBatchSizeIsTheDefault

A block with `window` 1 and `stride` 1 builds a dataset whose batch size equals
`TokenDatasetOptions{}.batch_size`.

### AKeyThatDoesNotFitIsRefused

`make_tokens_dataset` fails with `InvalidDescription` for a block without `window` ("the data
block", "has no 'window'"), a block without `file` ("the data block", "has no 'file'"), a
`window` of "four" ("'window'", "whole number"), a `window` of 0 ("'window'", "positive") and
a `shuffle` key ("the data block", "'shuffle'", "does not have").

### AMissingFileFails

A block whose `file` does not exist returns a status that is not ok. The case checks neither
the code nor the message.

## modules/vision/datasets/tests/test_datasets_voc_xml.cpp

Suites: `Unit_DatasetsVocXml`.

Tests `read_voc_xml` in `modules/vision/datasets/include/SushiAI/vision/datasets/voc_xml.hpp`
over files the cases write under the system's temporary directory. Unless a case says
otherwise the split is "train", listed in `ImageSets/Main/train.txt`. The sample annotation
names `first.jpg`, a size of 200 by 100 and some objects, each with a name, a difficult flag
and the box edges 11, 21, 60, 40.

### TheListedAnnotationsAreReadWithSortedClasses

The list file names second, an empty line, then first followed by " 1", with a CRLF on its
first line. The class names come out as "cat & co", "dog", and the samples as
`JPEGImages/second.jpg` with no object and `JPEGImages/first.jpg`. That sample has no class
index and two objects of the classes 1 and 0; its first box has the centre (35/200, 30/100)
and the size (50/200, 20/100). `Annotations/unlisted.xml` is not read.

### AnEmptySplitReadsEveryAnnotation

With an empty split the annotations `a.xml` and `b.xml` are read in that order, giving
`JPEGImages/a.jpg` and `JPEGImages/first.jpg`. `Annotations/readme.txt` is not read.

### ADifficultObjectIsDroppedUnlessKept

Over a difficult dog and a cat, the default options give the class names cat and one object.
`VOCDifficult::Kept` gives cat, dog and two objects.

### AStatedClassListGivesTheIndices

With `class_names` person, dog, cat the class count is 3 and a dog has the class 1. With
person, cat the read fails with `InvalidFile`, "Annotations/first.xml", "line 4",
"read_voc_xml" and "the class 'dog'". With dog, cat, dog it fails with `InvalidDescription`,
"read_voc_xml", "names 'dog' twice" and "at 0 and at 2".

### AMalformedAnnotationNamesItsFileAndLine

An annotation that closes `annotation` inside an open `size` fails with `InvalidFile`,
"Annotations/first.xml", "line 3", "parse_xml" and "closes '<size>'".

### AMissingElementOrNumberIsRefused

Each of these fails with `InvalidFile` and "first.xml": an annotation without `size`
("line 1", "read_voc_xml", "'annotation' has no 'size'"), a root named `image` ("line 1",
"the root element is 'image'"), a width of "wide" ("line 3", "'width' holds 'wide'"), an
object without `bndbox` ("line 4", "'object' has no 'bndbox'") and a box whose maximum lies
before its minimum ("line 7", "the box ends at (5, 2), before its start (9, 1)").

### TheListAndItsStemsMustExist

Without the list file the read fails with `FileNotFound`, "read_voc_xml", "the split 'train'"
and "ImageSets/Main/train.txt". A stem on line 3 without an annotation fails with
`FileNotFound`, "ImageSets/Main/train.txt", "line 3" and "Annotations/missing.xml". A list of
one empty line fails with `InvalidFile` and "holds no annotation".

## modules/language/datasets/tests/test_datasets_window_layout.cpp

Suites: `Unit_LanguageDatasetsWindowLayout`.

Tests `count_windows` and `first_token_of` over a `WindowLayout` in
`modules/language/datasets/include/SushiAI/language/datasets/window_layout.hpp`. A layout is
written here as {window, stride}.

### WindowsBeginOneStrideApart

`count_windows(10, {4, 3})` is 3, and `first_token_of` is 0, 3 and 6 for the windows 0, 1 and
2.

### OnlyWholeWindowsAreCounted

`count_windows` is 2 for 8 ids over {4, 4}, 2 for 11 ids over {4, 4}, 1 for 4 ids over {4, 1}
and 0 for 3 ids over {4, 1}.

### ALayoutOutOfBoundsIsRefused

`count_windows` fails with `InvalidDescription` for a window of 0 ("count_windows", "window of
0 tokens"), a stride of 0 ("stride of 0 tokens") and a window of `K_MAX_WINDOW_TOKENS + 1`
("window of").

## modules/vision/datasets/tests/test_datasets_xml_element.cpp

Suites: `Unit_DatasetsXmlElement`.

Tests `parse_xml` and `XMLElement` in
`modules/vision/datasets/include/SushiAI/vision/datasets/xml_element.hpp` over strings.

### ElementsTextAndLinesAreRead

A root `a` on line 1 with the children `b`, `c` and an empty `b` reads with no text of its own
and three children. The first `b` has the text "one" and the line 2, the `d` inside `c` has
the text "two", and the last `b` has the line 4 and no text. `find_child` returns the first
`b` for "b", the `c` for "c" and null for "e".

### TheFiveEntitiesAreReplaced

The text `a &amp; b &lt;c&gt; &quot;d&quot; &apos;e&apos;` reads as `a & b <c> "d" 'e'`.

### TheDeclarationCommentsAndAttributesAreSkipped

A document with a byte order mark, an XML declaration, comments before, inside and after the
root, and attributes on the root and its child reads as the root `a` on line 3 with the text
"t" and one child `b`.

### AClosingTagOfAnotherNameIsRefused

`</a>` on line 3 after `<b>` on line 2 fails with `InvalidFile`, "line 3", "parse_xml",
"'</a>' closes '<b>'" and "line 2".

### AnElementLeftOpenIsRefused

Each of these fails with `InvalidFile`: an element `b` left open ("line 3",
"the element 'b', opened on line 2, is not closed"), a comment left open ("line 2",
"a comment is not closed") and a tag without its end ("line 2", "has no '>'").

### AnUnknownEntityIsRefused

`&nbsp;` on line 3 fails with `InvalidFile`, "line 3" and "'&' opens none of". `&#65;` fails
with `InvalidFile` and "line 1".

### MarkupOutsideTheSubsetIsRefused

Each of these fails with `InvalidFile`: a CDATA section ("line 2", "CDATA"), a DOCTYPE
("line 2", "declaration") and a processing instruction inside the root ("line 3",
"processing instruction").

### AMalformedAttributeIsRefused

`k=v` without quotes fails with `InvalidFile`, "line 2", "attribute 'k'" and "not quoted".
`k` without a value fails with `InvalidFile`, "line 1" and "attribute 'k' has no '='".

### ADocumentNeedsExactlyOneRoot

Each of these fails with `InvalidFile`: two empty lines ("line 3", "holds no element"), bare
text ("line 1", "text stands where an element must open"), a second root ("line 2",
"content follows the root element 'a'") and the name `1a` ("line 1",
"a name must open with").

### NestingBeyondTheDepthIsRefused

`K_XML_MAX_DEPTH` + 1 elements opened one per line fail with `InvalidFile`, "line 65" and
"more than 64 elements are open at once".

## modules/vision/datasets/tests/test_datasets_yolo_txt.cpp

Suites: `Unit_DatasetsYoloTxt`.

Tests `read_yolo_txt` and `yolo_label_path` in
`modules/vision/datasets/include/SushiAI/vision/datasets/yolo_txt.hpp`. The reader cases write
one-byte image files and label text files under the system's temporary directory and read
with a class count of 3.

### TheLabelPathSwapsTheLastImagesFolder

`yolo_label_path` maps `data/images/train/a.jpg` to `data/labels/train/a.txt`,
`images/x/images/a.png` to `images/x/labels/a.txt` and `images/a.b.jpeg` to `labels/a.b.txt`.

### TheLabelPathKeepsAFolderThatIsNotImages

It maps `data/train/a.jpg` to `data/train/a.txt`, `data/images` to `data/images.txt` and
`a.jpg` to `a.txt`.

### TheImageFolderIsListedWithItsLabels

The split "train" lists `images/train/a.png`, `b.jpg` and `c.jpg` in that order and leaves
`notes.md` out, with a class count of 3 and no class names. The first sample has no stored
pixels, no class index and two objects of the classes 2 and 0; its first box is (0.5, 0.25,
0.2, 0.1) and has no polygon. The second sample has one object of class 1 with a polygon of
three vertices, the second being (0.9, 0.1).

### AnImageWithoutALabelFileHasNoObject

An image without a label file and an image with an empty one both have no object.

### AnImageListIsReadInItsOwnOrder

A `train.txt` naming `./images/all/z.jpg` with a CRLF, an empty line and `images/all/a.jpg`
between blanks gives two samples in that order. The first has no object and the second one
object of class 1. `unlisted.jpg` is not read.

### AnImageListWinsOverAnImageFolder

With `images/train` holding two images and `train.txt` naming one, the list has the one sample
`images/train/b.jpg`.

### AListedImageThatIsAbsentNamesItsLine

A listed image that does not exist fails with `FileNotFound`, "train.txt", "line 3" and
"gone.jpg".

### ABadLabelFileNamesItsPathAndLine

A label file whose second line holds "abc" as a coordinate fails with `InvalidFile`,
"labels/train/b.txt", "line 2" and "abc".

### AClassOutsideTheClassCountIsRefused

A label of class 3 fails with `InvalidFile`, "labels/train/a.txt" and "line 1".

### AClassCountBelowOneIsRefused

A class count of 0 fails with `InvalidDescription` and "class count is 0".

### ASplitWithNeitherFormIsNotFound

A split with no list and no folder fails with `FileNotFound`, "train.txt" and "images/train".

### ASplitWithoutAnImageIsRefused

An image folder holding only `readme.txt` fails with `InvalidFile`, "'train'" and "no image".

## modules/layers/train/tests/test_dense_losses.cpp

Suites: `Unit_DenseLosses`, `Unit_DenseLossesDeathTest`.

Tests the nodes `Train::emit_mean_of_elements`, `Train::emit_logistic_loss` and
`Train::emit_squared_error` add to a graph. Each case reads the operations of the whole graph in
emission order.

### TheMeanIsOneColumnSummedAndScaled

Over a [4, 3] input the mean is RESHAPE, SUM_ROWS, SCALE. The result has one element and the
reshaped value has the shape [12, 1].

### TheLogisticLossReadsTheLogitInItsStableForm

Over [4, 3] logits and targets the graph is RELU, MUL, ABS, SCALE, SOFTPLUS, SUB, ADD, RESHAPE,
SUM_ROWS, SCALE, and the loss is one float32 element. SOFTPLUS reads a SCALE whose operand is
the ABS node, and ABS reads the logits.

### TheSquaredErrorSquaresTheDifference

Over [4, 2] outputs and targets the graph is SUB, MUL, RESHAPE, SUM_ROWS, SCALE and the loss
has one element. MUL reads one value twice, and SUB reads the outputs and then the targets.

### OperandsOfTwoShapesAreFatal

`emit_squared_error` over a [4, 2] and a [4, 3] value stops the process with a report
containing "[4, 2]" and "[4, 3]".

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

## modules/layers/train/tests/test_dense_targets.cpp

Suites: `Unit_DenseTargets`.

Tests `Train::fill_dense_targets` and `Train::fill_binary_targets`. The fixture needs an engine.
A leaf holds 7 in every element before a fill.

### ValuesAreCopiedInOrderIntoEitherRealType

Six float32 values of shape [2, 3] fill a float32 leaf and a float64 leaf of the same shape,
and each leaf reads back the six values in order.

### AColumnLeafTakesOneLabelPerRow

`fill_binary_targets` writes the [4] labels 1, 0, 0, 1 into a [4, 1] leaf.

### AnotherCountOrTypeIsRefusedAndTheLeafKept

`fill_dense_targets` fails with `Incompatible` for six values into a leaf of four ("holds 6
values", "targets hold 4"), for int32 values ("int32", "float32") and for an int32 leaf
("targets are int32"). The float32 leaf still holds four 7s.

### ALabelThatIsNotZeroOrOneIsRefused

`fill_binary_targets` over the labels 1, 0, 0.5, 1 fails with `Incompatible` and a message
containing "element 2", "0.5" and "0 or 1". The leaf still holds four 7s.

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

## modules/core/graph/tests/test_depthwise_ops.cpp

Suites: `Unit_DepthwiseOps`.

Tests the three depthwise emitters of `GraphBuilder`: `depthwise_conv2d`,
`depthwise_conv2d_backward_input` and `depthwise_conv2d_backward_weight`. Most cases share one
geometry, a 3x3 window with unit stride and one pixel of padding, over a [2, 5, 5, 3] input and
[3, 3, 3] taps.

### DepthwiseConv2dWritesItsIdentityOverInputAndTapsInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D` node that reads the input and then the
taps and writes that one value.

### DepthwiseConv2dYieldsTheWindowGridOverThePaddedInput

The shared geometry yields [2, 5, 5, 3]. The same window with stride 2 on both axes yields
[2, 3, 3, 3].

### DepthwiseConv2dKeepsLayoutDTypeAndLifetime

Over a `Float64` input and `Float64` taps the result is tagged NHWC, has the element type
`Float64` and the lifetime `Activation`.

### DepthwiseConv2dStoresItsGeometryInTheGeometrySlots

For a 3x5 window with padding 1 on top and 2 on the left over a [2, 7, 9, 3] input, the node's
first eight attribute values are 3, 5, 1, 1, 1, 1, 1 and 2.

### DepthwiseConv2dRefusesARankFourWeight

Taps of the shape [3, 3, 3, 1] make the emitter throw `Error`, and the node count does not
change.

### DepthwiseConv2dRefusesTapsOfAnotherWindowOrChannelCount

Taps of the shapes [5, 3, 3], [3, 5, 3] and [3, 3, 4] each make the emitter throw `Error`
against the 3x3 geometry and three channels, and the node count does not change.

### DepthwiseConv2dRefusesAnNCHWInput

An input converted to NCHW with `layout_convert` makes the emitter throw `Error`.

### DepthwiseConv2dRefusesTapsOfAnotherDType

A `Float32` input with `Float64` taps makes the emitter throw `Error`.

### DepthwiseConv2dRefusesAWindowThatDoesNotFit

A 3x3 window without padding over a [2, 2, 2, 3] input makes the emitter throw `Error`.

### BackwardInputWritesItsIdentityOverGradientAndTapsInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT` node that reads the
gradient and then the taps.

### BackwardInputRecoversTheForwardInputShape

From a [2, 3, 3, 3] gradient, stride 2 and the input shape [2, 5, 5, 3], the result has the
shape [2, 5, 5, 3], the layout NHWC and the element type `Float32`.

### BackwardInputRefusesAGradientOnAnotherGrid

For the input shape [2, 5, 5, 3], gradients of the shapes [2, 4, 5, 3], [2, 5, 5, 4] and
[2, 5, 3] each make the emitter throw `Error`, and the node count does not change.

### BackwardWeightWritesItsIdentityOverGradientAndInputInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT` node that reads the
gradient and then the forward input.

### BackwardWeightYieldsTheTapShapeInTheOperandsDType

Over `Float64` operands the result has the shape [3, 3, 3], the layout NHWC, the element type
`Float64` and the lifetime `Activation`.

### BackwardWeightRefusesMismatchedBatchChannelsOrDType

Against a [2, 5, 5, 3] gradient, an input of another batch, of another channel count or of
another element type makes the emitter throw `Error`, as does a rank-3 gradient. The node count
does not change.

## modules/vision/detect/tests/test_detect_anchor_clustering.cpp

Suites: `Unit_DetectAnchorClustering`.

Tests `SeededSequence` and `cluster_anchors`. The clustering cases use a ladder of six box
sizes with 400 boxes per rung, 2400 boxes in all, each jittered by 2 %. A rung counts as
recovered when an anchor lies within 5 % of it.

### SequenceOfSeedZeroMatchesThePublishedNumbers

The first eight values of `SeededSequence(0).next()` equal the eight published SplitMix64
numbers the case lists, from 0xE220A8397B1DCDAF on.

### UnitDrawIsTheUpperBitsOfTheNextNumber

The first `next_unit()` of seed 0 has the bits of the upper 53 bits of 0xE220A8397B1DCDAF
divided by 2^53, and lies in [0, 1).

### LadderIsRecoveredWithinFivePercent

With the default seed and a count of 6, every rung is recovered. The mean best IoU is above
0.95 and not below its value before refinement, 2400 boxes are used and none rejected, and the
iteration and sweep counts stay within the options' limits.

### AnchorsAreInAscendingArea

Each anchor's area is larger than the area of the anchor before it.

### AnotherSeedStillRecoversTheLadder

With the seed 0x1234 every rung is recovered and the mean best IoU is above 0.95.

### OutliersLeaveEveryCentreWithinFivePercent

With 24 extra boxes of 0.9 x 0.9, 1 % of the ladder, 2424 boxes are used, every rung is still
recovered, and the mean best IoU is not below its value before refinement.

### TwoRunsAreBitIdentical

Two runs over the same boxes and options return anchors whose widths and heights have the same
bits, the same bits in both mean IoU numbers, and equal iteration and sweep counts.

### ZeroSweepsKeepTheFitBeforeRefinement

With `max_refinement_sweeps` 0 the result reports zero sweeps, and its mean best IoU equals the
value before refinement to 1e-9.

### RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

### RefusesFewerDistinctBoxesThanAnchors

Two distinct sizes, each repeated ten times, with a count of 3 fail with `Incompatible`.

### LeavesOutAZeroAndANaNSide

Three sizes repeated five times, plus one box with a zero side and one with a NaN side, give
2 rejected and 15 used boxes. The three anchors equal the three sizes to 1e-6 and the mean best
IoU is 1 to 1e-9.

### LeavesOutABoxAtTheSmallestSide

With `min_side` 0.125, the boxes 0.125 x 0.5 and 0.5 x 0.125 are rejected and 0.25 x 0.5 is the
one box used and the one anchor.

### RefusesOptionsOutsideTheirRange

A count of 0, a `max_iterations` of 0, a `max_refinement_sweeps` of -1, a `min_side` of -0.5
and a NaN `min_side` each fail with `InvalidDescription`.

## modules/vision/detect/tests/test_detect_anchor_cost.cpp

Suites: `Unit_DetectAnchorCost`.

Tests `measure_head_cost`
over a head with two classes whose delta the file builds by hand. Its box part sums to 5.25 and
its non-box part to 26, over a batch of 2. Every number is compared to 1e-12.

### SquaredErrorSplitsBoxFromNonBoxAndDividesByBatch

With the squared-error box loss and an objectness weight of 1, the total is 15.625, the non-box
share 13, the box share 2.625 and the stored cost 31.25.

### ObjectWeightScalesTheNonBoxShare

With an objectness weight of 2 the non-box share is 26 and the box share -10.375. The total
stays 15.625 and the stored cost 31.25.

### CompleteIoUStoredAddsTheIoUTermAndWeightedNonBoxSum

With the CompleteIoU box loss, a box weight of 0.75, an objectness weight of 2 and two
assignments whose IoU sums to 1, the stored cost is 52.375. The total is 15.625, the non-box
share 26 and the box share -10.375.

### CompleteIoUWithoutAssignmentsOmitsTheIoUTerm

A CompleteIoU head with no assignment and an objectness weight of 1 stores 26.

### ZeroDeltaReportsZeroCost

A delta of 28 zeros reports 0 in the total, the box share, the non-box share and the stored
cost.

## modules/vision/detect/tests/test_detect_anchor_count.cpp

Suites: `Unit_DetectAnchorCount`.

Tests `choose_anchor_count`, which clusters to one more anchor per head until the gain in mean
best IoU falls below `min_gain`. The file uses three box sets: a ladder of six sizes with 400
boxes per rung, one blob of 2000 boxes, and a set of two sizes.

### LadderWithTwoHeadsChoosesThreePerHead

The ladder with two heads gives three anchors per head and six anchors. There are four trials,
for 1 to 4 per head. The gain of the second and of the third trial is at least `min_gain`, and
the gain of the fourth is below it.

### TrialsHoldTheFitOfEachCountClusteredDirectly

Each trial's mean best IoU equals, to 1e-9, that of `cluster_anchors` called with the trial's
count. The chosen clustering's mean equals `measure_mean_best_iou` over its anchors and the
mean of the chosen trial.

### BlobChoosesOnePerHead

The blob with three heads gives one anchor per head and three anchors. There are two trials,
and the gain of the second is below `min_gain`.

### MaxPerHeadOfTwoStopsAtTwo

With `max_per_head` 2 the ladder with two heads gives two per head, four anchors and two
trials, and the clustering's mean equals the second trial's.

### EqualBoundsMakeOneTrial

With `min_per_head` and `max_per_head` both 2 and three heads, the choice is two per head, six
anchors and one trial.

### TwoSizesTakeTheSecondAnchorAtTheDefaultGain

Two sizes with one head and `max_per_head` 2 give two per head. The first trial's mean is
11.625 / 21 and the second's is 1, both to 1e-6.

### TwoSizesKeepOneAnchorWhenTheGainIsBelowMinGain

With `min_gain` 0.5 the same boxes give one anchor per head and one anchor whose mean is
11.625 / 21. Both trials are returned, and the second's mean is 1.

### IgnoresTheCountOfTheClusteringOptions

With `clustering.count` set to 17, the two sizes still give two per head and two anchors.

### RefusesAMinPerHeadAboveMaxPerHead

`min_per_head` 3 with `max_per_head` 2 fails with `InvalidDescription`.

### RefusesOptionsOutsideTheirRange

A `min_per_head` of 0, a `min_gain` of -0.01, a NaN `min_gain` and a clustering
`max_iterations` of 0 each fail with `InvalidDescription`.

### RefusesAHeadCountOutsideItsRange

A head count of 0, of -1 and of the largest `int64_t` each fail with `InvalidDescription`.

### RefusesFewerDistinctBoxesThanTheFirstCount

Two sizes with three heads fail with `Incompatible`.

### RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

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

## modules/vision/detect/tests/test_detect_anchor_delta_activated.cpp

Suites: `Unit_DetectAnchorDeltaActivated`.

Tests the delta of one anchor head whose entries arrive activated. The case holds two slots
and two classes over a 64-pixel network side. Slot 0 is assigned the truth. Slot 1 overlaps the
truth with an IoU of 0.8 and is not assigned. Both publish an objectness of 0.3.

### AnIgnoredSlotIsPulledTowardItsBestIoU

Slot 1's objectness delta is 0.8 - 0.3, to 1e-6, and its `assigned_row` is -1.

### TheObjectWeightScalesAnIgnoredSlot

With an object weight of 0.5, slot 1's objectness delta is 0.5 * (0.8 - 0.3).

### TheTargetOneLeavesAnIgnoredSlotAlone

With `ObjectnessTarget::One`, slot 1's objectness delta is exactly 0.

### ASlotWithoutAGatedClassIsNotIgnored

With both class entries of slot 1 at 0.2, its objectness delta is 0 - 0.3.

### AnAssignedSlotKeepsADeltaThatIsNotZero

Slot 0's `assigned_row` is 0 and its objectness delta is 0 - 0.3.

### AnAssignedSlotWithAZeroDeltaIsPulledTowardOne

With slot 0's objectness set to exactly 0, its objectness delta is 1.

### TheTargetOnePullsAnAssignedSlotTowardOne

With `ObjectnessTarget::One`, slot 0's objectness delta is 1 - 0.3.

### TheBoxDeltaIsTheWeightedGradientWithNoSizeFactor

With `scale_xy` 2 and a box written into slot 0, the four box deltas equal the gradient at the
decoded box against the truth, weighted by 0.75, within 1e-5 absolute plus 1e-5 relative. The
expected width entry is above 0.01 in magnitude. Slot 1's four box deltas are exactly 0.

### TheClassDeltasReadTheEntriesAsProbabilities

Slot 0's class deltas are 1 - 0.6 and 0 - 0.2. Slot 1's class deltas are exactly 0.

## modules/vision/detect/tests/test_detect_anchor_delta_gradient.cpp

Suites: `Unit_DetectAnchorDeltaGradient`.

Tests the two forms of the box gradient a head's training options select. Each case decodes
one slot to a chosen box against one truth on a 20-pixel network side and reads the four box
deltas. Hand values are compared to 1e-5.

### TheDefaultFormIsAnalytic

With `box_gradient` taken from a default `HeadTrainingOptions`, a 0.2 square centred in a 0.4
square gets a width delta and a height delta of 1.25.

### TheCompatibleFormDoublesTheSizeSlopeOfTheOverlap

For the same boxes the analytic form writes 1.25 into width and height and the compatible form
writes 2.5. Both write 0 into x.

### TheCompatibleFormFlipsTheEnclosingTermOfTheCentre

For two disjoint equal squares the x delta is 0.18 / 0.2809 in the analytic form and
0.88 / 0.2809 in the compatible form. Both width deltas are 0.7 * 0.25 / 0.2809, and the
compatible form's y and height deltas are 0.

### TheCompatibleFormSimplifiesTheAspectTerm

For a 0.2 by 0.1 box centred in a 0.4 square, the case computes the aspect term and its
trade-off in double and expects the width and height deltas of each form from them. The
compatible form's are 1.25 and 2.5 moved by the trade-off times the slope. The analytic form's
scale the overlap part by one minus the squared trade-off.

### TheCompatibleFormReproducesTheDumpedDeltas

For each of the twelve cells in `K_DUMPED_CELLS`, with a box weight of 0.05, a `max_delta` of 2,
a match threshold of 0.2 and `ObjectnessTarget::BestIoU`, the compatible form's IoU sum equals
the cell's IoU to 1e-6 and each box delta equals the cell's to 1e-4 relative.

### TheAnalyticFormDepartsFromTheDumpedSizeDeltas

For the first dumped cell, the analytic form's width and height deltas each differ from the
cell's by more than 10 % of the cell's value.

## modules/vision/detect/tests/test_detect_anchor_delta_iou.cpp

Suites: `Unit_DetectAnchorDeltaIou`.

Tests the delta of one anchor head when more than one anchor may take a truth and the box
deltas come from the IoU gradient. Truths are given in pixels of a 64-pixel network side. A box
delta is compared with a value the file computes in double, within 1e-5 absolute plus 1e-5
relative.

### TwoAnchorsAboveTheMatchThresholdAreBothAssigned

With a match threshold of 0.5, the best slot and a second slot of the truth's cell both get the
box deltas of their own decoded box against the truth, an objectness delta of 0.5 and a class
delta of 1. The other ten slots get an objectness delta of -0.5 and no box delta. The result
counts two assignments, and only the best slot records the truth in `assigned_row` and
`assigned_class`.

### AFurtherAssignmentAddsItsIoUToTheSum

With zero logits in both assigned slots, the IoU sum is 1 + 0.2 / 0.29 to 1e-6.

### AMatchThresholdOfOneAssignsTheBestAnchorAlone

At the default threshold the best slot gets its box deltas, the second slot gets none and an
objectness delta of -0.5, and the result counts one assignment.

### AnAnchorBelowTheMatchThresholdIsNotAssigned

With a match threshold of 0.75, above the second anchor's shape IoU of 0.2 / 0.29, the result
counts one assignment and the second slot has no box delta.

### AHeadWithoutTheBestAnchorStillTakesAMatchingTruth

Head 1, which does not own the truth's best anchor, gets box deltas, an objectness delta of 0.5
and a class delta of 1 at a match threshold of 0.5. The result counts one assignment, and
`assigned_row` and `assigned_class` stay -1.

### TwoPositiveClassDeltasHalveTheBoxDeltasOfTheSlot

Two truths of different classes in one slot, at a match threshold of 0.5 and a box weight of
0.07, leave the slot with the sum of the two box deltas divided by 2. Both class deltas are
0.5, and the result counts two assignments.

### AMatchThresholdOfOneDividesNothing

At the default threshold the same slot holds the undivided sum of the two box deltas.

### TheDivisionCoversSquaredErrorBoxes

With `BoxLoss::SquaredError`, a match threshold of 0.5 and two truths that are the anchor at
the cell centre, the width delta is 2 * k * (0 - 0.5) divided by 2, with
k = (2 - 0.5 * 0.5) * 0.75, and the other three box deltas are 0.

### ABoxDeltaBeyondTheBoundIsClipped

With `max_delta` 0.5, each box delta whose magnitude exceeds 0.5 is 0.5 with its sign, and the
others are unchanged. The case asserts that at least one component is clipped and at least one
is not.

### ANonFiniteBoxComponentAddsZero

With a width logit of 200, which decodes to infinity, the width delta is exactly 0, all four
box deltas are finite and the objectness delta is 0.5.

## modules/vision/detect/tests/test_detect_anchor_fit.cpp

Suites: `Unit_DetectAnchorFit`.

Tests `measure_anchor_fit`, which measures how well the anchors of an `AnchorLayout` fit a list
of box sizes. Every number is compared to 1e-9.

### OneBoxPerAnchorGivesOneEverywhere

One box at each of three anchors' sizes gives 3 boxes, a mean best IoU, a best possible recall
and a tenth percentile of 1, an unowned share of 0 and a share of 1/3 per anchor.

### HalfSizedBoxesGiveAQuarterAndAreRecalledAtAQuarter

Boxes at half each anchor's sides give a mean best IoU and a tenth percentile of 0.25. The best
possible recall is 1 with `recall_iou` 0.25 and 0 with `recall_iou` 0.3.

### V4TinyMasksLeaveASixthOfTheBoxesUnowned

For a six-anchor layout whose two heads list five of the anchors, 100 boxes at each anchor's
size give 600 boxes, an unowned share of 1/6, a best possible recall of 5/6, a mean best IoU of
1 and a share of 1/6 per anchor.

### MasksOverEveryAnchorLeaveNoBoxUnowned

With the heads listing all six anchors, the same boxes give an unowned share of 0 and a best
possible recall of 1.

### FourNumbersEqualTheBoxByBoxSums

For four anchors, of which no head lists anchor 0, and twelve boxes, the case finds each box's
best anchor with `shape_iou` and sums by hand. The mean best IoU, the recall at 0.5, the unowned
share, the tenth percentile, which is the second smallest best IoU, and the four anchor shares
equal those sums. The case also asserts that some boxes are unowned, some recalled and some
neither.

### TenthPercentileOfTenBoxesIsTheSmallest

Nine boxes at the anchor's size and one at half its sides give a tenth percentile of 0.25 and a
mean best IoU of 0.925.

### EqualAnchorsGiveTheBoxToTheLowerIndex

With two equal anchors, of which the one head lists anchor 1, one box at that size gives the
shares 1 and 0, an unowned share of 1 and a best possible recall of 0.

### RefusesAnEmptyBoxList

An empty box list fails with `Incompatible`.

### RefusesALayoutThatDoesNotValidate

A layout whose head names an anchor past the list fails with `InvalidDescription`.

### RefusesARecallIoUOutsideZeroToOne

A `recall_iou` of -0.1, of 1.5 and of NaN each fail with `InvalidDescription`.

## modules/vision/detect/tests/test_detect_anchor_head_assignment.cpp

Suites: `Unit_DetectAnchorHeadAssignment`.

Tests `assign_anchors_to_heads`, which gives each head, in the order of its stride, a run of
anchor indices in ascending area and returns an `AnchorLayout`.

### NineAnchorsOverThreeHeadsGiveThreeRunsOfThree

Nine anchors over the strides 32, 16 and 8 give the masks {6, 7, 8}, {3, 4, 5} and {0, 1, 2}.

### SixAnchorsOverTwoHeadsGiveTwoRunsOfThree

Six anchors of rising area over the strides 32 and 16 give {3, 4, 5} and {0, 1, 2}.

### ReproducesTheMasksOfFiveCfgs

For five named cases the file holds, each an anchor list in pixels, an input side, the strides
and the expected masks, the computed masks equal the expected ones.

### DiffersFromTheSecondMaskOfYolov4TinyByOneAnchor

For six rising anchors over the strides 32 and 16, the first computed mask is {3, 4, 5}. The
second is {0, 1, 2}, which is not {1, 2, 3}.

### TheRemainderGoesToTheHeadOfSmallestStride

Seven anchors over the strides 32 and 16 give {4, 5, 6} and {0, 1, 2, 3}. Over the strides 16
and 32 they give {0, 1, 2, 3} and {4, 5, 6}.

### ARemainderOfTwoGoesToTheTwoSmallestStrides

Eleven anchors over the strides 16, 32 and 8 give {4, 5, 6, 7}, {8, 9, 10} and {0, 1, 2, 3}.

### UnsortedAnchorsKeepTheirOrderAndAreNamedByArea

Four anchors whose ascending area is 3, 1, 2, 0 give the masks {2, 0} and {3, 1} over the
strides 32 and 16. The layout's anchors are the inputs in their given order, compared exactly.

### AnchorsOfEqualAreaGoInIndexOrder

Two anchors of one area over the strides 8 and 16 give {0} and {1}.

### TheResultIsAValidLayout

`validate_anchor_layout` is ok for the layout of a nine-anchor, three-head assignment.

### PerHeadSetsTheCountOfEachHead

With `per_head` {2, 4} over the strides 32 and 16, six anchors give {4, 5} and {0, 1, 2, 3}.

### RefusesAPerHeadThatDoesNotSumToTheCount

For six anchors, `per_head` {2, 3}, {3, 4} and two counts of the largest `int64_t` each fail
with `Incompatible`.

### RefusesAPerHeadOfAnotherLengthThanTheStrides

For two strides, `per_head` {6} and {2, 2, 2} each fail with `Incompatible`.

### RefusesAPerHeadWithAnEmptyHead

`per_head` {6, 0} and {7, -1} each fail with `InvalidDescription`.

### RefusesFewerAnchorsThanHeads

Two anchors over three strides fail with `Incompatible`.

### RefusesMissingAnchorsAndBadStrides

No anchor, no stride, a stride of 0 and a stride of -8 each fail with `InvalidDescription`.

### RefusesAnAnchorWithoutAFinitePositiveSide

An anchor with a zero height and an anchor with a NaN width each fail with
`InvalidDescription`.

## modules/vision/detect/tests/test_detect_anchor_head_truth.cpp

Suites: `Unit_DetectAnchorHeadTruth`.

Tests `validate_head_truth` over a batch of one image with two truth rows, three classes and a
64 by 64 network. The row under test is row 1, given by its pixel corners and its class. Every
refusal has the code `InvalidDescription`.

### AcceptsRowsWhoseCentreLiesInsideTheInput

A row with the corners -8 and 8 on both axes, whose centre is exactly 0, and the class 2 is
accepted.

### RefusesACentreAtTheRightEdge

A centre of exactly 1 along the columns is refused; the message holds "truth row 1 of image 0",
"centre" and "(1, 0.5)".

### RefusesACentreLeftOfTheInput

A centre of -0.25 along the columns is refused, and the message holds "(-0.25, 0.5)".

### RefusesACentreBelowTheInput

A centre of 1.25 along the rows is refused, and the message holds "(0.5, 1.25)".

### RefusesACentreAboveTheInput

A centre of -0.25 along the rows is refused, and the message holds "(0.5, -0.25)".

### RefusesACentreThatIsNotANumber

A NaN corner is refused with a message holding "truth row 1 of image 0" and "centre".

### RefusesAWidthThatIsNotPositive

A row of zero width is refused, and the message holds "size 0 by 0.5".

### RefusesAHeightThatIsNotPositive

A row whose bottom lies above its top is refused, and the message holds "size 0.5 by -0.25".

### RefusesAClassAtTheClassCount

The class 3 with three classes is refused, and the message holds "class 3" and "3 classes".

### RefusesANegativeClass

The class -1 is refused, and the message holds "class -1".

### LeavesAPaddingRowUnread

A row with an out-of-range centre and class is accepted when its `valid` flag is 0.

### RefusesArraysThatDoNotHoldTheirShape

A batch count of 2 against a truth of one image is refused with "2 images" and "batch of 1". A
truth one box value short is refused with "8 box values" and "7".

### RefusesANetworkSideBelowOne

A network size of 64 by 0 is refused with "network size" and "64 by 0".

### TheDeltaRefusesATruthOutsideTheGrid

`compute_head_delta` given the row with a centre of exactly 1 fails with the same message
parts, and leaves `delta` and `assigned_row` of its result empty.

## modules/vision/detect/tests/test_detect_anchor_layout.cpp

Suites: `Unit_DetectAnchorLayout`.

Tests `shape_iou` over two `AnchorSize` values and `validate_anchor_layout` over a layout of
six anchors split between two heads.

### ShapeIoUOfASizeWithItselfIsOne

`shape_iou` of 0.3 x 0.7 with itself is 1 to 1e-6.

### ShapeIoUOfAContainedSizeIsTheRatioOfTheAreas

`shape_iou` of 0.2 x 0.4 and 0.1 x 0.2 is 0.25 in either order.

### ShapeIoUOfCrossedSizesCountsTheSharedRectangle

`shape_iou` of 0.4 x 0.1 and 0.1 x 0.4 is 1/7.

### ShapeIoUWithAZeroAreaIsZero

`shape_iou` is exactly 0 when either size has a zero side, and for two default sizes.

### AcceptsOneHeadPerGroupOfAnchors

`validate_anchor_layout` is ok for the two-head layout.

### AcceptsASharedAnchorAndAnAnchorNoHeadUses

With the masks {3, 4, 5} and {1, 2, 3}, where anchor 3 is shared and anchor 0 unused, the
layout is accepted.

### RefusesAnIndexPastTheAnchors

A mask entry of 6 with six anchors fails with `InvalidDescription`.

### RefusesANegativeIndex

A mask entry of -1 fails with `InvalidDescription`.

### RefusesAnEmptyHead

A head with an empty mask fails with `InvalidDescription`.

### RefusesALayoutWithoutAnchorsOrWithoutHeads

A layout with no anchor and a layout with no head each fail with `InvalidDescription`.

### RefusesAnAnchorWithoutAFinitePositiveSide

An anchor with a zero width, one with a NaN height and one with an infinite width each fail
with `InvalidDescription`.

## modules/vision/detect/tests/test_detect_anchor_plan.cpp

Suites: `Unit_DetectAnchorPlan`.

Tests `plan_anchors`, which chooses or takes a count per head, clusters the boxes, assigns the
anchors to the heads by stride, measures the fit and derives the assignment thresholds. The
file uses a ladder of six sizes with 400 boxes per rung, one blob of 2000 boxes and a set of
two sizes.

### LadderWithTwoStridesPlansTwoHeadsOfThreeAnchors

The ladder under the strides 16 and 32 plans six anchors with the masks {0, 1, 2} and
{3, 4, 5}. The fit counts every box, has a best possible recall of 1, an unowned share of 0 and
a mean best IoU above 0.95. The thresholds equal `derive_assignment_thresholds` of the fit: an
ignore threshold of 0.7 and a match threshold at the fit's tenth percentile. There are four
trials, the last for 4 per head.

### ChosenCountEqualsTheStepsMadeByHand

The plan equals, to 1e-9, a plan the case builds from `choose_anchor_count`,
`assign_anchors_to_heads`, `measure_anchor_fit` and `derive_assignment_thresholds` called in
that order.

### GivenCountTriesNoCount

With `per_head` 3 the plan has no trial, and its layout is the one `cluster_anchors` to six
anchors and `assign_anchors_to_heads` give. Each head has three anchors and the best possible
recall is 1.

### GivenCountIgnoresTheBoundsOfTheCountChoice

With `per_head` 3, a plan made with a `min_per_head` of 0, a `max_per_head` of -1, a NaN
`min_gain` and a clustering count of 17 is identical to one made with the default count
options.

### StridesInAnyOrderGiveEachHeadTheAnchorsOfItsStride

Under the strides 32 and 16 the masks are {3, 4, 5} and {0, 1, 2}, and with the two masks
swapped the plan is identical to the plan for 16 and 32. Under 16, 64 and 32 with two per head,
the masks are {0, 1}, {4, 5} and {2, 3}.

### TwoCallsGiveTheSamePlan

Two calls with the seed 0x5EED and the same arguments return identical plans.

### BlobPlansOneAnchorPerHead

The blob under the strides 8, 16 and 32 plans the masks {0}, {1} and {2}, with two trials and a
best possible recall of 1.

### RefusesNoStrideAndAStrideOfZero

An empty stride list and the strides 16 and 0 each fail with `InvalidDescription`, with a
chosen count and with `per_head` 1.

### RefusesOptionsOutsideTheirRange

Six option sets each fail with `InvalidDescription`: a `per_head` of -1 and of the largest
`int64_t`, a `min_per_head` of 0, a clustering `max_iterations` of 0 with and without a given
`per_head`, and a `recall_iou` of 2.

### RefusesBoxesThatDisagreeWithTheCount

An empty box list, with a chosen and with a given count, fails with `Incompatible`. So do two
sizes under three strides with a chosen count, and two sizes under two strides with two per
head.

## modules/vision/detect/tests/test_detect_anchor_review.cpp

Suites: `Unit_DetectAnchorReview`.

Tests `review_anchors`, which measures a current `AnchorLayout` on a box list, clusters a
proposal with the same counts per head and returns `AnchorVerdict::Keep` or
`AnchorVerdict::Propose`. The boxes are a ladder of six sizes with 400 boxes per rung, and the
shifted ladder is the same one scaled by 0.75.

### FittedLayoutIsKeptOnItsOwnBoxes

A layout fitted to the ladder and reviewed on the ladder gets `Keep`. Its best possible recall
is 1, and the proposal's mean best IoU equals the current one to 1e-12.

### ShiftedBoxesGetAProposalWithBothFits

Reviewed on the shifted ladder, the layout gets `Propose`. The reported current fit and
proposed fit equal `measure_anchor_fit` called directly on each layout, and their mean best IoU
and recall equal sums the case makes by hand to 1e-6. The proposal's gain in mean best IoU is
above `min_gain`.

### ProposalRecoversTheShiftedRungs

The proposal has six anchors, and each lies within 5 % of the width and the height of its
shifted rung.

### ProposalKeepsTheCountsPerHead

For a current layout with 2 and 4 anchors under the strides 32 and 16, the proposal has the
same anchor count and the masks {4, 5} and {0, 1, 2, 3}.

### ProposalIsTheClusteringAssignedToTheHeads

With `clustering.count` set to 99 in the options, the proposal still equals the layout fitted
directly to the shifted boxes with three anchors per head.

### InputLayoutIsUnchanged

After a review that returns `Propose`, the reviewed layout holds the anchors and masks it held
before.

### RecallAndGainEachDecideTheVerdict

With `min_gain` 10 the shifted review has a current recall of 1 and returns `Keep`. With
`recall_iou` raised to 0.6 as well, the current recall is 0, which equals the hand sum, and the
verdict is `Propose`.

### RefusesOptionsOutsideTheirRange

A `min_gain` of -0.01 and of NaN, a `min_recall` of 1.5 and of -0.1, and a `recall_iou` of 2
each fail with `InvalidDescription`.

### RefusesALayoutWithoutHeads

A layout with one anchor and no head fails with `InvalidDescription`.

### RefusesBoxesStridesAndCountsThatDisagree

Three sizes with a layout of masks {0} and {1, 2} under two strides are accepted. An empty box
list, four copies of one box, one stride for two heads, and a layout whose masks {0, 1} and
{1, 2} hold four slots for three anchors each fail with `Incompatible`.

## modules/vision/detect/tests/test_detect_anchor_task.cpp

Suites: `Unit_DetectAnchorTask`.

Tests `read_anchor_task` and `write_anchor_task` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_task.hpp`. A task is
compared field by field, every float bit for bit. A section is compared by the JSON text
`write_json` gives it. The refusal cases start from a valid section of three classes, four
anchors, two heads of two anchors each and the criterion "iou", and replace one member.

### AThreeHeadTaskRoundTripsThroughItsSection

A task of 80 classes, nine anchors and three heads, with `DistanceIoU` and an exponent of
0.6f, comes back unchanged after `write_anchor_task`, `write_json`, `parse_json` and
`read_anchor_task`.

### ATwoHeadTaskRoundTripsThroughItsSection

A task of 3 classes, six anchors and two heads under the plain IoU criterion comes back
unchanged after the same four steps.

### AThreeHeadSectionRoundTripsThroughItsTask

A section of three heads over nine anchors with the criterion "distance_iou" is written back
with the same JSON text after `read_anchor_task` and `write_anchor_task`.

### ATwoHeadSectionRoundTripsThroughItsTask

A section of two heads over six anchors with the criterion "iou" is written back with the
same JSON text.

### ReadsEveryFieldOfASection

The three-head section reads as 80 classes, its nine anchors, the masks {6, 7, 8}, {3, 4, 5},
{0, 1, 2}, and the heads head_0 and head_1 with `Logit` and the scales 1.25 and 1.125 and
head_2 with `Activated` and 2.0. The criterion is `DistanceIoU` with the exponent 0.5.

### WritesTheTypeTheKeysInOrderAndEachFloatAsItsDouble

The written section has the keys `type`, `classes`, `anchors`, `heads`, `suppression` in that
order, the type `K_ANCHOR_TASK_TYPE` and 3 classes. A head has the keys `output`, `anchors`,
`coding`, `scale_xy`. The second head's scale equals `double(1.05f)` and the third anchor's
sides equal `double(0.09f)` and `double(0.14f)`.

### ReadsAndWritesEachCodingString

"logit" reads as `HeadCoding::Logit` and "activated" as `HeadCoding::Activated`, and each is
written back as the same string.

### ReadsAndWritesEachCriterionString

"iou" reads as `SuppressionCriterion::IoU` and "distance_iou", with an exponent of 0.75, as
`SuppressionCriterion::DistanceIoU`. Each is written back as the same string.

### TheExponentBelongsToTheCriterionThatReadsIt

A task under IoU with the exponent 0.25 writes a `suppression` object with the one key
`criterion`, and that section reads back with `K_DEFAULT_DISTANCE_EXPONENT`. Under
`DistanceIoU` the object has the keys `criterion`, `distance_exponent` and the exponent 0.25.

### ReadsAWholeNumberAsAScale

A `scale_xy` of 2 reads as 2.0f.

### AcceptsAnAnchorTwoHeadsShare

Heads with the anchors [1, 2] and [0, 1] read with those two masks.

### RefusesASectionThatIsNotAnObject

`read_anchor_task` fails with `InvalidDescription` and "task" for an array ("an array") and
for a null value ("null").

### RefusesAnotherTypeOrNone

The read fails with `InvalidDescription` for the type "anchor_free_detection" ("'type'",
"'anchor_free_detection'", "'anchor_detection'"), for the type 3 ("'type'",
"a whole number") and for a section without a type ("task has no 'type'").

### RefusesAnUnknownKeyAtEachLevel

The read fails with `InvalidDescription` for the key `clases` in the section ("'clases'",
"'classes'"), the key `scale` in a head ("'heads' entry 0", "'scale'", "'scale_xy'") and the
key `threshold` in `suppression` ("'suppression'", "'threshold'").

### RefusesASectionWithoutOneOfItsMembers

A section with all five members reads. Without `classes`, `anchors`, `heads` or `suppression`
it fails with `InvalidDescription` and "task has no" followed by the quoted key.

### RefusesAClassCountBelowOneOrOfAnotherKind

`classes` 0 fails with `InvalidDescription`, "'classes'", "at least 1" and "0". `classes` 3.5
fails with "'classes'" and "a whole number".

### RefusesAnAnchorThatIsNotAPairOfNumbers

The read fails with `InvalidDescription` for `anchors` given as a number ("'anchors'",
"an array"), an anchor of three values ("'anchors' entry 1", "[width, height]", "3 values"),
an anchor given as a number ("'anchors' entry 2", "[width, height]") and a height given as a
string ("'anchors' entry 0 height", "a string").

### RefusesAnAnchorSideThatIsNotPositiveAndFinite

The read fails with `InvalidDescription` for a width of 0.0 ("'anchors' entry 2",
"above zero"), a height of -0.5 ("'anchors' entry 1", "above zero"), a width of 1e-60
("'anchors' entry 0", "above zero"), a height of 1e60 ("'anchors' entry 3 height", "float32")
and an empty list ("'anchors'", "no anchor").

### RefusesNoHeadAndAHeadThatIsNotAnObject

The read fails with `InvalidDescription` for an empty `heads` ("'heads'", "no head"), for
`heads` given as an object ("'heads'", "an object") and for a second head given as the number
7 ("'heads' entry 1", "a whole number").

### RefusesAMaskIndexThatNamesNoAnchor

Over four anchors the read fails with `InvalidDescription` for the index 4
("'heads' entry 1 'anchors' entry 2", "anchor 4", "4 anchors"), the index -1
("'heads' entry 0 'anchors' entry 0", "anchor -1"), the index 1.5
("'heads' entry 0 'anchors' entry 1", "a whole number") and an empty mask
("'heads' entry 0 'anchors'", "no anchor").

### RefusesAHeadWithABadOutputCodingOrScale

The read fails with `InvalidDescription` for an empty `output` ("'heads' entry 0 'output'",
"empty string"), a head without `output` ("'heads' entry 0", "no 'output'"), the coding
"linear" ("'heads' entry 1 'coding'", "'linear'", "'logit', 'activated'"), a `scale_xy` of 0.0
("'heads' entry 0 'scale_xy'", "above zero") and a `scale_xy` given as a string
("'heads' entry 0 'scale_xy'", "a string").

### RefusesABadCriterionOrExponent

The read fails with `InvalidDescription` for the criterion "giou"
("'suppression' 'criterion'", "'giou'", "'iou', 'distance_iou'"), an exponent stated under
"iou" ("'suppression' 'distance_exponent'", "'iou'", "does not read"), "distance_iou" without
an exponent ("'suppression'", "no 'distance_exponent'") and `suppression` given as an array
("'suppression'", "an array").

### WriterRefusesAHeadOfAnotherClassCount

A task of 3 classes whose second head has 4 fails in `write_anchor_task` with
`InvalidDescription`, "'heads' entry 1", "4 classes", "'classes'" and "3".

### WriterRefusesTheValuesTheReaderRefuses

`write_anchor_task` fails with `InvalidDescription` for 0 classes ("'classes'",
"at least 1"), an anchor height of NaN ("'anchors' entry 4", "above zero"), a mask index of 6
over six anchors ("'heads' entry 0 'anchors' entry 1", "anchor 6", "6 anchors"), no head
("'heads'", "no head"), one head for two masks ("'heads'", "1 heads", "2 masks"), an empty
output ("'heads' entry 0 'output'", "empty string") and an infinite scale
("'heads' entry 1 'scale_xy'", "finite").

### WriterRefusesAnEnumeratorWithoutAString

`write_anchor_task` fails with `InvalidDescription` for a `HeadCoding` of 7
("'heads' entry 1 'coding'", "7", "'logit', 'activated'") and for a `SuppressionCriterion` of
9 ("'suppression' 'criterion'", "9", "'iou', 'distance_iou'").

### WriterRefusesAnExponentThatIsNotFiniteUnderDistanceIoU

With an infinite exponent the task writes under IoU. Under `DistanceIoU` the write fails with
`InvalidDescription`, "'suppression' 'distance_exponent'" and "finite".

## modules/vision/detect/tests/test_detect_anchor_thresholds.cpp

Suites: `Unit_DetectAnchorThresholds`.

Tests `AssignmentThresholds` and `derive_assignment_thresholds`. Each fit in the file is given
by its tenth percentile and its mean best IoU, and thresholds are compared to 1e-6.

### DefaultsAreOneAndSevenTenths

A default `AssignmentThresholds` has a match threshold of exactly 1 and an ignore threshold of
exactly 0.7.

### FitInsideBothRangesGivesItsOwnNumbers

A tenth percentile of 0.35 and a mean of 0.62 give a match threshold of 0.35 and an ignore
threshold of 0.62.

### FourFitsFollowTheFormula

For the fits (0.35, 0.62), (0.21, 0.51), (0.5, 0.69) and (0.25, 0.55), the match threshold is
the tenth percentile and the ignore threshold is the mean.

### MatchThresholdIsClampedAtBothEnds

Tenth percentiles of 0.05, 0 and 0.2 give a match threshold of 0.2. Tenth percentiles of 1 and
1.3 give 1.

### IgnoreThresholdIsClampedAtBothEnds

Means of 0.3, 0 and 0.5 give an ignore threshold of 0.5. Means of 0.7 and 0.95 give 0.7.

### PerfectFitGivesOneAndSevenTenths

A fit of 1 in both numbers gives a match threshold of 1 and an ignore threshold of 0.7.

## modules/vision/detect/tests/test_detect_anchor_training_section.cpp

Suites: `Unit_DetectAnchorTrainingSection`.

Tests `read_head_training_section` and `write_head_training_section` in
`modules/vision/detect/include/SushiAI/vision/detect/anchor/head_training_section.hpp`.
Options are compared field by field, every float bit for bit. The full section states the
type "anchor" and all nine option keys.

### TheDefaultsRoundTripThroughTheirSection

A default `HeadTrainingOptions` comes back unchanged after `write_head_training_section`,
`write_json`, `parse_json` and `read_head_training_section`.

### OptionsWithNoDefaultRoundTripThroughTheirSection

Options with the thresholds 0.213f and 0.7f, `CompleteIoU`, `DarknetCompatible`, `BestIoU`,
the weights 0.07f and 0.4f, a `max_delta` of 5.0f and a class gate of 0.3f come back unchanged
after the same four steps.

### AFullSectionRoundTripsThroughItsOptions

The full section, read and written back, gives the same JSON text.

### ReadsEveryFieldOfASection

The full section reads as the thresholds 0.25 and 0.5, `BoxLoss::CompleteIoU`,
`BoxGradient::DarknetCompatible`, `ObjectnessTarget::BestIoU`, a box weight of 0.0625, an
object weight of 1.5, a `max_delta` of 5.0 and a class gate of 0.125.

### AKeyLeftOutKeepsTheDefaultOfItsField

An empty section and a section with the type alone both read as the default options. A
section with `max_delta` 2.5 alone reads as the defaults with that one field changed.

### WritesTheTypeEveryKeyInOrderAndAFloatAsItsDouble

The written section has the keys `type`, `match_threshold`, `ignore_threshold`, `box_loss`,
`box_gradient`, `objectness_target`, `box_weight`, `object_weight`, `max_delta`, `class_gate`
in that order. Its type is `K_ANCHOR_OBJECTIVE_TYPE` and its `box_loss` "complete_iou". The
match threshold equals `double(0.213f)` and the box weight `double(0.07f)`.

### ReadsAndWritesEachEnumeratorString

The writer gives "squared_error" and "complete_iou" for `BoxLoss`, "analytic" and
"darknet_compatible" for `BoxGradient`, and "one" and "best_iou" for `ObjectnessTarget`. A
section of "squared_error", "analytic" and "one" reads as `SquaredError`, `Analytic` and
`One`.

### ReadsAWholeNumberAsAWeight

An `object_weight` of 2 reads as 2.0f.

### RefusesASectionThatIsNotAnObject

An array and a null value both fail with `InvalidDescription`, "objective" and "an object".

### RefusesAnotherType

The type "cross_entropy" fails with `InvalidDescription`, "objective 'type'",
"'cross_entropy'" and "'anchor'". The type 3 fails with "objective 'type'" and "a string".

### RefusesAnUnknownKey

The key `iou_normalizer` fails with `InvalidDescription`, "objective", "'iou_normalizer'",
"'type', 'match_threshold'" and "'class_gate'". The key `thresholds` fails with "objective"
and "'thresholds'".

### RefusesAValueOfAnotherKind

The read fails with `InvalidDescription` for a `box_weight` given as a string
("objective 'box_weight'", "a number"), an `ignore_threshold` given as an array
("objective 'ignore_threshold'", "a number") and a `box_loss` of 1 ("objective 'box_loss'",
"a string").

### RefusesAStringNoEnumeratorHas

The read fails with `InvalidDescription` for the `box_loss` "ciou" ("objective 'box_loss'",
"'ciou'", "'squared_error', 'complete_iou'"), the `box_gradient` "exact"
("objective 'box_gradient'", "'exact'", "'analytic', 'darknet_compatible'") and the
`objectness_target` "iou" ("objective 'objectness_target'", "'iou'", "'one', 'best_iou'").

### RefusesANumberFloat32CannotHold

A `max_delta` of 1e39 fails with `InvalidDescription`, "objective 'max_delta'" and "float32".

### RefusesTheValuesTheOptionsCheckRefuses

The read fails with `InvalidDescription` and "objective" for a `match_threshold` of 0.0
("match threshold", "(0, 1]"), an `ignore_threshold` of 1.5 ("ignore threshold", "1.5"), a
`class_gate` of -0.25 ("class gate", "-0.25") and a `box_weight` of -1.0 ("box weight",
"-1").

### WriterRefusesTheValuesTheOptionsCheckRefuses

`write_head_training_section` fails with `InvalidDescription` and "objective" for an object
weight of NaN ("object weight"), an infinite `max_delta` ("max_delta") and a `BoxLoss` of 7
("BoxLoss", "7").

## modules/vision/detect/tests/test_detect_anchors.cpp

Suites: `Unit_DetectAnchors`, `Unit_DetectAnchorLeaves`, `Unit_DetectLevelRows`.

Tests the anchor grid, the per-image anchor leaves and the copies between a level and the joined
anchor axis.

This file needs no device. Its three suites, `Unit_DetectAnchors`, `Unit_DetectAnchorLeaves` and
`Unit_DetectLevelRows`, cover `modules/vision/detect/include/SushiAI/vision/detect/anchors.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/anchor_leaves.hpp` and `modules/vision/detect/include/SushiAI/vision/detect/level_rows.hpp`.
`modules/vision/detect/README.md` quotes the reference lines each unit ports. One case reads
`loss.sacp` through `tests/common/reference_fixture.hpp` and reports itself skipped when the
file is absent.

### K_EXACT

`K_EXACT` is the tolerance 0. Every expected value in the file is a multiple of 0.5, which
float32 holds exactly.

### K_REFERENCE_LEVELS

`K_REFERENCE_LEVELS` holds the three levels `tests/fixtures/yolo26/MANIFEST.json` records for a
640 px input: 80 by 80 at stride 8, 40 by 40 at stride 16 and 20 by 20 at stride 32.

### K_SMALL_LEVELS

`K_SMALL_LEVELS` is a grid of 2 by 3 cells at stride 8 followed by 1 by 2 cells at stride 16,
eight anchors in all, so a case can write every expected value out.

### CountsOneAnchorPerCell

`count_anchors` returns 8 for `K_SMALL_LEVELS` and 8400 for `K_REFERENCE_LEVELS`.

### OrdersPointsByLevelThenRowThenColumn

`make_anchors` over `K_SMALL_LEVELS` writes the sixteen point values and the eight strides the
case lists: the first level row by row, x before y, then the second level.

### ScalesEachPointByItsStride

`scale_points` multiplies both coordinates of a point by its anchor's stride. The centres 0.5,
1.5 and 2.5 become 4, 12 and 20 on the first level, and 0.5 and 1.5 become 8 and 24 on the
second.

### ScaleRejectsStridesThatDoNotMatchThePoints

Seven strides for eight points throw with a message containing "[A] strides" and "got 16, 7 and
16 values".

### RejectsALevelWithoutCellsOrStride

`count_anchors` throws on a level of height 0 with a message containing "height 0" and
"width 3", and on a level of stride 0 with a message containing "stride 0".

### RejectsOutputsOfTheWrongSize

`make_anchors` given a point array of 15 values throws with a message containing "16 point
values" and "got 15".

### MatchesTheReferenceAnchorPoints

The points of `K_REFERENCE_LEVELS`, scaled to pixels, equal the tensor `anchors.points` of
`loss.sacp` at tolerance 0, and that tensor has shape [8400, 2]. The case is skipped when
`loss.sacp` is absent.

### GivesEachLevelItsStrideOverItsAnchorRange

The strides of `K_REFERENCE_LEVELS` are 8 over anchors 0 to 6399, 16 over 6400 to 7999 and 32
over 8000 to 8399. The offsets 0, 6400 and 8000 are the `anchor_offset` entries of the manifest.

### RepeatsThePointsAndTheStridesForEveryImage

`build_anchor_leaves` for a batch of two writes the eight points of `K_SMALL_LEVELS` twice. It
writes the strides with four values per anchor: 24 values of 8 and 8 values of 16, also twice.

### SplitsIntoALevelsPointsAndStrides

`split_level` cuts the second level, anchors 6 and 7, out of both leaves. The points come out as
[2, 1, 2, 2] and the strides as sixteen values of 16.

### RejectsABatchWithoutImagesAndAModelWithoutLevels

A batch of 0 throws with a message containing "positive batch" and "batch 0 and 2 levels". An
empty level list throws with a message containing "at least one level" and "batch 1 and 0
levels".

### JoinPlacesALevelAtItsAnchorOffset

`join_level` with two images, five anchors of two values and a level at anchors 3 and 4 writes
the level's rows into values 6 to 9 of each image and leaves the other values at 0.

### SplitReturnsTheRowsJoinPlaced

The joined array holds its own index in every value. `split_level` for anchors 1 to 3 returns
values 2 to 7 of image 0 and 12 to 17 of image 1.

### RejectsALevelPastTheLastAnchor

A level of two anchors at offset 4 of five throws with a message containing "2 anchors",
"offset 4" and "5 anchors".

### SplitRejectsALevelArrayOfTheWrongSize

`split_level` into an array of 10 values, where the placement needs 12, throws with a message
containing "12 level values", "20 joined values" and "got 10 and 20".

## modules/vision/detect/tests/test_detect_assigner.cpp

Suites: `Unit_DetectAssigner`, `Unit_DetectGroundTruth`, `Unit_DetectTargetLeaves`.

Tests the task-aligned assigner, its presets, the padded ground truth and the target leaves.

This file needs no device. Its three suites are `Unit_DetectAssigner`, `Unit_DetectGroundTruth`
and `Unit_DetectTargetLeaves`, over `modules/vision/detect/include/SushiAI/vision/detect/assigner.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/assigner_config.hpp`, `modules/vision/detect/include/SushiAI/vision/detect/ground_truth.hpp` and
`modules/vision/detect/include/SushiAI/vision/detect/target_leaves.hpp`. Most cases build one image on a 4 by 4 grid by
hand, with `alpha` and `beta` both 1, so the metric is the class score times the overlap and
each expected score is a fraction a comment derives. Three cases read `loss.sacp` and report
themselves skipped when the file is absent. `modules/vision/detect/README.md`, under
"Reference tensors the assigner tests read", lists the tensors.

### Tolerances

`K_EXACT` is 0, for copied values and multiples of 0.5. `K_SCORE_TOLERANCE` is 1e-6, eight
float32 steps at 1, for a target score computed by hand. `K_REFERENCE_SCORE_TOLERANCE` is 1e-5
for a reference target score, where `std::pow` and `std::atan` may round apart from torch.
`K_REFERENCE_BOX_TOLERANCE` is 1.3e-4, two float32 steps at 640, for a reference box in pixels.

### The hand-built grid

`K_GRID_SIDE` is 4 cells, `K_GRID_ANCHORS` is 16, `K_GRID_STRIDE` is 8 px and `K_CLASSES` is 2.
`K_FAR_BOX` is the box (100, 100, 101, 101), which no hand-built ground truth touches.

### Scene

`Scene` holds one image on that grid: anchor `a` sits in column `a % 4` and row `a / 4`, with
pixel centres 4, 12, 20 and 28. Its constructor gives every anchor one score for both classes
and the prediction `K_FAR_BOX`. `add_truth` appends a valid ground-truth box, `predict` sets one
anchor's predicted box, and `inputs` returns the scene as an `AssignerInputs`.

### make_hand_config

`make_hand_config` returns an `AssignerConfig` with the given `topk` and `topk2`, two classes,
`alpha` 1, `beta` 1 and `stride_val` 8.

### foreground_anchors

`foreground_anchors` returns the rows an `Assignment` marks as foreground, in ascending order.

### make_crowded_scene

`make_crowded_scene` returns a scene with one box of class 1 over the whole 32 px image. Every
anchor predicts that box and scores 0.1, except anchors 5, 10 and 2, which score 0.9, 0.6 and
0.3 in class 1.

### make_contested_scene

`make_contested_scene` returns a scene with two boxes, of classes 0 and 1, that both contain
anchor 5. Anchors 0, 1 and 4 predict the first box, anchors 6, 9 and 10 the second, and anchor 5
predicts the box the caller passes.

### box_row

`box_row` returns the four values of one row of a `[rows, 4]` array.

### make_idle_assignment

`make_idle_assignment` returns an `Assignment` of the given batch and anchor count with no
foreground and every array at zero.

### set_foreground

`set_foreground` marks one row of an `Assignment` as foreground and gives it a pixel box and two
class scores.

### Reference constants

`K_REFERENCE_LEVELS` holds the manifest's three levels for a 640 px input,
`K_REFERENCE_IMAGE_SIDE` is 640, and `K_SMALL_BOX_ROW` is 3, the ground-truth row of the
fixture that is smaller than a stride.

### pad_reference_truth

`pad_reference_truth` reads `batch.batch_idx`, `batch.cls` and `batch.bboxes` from `loss.sacp`
and pads them with `pad_ground_truth` for one 640 px image. It returns an empty string, or the
reason a tensor could not be read.

### count_holders

`count_holders` counts the foreground anchors of a mask whose target index is a given
ground-truth row. It is a template over the mask's element type, so it takes the assigner's
mask and the reference's.

### expect_reference_assignment

`expect_reference_assignment` runs one branch's assigner on the reference's scores, boxes and
anchor points and compares all five outputs: the mask, the indices and the labels for equality,
the boxes within `K_REFERENCE_BOX_TOLERANCE` and the scores within
`K_REFERENCE_SCORE_TOLERANCE`. It then compares the foreground count and the count of anchors
that hold box 3 with the two numbers the caller passes, and the second with the same count
taken from the reference. Equality of the mask and the indices relies on the fixture having no
tied metrics. The helper skips when `loss.sacp` is absent.

### OneToManyPresetMatchesTheReferenceSettings

`one_to_many_assigner_config(80, {8, 16, 32})` has `topk` 10, `topk2` 10, 80 classes, `alpha`
0.5, `beta` 6, `stride_val` 16 and `eps` 1e-9. These are the `assigner.one2many` values of
`loss.sacp` in `tests/fixtures/yolo26/MANIFEST.json`.

### OneToOnePresetMatchesTheReferenceSettings

`one_to_one_assigner_config(80, {8, 16, 32})` has `topk` 7, `topk2` 1, `alpha` 0.5, `beta` 6 and
`stride_val` 16, the `assigner.one2one` values of the manifest.

### PresetTakesStrideValFromTheSecondLevel

`stride_val` is 12 for the strides 4 and 12, 4 for the one stride 4, and 16 for an empty stride
list.

### ConfigDefaultsMatchTheReferenceConstructor

A default `AssignerConfig` has `topk` 13, `topk2` 13, 80 classes, `alpha` 1, `beta` 6,
`stride_val` 16 and `eps` 1e-9, the arguments of the reference's `TaskAlignedAssigner.__init__`.

### RejectsAConfigWithoutCandidatesOrWithANegativeMargin

Constructing a `TaskAlignedAssigner` throws for `topk` 0, with a message containing "positive
topk" and "got 0, 3 and 2", for `stride_val` -1, with "stride_val", "at least 0" and "-1", and
for `eps` -1, with "eps of at least 0" and "-1".

### TheMetricRaisesTheScoreToAlphaAndTheOverlapToBeta

The case uses the one-to-many preset, so `alpha` is 0.5 and `beta` is 6. Anchor 5 predicts the
box exactly with score 0.25, so its metric is sqrt(0.25) times 1, which is 0.5. Anchor 10
predicts a 24 px box with score 0.81; its overlap is 576 / 1024 - 32 / 2048, which is 35 / 64,
and its metric is 0.9 times (35 / 64)^6, about 0.024. With `topk` 1 the foreground is anchor 5
alone, and with `topk` 2 it is anchors 5 and 10. The target score of anchor 5 is 1 and that of
anchor 10 is its metric over 0.5. An assigner that used the scores or the overlaps unraised
gives another second score.

### NormalisesByTheLargestOverlapOfTheBox

Anchors 0 and 1 both predict a box offset by 8 px from a 16 px truth, with scores 0.8 and 0.4.
The overlap is 64 / 448 - 128 / 1152, which is 2 / 63. The target scores are that overlap and
half of it: each metric over the largest metric, times the largest overlap of the box.

### APredictionApartFromItsBoxHasNoOverlap

Anchors 0 and 1 score 0.9 and predict `K_FAR_BOX`, whose complete IoU with the box is negative
and is clamped to 0. With `topk` 2 the foreground is anchors 4 and 5, which predict the box.
With `topk` 3 the third place goes to anchor 0, the first candidate of metric zero, and its
target score is 0.

### AContestedAnchorOfEqualOverlapsGoesToTheFirstBox

Two ground-truth rows hold the same box with classes 0 and 1, and anchor 5 predicts it. Anchor 5
is the one foreground anchor, its target index and label are 0, its score in class 0 is 1 and
its score in class 1 is 0.

### AContestedAnchorCanGoToABoxThatDidNotKeepIt

With `topk` 1, boxes 0 and 1 both keep anchor 5 and box 2 keeps anchor 2. Anchor 5 predicts box
2 exactly, so its largest overlap is with box 2 and it goes there, with target index 2 and
label 1. Its score in class 1 is 0.1 / 0.9 and its score in class 0 is 0. The case pins that
the maximum overlap is taken over every box, including one that did not keep the anchor.

### AnImageWithoutGroundTruthGetsNoForeground

A scene with no ground truth has no foreground anchor. Every target label is 2, the class
count, and every index, box and score is 0.

### APaddingRowAssignsNothing

The crowded scene's one box is marked invalid. No anchor is foreground and every target score
is 0, although the box contains all sixteen anchors.

### KeepsTheTopKCandidatesWhenABoxHoldsMore

In the crowded scene every overlap is 1, so the metric is the class score. With `topk` 3 the
foreground is anchors 2, 5 and 10, with target scores 0.3 / 0.9, 1 and 0.6 / 0.9 in class 1 and
0 in class 0. Every target label is 1, the label of ground-truth row 0, on and off the
foreground.

### TheSecondTopKKeepsTheBestAnchorOfABox

With `topk` 3 and `topk2` 1 the crowded scene keeps anchor 5 alone, with target score 1.

### AContestedAnchorGoesToTheHigherOverlap

Anchor 5 predicts the second box of the contested scene. The foreground is anchors 0, 1, 4, 5,
6, 9 and 10, and the target index and label of anchor 5 are 1. Its score is 1 in class 1 and 0
in class 0, and its target box starts at x = 8.

### AContestedAnchorCanGoToTheFirstBox

The same scene with anchor 5 predicting the first box gives anchor 5 the target index 0, score
1 in class 0 and 0 in class 1.

### ASmallBoxIsEnlargedToStrideVal

A 2 px box centred at 14 contains no anchor centre. With `stride_val` 16 it is tested as the
span 6 to 22, which holds the centres 12 and 20 on both axes, so anchors 5, 6, 9 and 10 are
foreground. With `stride_val` 1 there is no foreground. The target box of anchor 5 still starts
at 13: the enlargement applies to the candidate test and not to the target.

### AssignsEachImageFromItsOwnRows

A batch of two holds an image with one invalid row and the crowded scene. The foreground rows
are 18, 21 and 26, which are anchors 2, 5 and 10 of image 1. The target label of row 0 is 0 and
that of row 16 is 1.

### RejectsTooFewAnchorsAndAnUnknownClass

`assign` with `topk` 17 on sixteen anchors throws with a message containing "16 anchors" and
"topk 17". A valid box of class 2 with two classes throws with "class 2 of 2".

### PadsLabelRowsPerImage

Three label rows, two for image 0 and one for image 2, are padded to a batch of three 64 by 32
images with `max_boxes` 2. The boxes are converted from normalised centre and size to pixel
corners, the labels are 3, 1, 0, 0, 7, 0 and the valid flags are 1, 1, 0, 0, 1, 0.

### MarksARowWithoutExtentInvalid

A row whose four values are 0 is padded in with its label 4 and the valid flag 0. The row after
it is valid.

### PadsNoRowsToABatchWithoutBoxes

With no label rows and a batch of 2, `max_boxes` is 0 and the labels, boxes and flags are empty.

### RejectsARowOutsideTheBatch

A row naming image 3 in a batch of 3 throws with a message containing "row 0 names image 3" and
"batch of 3".

### BuildsTheLossLeavesOfAnAssignment

The crowded scene is assigned with `topk` 3 and its leaves are built for an image 64 px wide and
32 px high. The three weights are 1, 2/3 and 1/3, so `target_scores_sum` is 2 and the
normalised weights of anchors 5 and 10 are 1/2 and 1/3. Row 5 has a box mask of four ones, the
target box (0, 0, 4, 4) in grid units, the target distances (1.5, 1.5, 2.5, 2.5) and an
`l1_scale` of 8 / 64 * 0.25 * 0.5 across and 8 / 32 * 0.25 * 0.5 down. The unequal width and
height separate the two `l1_scale` values.

### ReadsEachRowsAnchorAcrossImagesAndLevels

Two levels give five anchors, and a hand-built assignment for two images marks rows 1 and 9.
Row 1 is anchor 1 of image 0, at point (1.5, 0.5) on stride 8. Row 9 is anchor 4 of image 1, at
point (0.5, 0.5) on stride 16. Each row's target box, distances and `l1_scale` are computed
from its own point and stride, and `target_scores_sum` is 1.5. Row 4, the same anchor as row 9
in image 0, has a box mask of zeros.

### AnUnassignedRowCarriesTheSafeBoxAndNoWeight

Row 0 of the crowded scene is off the foreground. Its target box is (0, 0, 1, 1), and its box
mask, `l1_scale`, target distances, weight and normalised weight are 0. `safe_boxes` holds
(0, 0, 1, 1) in row 5 as well, which is a foreground row.

### TheScoreSumIsAtLeastOne

With no ground truth, `target_scores_sum` is 1.

### PadsTheReferenceLabelsToItsTargetBoxes

The reference labels pad to four valid rows. Gathering the padded boxes and labels by the
reference's `one2many.assignment.target_indices` gives its `target_boxes` within
`K_REFERENCE_BOX_TOLERANCE` and its `target_labels` exactly. The case is skipped when
`loss.sacp` is absent.

### ReproducesTheReferenceOneToManyAssignment

`expect_reference_assignment` runs the one-to-many preset and expects 31 foreground anchors,
the manifest's `foreground_anchor_count.one2many`, of which 4 hold box 3. The case is skipped
when `loss.sacp` is absent.

### ReproducesTheReferenceOneToOneAssignment

The same helper runs the one-to-one preset and expects 4 foreground anchors, the manifest's
`foreground_anchor_count.one2one`, of which 1 holds box 3. The case is skipped when `loss.sacp`
is absent.

## modules/vision/detect/tests/test_detect_average_precision.cpp

Suites: `Unit_AveragePrecision`.

Tests detection matching, the 101-point average precision and the validation table.

This file needs no device and reads no fixture. Every case in `Unit_AveragePrecision` builds
detection rows and ground-truth boxes by hand and calls `match_detections`,
`integrate_precision` or the `AveragePrecision` accumulator of
`modules/vision/detect/include/SushiAI/vision/detect/average_precision.hpp`. A comment in each case derives the expected
number from the recall and precision the detections produce.

### K_TOLERANCE

`K_TOLERANCE` is 1e-12, the absolute tolerance of a metric, about 4500 float64 steps at 1.

### K_FULL_AP

`K_FULL_AP` is 0.995, the area under an envelope that is 1 up to recall 1. The integration
samples 101 recall points and the envelope is 0 at the last one, so the area is
0.99 + 0.01 / 2 and not 1.

### The boxes

`K_BOX_A` is the 8 by 8 box at the origin. `K_BOX_B` and `K_BOX_FAR` are 8 by 8 boxes that
touch neither it nor each other. `K_BOX_LOOSE` is the 8 by 5 box whose IoU with `K_BOX_A` is
40 / 64, which is 0.625, and `K_BOX_NEAR` is the 8 by 7 box whose IoU with it is 56 / 64, which
is 0.875. `K_NO_VALUES` and `K_NO_CLASSES` are empty arrays.

### make_row

`make_row` returns one detection row: a box, then a score and a class.

### join

`join` lays several arrays end to end.

### match

`match` calls `match_detections` for one image and returns the flags. It sizes the array from
`K_DETECTION_VALUES` and fills it with 7 first, so a flag the function does not write is
neither 0 nor 1.

### add_image

The file-local `add_image` passes three vectors to `AveragePrecision::add_image`.

### summarize_image

`summarize_image` returns the table of an accumulator that holds one image.

### HoldsTheTenReferenceThresholds

`K_IOU_THRESHOLDS` has ten entries, and each equals a hexadecimal float literal that carries
the float32 bits of `torch.linspace(0.5, 0.95, 10)`. The comparison is exact, so a threshold
that rounds to a neighbouring float32 fails it.

### IntegratesAnEmptyCurveToOneHalf

`integrate_precision` of two empty curves is 0.5. With no points the envelope is the line from
1 at recall 0 to 0 at recall 1.

### IntegratesUnderThePrecisionEnvelope

Recall 0.5, 0.5, 1 with precision 1, 0.5, 2/3 integrates to
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3. The envelope lifts the dip to 0.5 up to the 2/3 that
follows it at higher recall.

### RejectsCurvesOfUnequalLength

Two recall values with one precision value throw with a message containing "1 and 2".

### PerfectDetectorScoresTheInterpolationCeiling

Two images each hold one box, of classes 0 and 1, and one detection that equals it. Both rows
have precision 1, recall 1 and `K_FULL_AP` in both AP columns, and the four means are the same
numbers.

### NoDetectionsScoreZero

One box of class 0 and no detection give one row, with label 0 and one instance. Precision,
recall and both AP columns are 0 in the row and in the means.

### FalsePositiveAboveTheTruePositiveHalvesPrecision

A miss at score 0.9 ranks above a hit at 0.6. Recall is 0 then 1 and precision 0 then 0.5, so
the AP is 0.99 * 0.5 + 0.01 * 0.25, which is 0.4975, at every threshold. The mean F1 is flat
below confidence 0.6, so the first confidence point is read: precision 0.5 and recall 1.

### ReadsPrecisionAndRecallAtTheBestF1Confidence

A hit at 0.9 ranks above a miss at 0.3. F1 rises from 2/3 at confidence 0.3 to 1 at 0.9 and is 0
above it, so the smoothed curve is largest at point 849 of 1000. Precision there is
1 - 0.5 * (0.9 - 849 / 999) / (0.9 - 0.3), and the case also compares it with the literal
0.9582082255854392. Recall is 1. The row and the means hold the same values. Reading the first
confidence point gives precision 0.5.

### TwoDetectionsOnOneBoxCountOneMatch

Two detections equal `K_BOX_A` and a third equals `K_BOX_B`, all of class 0. The flags are 1
for the first, 0 for the second and 1 for the third at all ten thresholds. The AP at IoU 0.5 is
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3, precision is 2/3 and recall is 1. Three matches give
0.995.

### ClassWithoutGroundTruthIsLeftOutOfTheMean

A detection of class 3 has no ground truth of its class. The table has one row, for class 0,
and the means are `K_FULL_AP`, 1 and 1. A row for class 3 halves every mean.

### ClassWithoutDetectionsCountsAsZeroInTheMean

Classes 0 and 1 both have a box and only class 0 has a detection. The row of class 1 has AP 0,
and the means are 0.4975 for both AP columns and 0.5 for precision and recall.

### EqualScoresKeepTheOrderTheyWereAddedIn

A miss and a hit share the score 0.5. Added miss first, the AP at IoU 0.5 is 0.4975; added hit
first, it is `K_FULL_AP`. The ranking is therefore stable.

### LooseDetectionCountsAtTheLowThresholdsOnly

A detection at IoU 0.625 has the flags 1, 1, 1 and then seven zeros. Its AP at IoU 0.5 is
`K_FULL_AP` and its mean over the ten thresholds is 3 * 0.995 / 10, which is 0.2985.

### ImageWithoutGroundTruthAddsFalsePositives

The second image has no ground truth and one detection at score 0.9, which ranks above the
first image's hit at 0.6. The summary counts 2 images and 1 instance, the class row counts 1
image, the AP at IoU 0.5 is 0.4975, precision is 0.5 and recall is 1.

### ImageWithoutDetectionsLowersRecall

The second image has a box and no detection. `image_count()` is 2, the class row counts 2
images and 2 instances, recall is 0.5, precision is 1, and both AP columns are 0.49 + 0.005,
which is 0.495.

### DetectionWhoseBestBoxIsTakenStaysUnmatched

The first detection takes `K_BOX_A`. The second overlaps `K_BOX_A` by 0.875 and `K_BOX_LOOSE`
by 40 / 56, about 0.714. Its flags are ten zeros: a detection is matched to its box of largest
IoU or to none.

### DetectionTakesTheBoxOfLargestIoU

The boxes are listed `K_BOX_LOOSE` then `K_BOX_A`, and one detection equals `K_BOX_NEAR`. Its
flags are eight ones and two zeros, which is the IoU 0.875 it has with the box listed second. A
matcher that took the first box passing 0.5 gives five ones.

### DetectionWithNaNBoxDoesNotMatch

A detection whose x1 is NaN has a NaN IoU with `K_BOX_A`, and its flags are ten zeros. The row
is built by hand because `make_row` takes a box constant.

### BoxGoesToItsFirstDetectionAtEachThreshold

A detection at IoU 0.625 ranks above one at IoU 1. The first holds the box at 0.5, 0.55 and
0.6; the second gets it from 0.65 up. The flags are 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 and
0, 0, 0, 1, 1, 1, 1, 1, 1, 1.

### DetectionOfAnotherClassDoesNotMatch

A detection of class 1 that equals a box of class 0 has ten zero flags.

### NoGroundTruthGivesNoRows

An empty accumulator has no rows. After an image with one detection and no ground truth it
counts 1 image, has no rows, and its four means are 0.

### ClearDropsEveryImage

An image with a miss is added and `clear()` is called. `image_count()` is 0. An image with a
hit added afterwards gives 1 image, 1 instance and AP `K_FULL_AP`, so nothing of the first
image remains.

### RejectsMalformedImages

`add_image` throws for a detection array of 5 values ("[N, 6]", "got 5"), for one box with two
classes ("4 values", "2 classes"), for a ground-truth class of -1 ("box 0", "negative class
-1"), for a detection class of 0.5 ("detection 0", "not a class index") and for a NaN score
("detection 0", "score"). The accumulator still holds no image and no rows afterwards.
`match_detections` throws for a box without a class ("0 classes") and for a flag array of 9
values ("need 10 values", "got 9").

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

## modules/vision/detect/tests/test_detect_iou_gradient.cpp

Suites: `Unit_DetectIouGradient`.

Tests the box gradient declared in
`modules/vision/detect/include/SushiAI/vision/detect/iou_gradient.hpp` in its two forms,
`BoxGradient::Analytic` and `BoxGradient::DarknetCompatible`. A box is a centre and a size.
The analytic form is compared with a central difference of step 1e-6, within 1e-6 relative
plus 1e-7 absolute. Hand values are compared to 1e-5.

### AnalyticMatchesTheDifferenceOnOverlappingBoxes

The analytic gradient equals the central difference for two pairs of boxes that overlap
without one holding the other.

### AnalyticMatchesTheDifferenceOnDisjointBoxes

The same holds for a pair apart along one axis and a pair apart along both.

### AnalyticMatchesTheDifferenceOnNestedBoxes

The same holds for a box inside its target and for a box around its target.

### AnalyticMatchesTheDifferenceOnSeededPairs

The same holds for 100 pairs drawn from a seeded sequence, with centres in [0.2, 0.8] and sides
in [0.1, 0.5]. A pair that lies near a kink of the measure is drawn again and not counted.

### DarknetCompatibleMatchesTheDumpedDeltas

For each of the twelve cells in `K_DUMPED_CELLS`, the compatible gradient times the box weight
0.05 equals each of the cell's deltas to 1e-4 relative.

### DarknetCompatibleDoublesTheSizeSlopeOfTheOverlap

For a 0.2 square centred in a 0.4 square, the analytic width and height components are 1.25
and the compatible ones 2.5. Both x components are 0.

### DarknetCompatibleFlipsTheEnclosingTermOfTheCentre

For two disjoint equal squares, the x component is 0.18 / 0.2809 in the analytic form and
0.88 / 0.2809 in the compatible form. Both width components are 0.7 * 0.25 / 0.2809, and the
compatible y and height components are 0.

### DarknetCompatibleSimplifiesTheAspectTerm

For a 0.2 by 0.1 box centred in a 0.4 square, the case computes the aspect term and its
trade-off in double, with `K_ASPECT_TRADE_OFF_EPS`, and expects the width and height components
of each form from them.

### BothFormsAreZeroForTwoBoxesAtOnePoint

For two boxes of zero size at one point, every component of both forms is exactly 0.

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

## modules/vision/detect/tests/test_detect_suppression_criterion.cpp

Suites: `Unit_DetectSuppressionCriterion`, `Unit_DetectSuppressionCriterionDeathTest`.

Tests `Detect::measure_suppression`, the six-argument `Detect::suppress_overlaps`, and the
criterion fields of `Detect::NMSConfig`. Boxes are corner pairs, and most cases measure against
one 10 by 10 square. Measures are compared to 1e-6.

### IoUOfOverlappingBoxesIsTheRatioOfAreas

The square and a copy shifted by 1 along x have an IoU of 90 / 110.

### IoUIsZeroForDisjointAndTouchingBoxes

A box 10 units away and a box that shares only an edge with the square each have an IoU of
exactly 0.

### IoUOfTwoEmptyBoxesIsZeroNotNaN

Two boxes of zero area at one point have an IoU that is not NaN and is exactly 0.

### DistanceIoUSubtractsTheNormalisedCentreDistance

For the square and a copy shifted down by 6, with the exponent 0.6, the measure is
0.25 - (36 / 356)^0.6.

### DistanceIoUOfDisjointBoxesIsNegative

For the square and a box 20 units to its right, the measure is -(0.4^0.6).

### ZeroExponentSubtractsOne

With the exponent 0 the measure of the square and its shifted copy is 0.25 - 1.

### DistanceIoUOfIdenticalBoxesIsOne

The square measured against itself gives 1.

### DistanceIoUOfCoincidentPointsIsTheIoU

Two boxes of zero area at one point measure exactly 0, and the value is not NaN.

### DistanceIoUIsSymmetric

The measure of the square and a diagonally shifted box is the same in both orders.

### DistanceIoUKeepsTheFartherOfTwoEqualIoUBoxes

Of three boxes scored 0.9, 0.8 and 0.7, at a threshold of 0, the DistanceIoU criterion keeps
rows 0 and 1. The IoU criterion keeps row 0 alone.

### IoUCriterionKeepsABoxAtTheThreshold

Two boxes with an IoU of exactly 0.5 are both kept at a threshold of 0.5. At a threshold of
0.25 only the first is kept.

### ZeroScoreRowsAreNeverKept

Of three boxes that do not overlap, scored 0.9, 0 and 0.5, the rows 0 and 2 are kept.

### EmptyInputKeepsNothing

With no box and no score, a `kept` vector that held one entry comes back empty.

### FourArgumentFormEqualsTheIoUCriterion

For four boxes that do not overlap, scored 0.1, 0.9, 0.5 and 0.5, the four-argument
`suppress_overlaps` keeps the rows 1, 2, 3, 0 in that order, and the six-argument form with the
IoU criterion keeps the same.

### SelectionAppliesTheConfiguredCriterion

`select_nms_detections` with the DistanceIoU criterion, a confidence threshold of 0.1 and an
IoU threshold of 0 over the three boxes of the equal-IoU case, given as centre and size,
returns one image with two detections: the corners (0, 0, 10, 10) scored 0.9 and
(0, 6, 10, 16) scored 0.8, both of class 0.

### NMSConfigDefaultsToTheIoUCriterion

A default `NMSConfig` has the criterion `SuppressionCriterion::IoU` and a `distance_exponent`
of 0.6.

### RejectsBoxesOfTheWrongSizeFatally

Five box values with two scores stop the process with a message holding "a suppression needs"
and "got 5 and 2 values".

### RejectsAnUnknownCriterionFatally

A criterion cast from 7 stops the process with a message holding "must be IoU or DistanceIoU"
and "got 7".

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

## modules/core/autograd/tests/test_dropout_backward.cpp

Suites: `Unit_DropoutBackward`.

Tests the backward rule of DROPOUT. The graph cases apply `GraphBuilder::dropout` with the rate
0.5 and the seed 11 to a [3, 4] value and take the `nll_loss` of the `log_softmax` of the
result.

### DropoutHasARule

`Ops::K_DROPOUT` is differentiable, does not block the gradient and has a backward rule.
`Autograd::K_RANDOM_RULES` has one row, for DROPOUT, and its rule is
`Autograd::dropout_backward`.

### TheInputGradientIsTheGradientTimesTheSavedMask

With x as a parameter, the gradient of x is written by a MUL node of two inputs whose second
input is the mask the forward node saved, and it has the shape [3, 4]. One node reads the mask
and the graph holds one DROPOUT node.

### AnInputThatNeedsNoGradientAddsNoNode

With x as an input, the gradient is `K_NO_VALUE` and no node reads the saved mask.

## modules/core/graph/tests/test_dropout_mask.cpp

Suites: `Unit_DropoutMask`.

Tests the functions of `modules/core/graph/include/SushiAI/graph/dropout_mask.hpp`:
`splitmix64_at`, `dropout_site_seed`, `dropout_bits`, `dropout_keeps` and `fill_keep_mask`.
Every case but the first two draws from the site seed 2026. A mask is compared as a string of
'1' for a kept element and '0' for a dropped one.

### TheSequenceOfSeedZeroIsThePublishedSplitMix64

`splitmix64_at(0, i)` for i from 0 to 3 is 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4,
0x06C45D188009454F and 0xF88BB8A8724C81EC.

### EachSiteOfATraceHasItsOwnSeed

`dropout_site_seed(0, 0)` and `dropout_site_seed(0, 1)` are the first two numbers of the case
above. `dropout_site_seed(7, 0)` is 0x63CBE1E459320DD7 and `dropout_site_seed(7, 1)` is
0x044C3CD7F43C661C.

### TheBitsOfASeedARunAndAnElementAreGolden

`dropout_bits` for elements 0 to 3 of runs 0 and 1 equals eight 64-bit constants written in the
test.

### TheMaskOfARunIsGolden

At a keep of 0.5 the first 32 elements of run 0 are "11110000011001111110011111110000" and those
of run 1 are "01110100100110000111111111000011".

### TheFillAgreesWithTheElementwiseQuestion

For 257 elements of run 5 at a keep of 0.7, each element of the filled mask is 1 exactly when
`dropout_keeps` returns true for it.

### AShorterFillIsAPrefixOfALongerOne

A fill of 40 elements of run 2 at a keep of 0.9 equals the first 40 elements of a fill of 96.

### TheSameInputsGiveTheSameMask

Two fills of 4096 elements of run 9 at a keep of 0.8 are equal.

### TwoRunsAndTwoSitesDrawDifferentMasks

Runs 0 and 1 at a keep of 0.5 differ in 2033 of 4096 elements. The masks of the site seeds
`dropout_site_seed(0, 0)` and `dropout_site_seed(0, 1)` are not equal.

### TheKeptFractionIsTheKeepProbability

Over 2^20 draws the kept fraction is within 0.002 of the keep: 0.9 and 0.5 in run 0, and 0.9 in
run 3.

### TheTwoEndsOfTheKeepRangeAreExact

A keep of 1 keeps all 64 elements and a keep of 0 keeps none.

## modules/core/graph/tests/test_dropout_ops.cpp

Suites: `Unit_DropoutOps`.

Tests `GraphBuilder::dropout`: the node it emits, the seed it stores, the operands it refuses
and its row in the lowering table. The accepted cases pass the rate 0.25.

### DropoutWritesItsValueAndTheMaskItSaves

Over a [2, 3, 4] float64 input the call is ok and emits one DROPOUT node that reads the input
and writes the output and then the saved mask. The node holds 0.25 in `K_DROPOUT_RATE_SLOT`.
Both outputs have the shape [2, 3, 4], the type float64 and the lifetime `Activation`, and the
graph has no refusal.

### TheSiteSeedSurvivesItsAttributeSlots

For the seeds 0xFEDCBA9876543210, 0 and all 64 bits set, `Graph::dropout_seed` of the emitted
node's attributes returns the seed that was passed.

### DropoutRefusesOperandsThatDoNotFit

An int32 input fails with `InvalidDescription` and "half, float32 or float64 input". The rates
0, 1 and -0.5 each fail with "rate in (0, 1)". After each the node count is unchanged and
`has_refusal()` is true.

### DropoutHasTheRowOfTheRandomTable

`Graph::K_RANDOM_LOWERINGS` has one row, for DROPOUT. `find_lowering` returns
`Graph::lower_dropout`, `lowering_row_count` is 1 and `is_unlowered` is false.

## modules/core/graph/tests/test_elementwise_ops.cpp

Suites: `Unit_ElementwiseOps`.

Tests the nodes the builder emits for the elementwise, comparison, selection and constant
operations.

This file needs no device. Every case in `Unit_ElementwiseOps` calls an emitter of
`modules/core/graph/include/SushiAI/graph/builder.hpp` on a host-side graph and reads the node and the value it
wrote. The emitters are `div`, `minimum`, `maximum`, `less_equal`, `abs`, `atan`, `softplus`,
`silu`, `select`, `activation_backward`, `constant` and `constant_like`. The cases that concern
several emitters loop over a table and label each iteration with `SCOPED_TRACE`, so a failure
names the emitter. `test_backward.cpp` covers the backward rules of the same operations.

### BinaryEmitter and UnaryEmitter

`BinaryEmitter` holds a label, the identity an emitter must write and a pointer to a
two-operand member of `GraphBuilder`. `UnaryEmitter` is the same for one operand.

### K_BINARY_EMITTERS

`K_BINARY_EMITTERS` lists `div`, `minimum`, `maximum` and `less_equal` with K_DIV, K_MINIMUM,
K_MAXIMUM and K_LESS_EQUAL.

### K_UNARY_EMITTERS

`K_UNARY_EMITTERS` lists `abs`, `atan`, `softplus` and `silu` with K_ABS, K_ATAN, K_SOFTPLUS and K_SILU.

### K_INPUT_READING_GRADIENTS

`K_INPUT_READING_GRADIENTS` lists K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, the two activation
gradients that read the forward input.

### nchw_input

`nchw_input` declares a [2, 2, 2, 2] input and converts it to NCHW. Every extent is 2, so the
permuted shape equals the original and two such values still agree in shape.

### expect_produced_by

`expect_produced_by` expects a value to be the single result of a node of a given operation
that reads a given operand list in order.

### expect_described_like

`expect_described_like` expects a value to have the shape, element type and layout of another
value and the lifetime ACTIVATION.

### BinaryEmittersWriteTheirIdentityOverBothOperandsInOrder

Each two-operand emitter writes one node of its own identity whose inputs are `a` then `b`.

### BinaryEmittersKeepShapeDTypeAndLayout

With two FLOAT64 NCHW operands, the result of each two-operand emitter has the first operand's
shape and is FLOAT64 and NCHW.

### BinaryEmittersRejectUnequalShapesWithoutBroadcasting

Each two-operand emitter throws for a [4, 3] operand with a [4, 1] one, in both orders, and
with a [3] one. These are the pairs a broadcasting library accepts. The node count is unchanged
after the three calls.

### BinaryEmittersRejectUnequalDtypes

Each two-operand emitter throws for a FLOAT32 operand with a FLOAT64 one, in both orders.

### LessEqualYieldsAMaskInTheOperandsOwnDType

The result of `less_equal` is FLOAT32 for FLOAT32 operands and FLOAT64 for FLOAT64 ones. The
mask has no element type of its own, which is what lets `mul` and `select` read it beside a
value of the same type.

### UnaryEmittersWriteTheirIdentityOverTheirOperand

Each single-operand emitter returns a value other than its operand, written by one node of its
own identity that reads the operand.

### UnaryEmittersKeepShapeDTypeAndLayout

With a FLOAT64 NCHW operand, the result of each single-operand emitter has the operand's shape
and is FLOAT64 and NCHW.

### SelectReadsItsConditionThenBothBranches

`select(condition, a, b)` writes one K_SELECT node whose inputs are the condition, `a` and `b`,
in that order.

### SelectKeepsShapeDTypeAndLayout

With three FLOAT64 NCHW operands the result has the shape, element type and layout of `a`.

### SelectRejectsAConditionOfAnotherShape

A [4, 1] condition over [4, 3] branches throws, and the node count is unchanged.

### SelectRejectsAConditionOfAnotherDType

An INT64 condition over FLOAT32 branches throws.

### SelectRejectsBranchesOfUnequalShape

A [4, 3] branch with a [4, 1] branch throws in both orders.

### SelectRejectsBranchesOfUnequalDType

A FLOAT32 branch with a FLOAT64 branch throws in both orders.

### ActivationBackwardEmitsTheInputReadingGradients

For K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, `activation_backward(gradient, dy, x)` writes one node
of that identity whose inputs are `dy` then `x`, and the result is described like `dy`.

### InputReadingGradientsRejectAGradientOfAnotherShape

A [4, 1] `dy` with a [4, 3] `x` throws for both gradients.

### InputReadingGradientsRejectAGradientOfAnotherDType

A FLOAT32 value with a FLOAT64 one throws for both gradients, in both orders.

### ActivationBackwardRejectsAnIdentityThatIsNotAnActivationGradient

`activation_backward` throws when the identity is K_SILU, K_SOFTPLUS or K_DIV. The first two are
forward identities of `Ops::K_ACTIVATIONS` and the third is not in the table.

### FinaliseAcceptsAGraphUsingEveryEmitter

Two NCHW inputs are chained through all four two-operand emitters and all four single-operand
ones, and the chain ends in a `select` on a `less_equal` mask. `Graph::finalise()` does not
throw.

### FinaliseRejectsOperandsInTwoLayouts

For each two-operand emitter, a node that reads one NCHW operand and one NHWC operand is
emitted, and `finalise()` throws. The emitter itself accepts the pair because the shapes are
equal.

### FinaliseRejectsASelectConditionInAnotherLayout

A `select` with an NHWC condition over NCHW branches is emitted, and `finalise()` throws.

### ResultsOfAParameterRequireAGradient

After `finalise()`, `silu(div(weight, constant))` requires a gradient and
`atan(maximum(constant, constant))` over an input does not.

### AComparisonOfAParameterRequiresNoGradient

After `finalise()`, the mask `less_equal(weight, bound)` requires no gradient, and neither does
a `select` that reads it as its condition over two inputs.

### ConstantWritesAnOperandFreeNodeHoldingItsValue

`constant([4, 3], 1e-7, FLOAT64)` writes one K_CONSTANT node with no input. Attribute slot 0
holds 1e-7 within 1e-20, and the value is [4, 3], FLOAT64, NHWC and ACTIVATION. `K_VALUE` and
`K_VALUE_TOLERANCE` are constants local to the case.

### ConstantDefaultsToFloat32

`constant([2], 1.0)` with no element type is FLOAT32.

### ConstantRejectsAnElementTypeThatIsNotRealFloating

`constant` throws for INT64 and for COMPLEX32, and the node count and the value count are
unchanged.

### ConstantLikeCopiesShapeDTypeAndLayoutWithoutReadingItsOperand

`constant_like(x, 0.5)` over a FLOAT64 NCHW value writes a K_CONSTANT node with no input whose
result is described like `x`. The value it names is not an operand of the node.

### ConstantLikeRejectsAValueThatIsNotRealFloating

`constant_like` over an INT64 value throws, and the node count and the value count are
unchanged.

### AConstantLikeAValueIsAnOperandThatFinaliseAccepts

`add(x, constant_like(x, 1e-7))` over an NCHW input passes `finalise()`, and the sum requires
no gradient.

### ZerosStillWritesTheZeroIdentity

`zeros` and `zeros_like` each write a K_ZERO node with no input. K_ZERO and K_CONSTANT are two
identities.

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

## modules/core/data/tests/test_epoch_permutation.cpp

Suites: `Unit_EpochPermutation`.

Tests `Data::EpochPermutation`. An order is read by calling `at` for every position from 0 to
the count. The seed is 42 unless a case says otherwise.

### EverySampleIsReadOnce

For the counts 1, 2, 3, 7, 16, 17, 1000 and 4097 and the epochs 0, 1 and 5, every value `at`
returns is below the count and no value is returned twice.

### AnOrderIsAFunctionOfTheSeedAndTheEpoch

Over 10 samples the order of seed 42, epoch 0 is 4, 9, 5, 2, 1, 3, 8, 7, 6, 0, and a second
read gives the same. Epoch 1 gives 2, 5, 9, 1, 4, 3, 6, 7, 8, 0, and seed 7 at epoch 0 gives
5, 4, 1, 2, 3, 9, 0, 6, 7, 8.

### CountIsTheCountItWasBuiltOver

`count()` of a permutation built over 17 samples is 17.

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

## modules/core/graph/tests/test_fusion.cpp

Suites: `Unit_Fusion`.

Tests the matches and the refusals of the static fusion pass on synthetic graphs.

This file needs no device. The static fusion pass is a pure function of the IR, as the memory
planner is, so every case is a synthetic graph and a table of expectations with no engine
anywhere.

The cases pin the set of refusals more than the matches. A pass that fuses too little is slow.
A pass that fuses too much is silently wrong: the graph still runs, the loss still falls, and
the only symptom is a model that converges somewhere slightly different. So most cases are
graphs the pass must decline, one per rule:

- an intermediate with a second consumer (fusing it means losing it),
- an intermediate the caller reads by id (the same loss, seen from outside),
- a spilled pre-activation read by a node that runs before the fused one.

The activation whose gradient differentiates at its input used to be a fourth entry on that
list. It no longer is. K_GEMM_BIAS_GELU folds it by spilling the pre-activation to a second
output, so the refusal became a condition: an INPUT-gradient activation may be folded only by an
entry that writes the tensor its gradient reads, and only if it writes the right one. That
condition has a compile-time half in `modules/core/graph/source/fusion.cpp`, where an entry breaking either half
fails a static_assert. A test cannot cover that half, because a test that fails to compile is
not a test. This file checks the pair of properties the assertion reads and, above all, that the
fold leaves the pre-activation written and readable by the backward node.

The cases fall into four groups, in file order:

- the catalog is well formed
- the matches
- the refusals, which are the half that matters
- the pass keeps the graph usable

### EveryFormRoundTripsThroughItsOperationIdentity

The lowering dispatches on `fused_form_op` and `fused_form_of`. A form that does not round-trip
would lower as "no recorder for this operation", which is loud, but only at run time and only on
a graph that happens to contain it.

### AnActivationFoldsForFreeOrWithASpillAndTheTwoAreDifferentFacts

`activation_survives_fusion(K_RELU)` is the property the static_assert in `modules/core/graph/source/fusion.cpp`
reads. If it ever flips for an activation that has a catalog entry, the static_assert catches
it, but only while the property stays true of the table the static_assert consults. This test
pins that.

`activation_survives_fusion(GELU)` is false because SushiBLAS's `gelu_backward` takes the
pre-activation. An entry that wrote only its result would differentiate at a tensor that never
reached memory. That has not changed and must not, because it is the premise the static_assert
reasons from.

`catalog_folds_activation(GELU)` is true because the conclusion drawn from that premise
changed. The one entry that spills folds GELU. "Survives fusion" and "the catalog folds it" are
now two questions with two answers, and a caller who reads the first as the second gets the
wrong answer for exactly one activation.

### AMatmulAndABiasAddBecomeOneNode

`report.fused().size()` is 1, and the fused node reads A, B and the bias and still writes the
value the caller named. That is the whole reason ids are preserved.

The `flag(0)` and `flag(1)` checks hold the transposition flags. They have to survive, because
the fused recorder hands them straight to `execute_gemm`, and losing one gives a wrong answer of
exactly the right shape.

### TheLongestMatchWinsOverTheShorterOneInsideIt

Both gemm+bias and gemm+bias+relu match this graph. Rooted at the bias_add, the shorter one
would win and the relu would be left outside for good, which is why the matcher walks roots in
reverse.

### TheCrossEntropyGradientsElementwiseTailBecomesOnePass

With `scale(sub(exp(log_probs), targets), 0.25)` observed, `fuse` reports one
`FusedForm::ExpSubScale`. The node that produces the scaled value reads `log_probs` and
`targets` and has the attribute 0.25, and the two intermediate values have no producer.

### ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch

(exp(a) - b) and (b - exp(a)) are different functions with the same shapes, so nothing
downstream could tell them apart. The link position in the catalog entry is what refuses, and
this test shows that the pass consults it.

### AnIntermediateWithASecondConsumerIsNotFusedAway

The `also` value is a second reader of the K_MATMUL's result. Fusing would mean recomputing the
result for that reader, or losing it.

### AnIntermediateTheCallerReadsByIDIsNotFusedAway

The `observed` list names `biased`, which is the logits of a real classifier. The training loop
reads it after every replay to score accuracy, and a fused form gives it no buffer to read.

`report.count(K_GEMM_BIAS)` is still 1: the pass falls back to the shorter form rooted at the
bias_add, whose own intermediate is free.

### AParameterIsProtectedWithoutBeingListed

A persistent value's buffer outlives the step whatever the graph says, so the caller must not
have to remember to list it. In this graph the "intermediate" is a parameter. That is an odd
graph, but it is the shape a caller would get wrong.

The assignment `lifetime = OPTIMIZER_STATE` forces the exp's result into a persistent class, as
a checkpointed activation would be.

### AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills

The graph is the trap the fusion design is arranged around. The differentiation pass has
already emitted `gelu_backward` reading the pre-activation, so that value has two consumers.
Every other entry would stop at that point under rule 1. This entry does not, for the same
reason it is allowed to exist: the fused node keeps writing that value, as a second output,
under the same id.

The check on the fused node's second output is the assertion that carries the test, and a plain
K_GEMM_BIAS_GELU with no spill would fail it. The fused node still writes the pre-activation as
its second output, and it is the same value, so `gelu_backward`'s operand needed no rewriting
and the backward rule needed no change.

`fused()[0].eliminated` holds only the K_MATMUL's result. A report that listed the spilled value
as eliminated would read like the report of the silently wrong version of this pass.

### ASpilledPreActivationReadBeforeTheFusedNodeIsRefused

The `early` reader shows the one new way the K_GEMM_BIAS_GELU entry could break the graph: a
wrong order, not a wrong number. The fused node is emitted at the K_GELU's position, so a reader
of the pre-activation sitting between the bias_add and the K_GELU would end up reading a value
written after it. Every pass downstream of fusion assumes that cannot happen.

`report.count(K_GEMM_BIAS)` is 1: the pass still falls back to the shorter form, as it does after
any other refusal of the three-node entry.

The closing loop over the nodes checks that the rewritten graph is still topological, which is
the property the refusal exists to protect.

### ASpilledPreActivationMayBeReadFromOutsideTheGraph

This case mirrors VALUE_IS_OBSERVED. That rule exists because a fused form gives an interior
value no buffer to read. The K_GEMM_BIAS_GELU entry gives it one, so naming the pre-activation as
observed must not refuse the fold.

### TheGELUFoldStillObeysEveryOtherRefusal

A spill licenses one interior value and no others. The K_MATMUL's result is not spilled, so a
second reader of it refuses the whole chain as it does for every other entry. The fallback is
nothing, because the same value blocks gemm+bias.

### DisablingTheGELUFormFallsBackToTheShorterOne

The last two assertions check that the GELU left outside is not also reported as unfoldable.
"Its gradient differentiates at its input" and "no catalog entry composes it" are both false for
this graph, and either would send a reader to change the wrong thing. The reason asserted just
before them, `FusionDecline::Disabled`, is the only one.

### ARecoverableActivationNextToAFusedGEMMIsFoldedNotReported

This case mirrors DisablingTheGELUFormFallsBackToTheShorterOne, and it is the reason ReLU's
backward rule reads the output. With the gradient reading the activation's own result, the
pre-activation has one consumer and folds with no special case anywhere.

### DisablingThePassLeavesTheGraphExactlyAsItWas

With `enabled` false over a linear trace followed by a ReLU, the report is not enabled and
lists nothing fused. The graph keeps its node count, its value count and its one MATMUL.

### DisablingOneFormFallsBackToTheShorterOne

With `FusedForm::GEMMBiasReLU` disabled, the report counts no `GEMMBiasReLU` and one
`GEMMBias`, and holds a decline of `FusionDecline::Disabled`.

### ValueIDsAndTheirDescriptionsSurviveTheRewrite

Everything a caller holds across the pass (a StepSpec, a Parameter's own id, a gradient)
depends on ids surviving. A pass that compacted ids would need a remap reaching into the
model's layer objects, and forgetting one would bind the optimizer to the wrong tensor.

### TheRewrittenGraphIsStillInTopologicalOrder

The memory planner's reachability closure and the lowering's single forward walk both rest on
topological order. The fused node is emitted at the root's position while it reads operands
from the chain's first node, and that is where a rewrite gets the order wrong.

### TheReportNamesEverySubtreeItCollapsed

After a default `fuse`, `to_string` holds "gemm+bias+relu" and "dropping value", and
`nodes_after` equals the graph's node count.

### StateLeavesAndTheRefusalMarkSurviveTheRewrite

The graph is a K_MATMUL, a K_BIAS_ADD whose bias is a buffer, and a K_RELU. The weight and the buffer
are bound to the states 21 and 22 and the graph is marked refused before `Graph::fuse` runs.
The report lists one fused form and the graph holds one K_GEMM_BIAS_RELU node, so the pass did
rebuild the graph. Both `state_leaf` lookups and the whole `state_leaves()` table are what they
were. The buffer keeps the lifetime `BUFFER` and no gradient, the weight still requires one,
and `has_refusal()` is still true.

## modules/core/autograd/tests/test_gather_backward.cpp

Suites: `Unit_GatherBackward`.

Tests the nodes `Autograd::differentiate` appends for GATHER_ROWS and SCATTER_ADD_ROWS. The
table has 7 rows of length 3, the index is int32, and the loss is the `nll_loss` of the
`log_softmax` of the rows against a `targets` input.

### TheGatherAndTheScatterAddEachHaveARule

`Ops::K_GATHER_ROWS` and `Ops::K_SCATTER_ADD_ROWS` are differentiable and
`Autograd::has_backward_rule` is true for both.

### TheScatteredRowsGradientIsOneGatherOverTheForwardIndex

With [5, 3] rows as a parameter scattered into 7 rows, the graph holds one GATHER_ROWS node
after differentiation. It reads a [7, 3] value and the forward index and writes the returned
gradient, whose shape is [5, 3]. The graph still holds one SCATTER_ADD_ROWS node.

### TheScatterIndexReceivesNoGradient

Asked for the rows and the index of a scatter-add, `differentiate` returns a value for the rows
and `K_NO_VALUE` for the index.

### FrozenScatteredRowsAppendNoGather

With the rows as an input and a `weight` parameter multiplied after the scatter-add,
differentiating at the weight adds no GATHER_ROWS node.

### TheTableGradientIsOneScatterAddOverTheForwardIndex

With the table as a parameter gathered by a [5] index, the graph holds one SCATTER_ADD_ROWS
node. It reads a [5, 3] value and the forward index and writes the returned gradient, a
float32 value of shape [7, 3] with the lifetime `Gradient`.

### TheIndexReceivesNoGradient

Asked for the table and the index of a gather, `differentiate` returns a value for the table
and `K_NO_VALUE` for the index, and the index does not require a gradient.

### ARankTwoIndexScattersTheGradientOfTheGathersShape

With a [2, 4] index and the gathered value reshaped to [8, 3], the one SCATTER_ADD_ROWS node
reads a [2, 4, 3] value and the forward index, and the gradient has the shape [7, 3].

### AFrozenTableAppendsNoScatterAdd

With the table as a buffer and a `weight` parameter multiplied after the gather, the weight
gets a gradient and the graph holds no SCATTER_ADD_ROWS node.

### TwoGathersOfOneTableAccumulateTheirScatterAdds

With two gathers of one table added together, the graph holds two SCATTER_ADD_ROWS nodes and
the table's gradient is written by an ACCUMULATE node.

## modules/core/graph/tests/test_gather_ops.cpp

Suites: `Unit_GatherOps`.

Tests `GraphBuilder::gather_rows` and `GraphBuilder::scatter_add_rows`: the node each emits,
the result shape and the operands each refuses. The table has 7 rows of length 3. A refusal is
`InvalidDescription`, leaves the node count unchanged and sets `has_refusal()`.

### GatherRowsWritesItsIdentityOverTheTableAndTheIndex

Over a [7, 3] table and a [5] int32 index the call is ok and the result is the one output of a
GATHER_ROWS node that reads the table and then the index. The result has the shape [5, 3], the
type float32 and the lifetime `Activation`.

### GatherRowsAppendsTheRowLengthToAnIndexOfAnyRank

Over a float64 table, a [2, 4] index gives a float64 [2, 4, 3] value and a [2, 1, 3, 1, 2]
index gives a [2, 1, 3, 1, 2, 3] value.

### GatherRowsRefusesATableThatIsNotRankTwo

A [7] table is refused with "rank-2 [V, D] table" and a [7, 2, 3] table with "[7, 2, 3]".

### GatherRowsRefusesAnIndexThatIsNotInt32

A float32 index and an int64 index are each refused with "INT32 index".

### GatherRowsRefusesATableThatIsNotFloating

An int32 table is refused with "half, float32 or float64 table".

### GatherRowsRefusesAnIndexWhoseResultWouldPassTheHighestRank

An index of rank `K_MAX_RANK`, shaped [1, 1, 1, 1, 1, 1], is refused with "rank 1 to 5".

### ScatterAddRowsWritesItsIdentityAndTheTableShape

Over [2, 4, 3] float64 rows and a [2, 4] index, `scatter_add_rows` with 7 table rows is ok. The
result is the one output of a SCATTER_ADD_ROWS node that reads the rows and then the index, and
it has the shape [7, 3], the type float64 and the lifetime `Activation`.

### ScatterAddRowsRefusesOperandsThatDoNotFit

Five calls are refused: rank-1 rows ("rank 2 or more"), int32 rows ("half, float32 or float64
rows"), a float32 index ("INT32 index"), a [4] index for five rows ("one index per row") and a
row count of 0 ("a table of 0 rows").

### AnInt32InputLeafRequiresNoGradient

After `graph.finalise()` the int32 index has the lifetime `Input` and does not require a
gradient, and the gathered value does.

### TheCastPassLeavesAnInt32LeafAndItsGatherAsDeclared

After `Graph::insert_casts` under a `MixedPrecisionPolicy`, the index is int32, the table and
the gathered value are float32 and the `matmul` that reads the gathered value is `Half`. The
gather still reads the table and the index. No reported cast has the index as its source or
GATHER_ROWS as its op, and `find_dtype_mismatches` is empty.

## modules/language/generation/tests/test_generation_greedy_sampler.cpp

Suites: `Unit_GenerationGreedySampler`, `Unit_GenerationGreedySamplerDeathTest`.

Tests `Language::Generation::GreedySampler` in
`modules/language/generation/include/SushiAI/language/generation/greedy_sampler.hpp`. Every
generator is seeded with 20261011.

### ThePickIsTheLargestLogitAndTheFirstOfEquals

`sample` returns 1 for the logits 0.5, 2, -1, returns 1 for 0.5, 2, 2, -1 and returns 0 for the
single logit -3. `name()` is `K_GREEDY_SAMPLER`.

### APickDrawsNothing

After one `sample`, the generator compares equal to a second generator built from the same
seed and never used.

### NoLogitIsFatal

`sample` over an empty span stops the process with a report containing "no logit". The case is
in the suite `Unit_GenerationGreedySamplerDeathTest`.

## modules/language/generation/tests/test_generation_nucleus_sampler.cpp

Suites: `Unit_GenerationNucleusSampler`.

Tests `Language::Generation::NucleusSampler` in
`modules/language/generation/include/SushiAI/language/generation/nucleus_sampler.hpp`. The
distribution cases count 20000 draws under the seed 20261011 over the logits ln 2, 0, 0, whose
probabilities are 1/2, 1/4 and 1/4, and compare each share within 0.02.

### TheFewestTokensThatReachTheMassAreDrawn

With a mass of 0.6 and a temperature of 1, token 0 has a share of 2/3, token 1 a share of 1/3
and token 2 is never drawn. `name()` is `K_NUCLEUS_SAMPLER`.

### AMassTheFirstTokenReachesKeepsItAlone

With a mass of 0.4, token 0 is drawn all 20000 times.

### AMassOfOneKeepsEveryToken

With a mass of 1, the shares are 1/2, 1/4 and 1/4.

### ASeedFixesTheSequence

A sampler of mass 0.9 and temperature 1.2 draws the same 64 tokens twice from the same seed
over eight logits from 0 to 0.7.

### MakeRefusesAMassOutsideItsRangeAndABadTemperature

`NucleusSampler::make` fails with `InvalidDescription` for a mass of 0 ("'nucleus'", "above
0"), a mass of 1.5 ("at most 1"), a mass that is not a number ("'nucleus'") and a temperature
of -2 ("temperature").

## modules/language/generation/tests/test_generation_position_feed.cpp

Suites: `Unit_GenerationPositionFeed`.

Tests `Language::Generation::Detail::make_position_feed`, which picks a `PositionFeed` from
the leaves a step is fed. The fixture is `Support::FedStepTest` of
`position_feed_test_support.hpp`; its `build_step(has_position, has_ring)` builds a step fed
"feed_position", or "feed_slot" with "feed_cosine" and "feed_sine", or both, or neither.

### AStepFedAPositionGetsTheTableFeed

Over a step fed only a position, the feed lists one leaf, named `K_FEED_POSITION`, and
`keeps_rows_on_slide()` is false.

### AStepFedACacheSlotGetsTheRingFeed

Over a step fed only the ring leaves, the feed lists three leaves, the first named
`K_FEED_SLOT`, and `keeps_rows_on_slide()` is true.

### LeavesWithoutARingPickTheTableFeed

With `CachedStepLeaves` that name a position and a mask and no ring, a step fed a position
gets a feed of one leaf.

### AStepFedBothOrNeitherIsRefused

`make_position_feed` fails with `InvalidDescription` over a step fed both ("CachedGenerator",
"both", "'feed_position'", "'feed_slot'") and over a step fed neither ("neither",
"'feed_position'", "'feed_slot'").

## modules/language/generation/tests/test_generation_ring_position_feed.cpp

Suites: `Unit_GenerationRingPositionFeed`.

Tests `Language::Generation::Detail::RingPositionFeed`. The fixture is `Support::FedStepTest`;
the feed is made over a window of four rows and three pairs, with an angle function that
writes position + pair as each cosine and the negation of that as each sine.

### TheFeedListsTheSlotThenTheAngles

`feeds()` lists three leaves in the order `K_FEED_SLOT`, `K_FEED_COSINE`, `K_FEED_SINE`. The
slot tensor is `Int32` with one element, the cosine tensor is `Float32` with three elements
and the sine tensor has three elements.

### PlaceWrapsTheSlotAndKeepsThePositionOfTheAngles

`place(11)` writes the slot 3, the cosines 11, 12, 13 and the sines -11, -12, -13. `place(4)`
writes the slot 0 and a first cosine of 4. `place(2)` writes the slot 2.

### EveryRowStaysRightAfterASlide

`keeps_rows_on_slide()` is true.

### ARingThatDoesNotFitIsRefused

Over a step fed both the position and the ring leaves, `RingPositionFeed::make` fails with
`InvalidDescription` for a null angle function ("no function", "'feed_cosine'",
"'feed_sine'"), for a sine leaf named "feed_tangent" ("'feed_tangent'"), for the cosine leaf
named as the slot ("'feed_cosine'", "int32 leaf of [1, 1]") and for the position leaf named
as the sine ("'feed_position'", "float32 leaf of [1, 3]").

## modules/language/generation/tests/test_generation_sampler_section.cpp

Suites: `Unit_GenerationSamplerSection`.

Tests `Language::Generation::make_sampler` in
`modules/language/generation/include/SushiAI/language/generation/sampler_section.hpp`, which
builds a sampler from a section. Every call passes the context "the task section 'sampler'".

### EachTypeBuildsItsSampler

The sections `{"type": "greedy"}`, `{"type": "temperature", "temperature": 0.7}`, `{"type":
"temperature"}`, `{"type": "top_k", "k": 40, "temperature": 2}` and `{"type": "nucleus", "p":
0.9}` build samplers whose names are `K_GREEDY_SAMPLER`, `K_TEMPERATURE_SAMPLER` twice,
`K_TOP_K_SAMPLER` and `K_NUCLEUS_SAMPLER`.

### AnUnknownTypeAndAnAbsentKeyAreRefused

`make_sampler` fails with `InvalidDescription` for the type "beam" ("'beam'", "'greedy',
'temperature', 'top_k', 'nucleus'"), a `top_k` section without `k` ("'sampler'", "has no
'k'"), a `nucleus` section without `p` ("has no 'p'"), a section without `type` ("has no
'type'") and a string in place of the section ("an object").

### AKeyTheTypeDoesNotReadIsRefused

`make_sampler` fails with `InvalidDescription` for a `greedy` section with `temperature`
("'temperature'", "does not have"), a `temperature` section with `k` ("'k'"), a `k` of 0
("keep 0 tokens"), a `k` of "many" ("'k'", "whole number") and a `p` of 2 ("at most 1").

## modules/language/generation/tests/test_generation_table_position_feed.cpp

Suites: `Unit_GenerationTablePositionFeed`.

Tests `Language::Generation::Detail::TablePositionFeed`. The fixture is
`Support::FedStepTest`.

### PlaceWritesThePositionIntoItsOneLeaf

Over a step fed a position, `feeds()` lists one leaf named `K_FEED_POSITION` whose tensor is
`Int32` with one element. `place(0)` writes 0 into it and `place(5)` writes 5.

### NoRowStaysRightAfterASlide

`keeps_rows_on_slide()` is false.

### ALeafThatDoesNotFitIsRefused

Over a step fed only the ring leaves, `TablePositionFeed::make` fails with
`InvalidDescription` for the leaf "feed_position" ("'feed_position'") and for the leaf
"feed_cosine" ("CachedGenerator", "'feed_cosine'", "int32 leaf of [1, 1]").

## modules/language/generation/tests/test_generation_task.cpp

Suites: `Unit_GenerationTask`.

Tests `Language::Generation::read_generation_task` in
`modules/language/generation/include/SushiAI/language/generation/generation_task.hpp`, the
reader of the `next_token_generation` task section. Every call passes the context "the task
section".

### EveryKeyIsRead

A section with `vocabulary` "vocab.savb", `window` 256, `tokens` 200, `seed` 7 and a `top_k`
sampler reads back with those four values and a sampler whose name is `K_TOP_K_SAMPLER`.

### TheOptionalKeysHaveDefaults

A section with only `type`, `vocabulary` and `window` reads `tokens` as
`K_DEFAULT_GENERATED_TOKENS`, `seed` as 0 and a sampler whose name is `K_GREEDY_SAMPLER`.

### AKeyThatDoesNotFitIsRefused

The reader fails with `InvalidDescription` for the type "anchor_detection"
("'anchor_detection'", "'next_token_generation'"), a section without `vocabulary` ("the task
section", "has no 'vocabulary'"), one without `window` ("has no 'window'"), a `window` of 0
("'window'", "positive"), a `tokens` of 0 ("'tokens'", "positive"), a `seed` of -1 ("'seed'"),
a `prompt` key ("the task section", "'prompt'", "does not have") and a sampler of the type
"beam" ("the task section 'sampler'", "'beam'").

### TheCacheIsReadAndDefaultsToTrue

`K_DEFAULT_GENERATION_CACHE` is true. `cache` reads as true from a section without the key,
false from `"cache": false` and true from `"cache": true`.

### ACacheThatIsNoFlagIsRefused

`"cache": 1` fails with `InvalidDescription` and a message containing "the task section",
"'cache'" and "a boolean".

## modules/language/generation/tests/test_generation_temperature_sampler.cpp

Suites: `Unit_GenerationTemperatureSampler`.

Tests `Language::Generation::TemperatureSampler` in
`modules/language/generation/include/SushiAI/language/generation/temperature_sampler.hpp`.
The distribution cases count 20000 draws under the seed 20261011 and compare each share
within 0.02.

### TheSharesAreTheSoftmax

At a temperature of 1 over the logits 0, ln 2, 0, the shares are 1/4, 1/2 and 1/4. `name()`
is `K_TEMPERATURE_SAMPLER`.

### ALowerTemperatureSharpensTheShares

At a temperature of 0.5 over the logits 0 and ln 2, the shares are 0.2 and 0.8.

### ASeedFixesTheSequence

Over eight logits from 0 to 0.7, two sequences of 64 draws from the same seed are equal, and
the sequence from the seed plus 1 differs from them.

### MakeRefusesATemperatureThatIsNotPositiveAndFinite

`TemperatureSampler::make` fails with `InvalidDescription` for 0 ("'temperature'",
"positive"), for -1 ("positive"), for infinity ("finite") and for a value that is not a number
("finite").

## modules/language/generation/tests/test_generation_top_k_sampler.cpp

Suites: `Unit_GenerationTopKSampler`.

Tests `Language::Generation::TopKSampler` in
`modules/language/generation/include/SushiAI/language/generation/top_k_sampler.hpp`. The
distribution cases count 20000 draws under the seed 20261011 and compare each share within
0.02.

### OnlyTheKLargestAreDrawnAndTheLowerIndexWinsATie

With `k` 2 and a temperature of 1 over the logits 0, ln 2, 0, -5, token 0 has a share of 1/3,
token 1 a share of 2/3, and tokens 2 and 3 are never drawn. `name()` is `K_TOP_K_SAMPLER`.

### KeepingOneTokenIsTheGreedyPick

With `k` 1 over the logits 0.3, 0.2, 0.9, 0.9, token 2 is drawn all 20000 times.

### AKAboveTheRowKeepsEveryToken

With `k` 50 over the logits 0 and ln 2, the shares are 1/3 and 2/3.

### ASeedFixesTheSequence

A sampler of `k` 4 and temperature 0.8 draws the same 64 tokens twice from the same seed over
eight logits from 0 to 0.7.

### MakeRefusesAKBelowOneAndABadTemperature

`TopKSampler::make` fails with `InvalidDescription` for a `k` of 0 ("'top_k'", "keep 0
tokens") and for a temperature of 0 ("'top_k'", "temperature").

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

## modules/core/autograd/tests/test_gradcheck.cpp

Suites: `Unit_GradCheck`.

Tests the gradient checker against closed-form functions and a wrong gradient.

This file needs no device. The gradient checker is the instrument the integration tier measures
with, so this file calibrates it against closed-form functions whose right answer is known. One
of them is a deliberately wrong gradient, which proves the check can fail.

### AcceptsAnExactGradient

`check_full_jacobian` of the cubic and its gradient at six coordinates passes with a largest
relative error below 1e-7 and six probes.

### RejectsAGradientThatIsWrongByATenthOfAPercent

This case is the reason for insisting on FLOAT64. A 1e-3 relative error is far above the noise
floor of about 1e-9 in FLOAT64. It would sit inside the floor of about 2.4e-5 that an fp32
difference quotient bottoms out at, and would pass unnoticed.

### LeavesTheProbePointUnchanged

The probe point holds the same values after `check_full_jacobian` and after
`check_directional`.

### DirectionalModeAgreesWithTheFullJacobian

At eight coordinates both `check_full_jacobian` and `check_directional` with 12 directions
pass, and the directional result reports 12 probes.

### DirectionalModeCatchesAWrongGradientToo

`check_directional` with 16 directions over `slightly_wrong_gradient` does not pass.

### DirectionalModeIsReproducibleForAGivenSeed

Two `check_directional` calls with the seed 12345 report the same largest relative error and
the same worst index.

### PushOffKinksMovesCoordinatesAwayFromZeroAndKeepsTheirSign

`push_off_kinks` over 0.0, 1e-12, -1e-12, 5.0, -5.0 leaves the first non-zero, the second
positive, the third negative and the last two unchanged. Every coordinate is then at least
`kink_margin * min_step * 0.999` from zero.

### AReLUFamilyFunctionNeedsToBeSampledOffTheKink

`options.min_step` is large enough that the difference survives the cancellation of summing it
against the O(1) terms. A step of 1e-4 keeps the pushed coordinates at 1e-3, where a central
difference of x^2 is still exact to 1e-9.

The `unshifted` point straddles the kink. The quotient taken across it reports the average of
the two one-sided derivatives, which no correct rule can match.

### AnEmptyProbePointIsRejected

`check_full_jacobian` over an empty point throws `Error`, and so does `check_directional` with
0 directions.

## modules/core/optim/tests/test_gradient_clip.cpp

Suites: `Unit_GradientClip`.

Tests the gradient clip of `Optim::Optimizer`: `enable_gradient_clip`, `set_clip_norm`, the
`clip_norm` key of `Optim::make_optimizer` and the update a clipped step applies. The fixture
needs an engine. Each slot has a float32 parameter of zeros and a stated gradient, and
`make_sgd` returns an SGD of rate 1 with no momentum and no decay. A parameter is compared
within 1e-6.

### ANormAboveTheThresholdScalesTheUpdate

With the gradient 3, 4, 0, 0 and a clip norm of 1, one step leaves the parameter at -0.6, -0.8,
0, 0. `gradient_clip()` is not null and its `applied_scale()` is 0.2.

### ANormBelowTheThresholdLeavesTheUpdate

With the same gradient and a clip norm of 10, one step leaves the parameter at -3, -4, 0, 0 and
`applied_scale()` is exactly 1.

### TheNormIsTakenOverEverySlot

With the gradients 3, 0 and 0, 4, 0 in two slots and a clip norm of 2.5, one step leaves the
parameters at -1.5, 0 and 0, -2, 0. `gradient_count()` is 2.

### AThresholdSetAtRunTimeIsTheNextUpdates

With the gradient 3, 4 and a clip norm of 5, the first step leaves -3, -4. After
`set_clip_norm(2.5)`, `max_norm()` is 2.5 and the second step leaves -4.5, -6.

### TheMomentumKernelReadsTheFactor

An SGD of rate 1 and momentum 0.5 with a clip norm of 1 leaves the parameter at -0.6, -0.8
after its first step.

### AdamWReadsTheSameFactor

An AdamW of default options over the gradient 3, 4 with a clip norm of 1 reports an
`applied_scale()` of 0.2 after one step.

### AClipAddsItsTasksAndItsAbsenceAddsNone

Over two slots, `record_into_step` returns 2 for an SGD without a clip, whose
`gradient_clip()` is null, and 5 for one with a clip.

### AClipRefusesWhatItCannotCompute

`enable_gradient_clip` fails with `InvalidDescription` and "positive and finite" for the norms
0 and -1, and `gradient_clip()` stays null. On an SGD made with `UpdateKernel::Separate` it
fails with a message containing "UpdateKernel::Fused".

### ANormSetWithoutAClipIsFatal

`set_clip_norm(1.0)` on an optimizer without a clip stops the process with a report containing
"no gradient clip". After a clip is enabled, `set_clip_norm(0.0)` stops it with "clip norm" and
"positive and finite".

### TheKeyClipNormEnablesTheClip

For the types `sgd` and `adamw`, `make_optimizer` with `{"clip_norm": 0.5}` returns an
optimizer whose clip has the `max_norm()` 0.5, and with `{}` one whose `gradient_clip()` is
null. `{"clip_norm": 0}` fails with `InvalidDescription`, "optimizer 'clip_norm'" and "positive
and finite".

## modules/vision/training/tests/test_head_delta_loss.cpp

Suites: `Unit_HeadDeltaLoss`.

Tests the IR that `emit_head_delta_loss` of the vision training module emits over a head
parameter and a delta input of the same shape, [2, 3, 3, 7] in this file, with the scale 0.25.

### EmitsFourNodesEndingInAScaleOfOneElement

The emitter adds `sushiai.mul`, `sushiai.reshape`, `sushiai.sum_rows` and `sushiai.scale`, in
that order. MUL reads the delta and then the head, and each later node reads the one before it.
The RESHAPE result has the shape [126, 1], the loss has the shape [1], and the SCALE node's
attribute 0 is -0.25 to 1e-12.

### TheResultHasTheElementTypeOfTheHead

Over a `Float64` head and a `Float64` delta, the loss has the element type `Float64`.

### TheResultIsARootAutogradDifferentiates

`Autograd::differentiate` of the loss with respect to the head does not throw and returns one
gradient of the head's shape. The delta's `requires_grad` is false.

### ADeltaOfAnotherShapeThrowsNamingBothShapes

A delta of the shape [2, 3, 3, 6] throws an error naming "MUL needs equally shaped operands",
"[2, 3, 3, 6]" and "[2, 3, 3, 7]", and adds no node.

### ADeltaOfAnotherElementTypeThrowsNamingBothTypes

A `Float64` delta against a `Float32` head throws an error naming
"MUL needs operands of one element type", "float32" and "float64", and adds no node.

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

## modules/vision/imaging/tests/test_imaging_affine.cpp

Suites: `Unit_ImagingAffine`.

Tests the plan and the application of the scale and translation warp to an image and its
labels.

This file needs no device. `Unit_ImagingAffine` covers `modules/vision/imaging/include/SushiAI/vision/imaging/affine.hpp`.
`plan_affine` draws a scale and a translation per axis and stores the 2 by 3 matrix;
`apply_affine` warps an image with bilinear weights, or warps, clips and filters a label set.
The matrix has no rotation and no shear, and `apply_affine` refuses a plan that has one.

### K_PIXEL_TOLERANCE

`K_PIXEL_TOLERANCE` is 1e-3 px, the tolerance of a box edge or a matrix term.

### Matrix3 and multiply

`Matrix3` is a row-major 3 by 3 float matrix, the form the reference builds its factors in.
`multiply` returns the product of two of them with each term summed left to right.

### make_pattern_image

`make_pattern_image` returns an image whose byte at row y, column x, channel c is
(31 y + 7 x + 53 c + 1) modulo 256.

### make_gray_image

`make_gray_image` returns a one-channel image holding the given bytes.

### make_plan

`make_plan` returns an `AffinePlan` with a given scale and matrix translation, built without a
draw. A case uses it to state the matrix it tests.

### make_labels

`make_labels` returns a label set with one object per box and the classes 0, 1, 2 and so on.

### expect_box_near

`expect_box_near` compares the four edges of two boxes within `K_PIXEL_TOLERANCE`.

### bytes_of

`bytes_of` returns the bytes of an image.

### ZeroRangesPutTheSourceCentreOnTheOutputCentre

With the scale and translate ranges at 0, a 1200 by 1000 source and a 480 by 640 output give
scale 1, translation (320, 240) and the matrix [1 0 -180; 0 1 -360]: the source centre
(500, 600) maps to the output centre.

### MatrixEqualsTranslationTimesScaleTimesCentre

The case replays the three draws on a copy of the generator and builds the centring, scaling
and translation matrices. The plan's matrix equals translation times scaling times centring.
The seed is required to give a scale more than 0.05 from 1, because at scale 1 the product is
the same in any order.

### PlanConsumesExactlyThreeDraws

After `plan_affine` the generator equals a copy advanced by three draws.

### PlanCarriesSizesBorderValueAndFilter

The plan holds the source size, the output size, the border value 37 and the filter fields of
the settings.

### DrawnParametersStayInRangeAndCentreOnTheirMeans

Over 20000 plans from 200 seeds, the scale lies in [0.5, 1.5) and each translation in [0.4,
0.6] of its output side. The means are 1 within 0.012 and 0.5 within 0.0025, the extremes come
within 0.01 and 0.002 of the range ends, and the x and y fractions are within 1e-6 of each
other in fewer than 1 percent of the plans, so each axis has its own draw.

### PlanRejectsSettingsAndSizesOutsideTheirRanges

A scale half-range of 1, -0.1 or NaN and a translate half-range of 0.6 or -0.1 throw with
"validate_affine" or the field name. A source of 0 by 16 and an output of 16 by 0 throw with
"plan_affine" and the size. No rejected call makes a draw, and the default settings pass.

### IdentityPlanLeavesTheImageUnchanged

With both ranges at 0 and equal source and output sizes, the warped image has the source's
bytes.

### IdentityPlanLeavesBoxesUnchanged

Under the same plan three boxes keep their classes and their edges.

### TranslationMovesPixelsAndFillsTheUncoveredBorder

A 3 by 4 source shifted by (2, 1) into a 5 by 6 output puts source pixel (y - 1, x - 2) at
(y, x) and fills every other pixel with 114.

### ScaleByTwoInterpolatesAndFadesIntoTheBorder

A 2 by 2 image scaled by 2 with border value 100 gives the sixteen bytes the case lists. Odd
output coordinates fall halfway between two source pixels, and the last row and column blend
with the border.

### HalfPixelBlendRoundsToNearest

A row (10, 13) shifted by half a pixel gives 62, 12 and 64: the means 62, 11.5 and 63.5 with
the border 114, rounded half up.

### FractionRoundsToTheNearestOf2048Weights

With an offset of 6141 / 8192 the fraction times 2048 is 512.75, so the weights are 1535 and
513. The three output bytes are 90, 19 and 42, computed from those weights; the weights 1536
and 512 give other bytes.

### OutputFarFromTheSourceIsAllBorder

A translation of (1000, -1000) leaves every output byte at the border value 9.

### ImageWarpRejectsAPlanItCannotApply

`apply_affine` on an image throws for a plan made for another source size, a matrix with an
off-diagonal term, a diagonal term of 0, a matrix scale that is not the plan's scale, an
infinite translation, a source without pixels, a source of five channels, the destination as
its own source and a view of two rows of the destination. The aliased image keeps its bytes.

### TranslationMovesBoxesByTheOffset

A translation of (30, -10) maps (10, 20, 50, 60) to (40, 10, 80, 50).

### ScaleMultipliesBoxCornersBeforeTheOffset

Scale 2 with translation (10, -20) maps (5, 20, 25, 40) to (20, 20, 60, 60).

### WarpedBoxIsTheHullOfItsFourWarpedCorners

The case maps the four corners of a box through the plan's matrix and takes their hull. The
warped box equals the hull and the literal (30, 40, 80, 90).

### BoxesAreClippedThenFilteredAndSurvivorsKeepTheirOrder

Five 40 by 40 boxes are shifted right by 80 in a 100 px wide output, which leaves widths of 10,
5, 3, 2 and 0 inside it. The first two remain, in order, clipped to x2 = 100. Width 3 fails the
area ratio at 0.075, width 2 fails the minimum side and width 0 fails both.

### FilterComparesAgainstTheSourceBoxTimesTheScale

At scale 0.25 a 40 by 40 box becomes 10 by 10 and is kept: its ratio is 1 against the scaled
box and 0.0625 against the unscaled one. At scale 2 a 20 by 20 box that keeps 3 by 40 inside
the output is removed: its ratio is 0.075 against the scaled box and 0.3 against the unscaled
one.

### BoxWarpUsesThePlansFilter

A 4 px wide box is kept with the default filter and removed when the plan's `min_side` is 4.

### BoxWarpRejectsARotatedMatrixAndAScaleThatIsNotThePlans

`apply_affine` on labels throws for an off-diagonal matrix term and for a matrix scale that is
not the plan's scale.

### FiltersDefaultToTheBoxAndThePolygonAreaRatios

The settings and a plan made from them have `filter.min_area_ratio` 0.10 and
`polygon_filter.min_area_ratio` 0.01, and a default `AffinePlan` has the second value too.

### PolygonVerticesAreScaledThenTranslated

Scale 2 with translation (10, -5) maps the vertices (5, 10), (25, 12) and (15, 30) to (20, 15),
(60, 19) and (40, 55), and the box to (20, 15, 60, 55).

### PolygonObjectTakesTheBoxOfItsVisiblePart

A triangle that reaches outside a 100 by 80 output gets the box (0, 0, 36.666664, 33), the
reference's `segment2box` result. The same bounds without a polygon are clipped to
(0, 0, 70, 60).

### WarpedPolygonIsClampedToItsNewBox

The same triangle's vertices become (0, 0), (36.666664, 0) and (0, 33): each is clamped to the
box rebuilt from the polygon.

### WarpedPolygonStaysInsideTheOutputWhenACrossingRoundsOutside

In float the edge from x = -0.03 to x = 0.25 crosses x = 0 at -1.86e-9. The box's x1 is exactly
0, and every vertex lies inside the output and between the box's x1 and x2.

### PolygonObjectIsFilteredAtTheLowerAreaRatio

A box that keeps an area ratio of 0.075 is kept when its object has a polygon and removed when
it has none.

### PolygonWarpUsesThePlansPolygonFilter

The same object is removed when `polygon_filter.min_area_ratio` is 0.08, with
`filter.min_area_ratio` set to 0. The polygon filter decides for an object with a polygon.

### PolygonOutsideTheOutputDropsItsObject

With a translation of 500, the object whose polygon leaves the output is removed and the one
the translation moves into it is kept, with class 1 and the box (40, 10, 80, 50).

## modules/vision/imaging/tests/test_imaging_boxes.cpp

Suites: `Unit_ImagingBoxes`.

Tests the two box forms, their conversions and the candidate filter.

This file needs no device. `Unit_ImagingBoxes` covers `modules/vision/imaging/include/SushiAI/vision/imaging/boxes.hpp`: a
`NormalizedBox` is a centre and an extent in fractions of the image, a `PixelBox` is two corners
in pixels, and `is_box_candidate` decides whether a box survives an augmentation.

### Tolerances and K_SIZE

`K_PIXEL_TOLERANCE` is 1e-3 px and `K_FRACTION_TOLERANCE` is 1e-6 of the image.
`K_EPSILON_TOLERANCE` is 1e-22, for the filter's default epsilon of 1e-16. `K_SIZE` is 480 rows
by 640 columns, the size the hand cases use.

### expect_pixel_box

`expect_pixel_box` compares the four corners of a `PixelBox` within `K_PIXEL_TOLERANCE`.

### expect_normalized_box

`expect_normalized_box` compares the four values of a `NormalizedBox` within
`K_FRACTION_TOLERANCE`.

### box_of_sides

`box_of_sides` returns a box with its left top corner at the origin and the given width and
height.

### ToPixelBoxPlacesCornersFromCentreAndExtent

The normalized box (0.5, 0.25, 0.5, 0.25) in `K_SIZE` is the pixel box (160, 60, 480, 180).

### ToPixelBoxUsesWidthForXAndHeightForY

A box covering a 100 by 200 image is (0, 0, 200, 100). The two sides differ, so a conversion
that swapped them gives other corners.

### ToPixelBoxDoesNotClip

A box centred on the left edge converts to (-50, 75, 50, 125). The conversion leaves the
negative corner as it is.

### ToNormalizedBoxDividesCentreAndExtentByTheImageSize

The pixel box (160, 60, 480, 180) in `K_SIZE` is the normalized box (0.5, 0.25, 0.5, 0.25).

### NormalizedToPixelAndBackReturnsTheBox

Four normalized boxes, one of them all zeros, come back from the two conversions within
`K_FRACTION_TOLERANCE`.

### PixelToNormalizedAndBackReturnsTheBox

The pixel box (12.5, 7.25, 630, 471.75) comes back within `K_PIXEL_TOLERANCE`.

### ScaleBoxMultipliesEachAxisByItsOwnFactor

`scale_box` by 2 across and 0.5 down maps (10, 20, 30, 40) to (20, 10, 60, 20).

### TranslateBoxAddsEachAxisItsOwnOffset

`translate_box` by 5 across and -25 down maps (10, 20, 30, 40) to (15, -5, 35, 15).

### ClipBoxClampsEachEdgeToTheImage

`clip_box` maps (-10, 20, 700, 500) in `K_SIZE` to (0, 20, 640, 480).

### ClipBoxLeavesABoxInsideTheImageAlone

The box (0, 0, 640, 480) is unchanged.

### ClipBoxCollapsesABoxOutsideTheImageOntoTheEdge

A box to the right of the image clips to (640, 10, 640, 20) and one above it to
(10, 0, 20, 0). Both have area 0.

### BoxAreaIsWidthTimesHeight

`box_area` of (1, 2, 4, 6) is 12.

### PolygonFilterLowersOnlyTheAreaRatioToOneHundredth

`make_polygon_filter` returns a `CandidateFilter` whose `min_area_ratio` is 0.01, which is
`K_POLYGON_MIN_AREA_RATIO`. Its `min_side`, `max_aspect_ratio` and `epsilon` equal those of a
default filter.

### PolygonFilterKeepsABoxThePlainFilterRejects

A box that goes from 100 by 100 to 50 by 10 keeps 5 percent of its area. The default filter
rejects it and the polygon filter keeps it.

### CandidateFilterDefaultsAreTheReferenceThresholds

A default `CandidateFilter` has `min_side` 2, `max_aspect_ratio` 100, `min_area_ratio` 0.10 and
`epsilon` 1e-16.

### CandidateNeedsAWidthAboveTheMinimumSide

A width of 2 is rejected and a width of 2.25 is kept. The comparison is strict.

### CandidateNeedsAHeightAboveTheMinimumSide

A height of 2 is rejected and a height of 2.25 is kept.

### CandidateNeedsMoreThanTheMinimumAreaRatio

Against a 100 by 100 box, a 10 by 100 box has a ratio of 0.10 in float and is rejected. A width
of 10.5 is kept and a width of 9.5 is rejected.

### CandidateAreaRatioFollowsTheFilterField

With `min_area_ratio` set to 0.01, the 10 by 100 box is kept and a 10 by 10 box, whose ratio is
0.01, is rejected.

### CandidateNeedsAnAspectRatioBelowTheMaximum

A 300 by 3 box and a 3 by 300 box have an aspect ratio of 100 in float and are rejected. Boxes
of 297 by 3 and 3 by 297 are kept. The filter takes the larger of the two ratios.

### CandidateMeasuresSidesNotPositions

A 50 by 40 box moved to another position is kept. The filter reads widths and heights and no
coordinate.

### CandidateEpsilonIsAddedToTheAreaBefore

With `epsilon` 100 and `min_area_ratio` 0.6, a 10 by 10 box against itself has the ratio
100 / (100 + 100), which is 0.5, and is rejected. Without the epsilon in the denominator the
ratio is 1.

### CandidateEpsilonIsAddedToTheHeightOfAWideBox

With `epsilon` 1, a 300 by 3 box has the ratio 300 / (3 + 1), which is 75, and is kept.

### CandidateEpsilonIsAddedToTheWidthOfATallBox

With `epsilon` 1, a 3 by 300 box has the ratio 75 and is kept.

### CandidateWithAnEmptyBeforeBoxIsKeptAndAnEmptyAfterBoxIsNot

An empty box before and a 10 by 10 box after is kept. A 10 by 10 box before and an empty box
after is rejected.

## modules/vision/imaging/tests/test_imaging_flip.cpp

Suites: `Unit_ImagingFlip`.

Tests the plan and the application of the horizontal flip to an image and its labels.

This file needs no device. `Unit_ImagingFlip` covers `modules/vision/imaging/include/SushiAI/vision/imaging/flip.hpp`.
`plan_flip` makes the one random draw and records the decision in a `FlipPlan`; `apply_flip`
then mirrors an image or a label set from the plan and draws nothing.

### Constants

`K_BOX_TOLERANCE` is 1e-4 px. `K_SEED_COUNT` is 10000, the seeds 0 to 9999 a statistical case
draws from. `K_FREQUENCY_TOLERANCE` is 0.03, six standard deviations of a frequency of 0.5 over
10000 draws.

### make_counted_image

`make_counted_image` returns an image whose byte at offset i is i modulo 256.

### expect_same_image

`expect_same_image` expects two images to have the same size, channel count and bytes.

### expect_box

`expect_box` compares the four coordinates of two boxes within `K_BOX_TOLERANCE`.

### PlanRecordsTheSizeAndMakesExactlyOneDraw

The plan holds the source size 480 by 640. Its `is_flipped` equals one `bernoulli` draw of a
second generator with the same seed, and the two generators are equal afterwards.

### ProbabilityZeroNeverFlipsAndOneAlwaysFlips

Over 10000 seeds, probability 0 never flips and probability 1 always does, and both leave the
generator in the same state.

### DefaultProbabilityFlipsAboutHalfOfTheSeeds

With the default settings the flip frequency over 10000 seeds is 0.5 within
`K_FREQUENCY_TOLERANCE`.

### PlanRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5 and -0.1 throw with a message containing "validate_flip" and
"probability", and the generator is unchanged.

### PlanRejectsAProbabilityThatIsNotANumberBeforeAnyDraw

A NaN probability throws and the generator is unchanged.

### PlanRejectsASideThatIsNotPositiveBeforeAnyDraw

Sizes of 0 by 8 and 8 by -2 throw with a message containing "plan_flip", "source size" and the
size, and the generator is unchanged.

### SourceWithoutPixelsIsRejected

`apply_flip` on a default `ImageView` throws with "apply_flip" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_flip` with the image as both source and destination throws with "destination holds the
source's pixels", and the image is unchanged.

### ViewInsideTheDestinationIsRejected

A view of three rows of the destination is refused with the same message, and the image is
unchanged.

### FlippedImageHoldsEachRowInReverseColumnOrder

For a 2 by 3 image of three channels, pixel x of each destination row equals pixel 2 - x of the
source row, channel by channel. The first pixel of row 0 is bytes 6, 7, 8: the pixels are
reversed and the channels inside a pixel are not.

### UnflippedImageIsCopiedUnchanged

A plan with `is_flipped` false copies the source.

### FlippingAnImageTwiceRestoresIt

A 4 by 7 image of three channels and a 3 by 4 image of one channel both come back after two
flips.

### ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 3 source with a plan for 2 by 4 or for 5 by 3 throws with a message naming the plan's
size and "2x3".

### FlippedBoxesMirrorAboutTheWidthAndKeepTheirOrder

In a 640 px wide image the box (100, 50, 300, 200) becomes (340, 50, 540, 200), and a box
spanning the whole width is unchanged. Classes and order are kept, and x1 stays at or below x2.

### UnflippedBoxesAreReturnedUnchanged

An unflipped plan returns the labels as they are.

### FlippingBoxesTwiceRestoresThem

Two boxes come back after two flips within `K_BOX_TOLERANCE`.

### FlippedPolygonMirrorsEveryVertexAboutTheWidth

In a 640 px wide image the vertices at x = 100, 300 and 150.5 move to 540, 340 and 489.5, and
every y is kept. An object without a polygon still has none.

### UnflippedPolygonIsReturnedUnchanged

An unflipped plan returns the polygon's vertices as they are.

### ValidateAcceptsZeroAndOne

`validate_flip` does not throw for probability 0, 1 or the default.

## modules/vision/imaging/tests/test_imaging_gray.cpp

Suites: `Unit_ImagingGray`.

Tests the expansion of a one-channel image to three equal channels.

This file needs no device. `Unit_ImagingGray` covers `expand_gray` of
`modules/vision/imaging/include/SushiAI/vision/imaging/gray.hpp`.

### make_counted

`make_counted` returns an image whose byte at offset i is 17 * i + 3 modulo 256.

### ExpandCopiesEachLevelIntoThreeChannels

A 3 by 5 grey image expands to a 3 by 5 image of `K_RGB_CHANNELS` channels, and all three
channels of every pixel hold the pixel's grey level.

### ExpandRewritesAReusedDestination

A destination that was 9 by 9 with four channels and filled with 255 becomes 2 by 2 with three
channels, and every byte holds its pixel's grey level.

### ExpandRejectsASourceThatIsNotGrey

Sources of three and of four channels throw with "expand_gray" and the channel count.
`K_GRAY_CHANNELS` is 1.

### ExpandRejectsASourceWithoutPixelsAndOneInTheDestination

An empty view throws with "no pixels". The destination's own view, and a view of two of its
rows, throw with "destination holds the source's pixels". The image keeps its size, channel
count and bytes.

## modules/vision/imaging/tests/test_imaging_hsv.cpp

Suites: `Unit_ImagingHsv`.

Tests the hue, saturation and value jitter: its plan, its tables and both colour conversions.

This file needs no device. `Unit_ImagingHsv` covers `modules/vision/imaging/include/SushiAI/vision/imaging/hsv.hpp`.
`plan_hsv` draws three gains, `make_hsv_tables` turns a plan into three 256-entry tables, and
`apply_hsv` converts each pixel with `to_hsv`, looks it up and converts back with `to_rgb`. Hue
runs over `K_HUE_STEPS` (180) steps and every level is an integer, so the expected colours are
exact and a comment in the case derives them.

### Constants

`K_SEED_COUNT` is 10000, the seeds a statistical case draws from. `K_ROUND_TRIP_LEVELS` is 4,
the largest level error of `to_hsv` followed by `to_rgb`. `K_SWEEP_STEP` is 5, the step between
the levels of the colour sweep.

### make_row and pixel_at

`make_row` returns a one-row RGB image of the given pixels, and `pixel_at` returns the pixel at
a column of such an image.

### jitter

`jitter` applies a plan to a single pixel through `apply_hsv` and returns the result.

### expect_rgb and expect_hsv

`expect_rgb` and `expect_hsv` compare the three levels of a pixel for equality.

### level_distance

`level_distance` returns the absolute difference of two levels.

### PlanDrawsHueThenSaturationThenValueAndNothingMore

The plan's three gains equal three `uniform(-1, 1)` draws of a second generator, each times its
setting, in the order hue, saturation, value. The comparison is exact, and the two generators
are equal afterwards.

### DrawnGainsStayInsideTheConfiguredRangesOverManySeeds

Over 10000 seeds every drawn gain lies in [-gain, gain). The smallest of each is below -0.9
times its gain and the largest above 0.9 times it, so the draws cover the range.

### ZeroGainsDrawThreeTimesAndGiveAZeroPlan

With three gains of 0 the plan is three zeros and the generator has advanced by three draws.
The draw count does not depend on the settings.

### PlanRejectsANegativeGain

A saturation gain of -0.5 throws with "validate_hsv" and "gain", and the generator is
unchanged.

### PlanRejectsAGainThatIsNotFiniteBeforeAnyDraw

An infinite hue gain and a NaN value gain throw, and the generator is unchanged.

### SourceWithoutPixelsIsRejected

`apply_hsv` on a default view throws with "apply_hsv" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_hsv` with the image as source and destination throws with "destination holds the
source's pixels", and both pixels are unchanged.

### ViewInsideTheDestinationIsRejected

A view of the destination's second pixel is refused with the same message, and the image keeps
its size and pixels.

### ZeroPlanGivesIdentityTables

For a zero plan the saturation and value tables map every level to itself. The hue table maps
x to x below 180 and to x - 180 from 180 up.

### HueTableShiftsByAFractionOfATurnAndWrapsBothWays

A hue gain of 0.25 shifts by 45 steps: 0 maps to 45, 134 to 179 and 135 to 0. A gain of -0.25
maps 0 to 135 and 45 to 0. Every entry of both tables is below 180.

### SaturationAndValueTablesScaleTruncateAndClamp

A saturation gain of 0.5 multiplies by 1.5: 3 maps to 4, which is 4.5 truncated, and 170 and
above map to 255. A value gain of 0.25 multiplies by 1.25: 203 maps to 253 and 204 to 255. A
saturation gain of -0.5 maps 255 to 127, and a value gain of -2 maps every level to 0.

### PrimaryColoursHaveHuesSixtyStepsApart

`to_hsv` gives red, green and blue the hues 0, 60 and 120 at saturation 255 and value 255.

### GreyHasHueZeroAndBlackHasSaturationZero

Black, a mid grey and white have hue 0 and saturation 0, and their value is their level.

### MixedColourConvertsToTheHandComputedLevels

(100, 50, 50) converts to hue 0, saturation 128, value 100. (200, 100, 50) converts to hue 10,
saturation 191, value 200. The comments give the integer arithmetic.

### EachSectorConvertsBackToTheHandComputedColour

`to_rgb` is checked at eight HSV values, one or more in each of the six hue sectors, among them
hue 178 in the last sector and a colour that is not fully saturated.

### PureHueShiftOnRedGivesTheHandComputedColours

Red with a hue gain of 0.25 becomes (128, 255, 0), with -0.25 it becomes (128, 0, 255) and with
0.5 it becomes (0, 255, 255).

### ValueGainClampsAtFullAndAtZero

(100, 50, 50) with a value gain of 2 becomes (255, 127, 127), and with -2 it becomes black.

### SaturationGainClampsAtFullAndAtZero

(100, 50, 50) with a saturation gain of 2 becomes (100, 0, 0), and with -2 it becomes the grey
(100, 100, 100).

### WhiteStaysWhiteUnderASaturationGain

White with a hue gain of 0.3 and a saturation gain of 0.7 stays white. Its saturation is 0, and
0 times any factor is 0.

### ZeroPlanIsTheIdentityOnGreysPrimariesAndSecondaries

A zero plan returns nine colours unchanged: three greys, the three primaries and the three
secondaries.

### ZeroPlanCopiesColoursTheRoundTripWouldChange

`to_rgb(to_hsv(.))` changes the blue level of (12, 200, 99) to 100 and of (250, 3, 180) to 176,
which the case asserts first. A zero plan returns both colours unchanged, so `apply_hsv` copies
the image for an identity plan and does not send it through the conversion.

### PlanOfZeroGainsCopiesTheImage

Over 64 seeds a plan drawn from three zero gains satisfies `is_hsv_identity`, and the same two
colours come back unchanged.

### PlanWithOneGainAwayFromZeroIsNotAnIdentity

`is_hsv_identity` is true for a zero plan and for one with negative zeros, and false when any
one gain is 1e-6 or -1e-6. A value gain of 1e-6 changes no table entry, and (12, 200, 99) still
comes back as (12, 200, 100): the image went through the conversion.

### ValidateAcceptsZeroGainsAndTheDefaults

`validate_hsv` does not throw for three zero gains or for the defaults.

### RoundTripMovesNoLevelByMoreThanItsBound

Over a sweep of every fifth level on each channel, `to_hsv` gives a hue below 180 and `to_rgb`
of it differs from the original by at most `K_ROUND_TRIP_LEVELS` on any channel.

### ImageWithoutThreeChannelsIsCopiedUnchanged

A one-channel image with a plan that is not the identity comes back with the same size, one
channel and the same bytes.

### SamePlanGivesTheSameImageTwice

Two applications of one plan to one image give the same bytes.

## modules/vision/imaging/tests/test_imaging_image.cpp

Suites: `Unit_ImagingImage`.

Tests the owned image, the view over foreign pixels and the checks both share.

This file needs no device. `Unit_ImagingImage` covers `modules/vision/imaging/include/SushiAI/vision/imaging/image.hpp`:
`ImageSize`, `ImageView`, `Image`, `overlaps` and `check_source`. Pixels are bytes in
height, width, channel order.

Eleven `static_assert` lines at file scope hold the parts that are constant expressions:
equality of `ImageSize`, `is_valid_side`, `is_valid_size` and `is_valid_channel_count` at both
ends of their ranges and one past each, and `row_bytes()` and `byte_count()` of a view.

### write_offsets

`write_offsets` writes each byte's own offset, modulo 251, into an image.

### ImageSizeComparesHeightAndWidth

Two sizes are equal when both sides are, and 3 by 4 differs from 4 by 3.

### ADefaultImageHasNoPixels

A default `Image` has height, width, channel count, row bytes, byte count and allocated bytes
of 0.

### AConstructedImageReportsItsExtents

An image of 4 by 6 with three channels reports those extents, 18 row bytes and 72 bytes, and
its data pointer is not null.

### AConstructedImageIsZero

Every byte of a constructed image is 0.

### TheLargestSideAndChannelCountAreAccepted

An image of `K_MAX_IMAGE_SIDE` by 1 and one of 1 by `K_MAX_IMAGE_SIDE` with `K_MAX_CHANNELS`
channels are constructed and report their byte counts.

### ConstructionRejectsAHeightOutsideItsBounds

Heights of 0, -2 and 16385 throw with a message containing "Image:" and the size.

### ConstructionRejectsAWidthOutsideItsBounds

Widths of 0, -7 and 16385 throw with a message containing "Image:" and the size.

### ConstructionRejectsAChannelCountOutsideItsBounds

Channel counts of 0, -1 and 5 throw with a message containing "Image:" and the count; the
first also contains "[1, 4]".

### RowsStartOneRowOfBytesApart

`row(y)` of the image, of a const reference to it and of its view all equal the data pointer
plus 15 * y for a 5 px wide image of three channels.

### AByteWrittenThroughARowIsReadAtItsHwcOffset

A byte written at row 2, column 1, channel 2 of a 3 by 4 image is byte 29 of the data, and
every other byte is 0.

### FillSetsEveryByte

`fill(114)` sets every byte to 114.

### FillOnAnEmptyImageDoesNothing

`fill` on a default image leaves the byte count at 0.

### ResetGivesTheImageNewExtents

`reset` to 5 by 3 with four channels gives 12 row bytes and 60 bytes, and every byte of the new
extent can be written and read back.

### ResetOnADefaultImageAllocates

`reset` on a default image gives a data pointer that is not null and 18 bytes.

### ResetKeepsTheAllocationWhenItIsLargeEnough

An 8 by 8 image of three channels keeps its data pointer through a reset to 6 bytes and through
one to 192 bytes, the count it was constructed with.

### ResetRejectsTheBoundsTheConstructorRejects

`reset` throws for a height of 0, a width of 16385 and five channels, with a message
containing "Image::reset".

### ARejectedResetLeavesTheImageAsItWas

After a `reset` that throws, the image has its size, channel count, data pointer and bytes.

### AViewOfAnImageDescribesTheSamePixels

`view()` has the image's data pointer, size, channel count, row bytes and byte count.

### ADefaultViewIsEmpty

A default `ImageView` has a null pointer, size 0 by 0 and no bytes.

### AViewOverForeignBytesAddressesThemInHwcOrder

A view over a twelve-byte array as 2 by 3 with two channels has 6 row bytes, and row 1,
column 2, channel 1 is byte 11.

### ACopyOwnsItsOwnPixels

A copy has another data pointer, and filling the copy leaves the original's bytes as they
were.

### AllocatedBytesKeepsTheLargestSizeTheImageHad

After a reset from 48 bytes to 4, `byte_count()` is 4 and `allocated_bytes()` is 48.

### OverlapsFindsTheWholeImageAndAViewInsideIt

`overlaps` is true for the image's own view, for a view of two of its rows and for a view of
its last byte.

### OverlapsIsFalseForBytesOfAnotherContainer

A view over an array the image does not own does not overlap it.

### OverlapsCountsTheAllocationBeyondTheCurrentSize

A view of the last row of a 4 by 4 image still overlaps it after the image is reset to one
byte. The test is against the allocation and not the current size.

### OverlapsIsFalseForSeparateOrEmptyOperands

`overlaps` is false for two separate images, for a default view, for an unallocated image, and
for views with a height of 0 or no channels.

### CheckSourceAcceptsAViewOfAnotherImage

`check_source` does not throw for a view of another image, with an allocated or an unallocated
destination.

### CheckSourceRejectsEachFaultUnderTheCallersName

`check_source` throws with the caller's name as prefix for a view without pixels, a height of
0, a width of 16385, five channels, the destination's own view and a view of two of its rows.

## modules/vision/imaging/tests/test_imaging_label_set.cpp

Suites: `Unit_ImagingLabelSet`.

Tests the label set that holds classes, boxes and polygons in step, and its conversions.

This file needs no device. `Unit_ImagingLabelSet` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/label_set.hpp`. `Labels` holds pixel boxes and `FileLabels` holds
normalized boxes; both keep one class, one box and one polygon per object, and a polygon may be
empty.

### Tolerances and K_SIZE

`K_PIXEL_TOLERANCE` is 1e-3 px, `K_FRACTION_TOLERANCE` is 1e-6 of the image, and `K_SIZE` is
480 rows by 640 columns.

### expect_pixel_box, expect_normalized_box and expect_point

The three helpers compare a `PixelBox`, a `NormalizedBox` and a `Point` with expected values.
`expect_point` takes its tolerance as an argument.

### make_diagonal

`make_diagonal` returns a polygon of a given number of vertices spread along the diagonal from
(0, 0) towards (1, 1).

### DefaultSetHoldsNoObject

A default `Labels` is empty, has size 0 and has no class, box or polygon.

### ConstructorsRejectListsOfDifferentLengths

Two classes with one box, one class with two boxes, two objects with one polygon, and no class
with one box all throw with a message containing "LabelSet" and the two counts.

### SetWithoutPolygonsHoldsOneEmptyPolygonPerObject

A set built from two classes and two boxes has two polygons, both empty.

### AddAppendsOneEntryToEachList

`add` with and without a polygon grows the classes, the boxes and the polygons together. The
object added without one has an empty polygon.

### KeepIfRemovesAnObjectFromAllThreeLists

`keep_if` with a predicate that keeps even classes leaves classes 2 and 4 with their own boxes
and polygons.

### ToPixelLabelsKeepsClassesAndOrder

`to_pixel_labels` converts three normalized boxes in `K_SIZE` and keeps the classes 7, 2, 7 in
order.

### ToPixelLabelsScalesVerticesByWidthAndHeight

A vertex at (0.25, 0.25) becomes (160, 120): x is multiplied by the width and y by the height.
An empty polygon stays empty.

### ToNormalizedLabelsKeepsClassesAndOrder

`to_normalized_labels` converts two pixel boxes and the vertices of one polygon, and keeps
classes and order.

### LabelConversionsOfABackgroundImageAreEmpty

Both conversions of an empty set are empty.

### ClipLabelsClipsBoxesAndVerticesAndRemovesNone

`clip_labels` clips each box to the image and clamps each vertex onto it. A box wholly outside
collapses to zero width and stays in the set.

### RemoveEmptyBoxesDropsZeroAreaAndKeepsOrderClassesAndPolygons

Of five objects, the two with zero width or zero height are removed. The classes 10, 12 and 14
remain in order with their boxes and polygons.

### RemoveEmptyBoxesDropsABoxOfNegativeArea

A box whose x1 is above its x2 is removed.

### ResamplePolygonsGivesEachPolygonAThousandPoints

`resample_polygons` gives a triangle and a polygon of 40 vertices `K_POLYGON_POINTS` points
each. An empty polygon stays empty, and classes and boxes are unchanged.

### ResamplePolygonsUsesTheLongestPolygonPlusOneAboveAThousand

With one polygon of 1200 vertices in the set, both polygons are resampled to 1201 points.

### ResamplePolygonsLeavesASetWithoutPolygonsAlone

A set whose one polygon is empty comes back with the polygon empty and the box unchanged.

## modules/vision/imaging/tests/test_imaging_labels.cpp

Suites: `Unit_ImagingLabels`.

Tests the parser of a label file's text.

This file needs no device and reads no file. `Unit_ImagingLabels` passes strings to
`parse_label_text` of `modules/vision/imaging/include/SushiAI/vision/imaging/labels.hpp`. A row is a class and four
normalized box values, or a class and the x, y pairs of a polygon. The refusal cases check that
the message names the line and quotes the field.

### K_CLASS_COUNT and K_FRACTION_TOLERANCE

`K_CLASS_COUNT` is 80, the class count the cases parse with. `K_FRACTION_TOLERANCE` is 1e-7,
the tolerance of a parsed value.

### expect_box

`expect_box` compares the four values of a `NormalizedBox` within `K_FRACTION_TOLERANCE`.

### ParsesBoxRowsInFileOrder

Two rows give two objects, with classes 0 and 3 and the four values of each row.

### EmptyTextIsABackgroundImage

An empty string gives an empty set.

### TextOfBlankLinesIsABackgroundImage

Text of newlines, spaces, a tab and carriage returns gives an empty set.

### AcceptsTabsRepeatedSpacesAndBlankLinesBetweenRows

Rows with leading spaces, tabs, repeated spaces, a blank line between them and no final newline
parse to two objects.

### AcceptsCarriageReturnLineEndings

Rows ended by "\r\n" and by a lone "\r" parse to three objects.

### AcceptsVerticalTabAndFormFeedLineEndings

A vertical tab and a form feed each end a row.

### AcceptsTheSeparatorBytesAsLineEndings

The bytes 0x1c, 0x1d and 0x1e each end a row: four rows separated by them parse to four
objects.

### AcceptsTheUnitSeparatorByteBetweenFields

The byte 0x1f separates fields, alone or repeated.

### AcceptsExponentsBareDecimalPointsAndAPlusSign

"+7.0 5e-1 .5 2.5E-1 +0.125" parses to class 7 and the box (0.5, 0.5, 0.25, 0.125).

### RejectsARowOfFourColumns

A row of four columns throws with "parse_label_text: line 1", "5 columns" and "this row has 4".

### RejectsARowOfSixColumns

A row of six columns throws with "5 columns" and "this row has 6".

### RejectsARowOfOneColumn

A row of one column throws with "this row has 1".

### NamesTheLineOfTheRejectedRow

A short row after one row and one blank line is reported as "line 3". Blank lines count.

### RejectsAFieldThatIsNotANumber

The field "abc" throws with "'abc'" and "not a number".

### RejectsAFieldWithCharactersAfterTheNumber

The field "0.5x" throws with "'0.5x'".

### RejectsADecimalComma

The field "0,5" throws with "'0,5'".

### RejectsADoubledSign

The field "+-0.5" throws with "'+-0.5'".

### RejectsNotANumberAndInfinity

The fields "nan", "inf" and "1e999" each throw with the field quoted.

### RejectsAFiniteValueAboveTheFloatRange

The field "1e39", finite as a double and not as a float, throws with "not a finite number".

### RejectsANonzeroValueBelowTheSmallestNormalDouble

The fields "1e-310" and "1e-999" throw with the field quoted.

### AcceptsZeroAndTheSmallestNormalDouble

"0.0" and "2.2250738585072014e-308" are accepted and both store 0 within the tolerance.

### RejectsACoordinateBelowTheTolerance

Values of -0.02 and -0.5 throw with "negative".

### RejectsACoordinateAboveTheTolerance

Values of 1.02 and 640 throw with "above". The second is a pixel value in a file that must hold
fractions.

### StoresACoordinateInsideTheToleranceUnclipped

Values of 1.005 and -0.005 are stored as they are.

### AcceptsACoordinateOnEitherToleranceBound

Values of 1.01 and -0.01 are accepted and stored as they are.

### RejectsANegativeClass

Class -1 throws with "class" and "negative".

### RejectsAClassWithAFraction

Class 1.5 throws with "class" and "whole number".

### RejectsAClassAtOrAboveTheClassCount

Class 80 of 80 throws with "out of range" and "80 classes", and class 1 of 1 with "1 classes".

### AcceptsTheLastClass

Class 79 of 80 is accepted.

### RemovesRepeatedRowsKeepingTheFirstAndTheFileOrder

Five rows of which two repeat earlier ones give three objects, with classes 5, 2, 9 in file
order.

### RemovesARowThatRepeatsAnotherInADifferentSpelling

"5.0 5e-1 0.50 .25 0.250" repeats "5 0.5 0.5 0.25 0.25" and is removed. Rows are compared by
value and not by text.

### KeepsRowsThatDifferInOneValue

Three rows that differ in the class or in one box value are all kept.

### ReducesAPolygonRowToTheBoxAroundItsPoints

A row of three points gives the box (0.5, 0.5, 0.5, 0.5) around them.

### KeepsThePointsOfAPolygonRowInFileOrder

A row of five points gives a polygon of those five points in file order.

### GivesABoxRowNoPolygon

Each object of two box rows has an empty polygon.

### ReducesAPolygonOfManyPointsToItsBounds

The five points of the earlier case give the box (0.5, 0.375, 0.75, 0.5).

### RejectsABoxRowInAPolygonFile

A box row after a polygon row throws with "line 2" and "polygon".

### RejectsAPolygonRowWithAHalfPoint

A polygon row of eight columns throws with "polygon" and "this row has 8".

### RejectsAPolygonRowOfFewerThanThreePoints

A row of three columns in a polygon file throws with "line 2", "three points" and "this row
has 3".

### RejectsAPolygonWhoseBoxLeavesTheImage

A polygon with a vertex at x = 1.5 throws with "above".

### RemovesARepeatedPolygonAndKeepsOneWithTheSameBox

Of three polygon rows the second repeats the first and is removed. The third has other points
and the same box, and is kept: both remaining boxes are (0.5, 0.5, 0.5, 0.5) and their second
vertices differ.

## modules/vision/imaging/tests/test_imaging_letterbox.cpp

Suites: `Unit_ImagingLetterbox`.

Tests the plan and the application of the letterbox to an image and its labels.

This file needs no device. `Unit_ImagingLetterbox` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp`. `plan_letterbox` computes the ratio, the resized size
and the four paddings, and draws no random number. `apply_letterbox` resizes with
`resize_bilinear` of `modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp`, pads, and moves boxes and polygons
by the same plan.

### K_BOX_TOLERANCE

`K_BOX_TOLERANCE` is 1e-4 px.

### make_counted_image and make_flat_image

`make_counted_image` returns an image whose byte at offset i is a first value plus i, modulo
256. `make_flat_image` returns an image with every byte at one level.

### is_content

`is_content` reports whether a pixel of the target lies inside the plan's resized image.

### expect_two_levels

`expect_two_levels` expects every content byte of an image to be one level and every padding
byte another.

### WideImageIsPaddedAboveAndBelowWithTheOddRowBelow

A 50 by 200 source in a 100 by 100 target has ratio 0.5 and resizes to 25 by 100. The 75 rows
of padding split into 37 above and 38 below, and the plan records the source and target sizes.

### TallImageIsPaddedLeftAndRightWithTheOddColumnRight

A 200 by 50 source resizes to 100 by 25 with 37 columns left and 38 right.

### ExactFitHasRatioOneAndNoPadding

A 64 by 64 source in a 64 by 64 target has ratio 1 and four paddings of 0.

### SmallImageIsOnlyPaddedUnlessScaleUpIsSet

A 20 by 40 source in an 80 by 80 target keeps its size by default, with paddings 30, 30, 20,
20. With `scale_up` set the ratio is 2, the resized size is 40 by 80 and the paddings are 20
above and below.

### ResizedSideRoundsAHalfToTheEvenInteger

At ratio 0.5 a height of 5 resizes to 2 and a height of 7 to 4: 2.5 and 3.5 round to the even
integer.

### RectangularStrideTargetPadsOnlyTheShortAxis

A 480 by 640 source in a 512 by 640 target has ratio 1 and 16 rows above and below.

### PlanCopiesThePaddingValueOfTheSettings

The plan's `padding_value` is that of the settings, at the default and at 7.

### PlanRejectsASideThatIsNotPositive

A source of 0 by 8 and a target of 8 by -1 throw with "plan_letterbox" and the size.

### PlanRejectsASideAboveTheImageBound

A source height and a target width of 16385 throw with the size and "16384".

### ValidateAcceptsEverySetting

`validate_letterbox` does not throw with `scale_up` set and a padding value of 255, or for the
defaults.

### PlanRejectsAResizedSideOfZeroPixels

A 1 by 1000 source in an 8 by 8 target resizes to a height of 0, and the plan throws with "zero
pixels".

### ImageWithoutResizeKeepsItsPixelsInsideThePadding

A 2 by 4 source in a 4 by 6 target is placed at row 1, column 1. Every content byte equals the
source byte and every other byte is the padding value.

### ExactFitCopiesEveryPixel

A plan without resize or padding copies the source.

### WideImageIsResizedThenPaddedWithTheFillValue

A flat 50 by 200 image at level 200 gives a 100 by 100 result that is 200 inside the content
rectangle and 114, the default padding value, outside it.

### TallImageIsResizedThenPaddedWithACustomFillValue

A flat one-channel 200 by 50 image at level 33 with a padding value of 7 gives 33 inside and 7
outside.

### ResizedContentEqualsTheBilinearResizeOfTheSource

For a counted 50 by 200 image, every content byte equals the byte `resize_bilinear` gives for
the plan's resized size, and every padding byte is 114. The case also expects the content not
to be flat, since a flat rectangle cannot tell a resize from a fill.

### SourceWithoutPixelsIsRejected

`apply_letterbox` on a default view throws with "apply_letterbox" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_letterbox` with the image as source and destination throws with "destination holds the
source's pixels", and the image keeps its size.

### ViewInsideTheDestinationIsRejected

A view of two rows of the destination is refused with the same message, and the image keeps its
size and its bytes.

### ResizedContentBesideSidePaddingEqualsTheBilinearResize

A counted 200 by 50 image in a 100 by 101 target has 38 columns of padding on each side. The
destination starts as a 120 by 120 image filled with 255. After the call no byte differs from
the bilinear resize placed between two bands of 114.

### DestinationIsFullyRewrittenWhenItIsReused

A destination that was 9 by 9 and filled with 255 becomes 4 by 6 with content 50 and padding
114.

### ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 4 source with a plan made for 4 by 4 throws with "plan was made for".

### PlanWhosePaddingsDoNotTileTheTargetIsRejected

A plan whose right padding is raised by 1 throws with "tile".

### BoxesMoveByTheRatioThenTheLeftAndTopPadding

In the tall plan a box maps by x * 0.5 + 37 and y * 0.5, and in the wide plan by x * 0.5 and
y * 0.5 + 37. Classes and order are kept.

### BoxesOutsideTheImageAreNeitherClippedNorRemoved

A box that extends past the source and a box of zero size are both moved by the plan and kept.

### ExactFitLeavesBoxesWhereTheyWere

With ratio 1 and no padding a box keeps its four coordinates.

### PolygonMovesByTheRatioThenTheLeftAndTopPadding

In the tall plan the vertices (10, 20), (30, 40) and (-4, 210) move to (42, 10), (52, 20) and
(35, 105). An object without a polygon still has none.

## modules/vision/imaging/tests/test_imaging_mask_pipeline.cpp

Suites: `Unit_ImagingAffineNearest`, `Unit_ImagingMaskPipeline`.

Tests `apply_affine_nearest` in
`modules/vision/imaging/include/SushiAI/vision/imaging/affine_nearest.hpp` and
`TrainingMaskPipeline` in
`modules/vision/imaging/include/SushiAI/vision/imaging/mask_pipeline.hpp`. Every plan scales by
one factor, translates and has the border value 255. The pipeline cases take grey RGB images
whose one box of the image's index covers the whole image, masks held in memory, and the seed
20261011.

### AnIdentityPlanCopiesTheSource

A plan of scale 1 and no translation over a mask of 3 rows and 4 columns gives an image of
that size and one channel holding the source's bytes.

### ATranslationFillsWhatItUncoversWithTheBorderValue

A translation of 2 columns and 1 row over a mask of 2 rows and 4 columns gives a first row of
255. In the second row the first two columns are 255 and column x holds the source's row 0,
column x - 2.

### AScaleRepeatsPixelsAndBlendsNone

The row 3, 50, 7, 120 warped at scale 2 into 8 columns is 3, 50, 50, 7, 7, 120, 120, 255.

### EveryChannelOfAPixelMovesTogether

An RGB row of the pixels (1, 2, 3) and (4, 5, 6) translated by one column is (255, 255, 255),
(1, 2, 3).

### RefusesWhatItCannotWarp

`apply_affine_nearest` fails with `InvalidDescription` for an empty view ("holds no image"),
for an output that is the source image ("share pixels") and for a matrix with a shear term of
0.5 ("off-diagonal"). A plan whose source size is 3 by 3 over a 2 by 2 image fails with
`Incompatible`, "3x3" and "2x2".

### AFlippedSampleHasItsMaskMirrored

With no mosaic, an identity warp and a flip probability of 1 over an 8 by 8 image, the trace
of sample 0 is flipped and has no mosaic. `make_mask` over that trace gives an 8 by 8 mask
whose pixel (y, x) holds the source mask's pixel (y, 7 - x).

### AMosaicMaskFollowsTheBoxesOfItsSample

Four images of four sizes have masks filled with their own index, and the training size is 32
with the default settings. For each sample of the epochs 0, 1 and 2 the trace has the mosaic
enabled and the mask is 32 by 32 with one channel. Every mask pixel at least 2 pixels inside a
box of the sample holds that box's class, and at least one pixel is compared.

### RefusesAMaskItCannotReplay

`make_mask` fails with `Incompatible` for a mask of three channels ("3 channels") and for a
mask of 8 rows and 4 columns under an 8 by 8 image ("8x4", "8x8"). When the mask source fails
it returns that failure: `InvalidFile` and "mask 0 is unreadable".

## modules/vision/imaging/tests/test_imaging_mosaic.cpp

Suites: `Unit_ImagingMosaic`.

Tests the draw, the plan and the application of the four-image mosaic.

This file needs no device. `Unit_ImagingMosaic` covers `modules/vision/imaging/include/SushiAI/vision/imaging/mosaic.hpp`.
`draw_mosaic` asks a `PartnerSource` for three partners, draws the gate and then the centre.
`plan_mosaic` turns a centre and four patch sizes into four placements on a canvas of twice the
image size. `apply_mosaic` copies the patches onto the canvas, or shifts, joins, clips and
prunes their labels. The two placement tables are lines 525 to 538 of the reference's
`data/augment.py`, worked by hand.

### K_SIDE and K_BOX_TOLERANCE

`K_SIDE` is 8, the image size of the hand cases, so the canvas is 16 by 16. `K_BOX_TOLERANCE`
is 1e-4 px.

### PatchSizes and Placements

`PatchSizes` holds four `ImageSize` values and `Placements` holds four `MosaicPlacement`
values, patch 0 first.

### The low centre

`K_LOW_CENTER_X` and `K_LOW_CENTER_Y` are 5 and 11, a centre that cuts patch 0 on the left and
patches 2 and 3 at the bottom. `K_LOW_SIZES` are the patch sizes used with it; patch 1 is
shorter and patch 2 narrower than the others. `K_LOW_PLACEMENTS` are the four expected
placements.

### The high centre

`K_HIGH_CENTER_X` and `K_HIGH_CENTER_Y` are 10 and 6, a centre that cuts patches 0 and 1 at the
top and patches 1 and 3 on the right. `K_HIGH_SIZES` are four 8 by 8 patches and
`K_HIGH_PLACEMENTS` are the four expected placements.

### make_draw

`make_draw` returns an enabled `MosaicDraw` with a given centre and the partners 0, 0, 0.

### make_constant and make_textured

`make_constant` returns a three-channel image of one colour. `make_textured` returns one whose
every byte depends on its row, column, channel and a salt, so a copy from the wrong position
shows.

### view_all

`view_all` returns the views of four images in patch order.

### count_wrong_bytes

`count_wrong_bytes` counts the canvas bytes that differ from what four placements name: the
patch byte at the placement's offset inside a placement, and the fill value elsewhere.

### expect_placements

`expect_placements` compares all ten fields of a plan's four placements with the expected ones.

### expect_box and expect_point

`expect_box` and `expect_point` compare a box and a vertex within `K_BOX_TOLERANCE`.

### RecordingPartners

`RecordingPartners` is a `PartnerSource` that returns the partners 4, 5 and 6, makes no draw
and records every index it is asked for.

### pixel_at

`pixel_at` returns the three bytes of one canvas pixel.

### SettingsDefaultToTheReferenceProbabilityAndFill

Default `MosaicSettings` have probability 1 and fill value 114, and `K_MOSAIC_PATCHES` is 4.

### DrawMakesPartnersThenGateThenRowThenColumn

With a `DatasetPartners` of 1000 and image size 640, the draw equals a replay on a copied
generator: three `index(1000)` draws, one `bernoulli(1)`, then `uniform(320, 960)` truncated
for the centre row and again for the centre column. The generators are equal afterwards, so
the call makes six draws.

### DisabledDrawMakesFourDrawsAndLeavesTheFieldsZero

With probability 0 the draw is disabled, its partners and centre are 0, and the generator has
advanced by four draws: the three partners and the gate.

### CentreStaysInTheReferenceRangeOverManySeeds

Over 5000 seeds at image size 640 both centre coordinates lie in 320 to 959. Their mean is
639.5 within 10, the smallest is at most 330 and the largest at least 949.

### CentreOfAnOddSizeUsesTheFlooredBorder

At image size 5 the centre lies in 3 to 6 and all four values occur over 400 seeds. The
reference computes the border as -5 // 2, which is -3, so the range is `int(uniform(3, 7))`.

### PartnersCoverThePoolAndNeverLeaveIt

Over 300 seeds and a pool of 7, every partner is below 7 and every index occurs.

### GateOpensWithItsProbability

At probability 0.25 the draw is enabled for 0.25 of 4000 seeds, within 0.04.

### PlanCutsPatchesAtTheLeftAndBottomEdges

`plan_mosaic` for the low centre gives `K_LOW_PLACEMENTS` and records the image size.

### PlanCutsPatchesAtTheTopAndRightEdges

`plan_mosaic` for the high centre gives `K_HIGH_PLACEMENTS`.

### PlanCarriesTheFillValueOfTheSettings

The plan's `fill_value` is 9 when the settings say 9 and 114 by default.

### CanvasIsTwiceTheSizeAndFilledWhereNoPatchLands

The canvas is 16 by 16 with three channels. Its two outer corners, a pixel in the row above the
shorter patch 1 and a pixel in the column left of the narrower patch 2 hold the fill value.

### EachColourLandsInItsQuadrantAroundTheCentre

Four patches of one colour each are placed around the low centre. The four pixels that meet at
the centre are red, green, blue and yellow, in patch order, and `count_wrong_bytes` is 0.

### CutPatchesKeepThePixelsNearestTheCentre

Four textured patches around the high centre with fill value 3 give a canvas with no wrong
byte. The texture makes the test fail for a patch copied from its far corner.

### ApplyOverwritesWhatTheDestinationHeld

A destination that was 40 by 40 and filled with 200 becomes the 16 by 16 canvas with no wrong
byte.

### LabelsAreShiftedJoinedInPatchOrderClippedAndPruned

Five boxes in patches 0, 1 and 3 are shifted by their placements' offsets and joined in patch
order. The second box of patch 0 lies wholly left of the canvas, has no area after the clip and
is removed. The four that remain have the classes 3, 2, 7, 5 and the boxes the case lists.

### LabelsOfFourBackgroundPatchesAreEmpty

Four empty label sets join to an empty one.

### DrawRejectsANonPositiveSizeBeforeAnyDraw

Image sizes of 0 and -4 throw with "validate_mosaic_size" and the value, and the generator is
unchanged.

### PlanRejectsADisabledDrawAndAnEmptyPatch

`plan_mosaic` throws for a disabled draw ("not enabled"), for a patch of height 0 ("patch 2")
and for a centre column of 17 on a 16 px canvas ("centre").

### ApplyRejectsPatchesThePlanWasNotMadeFor

`apply_mosaic` throws for a one-channel patch among three-channel ones ("patch 1", "channels")
and for a patch of another size than the plan's ("patch 3").

### DrawnCentreIsWhereTheFourColoursMeet

Over 32 seeds the case draws, plans and applies a mosaic of four 8 by 8 patches of one colour
each. The centre lies in 4 to 11, and the four pixels around it are red, green, blue and
yellow. The draw and the plan therefore agree on which coordinate is the row.

### DrawRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5, -0.25 and NaN throw with "validate_mosaic" and "probability", the
default settings pass `validate_mosaic`, and the generator is unchanged.

### DrawRejectsASizeWhoseCanvasIsTooLarge

An image size of 8193, whose canvas would exceed `K_MAX_IMAGE_SIDE`, throws with "image_size of
8193" and "[1, 8192]". An image size of 8192 is accepted.

### DrawAsksThePartnerSourceForEverySample

With a `RecordingPartners`, one draw with the gate open and one with it closed record the
indices 17 and 23. The open draw holds the partners 4, 5, 6 and the closed one 0, 0, 0.

### PolygonsAreShiftedClampedAndPrunedWithTheirObjects

Polygons move with their patch, by (-3, 3) for patch 0 and (5, 11) for patch 3, and are clamped
to the 16 by 16 canvas. The object whose box leaves the canvas is removed with its polygon, and
the object without a polygon still has none.

### PlanRejectsASizeOutOfRangeAndAPatchSideAboveTheLimit

`plan_mosaic` throws for an image size of 0 and of 8193 ("image_size") and for a patch height
of 16385 ("patch 1").

### ApplyRejectsAPlanWithoutAValidSizeForPixelsAndForLabels

Both forms of `apply_mosaic` throw with "image_size" for a default plan and for one whose image
size is 8193.

### ApplyRejectsAPatchWithoutPixels

A default view as patch 2 throws with "apply_mosaic: patch 2" and "no pixels".

### ApplyRejectsAPatchThatLiesAnywhereInTheDestination

A patch that views the start of the destination, and one that views its row 20, throw with
"destination holds the source's pixels" under the patch's number. The destination keeps its
40 by 40 size and its bytes.

### ApplyRejectsAPlacementOutsideTheCanvasAndOneThatResamples

A placement whose `canvas_x2` is 17 throws with "patch 1" and "outside a canvas". A placement
whose patch span is one pixel narrower than its canvas span throws with "patch 3" and
"resamples nothing".

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

## modules/vision/imaging/tests/test_imaging_pipeline.cpp

Suites: `Unit_ImagingPipeline`.

Tests the training and the validation pipeline that compose the imaging transforms into a
sample.

This file needs no device and reads no file. `Unit_ImagingPipeline` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/pipeline.hpp`. A `TrainingPipeline` loads an image from a
`SampleSource`, draws and applies mosaic, affine, colour jitter and flip, and returns a
`TrainingTrace` of the plans it used. A `ValidationPipeline` resizes the long side and
letterboxes. The cases replay a sample's draws on a copy of its generator and rebuild the
sample from the trace with the transform functions, so the pipeline is compared with its parts
and not with stored images.

### Constants

`K_SIDE` is 32, the training size of the cases, so a mosaic canvas is 64 by 64.
`K_REPLAY_TOLERANCE` is 0: a replayed draw repeats the same arithmetic. `K_BOX_TOLERANCE` is
1e-4 px. `K_MIN_SCALE_GAP` is 2^-24, one step of the 24-bit uniform over the default scale
range of width 1.

### MemorySource

`MemorySource` is a `SampleSource` that holds images and labels in memory and records the index
of every `load`. `loads` returns the recorded indices and `clear_loads` forgets them.

### FixedPartners

`FixedPartners` is a `PartnerSource` that returns the three partners it was constructed with
and makes no draw.

### make_textured, make_constant and make_flat

`make_textured` returns a three-channel image whose every byte depends on its row, column,
channel and a salt. `make_constant` returns a three-channel image of one colour. `make_flat`
returns an image of a given channel count with every byte at one level.

### one_object

`one_object` returns the `FileLabels` of one object of a given class and box.

### fill_source

`fill_source` adds five textured images to a source: 32 by 64, 64 by 64, 64 by 32, 16 by 32 and
32 by 32. The first four carry one object each and the last carries none.

### make_settings

`make_settings` returns default `TrainingSettings` with the image size `K_SIDE`.

### make_still_settings

`make_still_settings` returns settings under which nothing moves: one epoch with
`close_mosaic` 1, so epoch 0 has no mosaic, affine ranges of 0, HSV gains of 0 and a flip
probability of 0.

### whole_dataset

`whole_dataset` returns a `DatasetPartners` over every image of a source.

### images_match

`images_match` returns a GoogleTest assertion result that says whether two images have the same
size, channel count and bytes, and how they differ when they do not.

### expect_same_labels

`expect_same_labels` expects two label sets to hold the same classes, box edges and polygon
vertices at `K_REPLAY_TOLERANCE`.

### expect_same_affine and expect_same_hsv

`expect_same_affine` compares the sizes and the three drawn values of two `AffinePlan` objects.
`expect_same_hsv` compares the three gains of two `HSVPlan` objects.

### load_patch

`load_patch` loads one image the way a pipeline does: `resize_long_side` to `K_SIDE`,
`expand_gray` when the image has one channel, and `to_pixel_labels` of the resampled polygons.

### compose_from_trace

`compose_from_trace` builds a sample from a `TrainingTrace` by calling the transforms in the
reference's order, mosaic, affine, colour jitter, flip (`data/augment.py`, lines 2870 to 2912),
each with the plan the trace holds.

### SettingsDefaultToTheReferenceRun

Default `TrainingSettings` have image size 640, 100 epochs, `close_mosaic` 10, batch size 16
and seed 0. Default `ValidationSettings` have image size 640 and a letterbox that does not
scale up.

### MosaicClosesForTheLastTenEpochsOfAHundred

`is_mosaic_closed` is false for epochs 0 and 89 of 100 with `close_mosaic` 10 and true for 90
and 99. With `close_mosaic` 0 it is false for epoch 99, and with 5 epochs and `close_mosaic`
10 it is true for epoch 0.

### SampleRandomDerivesFromTheSeedTheEpochAndTheIndex

`make_sample_random(2026, 3, 17)` equals `Random(2026).derive(3).derive(17)`. Another seed,
epoch or index gives another generator, and so do the epoch and the index swapped.

### TrainingDrawsMosaicThenAffineThenHSVThenFlip

The trace of sample 2 in epoch 5 equals a replay on `make_sample_random(41, 5, 2)`:
`draw_mosaic`, then `plan_affine` from the 64 by 64 canvas to 32 by 32, then `plan_hsv`, then
`plan_flip`.

### TrainingLoadsTheSampleThenItsPartnersInDrawOrder

The source records four loads: index 3, then the trace's three partners in order.

### TrainingAppliesItsPlansInTheReferenceOrder

For every index of the source in epochs 0 and 1, the sample's image and labels equal what
`compose_from_trace` builds from the trace.

### TrainingAppliesAForcedFlipAndSkipsAForbiddenOne

Two pipelines with one seed and flip probabilities of 1 and 0 give images that are mirrors of
each other, pixel for pixel. The flipped sample also equals `compose_from_trace` of its trace.

### TrainingProducesTheTrainingSizeAndTheSizesOfItsOwnImage

The sample is 32 by 32 with `K_RGB_CHANNELS` channels. `original_size` is 32 by 64, the stored
size of image 0, and `loaded_size` is 16 by 32, its size after the long-side resize. The affine
plan maps 64 by 64 to 32 by 32.

### MosaicSampleKeepsItsLargestQuadrantBoxInsideTheTrainingSize

Four 32 by 32 images each carry a box over the whole image, and the affine ranges are 0, so
the warp shows canvas rows and columns 16 to 48. Over eight seeds every box of the sample lies
inside the 32 by 32 output with positive width and height, and exactly one box equals the
quadrant that is at least 16 by 16, computed from the trace's centre.

### ClosedEpochSkipsTheMosaicAndItsDraws

With 10 epochs and `close_mosaic` 3, epoch 7 is closed. The trace says so, the mosaic is not
enabled, the source records one load, and the affine, HSV and flip plans equal a replay that
starts with `plan_affine` on the sample's generator. No mosaic draw precedes them. The sample
equals `compose_from_trace`.

### SampleWithoutMosaicKeepsTheBoxOfItsOwnImage

In a closed epoch with affine ranges of 0, the box (8, 4, 24, 12) of the 16 by 32 loaded image
becomes (8, 12, 24, 20) in the 32 by 32 output: the image is centred, which adds 8 to y. The
box is symmetric about the centre column, so a flip leaves it where it is.

### EpochBeforeTheClosedOnesStillMakesAMosaic

Epoch 6 of the same settings is open: the mosaic is enabled and the source records four loads.

### ClosedGateInAnOpenEpochWarpsTheLoadedImage

With a mosaic probability of 0 in an open epoch, the trace is not closed and the mosaic is not
enabled. The affine plan's source is the 32 by 16 loaded image, the source records one load,
and the sample equals `compose_from_trace`.

### SampleDependsOnTheSeedTheEpochAndTheIndexAlone

The same seed, epoch and index give the same image, labels, affine plan and partners a second
time. Another seed and another epoch each give another image and an affine scale that differs
by more than `K_MIN_SCALE_GAP`.

### EarlierSamplesDoNotShiftTheDrawsOfALaterOne

A pipeline that already made a mosaic sample and a closed-epoch sample, which differ by six
draws, gives the same sample for index 0 of epoch 0 as a new pipeline does.

### DefaultPartnersComeFromTheImagesThePipelineLoaded

A pipeline built without a partner source gives the first sample, index 3, the partners
3, 3, 3. The second sample's partners are each 3 or 1, the two indices loaded so far.

### GivenPartnerSourceChoosesThePartners

With `FixedPartners` of 4, 4, 0 the trace holds those partners and the source records the loads
2, 4, 4, 0.

### TrainingRejectsEverySettingOutOfRangeAtConstruction

Constructing a `TrainingPipeline` throws for an image size of 0 or 8193, 0 epochs, a
`close_mosaic` of -1, a batch size of 0, a mosaic probability of 1.5, an affine scale range of
1, a negative HSV gain and a NaN flip probability, each under the name of the validator that
found it. No rejected construction loads an image, and `validate_training` accepts the test
settings.

### TrainingRejectsAnEmptySourceAndANullPartnerSource

An empty source throws with "TrainingPipeline" and "no image", with or without a partner
source. A null partner source throws with "partner source is null".

### TrainingRejectsIndicesAndEpochsOutsideTheirRange

`make_sample` throws for index 5 of five images and for epochs -1 and 100 of 100.

### TrainingExpandsAGreyImageToRGB

A one-channel image at level 77 under the still settings gives a 32 by 32 sample of three
channels whose centre pixel is (77, 77, 77).

### TrainingComposesAMosaicOfGreyAndColourImages

A source of two one-channel images and one colour image, with fixed partners, gives a
three-channel 32 by 32 sample that equals `compose_from_trace`.

### TrainingRejectsAnImageOfFourChannels

A four-channel image throws from `make_sample` with "4 channels" and "image 0".

### TrainingCarriesAPolygonAndTakesTheBoxFromIt

Under the still settings a triangle in a 32 by 32 image keeps its place. The sample's one
object has class 5, the box (8, 8, 24, 24) and a polygon of `K_POLYGON_POINTS` points that
starts at (8, 8).

### ValidationResizesTheLongSideThenOnlyPads

An 8 by 16 image of one colour is loaded as 16 by 32 and letterboxed into 32 by 32 with ratio 1
and 8 rows above and below. The sample's rows 8 to 23 hold the colour and the other rows hold
114. The source records one load.

### ValidationMovesBoxesByThePaddingWithoutClipping

The box (8, 4, 24, 12) of the loaded image becomes (8, 12, 24, 20): the top padding adds 8 to
y.

### ValidationIntoARectangularTargetAddsNoPadding

With a 16 by 32 target for a 16 by 32 image the four paddings are 0 and the sample's image
equals the source image.

### ValidationRepeatsItsSampleAfterAnotherIndex

Sample 0 made again after sample 2 has the same image and labels as the first time, and sample
2 has another image.

### ValidationExpandsAGreyImageToRGB

A one-channel 8 by 16 image at level 61 gives a three-channel 32 by 32 sample with 61 in the
content rows and 114 in the padding.

### ValidationCarriesAPolygonThroughTheLetterbox

A triangle's polygon is resampled to `K_POLYGON_POINTS` points, and its first vertex, (8, 4) on
the loaded image, becomes (8, 12).

### ValidationRejectsSettingsAndIndicesOutsideTheirRange

Constructing a `ValidationPipeline` throws for an image size of 0 ("validate_validation") and
for an empty source ("no image"). `make_sample` throws for index 9 of five images.

## modules/vision/imaging/tests/test_imaging_polygon.cpp

Suites: `Unit_ImagingPolygon`.

Tests polygon resampling, mapping, clamping, the point test and the visible box.

This file needs no device. `Unit_ImagingPolygon` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/polygon.hpp`. The expected vertices and boxes are the outputs of the
reference's `resample_segments` and `segment2box` on the same inputs, written as literals; a
comment in each case names the call.

### Tolerances and K_IMAGE

`K_POINT_TOLERANCE` is 1e-6, for a resampled coordinate of magnitude up to 4.
`K_BOX_TOLERANCE` is 1e-4 px, for a box edge up to 100. `K_IMAGE` is 80 rows by 100 columns.

### expect_polygon

`expect_polygon` compares a polygon with a flat list of expected x, y values.

### expect_box

`expect_box` compares the four edges of a `PixelBox` within `K_BOX_TOLERANCE`.

### make_square, make_triangle and make_hexagon

`make_square` returns the square from (0, 0) to (4, 4). `make_triangle` returns a triangle in
fractions of an image. `make_hexagon` returns six vertices with integer coordinates.

### PointCountIsAThousandOrTheLongestPlusOne

`K_POLYGON_POINTS` is 1000. `resampled_point_count` is 1000 for a longest polygon of 0, 3 or
1000 vertices and 1002 for 1001.

### ResampleMergesTheVerticesWithEvenPositions

The square resampled to 8 and to 11 points keeps its four vertices and gains the evenly spaced
positions between them.

### ResampleToOneMoreThanTheClosedLengthRepeatsTheFirstVertex

The square resampled to 6 points gains one point, at the first vertex.

### ResampleInterpolatesAlongTheEdges

The triangle resampled to 9 points holds interpolated points such as (0.55, 0.275) on its first
edge.

### ResampleOfALongPolygonTakesEvenPositionsOnly

The hexagon, closed, has seven points. Resampled to 4 it gives every second vertex and the
first again; resampled to 7 it gives the six vertices and the first again.

### ResampleLeavesAPolygonOfTheCountAndAnEmptyOneAlone

The triangle resampled to 3 points is unchanged, and an empty polygon stays empty.

### ResampleToAThousandPointsKeepsTheOutline

The triangle resampled to 1000 points has 1000 points, starts and ends at x = 0.1, and has the
bounds (0.1, 0.2, 0.7, 0.9).

### ResampleRejectsACountOfZero

A count of 0 throws with "resample_polygon" and "point count of 0".

### MapScalesThenTranslatesEachAxis

`map_polygon` with scales 2 and 0.5 and offsets 10 and -1 maps the square to the corners
(10, -1), (18, -1), (18, 1), (10, 1).

### ClampMovesVerticesOntoTheBounds

`clamp_polygon` to the box (0, 0, 4, 5) moves each coordinate outside onto the nearest bound
and leaves the vertex inside alone.

### PointInsideOrOnTheOutlineIsInside

`is_point_inside` is true for the centre of the square, a point on an edge and two vertices.

### PointOutsideIsOutside

`is_point_inside` is false for four points outside the square, for a point in the notch of a
concave polygon and for any point against an empty polygon. Two other points of the concave
polygon are inside.

### VisibleBoxOfAPolygonInsideTheImageIsItsBounds

A triangle inside `K_IMAGE` has the visible box (10, 20, 50, 60).

### VisibleBoxOfAPolygonOutsideTheImageIsZero

A triangle above and left of the image, one to its right and an empty polygon each give four
zeros.

### VisibleBoxUsesTheCrossingsOfTheImageLines

A triangle crossing the top and left edges gives (0, 0, 60, 50), and one crossing the right
and bottom edges gives (90, 70, 100, 80).

### VisibleBoxIncludesTheImageCornersThePolygonContains

A triangle that contains the image's origin gives (0, 0, 36.666664, 33), and a square around
the whole image gives the image, (0, 0, 100, 80).

## modules/vision/imaging/tests/test_imaging_random.cpp

Suites: `Unit_ImagingRandom`.

Tests the generator every augmentation draws from.

This file needs no device. `Unit_ImagingRandom` covers `Random` of
`modules/vision/imaging/include/SushiAI/vision/imaging/random.hpp`: a 64-bit counter advanced by `K_INCREMENT` and passed
through `mix`. The first cases follow the arithmetic by hand, with each intermediate value as a
literal, so the sequence is fixed by this file and not by a second copy of the code. The
statistical cases use one seed and state their bound as a multiple of the standard error.

### Constants

`K_SEED` is 2026, the seed of every statistical case. `K_SAMPLES` is 65536. `K_UNIFORM_VALUES`
is 2^24, the number of values `uniform()` can return.

### MixFollowsTheFinaliserStepByStep

For the value 0x9E3779B97F4A7C15 the case asserts each of the five steps of `mix`: the shift by
30 and xor, the product with `K_FIRST_MULTIPLIER`, the shift by 27 and xor, the product with
`K_SECOND_MULTIPLIER` and the shift by 31 and xor. `mix` returns the last literal.

### MixOfTheSecondCounterFollowsTheFinaliserStepByStep

The same five steps for 0x3C6EF372FE94F82A, the second counter of seed 0.

### SeedZeroReproducesTheHandComputedSequence

The first four `next_u64` values of seed 0 are the literals the case lists, and the counter
after draw n is n times `K_INCREMENT` modulo 2^64.

### TheCounterStartsAtTheSeed

`state()` of a new generator is its seed. The first draw of seed 42 is
`mix(42 + K_INCREMENT)`, and the counter is then `42 + K_INCREMENT`.

### NextU32IsTheHighHalfOfOneDraw

The first two `next_u32` values of seed 0 are the high 32 bits of its first two draws, and two
calls advance the counter twice.

### UniformIsTheTop24BitsScaledByTwoToTheMinus24

The first three `uniform()` values of seed 0 are 14819496, 7239838 and 443485 over 2^24, the
top 24 bits of each draw.

### UniformStaysInTheHalfOpenUnitInterval

65536 draws are all at or above 0 and below 1.

### UniformHasTheMeanAndVarianceOfAUniformVariable

Over 65536 draws the mean is 0.5 within 0.006 and the variance 1/12 within 0.0015, five
standard errors each.

### RangedUniformMapsTheUnitDrawLinearly

`uniform(-1, 1)` and `uniform(0.5, 1.5)` of seed 0 return `low + (high - low) * u` for the
first two unit draws, as exact floats, and make one draw each.

### RangedUniformStaysInItsHalfOpenInterval

65536 draws of `uniform(0.4, 0.6)` are all at or above 0.4 and below 0.6.

### RangedUniformNeverReturnsItsUpperBound

With `high` the float after `low`, 1024 draws all return `low`. A sum that rounds up to `high`
is brought back.

### RangedUniformOverAnEmptyRangeStillDraws

`uniform(0.25, 0.25)` returns 0.25 and advances the generator by one draw.

### IndexIsTheHighWordOfTheProduct

`index(10)` of seed 0 returns 8, 4 and 0: the high 32 bits of each 32-bit draw times 10.

### IndexOfOneIsAlwaysZeroAndStillDraws

64 calls of `index(1)` return 0 and advance the generator 64 times.

### IndexStaysBelowTheCountAndFillsEveryBucketEvenly

70000 draws of `index(7)` are all below 7, and each bucket holds 10000 within 500, above five
standard deviations of 93. `K_BUCKETS` and `K_DRAWS` are local to the case.

### IndexReachesTheLargestCount

`index(0xFFFFFFFF)` of seed 0 returns 0xE220A838.

### BernoulliComparesOneUniformDrawWithTheProbability

`bernoulli(0.5)` of seed 0 returns false, true, true for the unit draws 0.8833, 0.4315 and
0.0264, one draw per call.

### BernoulliNeverSucceedsAtZeroAndAlwaysSucceedsAtOne

Over 65536 draws, probability 0 never succeeds and probability 1 always does, and both make
one draw per call.

### AZeroDrawIsTheLowerBoundAndFailsAZeroProbability

`mix(0)` is 0, so a seed of minus `K_INCREMENT` makes the first draw 64 zero bits. `uniform()`
returns 0, `bernoulli(0)` returns false and `uniform(0.4, 0.6)` returns 0.4. A draw of exactly
0 does not pass a probability of 0.

### BernoulliSucceedsAtTheRateOfItsProbability

Over 65536 draws `bernoulli(0.25)` succeeds at a rate of 0.25 within 0.01.

### DeriveStartsAtTheMixOfTheStateAndTheKeyedMix

`Random(0).derive(0).state()` is 0x48218226FF3CD4BF. The case follows the five steps of `mix`
from the keyed value 0xE220A8397B1DCDAF.

### DeriveDependsOnTheKeyAndOnTheParentState

`Random(0).derive(1)` and `Random(7).derive(5)` have the two states the case lists. Two keys
of one parent give different generators, and so does one key of two parents.

### DeriveLeavesTheParentUntouched

After `derive(3)` the parent's state is its seed, the child differs from the parent, and a
second `derive(3)` equals the first.

### ACopyReplaysTheSequenceFromThePointOfTheCopy

A copy taken after one draw equals the original, and their next 16 draws are equal.

### EqualityComparesTheCounter

Two generators of one seed are equal, and are unequal after one of them draws.

## modules/vision/imaging/tests/test_imaging_raw_cache.cpp

Suites: `Unit_ImagingRawCache`.

Tests the raw image file: its bytes, its header checks and its file reader and writer.

This file needs no device. `Unit_ImagingRawCache` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/raw_cache.hpp`. A file is a 24-byte header of six little-endian
32-bit words (magic, version, height, width, channel count, channel order) followed by the
pixels. Most cases read and write string streams. The cases from ImageRoundTripsThroughAFile on
write files in a directory of their own under the system's temporary directory, which
`ScratchDirectory` creates empty and removes when the case ends.

### K_RGB

`K_RGB` is 1, the channel order word of an RGB file.

### append_u32

`append_u32` appends a value to a byte string as four little-endian bytes.

### header_bytes and layout_bytes

`header_bytes` returns the 24 header bytes for six given fields. `layout_bytes` returns them
for `K_RAW_IMAGE_MAGIC`, `K_RAW_IMAGE_VERSION` and a given height, width, channel count and
order.

### make_image

`make_image` returns an image whose byte at offset i is 7 * i + 3 modulo 256.

### bytes_of

`bytes_of` returns a copy of an image's bytes.

### written_bytes

`written_bytes` returns the bytes `write_raw_image` produces for an image.

### expect_untouched

`expect_untouched` expects an image to be 1 by 1 with one channel still, the destination a
failed read must not replace.

### ScratchDirectory

`ScratchDirectory` owns one directory under `std::filesystem::temp_directory_path()`. It
removes what the path held and creates the directory in its constructor, and removes it in its
destructor. `write` puts bytes in a file of the directory and `file` returns a path inside it.
The class cannot be copied or moved.

### WriterEmitsTheDocumentedBytes

A 1 by 2 RGB image is written as "SAIM", the words 1, 1, 2, 3, 1 and the six pixel bytes. The
case lists all thirty bytes.

### HeaderSizeAndMagicMatchTheDocumentedLayout

`K_RAW_IMAGE_HEADER_BYTES` is 24 and the first four header bytes are "SAIM".

### RGBImageRoundTripsThroughAStream

A 5 by 7 RGB image written and read back has the header fields 5, 7, 3 and
`ChannelOrder::RGB`, `pixel_bytes()` of 105 and the same bytes.

### GrayImageRoundTripsThroughAStream

A 4 by 3 one-channel image round-trips with `ChannelOrder::Gray`.

### BGRFileIsWrittenAsGivenAndReadAsRGB

Written with `ChannelOrder::BGR`, the pixel bytes 1 to 6 are stored as given. Read back they
are 3, 2, 1, 6, 5, 4: the reader swaps the first and third channel so the image is RGB.

### ReadReusesADestinationOfAnotherSize

A destination that was 9 by 9 with one channel becomes 2 by 2 with three channels and holds the
file's bytes.

### HeaderRejectsAWrongMagic

A header that starts with the PNG signature bytes throws with "magic" and "not a raw image
file".

### HeaderRejectsANewerVersionAndNamesBoth

Version 2 throws with "format version 2" and "reads version 1".

### HeaderRejectsVersionZero

Version 0 throws with "format version 0" and "reads version 1".

### HeaderRejectsATruncatedHeader

A stream of 23 bytes throws with "header ended" and "23 of 24".

### HeaderRejectsAnEmptyStream

An empty stream throws with "header ended" and "0 of 24".

### HeaderRejectsAZeroHeight

A height of 0 throws with "height 0" and "16384".

### HeaderRejectsAZeroWidth

A width of 0 throws with "width 0" and "16384".

### HeaderRejectsAHeightAboveTheLimit

A height of 16385 throws with "height 16385".

### HeaderRejectsAWidthAboveTheLimit

A width of 16385 throws with "width 16385".

### HeaderAcceptsBothSidesAtTheLimit

A 16384 by 16384 RGB header is accepted, and `pixel_bytes()` is 805306368.

### HeaderRejectsExtentsWhoseProductWouldOverflow

Extents of 0xFFFFFFFF throw with "height 4294967295". The extent is checked before any product
is formed.

### HeaderRejectsAnExtentWithTheSignBitSet

A width of 0x80000001 throws with "width 2147483649". The word is read as unsigned and is not
taken for a negative number.

### HeaderRejectsAnUnknownChannelOrder

A channel order of 3 throws with "channel order 3" and "unknown".

### HeaderRejectsThreeChannelsInAGrayFile

Three channels with order 0 throw with "channel count 3" and "channel order 0".

### HeaderRejectsOneChannelInAnRGBFile

One channel with order 1 throws with "channel count 1" and "channel order 1".

### HeaderRejectsFourChannels

Four channels throw with "channel count 4".

### HeaderRejectsZeroChannels

Zero channels with order 2 throw with "channel count 0" and "channel order 2".

### PixelsRejectAStreamThatEndsEarly

A stream one byte short of its pixels throws with "pixels ended" and "11 of 12".

### PixelsRejectAHeaderTheReaderDidNotValidate

`read_raw_image_pixels` given a hand-built header of height 0 throws with "height 0" and leaves
the destination as it was.

### WriterRejectsAnEmptyView

A default view throws with "no pixels" and nothing is written.

### WriterRejectsAChannelCountThatDoesNotFitTheOrder

One channel with `RGB` or `BGR` and three channels with `Gray` throw with the channel count,
and nothing is written.

### WriterRejectsASideAboveTheLimit

A view 16385 px wide throws with "width 16385" and nothing is written.

### WriterReportsAFailedStream

Writing to a stream whose bad bit is set throws with "stream failed".

### ImageRoundTripsThroughAFile

A 6 by 4 RGB image written with `write_raw_image_file` gives a file of 24 + 72 bytes, and
`read_raw_image_file` returns the same extents and bytes.

### FileWriterReplacesALongerFile

Writing a 1 by 1 image over a file that held an 8 by 8 one leaves a file of 24 + 3 bytes that
reads back as the 1 by 1 image.

### FileWriterLeavesTheFileAloneWhenItRejectsTheImage

A write that throws for its channel count leaves an existing four-byte file at four bytes. The
image is checked before the file is opened.

### FileWriterRejectsAPathItCannotOpen

A path inside a directory that does not exist throws with "cannot open".

### FileReaderRejectsAMissingFile

A missing file throws with "cannot open" and the file name, and the destination is untouched.

### FileReaderRejectsATruncatedHeader

A file of 10 bytes throws with "header ended" and "10 of 24", and the destination is untouched.

### FileReaderRejectsTruncatedPixelsBeforeResettingTheDestination

A file one byte short throws with "holds 35 bytes" and "declares 36", and the destination is
untouched.

### FileReaderRejectsTrailingBytesBeforeResettingTheDestination

A file with one byte after its pixels throws with "holds 37 bytes" and "declares 36", and the
destination is untouched.

### FileReaderAllocatesNothingForALargeImageTheFileDoesNotHold

A file that is only a header declaring 16384 by 16384 pixels throws with "holds 24 bytes" and
"declares 805306392". The destination is untouched, so the reader compares the file size with
the header before it allocates.

### FileReaderRejectsAMalformedHeaderBeforeResettingTheDestination

A file whose header has extents of 0xFFFFFFFF throws with "height 4294967295", and the
destination is untouched.

## modules/vision/imaging/tests/test_imaging_rectangle.cpp

Suites: `Unit_ImagingRectangle`.

Tests the plan that orders a dataset by aspect ratio and gives each batch one shape.

This file needs no device. `Unit_ImagingRectangle` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/rectangle.hpp`. The expected shapes are what lines 405 to 427 of the
reference's `data/base.py` give for the same sizes.

### make_sizes

`make_sizes` returns seven image sizes with height over width below, at and above 1. Images 0
and 5 share the ratio 0.75.

### make_settings

`make_settings` returns `RectangleSettings` with a batch size, an image size and a pad, at the
default stride of 32.

### SettingsDefaultToTheReferenceValidationRun

Default settings are image size 640, batch size 32, stride 32 and pad 0.5.

### OrderRisesWithTheAspectRatioAndKeepsTies

`plan_rectangular_batches` orders the seven images as 6, 3, 0, 5, 2, 1, 4. Images 0 and 5 keep
their dataset order.

### ShapesMatchTheReferenceArithmetic

With batch 3 and pad 0.5 the shapes are 512 by 672, 672 by 672 and 672 by 448. With batch 2 and
pad 0 they are 448 by 640, 480 by 640, 640 by 640 and 640 by 448. With one batch of 7 at image
size 320 the shape is 352 by 352.

### ShapeAtNamesTheBatchOfAPosition

`shape_at` returns the first batch's shape for positions 0 and 2, the second's for position 3
and the third's for position 6.

### EverySideIsAMultipleOfTheStride

With stride 64 and image size 600, both sides of all four shapes are multiples of 64 and at
least 64.

### OneWideImageGivesOneWideShape

One 480 by 640 image gives the order {0} and the shape 512 by 672:
ceil(0.75 * 640 / 32 + 0.5) * 32 and ceil(20.5) * 32.

### RejectsSettingsAndSizesOutsideTheirRanges

`validate_rectangle` throws for a batch size, a stride or an image size of 0 and for a pad that
is negative or NaN. `plan_rectangular_batches` throws for an empty size list ("no image") and
for a size of 0 by 500 ("image 2", "0x500"). Default settings pass.

### RejectsAShapeAboveTheImageBound

An image size of `K_MAX_IMAGE_SIDE` with pad 0.5 gives a side above the bound, and the call
throws with "batch side" and "16384".

## modules/vision/imaging/tests/test_imaging_resize.cpp

Suites: `Unit_ImagingResize`.

Tests the bilinear resize and the fit of an image's long side.

This file needs no device. `Unit_ImagingResize` covers `modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp`:
`resize_bilinear`, `fit_long_side` and `resize_long_side`. The kernel samples at pixel centres
with weights rounded to `K_BILINEAR_WEIGHT_BITS` bits and rounds its result half up. The first
cases compare whole images with values a comment derives.

### Bytes

`Bytes` is a vector of bytes, an image's contents in height, width, channel order.

### make_image and bytes_of

`make_image` returns an image holding the given bytes, and `bytes_of` returns an image's bytes.

### resized_bytes

`resized_bytes` resizes a one-channel source to a size and returns the bytes. It also expects
the destination to have that size and one channel.

### WeightBitsGiveAWeightSumOf2048

`1 << K_BILINEAR_WEIGHT_BITS` is 2048.

### TwoByTwoToFourByFourMatchesTheHandComputedImage

Along one axis the four outputs of a pair (a, b) are a, (3a + b) / 4, (a + 3b) / 4 and b. The
case lists all sixteen output bytes, among them 58.75 rounded to 59 and 76.25 rounded to 76.

### FourByFourToTwoByTwoAveragesEachTwoByTwoBlock

Each output reads position 0.5 or 2.5 on both axes, so it is the mean of a 2 by 2 block. The
sums 142, 220, 460 and 541 give 36, 55, 115 and 135: 35.5 rounds up and 135.25 rounds down.

### ResizeToTheSameSizeCopiesEveryPixel

A 3 by 3 image resized to 3 by 3 is unchanged.

### AnEnlargedRowClampsAtBothEdges

A row (10, 250) enlarged to four columns starts at 10 and ends at 250. Its outer columns read
positions outside the two pixel centres.

### AReducedRowUsesWeightsRoundedToElevenBits

Eight columns reduced to three read positions 0.8333, 3.5 and 6.1667, with upper weights 1707,
1024 and 341 of 2048. The case computes the three bytes from those weights, checks they are 3,
41 and 209, and compares the resize with them.

### EachChannelIsInterpolatedOnItsOwn

A two-pixel row of three channels enlarged to four pixels gives each channel the values of its
own pair. Channel 2 falls from 255 to 3 while the others rise.

### AConstantImageStaysConstantWhenEnlarged

A 5 by 7 image of the colour (17, 114, 255) enlarged to 13 by 23 holds that colour in every
pixel.

### AConstantImageStaysConstantWhenReduced

A 31 by 17 image at level 201 reduced to 4 by 5 holds 201 in every byte.

### AHorizontalGradientStaysMonotoneAtEveryWidth

A rising row of eight levels resized to every width from 1 to 40 never falls from one column to
the next.

### AnEnlargedGradientKeepsItsEndValues

The same row enlarged to 20 columns starts at 0 and ends at 255.

### AVerticalGradientStaysMonotoneAndRowsStayUniform

A 4 by 3 image whose rows are each one level is resized to every height from 1 to 17 at width
5. Every output row is one level, and the levels never fall from one row to the next.

### TheDestinationIsResetToTheRequestedSizeAndChannels

A destination that was 9 by 9 with one channel becomes 2 by 2 with the source's three channels.

### ResizeRejectsASourceWithoutPixels

`resize_bilinear` on a default view throws with "resize_bilinear", "source" and "no pixels".

### ResizeRejectsASizeOutsideTheImageBounds

Destination sizes of 0 by 2, 2 by -1 and 2 by 16385 throw with "destination size" and the size.

### ResizeRejectsASourceWithExtentsOutsideTheImageBounds

A source view of 0 by 2, of 2 by 16385 and of five channels throws with "source size" or
"source channel count".

### ResizeRejectsADestinationThatHoldsTheSource

`resize_bilinear` with the image as source and destination throws with "destination holds the
source's pixels". The image is still 2 by 2 with its bytes, so the refusal comes before the
reset.

### FitLongSideLeavesAnImageWhoseLongSideAlreadyFits

`fit_long_side` at 640 returns 480 by 640, 640 by 427 and 640 by 640 unchanged.

### FitLongSideReducesALargeImage

1080 by 1920 becomes 360 by 640, and 1920 by 1080 becomes 640 by 360.

### FitLongSideEnlargesASmallImage

100 by 50 becomes 640 by 320, and 500 by 375 becomes 640 by 480.

### FitLongSideRoundsTheShortSideUp

333 by 500 becomes 427 by 640, 7 by 3 becomes 640 by 275 and 3 by 1000 becomes 2 by 640: the
short side is the ceiling of 426.24, 274.29 and 1.92.

### FitLongSideCapsBothSidesAtTheImageSize

1281 by 1280 becomes 640 by 640. The short side would be the ceiling of 639.5.

### FitLongSideAlwaysPutsTheLongSideAtTheImageSize

Over heights and widths from 1 to 1499 in steps of 7 and 11, the long side of the result is 640
and the short side lies in [1, 640].

### ResizeLongSideCopiesAnImageThatAlreadyFits

A 2 by 4 image with image size 4 comes out as a 2 by 4 image with the same bytes under another
data pointer. The destination, which was 3 by 3 with three channels, takes the source's one
channel.

### ResizeLongSideEnlargesToTheFittedSize

A 1 by 2 image with image size 4 becomes 2 by 4, and both rows are 0, 25, 75, 100.

### ResizeLongSideReducesToTheFittedSize

The 4 by 4 image of the block-average case with image size 2 becomes the 2 by 2 image 36, 55,
115, 135.

### ResizeLongSideRejectsAnImageSizeOutsideTheImageBounds

Image sizes of 0 and 16385 throw with "resize_long_side" and the value.

### ResizeLongSideRejectsASourceWithoutPixels

A default view throws with "resize_long_side" and "no pixels".

### ResizeLongSideRejectsADestinationThatHoldsTheSource

The image as source and destination throws with "destination holds the source's pixels".

### ResizesRejectAViewInsideTheDestination

`resize_bilinear` and `resize_long_side` both refuse a view of two rows of the destination, and
the image keeps its size and its bytes.

### ResizeLongSideRejectsASourceWithExtentsOutsideTheImageBounds

A source view of 0 by 2 and one of five channels throw with "source size" and "source channel
count".

### NearestRepeatsEachPixelWhenEnlarged

`resize_nearest` of the 2 by 2 image 1, 2, 3, 4 to 4 by 4 gives one channel with the rows
1 1 2 2, 1 1 2 2, 3 3 4 4, 3 3 4 4.

### NearestReadsThePixelUnderEachCentreWhenReduced

The row 10, 11, 12, 13, 14 resized to two columns is 11, 13.

### NearestKeepsClassIndicesUnblended

The row 0, 255 resized to seven columns is 0, 0, 0, 255, 255, 255, 255.

### NearestCopiesEveryChannelOfAPixel

An RGB row of the pixels (1, 2, 3) and (4, 5, 6), resized to 2 by 2 into a destination that
was 9 by 9 with one channel, gives a 2 by 2 image of three channels holding that row twice.

### NearestRejectsABadSourceABadSizeAndAnAliasedDestination

`resize_nearest` raises an error for an empty view ("resize_nearest", "source", "no pixels"),
for a size of 0 by 2 ("Image::reset", "0x2") and for a destination that is the source image
("resize_nearest", "destination holds the source's pixels"). The source still holds its four
bytes of 9.

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

## modules/layers/train/tests/test_input_match.cpp

Suites: `Unit_InputMatch`.

Tests `Train::describe_input_leaf` and `Train::match_input`. The dataset is a local `Published`
class named "published" that publishes the input it is built with, no role and no batch.

### ALeafIsDescribedWithoutItsBatchAxis

A [4, 16] int32 leaf is described as [16] int32, a [2, 8, 8, 3] float32 leaf as [8, 8, 3]
float32 and a [4] float64 leaf as an empty shape and float64.

### ALeafWithNoBatchAxisIsRefused

A leaf of rank 0 fails with `InvalidDescription` and "no batch extent".

### ADatasetThatPublishesTheInputIsAccepted

`match_input` is ok for [16] int32 against a dataset that publishes [16] int32, and for [16]
float32 against one that publishes [16] float32.

### AnotherElementTypeIsRefusedWithBothTypes

A float32 input against the int32 dataset fails with `Incompatible` and a message containing
`'published'`, "as int32 [16] per sample" and "compiled for float32 [16] per sample". An int32
input against the float32 dataset fails with the two types the other way round.

### AnotherShapeIsRefusedWithBothShapes

A [32] int32 input against the [16] int32 dataset fails with `Incompatible`, "as int32 [16] per
sample" and "compiled for int32 [32] per sample".

## modules/core/graph/tests/test_ir.cpp

Suites: `Unit_Ir`, `Unit_IR`.

Tests the IR's single-assignment and ordering rules and the builder's shape inference.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### AddValueRecordsShapeDTypeAndLifetime

`add_value` of a [2, 3] `Float64` parameter named "w" gives a graph of one value with that
shape, element type, lifetime and debug name and no producer.

### AddNodeClaimsItsOutputs

After `add_node` of an ADD over two inputs, the output value's producer is that node and the
node has two inputs.

### ASecondProducerForOneValueThrows

Single assignment is the property that lets differentiation skip alias analysis entirely, so a
double write is rejected and not tolerated.

### ReferencingAnUnknownValueThrows

`add_node` throws `Error` when it reads the value 99 and when it writes the value 99.

### PropagateRequiresGradFlowsForwardThroughTheWholeChain

Before `finalise`, the MATMUL of an input and a parameter and the ReLU after it do not
require a gradient. After it the input still does not, and the parameter, the product and the
ReLU do.

### PropagateRequiresGradLeavesAFrozenBranchAlone

After `finalise`, a ReLU of the input does not require a gradient and the MATMUL of the input
and a parameter does.

### BuilderInfersMatmulShapesForEveryTranspositionCase

With [4, 3] or [3, 4] on the left and [3, 5] or [5, 3] on the right, `matmul` gives [4, 5]
for each of the four combinations of the two transpose flags.

### BuilderRejectsAContractionMismatch

`matmul` of [4, 3] and [7, 5] throws `Error`.

### BuilderChecksBiasAndLossShapes

Over a [4, 5] value, `bias_add` with a [5] bias gives [4, 5] and with a [4] bias throws
`Error`. `sum_rows` gives [5]. `nll_loss` with [4, 5] targets gives [1] and with [4, 6]
targets throws `Error`.

### BuilderRejectsMixedElementTypes

`add` and `matmul` of a `Float32` and a `Float64` operand both throw `Error`.

### NodesAreEmittedInTopologicalOrder

The lowering relies on insertion order as a topological order, and differentiation relies on
its reverse, so the test asserts it.

### InsertNodeRejectsAResultThatIsAlreadyReadEarlier

The graph exercises the half of `insert_node`'s ordering guard that single assignment does not
already cover. A result with no producer passes the single-assignment check whatever its
position. Without this half the function would accept a producer placed after its own reader,
which is the arrangement it exists to prevent, and `MemoryPlan::build()` would then reject a
graph `insert_node` had approved.

`node_count()` is still 1 after the throw. The insert was rejected, not half-applied: nothing
was inserted and no producer was claimed.

The final `EXPECT_NO_THROW` inserts the same node ahead of that reader and succeeds. That makes
the rejection a statement about position and not about the node.

### InsertNodeRejectsAnOperandProducedNoEarlierThanItself

With one RELU node writing value 1, `insert_node` at 0 of a TANH that reads value 1 raises an
error holding "sushiai.tanh", "inserted at 0", "value 1", "node 0 writes" and "topological".
The graph still has one node and the TANH's output has no producer.

### AttributesHoldAConvolutionsGeometryAndItsWindowGrid

`Attributes::K_CAPACITY` is ten, not eight. The eight of CV-5 are the geometry: kernel_h/w,
stride_h/w, dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents out_h and
out_w, and they have to be carried and cannot be re-derived. Deriving them from the geometry
assumes symmetric padding, and an asymmetric window grid can derive to its own transpose with
the same window count, which every downstream shape check accepts.

Nothing downstream caps the ten. Nothing copies an attribute into a fixed-width metadata array:
the lowering reads each input through `LoweringContext::input` and passes the tensor straight to
SushiBLAS.

### ABufferLeafIsPersistentAndNeedsNoGradient

The nine cases from here are in the suite `Unit_Ir`; the cases above are in `Unit_IR`.

`GraphBuilder::buffer` declares a value of the shape, type and name it was given, with the
lifetime `Lifetime::Buffer` and `requires_grad` false. Its producer is a K_LEAF node. A K_BIAS_ADD
that reads an input and a buffer does not require a gradient either. `is_persistent(BUFFER)` is
true and `lifetime_name(BUFFER)` is "buffer".

### ActivationAndScratchAreTheOnlyLifetimesThatDoNotPersist

`Graph::is_persistent` is true for PARAMETER, GRADIENT, OPTIMIZER_STATE, INPUT and BUFFER, and
false for ACTIVATION and SCRATCH.

### StateLeafReturnsTheBoundLeafAndNoValueForAnUnboundState

`Graph::state_leaf` returns `K_NO_VALUE` before any binding. After `bind_state_leaf` of state 7
to a buffer and state 5 to a parameter, it returns each leaf for its state and `K_NO_VALUE` for
state 6 and for `K_NO_STATE`. `state_leaves()` holds the two pairs in the order they were
bound, not in the order of the state numbers.

### BindStateLeafRefusesASecondBindingOfAStateOrOfALeaf

With state 7 bound to a leaf, four bindings throw: state 7 to another leaf ("State 7",
"already has leaf"), state 8 to the bound leaf ("already stands for state 7"), `K_NO_STATE` to
a leaf ("no identity"), and state 9 to value 99 ("value 99", "does not exist"). The table
still holds the first pair alone.

### WithoutNodesKeepsEveryValueAndStateLeafAndHoldsNoNode

`Graph::without_nodes()` returns a graph with no node and the same number of values. Every
value keeps its shape, element type, layout, lifetime, `requires_grad` and debug name, and its
producer is `K_NO_NODE`. The fixture holds an NCHW value, a parameter that requires a gradient
and a HALF buffer, and each is checked by name. The copy's `state_leaves()` equals the
source's. The source graph still has its nodes and producers.

### ARefusalIsReportedAfterItIsMarkedAndSurvivesWithoutNodes

`has_refusal()` is false for a new graph and for its `without_nodes()` copy. After
`mark_refused()` it is true for both.

### StateIdentityDrawsOneIdentityPerHolder

`Graph::StateIdentity` in `modules/core/graph/include/SushiAI/graph/state_id.hpp` holds the identity of one piece
of model state. `peek()` returns `K_NO_STATE` before the first `get()`. `get()` returns the
same identity on every call, `peek()` then returns it too, and a second holder's `get()`
returns a different one.

### ACopiedStateIdentityDoesNotShareTheSourcesIdentity

A holder copy-constructed from one that has drawn peeks `K_NO_STATE` and then draws a
different identity. A holder that had drawn and is then copy-assigned from the source also
peeks `K_NO_STATE`, and its next identity differs from the source's and from the one it had.
The source keeps its identity.

### AMovedStateIdentityCarriesItsIdentityAway

After a move construction the target peeks the source's identity and the source peeks
`K_NO_STATE`. A move assignment over a holder that had drawn does the same.

## modules/core/config/tests/test_json_events.cpp

Suites: `Unit_JsonEvents`.

Tests `Config::read_json_events`. Each case writes a file under the system's temporary
directory and removes it at its end. A local `EventLog` implements `Config::JSONEvents`, keeps
every event as one line of text and can return false at a chosen event.

### EventsArriveInTheOrderOfTheText

A document with an object, an array, two whole numbers, a real, a string, both booleans, a null
and a nested object gives 16 events in the order of the text. Keys arrive as `key` events
before their values.

### AWholeNumberAboveInt64IsAReal

In `[9223372036854775807, 18446744073709551615]` the first entry arrives as
"whole 9223372036854775807" and the second as a real. The read gives four events.

### ADocumentLongerThanTheBufferArrivesWhole

An array of 40000 whole numbers, whose text is longer than twice
`K_JSON_EVENT_BUFFER_BYTES`, gives 40002 events. The last number is "whole 39999" and the last
event is the end of the array.

### AHandlerThatStopsEndsTheReadWithSuccess

A log that returns false at its fourth event receives four events from `{"a": [1, 2, 3]}`, and
the read is ok.

### TextThatIsNoJSONIsRefusedWithItsOffset

Three files fail with `InvalidFile` and "not valid JSON": a document cut short, which also
names the file and "at byte 12", a document with trailing text ("at byte 5") and an empty file
("at byte").

### AnAbsentFileIsRefusedByItsPath

A path with no file fails with `FileNotFound` and a message containing the file name and "could
not open".

## modules/core/config/tests/test_json_writer.cpp

Suites: `Unit_JsonWriter`.

Tests `Config::write_json`. A depth of 0 writes the whole document on one line. The cases that
read text back parse it with `Config::parse_json`, and a real is compared bit for bit.

### WritesEveryKindOnOneLineInTheOrderOfTheKeys

At depth 0 an object of nine keys is written on one line that ends with a line feed. The keys
keep the order they were stated in, `zebra` before `apple`, an empty object is `{}` and an
empty array is `[]`.

### ExpandsTheFirstTwoLevelsAndKeepsDeeperOnesOnOneLine

At the default depth the root object and the values one level below it are expanded with four
spaces of indent per level. An array of numbers at the second level, each object inside
`layers` and an empty array stay on one line. The whole text is compared.

### EndsEveryLineWithALineFeedAndTheTextWithOne

The text ends with one line feed, the character before it is not a line feed, and the text
holds no carriage return.

### WritesARootScalarAlone

`Value::integer(-3)` is written as "-3" and a line feed.

### SpellsNumbersInTheShortestFormThatReadsBack

`[7, -3, 0, 0.001, 1e-3, 2.5, 1.0, 1e300]` is written as
`[7, -3, 0, 0.001, 0.001, 2.5, 1.0, 1e+300]`.

### AnIntegerReadsBackAsTheSameInteger

Eight integers, among them 9007199254740993 and the largest and the smallest int64, each read
back as `Kind::Int` with the same value.

### ARealReadsBackAsTheSameBits

Nineteen reals, among them -0.0, 0.1 + 0.2, 1e22, the largest and the lowest double, the
smallest normal, the smallest denormal and epsilon, each read back as `Kind::Real` with the
same bits.

### EscapesQuotesBackslashesAndControlCharacters

A string with a quote, a backslash, a solidus, the five short control escapes, the bytes 0x01,
0x1F, 0x7F and a NUL is written with `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, `\u0001`,
`\u001f` and `\u0000`. The solidus and 0x7F are written as they are. The text reads back to the
same string.

### WritesUTF8AsItsOwnBytes

A string holding U+011F, U+65E5 and U+1F363 is written between quotes as its own bytes and
reads back to the same string.

### EscapesAKeyAsItEscapesAString

An object whose key holds quotes and a tab is written with `\"` and `\t` in the key, and its
keys read back equal.

### RefusesANumberThatIsNotFinite

A NaN fails with `InvalidDescription` and "not a number". Plus and minus infinity each fail
with "infinite".

### RefusesAStringThatIsNotUTF8

Six byte sequences fail with `InvalidDescription` and "UTF-8" as a string, and with "UTF-8" and
"key 0" as the key of an object: 0xFF, a lone continuation byte, an overlong solidus, a
three-byte form cut short, the surrogate U+D800 and U+110000.

### RefusesAnObjectThatStatesAKeyTwice

An object with the keys `rate`, `units`, `rate` fails with `InvalidDescription`, `'rate'` and
"twice".

### ARefusalNamesTheWayToTheValue

A NaN under `rate` in entry 1 of the array under `layers` fails with a message containing
"key 'layers'", "entry 1", "key 'rate'" and "not a number".

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

## modules/core/autograd/tests/test_layer_norm_backward.cpp

Suites: `Unit_LayerNormBackward`.

Tests the nodes `Autograd::differentiate` appends for LAYER_NORM. The graph normalises a
[5, 3] value with a [3] gain and a [3] bias at the epsilon 1e-5 and takes the `nll_loss` of
the `log_softmax` of the result.

### TheForwardHasARuleAndItsGradientHasNone

`Ops::K_LAYER_NORM` is differentiable and has a backward rule. `Ops::K_LAYER_NORM_BACKWARD` is
not differentiable and has none.

### OneGradientNodeServesTheThreeOperands

With x as a parameter, differentiating at x, the gain and the bias adds one
LAYER_NORM_BACKWARD node of five inputs. Inputs 1 to 4 are x, the gain, the saved mean and the
saved inverse deviation, and its outputs are the three returned gradients, of the shapes
[5, 3], [3] and [3].

### AnInputThatNeedsNoGradientReceivesNone

With x as an input, the gradient of x is `K_NO_VALUE`, the gain has a gradient and the graph
holds one LAYER_NORM_BACKWARD node.

## modules/core/graph/tests/test_layer_norm_ops.cpp

Suites: `Unit_LayerNormOps`.

Tests `GraphBuilder::layer_norm` and `GraphBuilder::layer_norm_backward`. The leaves are a
[2, 3, 4] input, a [4] gain and a [4] bias, and the epsilon is 1e-5. A refusal is
`InvalidDescription`, leaves the node count unchanged and sets `has_refusal()`.

### LayerNormWritesItsValueAndTheTwoRowStatistics

Over float64 leaves the call is ok and emits one LAYER_NORM node that reads x, the gain and the
bias and writes the output, the saved mean and the saved inverse deviation, with the epsilon in
attribute slot 0. The output has the shape [2, 3, 4] and both saves the shape [6]. The output
and the saved mean are float64, the output has the lifetime `Activation` and the graph has no
refusal.

### ARankOneInputIsOneRow

Over a [4] input the output has the shape [4] and the saved mean the shape [1].

### LayerNormRefusesOperandsThatDoNotFit

Four calls are refused: an int32 input ("half, float32 or float64 input"), a [3] gain ("gamma
as [4]"), a float64 bias under a float32 input ("beta as [4] of float32") and an epsilon of 0
("epsilon must be positive").

### LayerNormBackwardWritesTheThreeGradients

`layer_norm_backward` is ok and emits one LAYER_NORM_BACKWARD node that reads dy, x, the gain,
the saved mean and the saved inverse deviation and writes dx, dgamma and dbeta, of the shapes
[2, 3, 4], [4] and [4].

### LayerNormBackwardRefusesOperandsThatDoNotFit

A [6, 4] gradient is refused with "operand of [2, 3, 4]" and a [4] saved mean with "operand of
[6]".

## test_layer_registry.cpp

Suites: `Unit_LayerRegistry`.

Tests the layer registry: each row's factory, the keys it reads and the model it builds.

This file needs no device. `NN::K_LAYER_REGISTRY` in `modules/layers/nn/include/SushiAI/nn/layer_registry.hpp` is
a constant table of fourteen rows, and a factory turns a per-sample input shape and a parsed
JSON object into an `NN::Layer`. A case therefore needs a `Shape`, a string of JSON and, for
the model cases, a host-side graph.

The cases fall into three groups, in file order:

- every row of the table: that it has a factory, builds its own layer type and refuses an
  unknown key
- the keys of each factory, their defaults and the values they refuse
- the convolutional classifier of `K_CONV_MODEL`, built, traced, named and differentiated

### K_CONV_MODEL

`K_CONV_MODEL` is a model file as a string: 784 input features, then `unflatten` to
28 x 28 x 1, `conv2d` with 8 filters, `relu`, `maxpool2d`, `conv2d` with 16 filters, `relu`,
`maxpool2d`, `flatten` and `linear` with 10 units. Both convolutions have kernel 3 and
padding 1, and both pools have kernel 2. The layers of
`applications/examples/mnist_cnn/model.json` are the same nine, with the optional keys spelled out.

### K_CONV_MODEL_PARAMETERS

`K_CONV_MODEL_PARAMETERS` lists the six parameters that model declares, as "name:extents":
a weight and a bias under the indices 1, 4 and 8. The two tests that compare names read the
expected list from here.

### MinimalCase

`MinimalCase` holds one registry key, a layer object without its "type", the per-sample shape
that enters the layer and the per-sample shape the layer must report.

### K_MINIMAL_CASES

`K_MINIMAL_CASES` holds one `MinimalCase` per registry row, in registry order. Each object
carries the required keys of its row and no other.

### attributes_from

`attributes_from` parses a string as the attributes of one layer.

### make_layer

`make_layer` looks a type up in `NN::K_LAYER_REGISTRY` with `Config::find_entry` and calls the
row's factory with the context "layer 0 '<type>'". The cases that assert a message expect that
context in it.

### wrapped

`wrapped<L>` returns the layer of type `L` inside an `NN::LayerAdapter<L>`, or null when the
`Layer` wraps another type. The `dynamic_cast` is how a case tells which layer type a factory
built and reads the fields the factory set.

### square_window

`square_window` returns a geometry with a square kernel and one stride and one padding for
both axes.

### make_conv

`make_conv` returns a `Conv2d` with the given channel counts and geometry, its parameters
undeclared.

### expect_pool_window_spellings_agree

`expect_pool_window_spellings_agree<Pool>` builds a pool from `"kernel": 3` and from
`"kernel": [3, 3]`. It expects both to wrap a `Pool`, the first to hold a 3x3 kernel, and the
two to hold the same kernel and the same stride.

### make_conv_stack

`make_conv_stack` parses `K_CONV_MODEL` and returns the `LayerStack` that `build_layer_stack`
makes of it.

### traced_names

`traced_names` traces a model over a [4, 784] input and returns its parameters as
"name:extents" strings. It is a template, so it takes a `LayerStack` and a `Sequential`.

### EveryRowHasAMinimalCase

`K_MINIMAL_CASES` has as many entries as `NN::K_LAYER_REGISTRY`, and entry `i` names the type of
row `i`. A row added to the registry without a case here fails this test, so the two tests
that loop over the cases cover every factory.

### TheTypeStringsAreTheOnesModelFilesSpell

The `type` fields of `NN::K_LAYER_REGISTRY` are, in order, "linear", "gelu", "relu", "tanh",
"sigmoid", "leaky_relu", "unflatten", "conv2d", "maxpool2d", "avgpool2d", "global_avgpool",
"flatten", "upsample" and "silu". A model file names a layer by one of these strings, so a
renamed row fails here.

### EachFactoryBuildsFromAMinimalObject

For every case the factory returns a layer whose `output_shape()` is the shape the case lists.

### EachKeyBuildsItsOwnLayerType

Each of the fourteen keys builds an adapter over the layer type it names: `wrapped<L>` is not
null for the expected `L`. Two rows that pointed at one factory would fail for one of them.

### EachFactoryRejectsAnUnknownKey

For every case, an object holding only `"no_such_key"` throws with a message containing the
layer's context and "'no_such_key'".

### Conv2dTakesItsInputChannelsFromTheInputShape

A `conv2d` built over a [8, 8, 7] input has `in_channels` 7 and `out_channels` 5. An object
that sets `"in_channels"` is refused by name, so a model file cannot state a channel count
that disagrees with the layer before it.

### Conv2dRefusesAnInputThatIsNotAnImage

A `conv2d` given the per-sample shape [784] throws. The message contains the context, "[784]",
"[H, W, C]" and "'unflatten'", the layer that gives a feature vector its image shape.

### Conv2dDefaultsItsOptionalKeys

With only "filters" and "kernel" set, the layer has stride 1, padding 0 and dilation 1 on both
axes, and `use_bias` is true.

### Conv2dReadsEveryKeyItAccepts

The object sets all six keys, four of them as [h, w] pairs with unequal entries. The layer
holds kernel 3 by 5, stride 2 by 1, padding 1 by 2, dilation 1 by 2 and `use_bias` false, and
over a [16, 16, 2] input it reports [8, 12, 5].

### AKernelAsAnIntegerAndAsAPairAgree

For `conv2d`, `maxpool2d` and `avgpool2d`, `"kernel": 3` and `"kernel": [3, 3]` give the same
output shape. The case then reads the stored window: the two convolutions both hold a 3x3
kernel, and `expect_pool_window_spellings_agree` checks the two pools. Equal output shapes
alone would also hold for two layers that both stored a wrong window.

### AWindowSettingOfTheWrongFormIsRefused

A one-entry array, a three-entry array, a string and the real number 1.5 are each refused.
Every message names the layer and the key, as in "layer 0 'maxpool2d' 'kernel'", and contains
"two-entry [h, w]". The two array messages also give the entry count, "1 entries" and
"3 entries".

### APoolStrideDefaultsToItsKernel

A `maxpool2d` with `"kernel": [2, 3]` and no stride holds stride 2 by 3 and padding 0, and
maps [12, 12, 2] to [6, 4, 2]. An `avgpool2d` with kernel 3, stride 1 and padding 1 holds
those values and keeps [12, 12, 2].

### Conv2dRejectsANonPositiveExtent

Eight objects are refused: a filter count of 0 and of -4, a kernel of 0, of [3, 0] and of -3,
a stride of 0 and of [1, -1], and a dilation of 0. Each message contains the context, "Conv2d"
and "positive".

### ThePoolsRejectANonPositiveExtent

For both pools, a kernel of 0, of [2, 0] and of -2 and a stride of 0 and of [-1, 2] are
refused with a message containing the context and "positive".

### AWindowedLayerRejectsANegativePadding

For `conv2d` and both pools, a padding of -1, of [0, -1] and of [-1, 0] is refused with a
message containing the context and "padding". The two pairs put the negative entry on each
axis in turn.

### AFactoryRejectsAMissingRequiredKey

A `conv2d` without "filters", a `conv2d` without "kernel", each pool without "kernel" and a
`linear` without "units" are refused. Each message contains the context and "has no" followed
by the quoted key.

### LeakyReLURejectsASlopeOutsideItsRange

Slopes of 0, 1 and -0.1 are refused when the layer is built, with a message containing the
context and "slope".

### UpsampleRejectsAScaleBelowOne

A scale of 0 is refused when the layer is built, with a message containing the context,
"Upsample2d" and "scale".

### LeakyReLUAndUpsampleReadTheirOptionalKey

A `leaky_relu` built from an empty object has slope 0.01, and one built with `"slope": 0.25`
has 0.25. An `upsample` built from an empty object maps [4, 4, 2] to [8, 8, 2], and one built
with `"scale": 3` maps it to [12, 12, 2].

### UnflattenRefusesAnInputOfAnotherSize

A 28 x 28 x 1 `unflatten` given [783] throws with a message containing the context, "[783]"
and "[784]". An object without "channels" throws with a message containing the context and
"channels".

### TheConvModelBuildsAndReportsItsOutputShape

The stack has nine layers and its output shape is [10]. The shapes after layers 1, 3, 6 and 7
are [28, 28, 8], [14, 14, 8], [7, 7, 16] and [784], so each factory received the shape its
predecessor reported.

### TheConvModelTracesToTheShapeItReports

The stack traces a [4, 784] input to a [4, 10] value.

### TheConvModelNamesItsParametersByLayerPosition

The traced stack's parameters are `K_CONV_MODEL_PARAMETERS`: indices 1, 4 and 8, with gaps at
the layers that own no parameter.

### ASequentialOfTheSameLayersYieldsTheSameNames

An `NN::Sequential` of the same nine layer types is traced over the same input. Its names
equal `K_CONV_MODEL_PARAMETERS`, and the stack's names equal the `Sequential`'s. A checkpoint
written from one model therefore names its parameters as the other reads them.

### CrossEntropyDifferentiatesToEveryParameterOfTheConvModel

The stack's output goes into `NN::CrossEntropyLoss`, and the loss is differentiated against
the six parameters the walk yields. Every gradient is a value other than `K_NO_VALUE` with the
shape of its parameter. The case checks that the graph differentiates; it computes no number.

### ASiLURowTracesOneSiLUNode

The layer of a `silu` row over [8, 8, 2], traced over [4, 8, 8, 2], adds one node. Its
operation is `Ops::K_SILU` and the result keeps the shape [4, 8, 8, 2].

### AVersionTwoChainStartsFromTheInputShape

A version 2 document over the input shape [8, 8, 2] holds a `conv2d` of 5 filters and kernel
3, a `silu` and a `flatten`. `NN::build_layer_stack` builds three layers. Layer 0 reports
[6, 6, 5], which requires the factory to have received the rank-3 shape and not the count 128,
and the stack reports [180].

### AChainRefusesARowThatReadsTwoRows

Row 2 with `"from": [-1, 0]` makes `build_layer_stack` throw with a message containing "layer
2 'relu'", "the row before it" and "build_model".

### AChainRefusesARowThatSkipsBack

Row 2 with `"from": 0` throws with "layer 2 'relu'" and "the row before it".

### AChainRefusesALaterRowThatReadsTheInput

Row 1 with `"from": "input"` throws with "layer 1 'tanh'" and "the row before it".

### AChainRefusesOutputsThatPublishAnEarlierRow

A two-row document whose `outputs` is `[{"node": 0}]` throws with a message containing
"'outputs'", "last row" and "build_model".

### AChainAcceptsOutputsThatPublishTheLastRow

The same document with `[{"node": 1}]` builds a stack of two layers.

### AChainRefusesARowThatReadsANamedPort

Row 1 with `"from": {"node": 0, "port": "output"}` throws with "layer 1 'tanh'" and "the row
before it". The row it names is the row above; the port is what the chain refuses.

## modules/layers/nn/tests/test_layer_stack.cpp

Suites: `Unit_LayerStack`.

Tests the runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread.

This file needs no device. A LayerStack is host-lane, and the first property this file pins,
that it names its parameters exactly as Sequential does, is a property of the walk and not of
any memory. The second is the per-sample `Shape` that `NN::Layer::output_shape()` reports
and `NN::build_layer_stack` hands from each layer to the next factory.

The comparison is written as "trace both, walk both, compare the two lists" and not as "check
LayerStack produces these literal names", so it fails if either path changes its naming. A test
against literals would let both drift together and still pass.

### chain_row

`chain_row` returns one `Config::LayerSpec` of a given type and attributes that reads one given
source row.

### hand_built_chain

`hand_built_chain` fills a `Config::ModelSpec` field by field, with no parser involved: the
input shape [6] and the rows `linear` 8, `gelu` and `linear` 4. Row 0 reads
`Config::K_MODEL_INPUT` and each later row reads the row above it.

### NamesItsParametersExactlyAsSequentialDoes

The second `EXPECT_EQ` states the names once as literals too, so the `expected` list that both
walks are compared through cannot become trivially equal by both paths going empty.

### KeepsTheIndexGapAParameterlessActivationLeaves

The gap is the whole reason the naming matches: indices track layer positions, not parameter
positions. A stack that renumbered would produce "1.weight" for the second Linear and load
nothing.

### ReportsTheLastLayersOutputWidth

The stack of `make_stack` has a size of 3, is not empty and reports 4 output features.

### TracesTheLayersInOrder

`forward` of that stack over a [4, 6] input gives a [4, 4] value.

### RefusesToTraceWhenEmpty

`forward` of an empty stack raises an error holding "no layers" and "identity".

### RefusesAnOutputWidthWhenEmpty

`output_features` of an empty stack raises an error holding "no layers".

### RefusesANullLayer

`push(nullptr)` raises an error holding "null layer" and "index 0".

### ReportsAnOutOfRangeLayerIndex

On a stack of three layers `layer(2)` does not throw, and `layer(3)` raises an error holding
"layer 3" and "holds 3".

### TracingTwiceDoesNotDuplicateParameters

`Parameter::declare` is idempotent, and the run-time path must inherit that and not reintroduce
the duplicate-on-retrace bug at this layer. Both traces go into one graph, because a
parameter's id names a value in the graph that declared it. Retracing into a second graph would
reuse an index that means something else there.

### AnAdapterBuiltFromAShapeReportsIt

A `LayerAdapter<GELU>` constructed with the shape [4, 5, 3] returns that shape from
`output_shape()`.

### AnAdapterBuiltFromAWidthReportsARankOneShape

A `LayerAdapter<Linear>` constructed with the width 8 reports the shape [8] and
`output_features()` 8. `make_stack` builds its layers through this constructor.

### OutputFeaturesEqualsTheElementCountOfTheOutputShape

Through an `Layer` reference, `output_features()` of an adapter of shape [4, 5, 3] is 60 and
equals `output_shape().num_elements()`. `output_features()` is a non-virtual function of
`Layer` in `modules/layers/nn/include/SushiAI/nn/layer.hpp`, derived from the one virtual.

### ReportsTheLastLayersOutputShape

The stack of `make_stack` reports [4]. With a `GELU` adapter of shape [2, 2, 1] pushed after
it, the stack reports [2, 2, 1] and `output_features()` 4.

### RefusesAnOutputShapeWhenEmpty

`output_shape()` of an empty stack throws with a message containing "no layers" and "output
shape".

### BuildLayerStackThreadsEachOutputShapeIntoTheNextLayer

The spec of `hand_built_chain` builds three layers that report [8], [8] and [4]. The last
layer is cast to `LayerAdapter<Linear>` and its `in_features` is 8, the width the activation
before it reported. The spec never states that number, so it can only come from the shape
`build_layer_stack` hands on.

### BuildLayerStackSeedsTheChainFromTheInputShape

The spec of `hand_built_chain` has its `input_shape` replaced by [5]. The first layer's
`in_features` is 5, so `build_layer_stack` seeds the chain from the shape.

### BuildLayerStackRefusesAHandBuiltRowWithoutASource

With the `from` list of row 1 cleared, `build_layer_stack` throws with a message containing
"layer 1 'gelu'" and "the row before it".

### OutputShapeOfAnInputIsTheLastLayersOutputShape

`LayerStack::output_shape([6])` of the built chain is [4] and equals the argument-free
`output_shape()`.

### RefusesAnOutputShapeOfAnInputWhenEmpty

`output_shape([6])` of an empty stack throws with a message containing "no layers" and "output
shape".

### AnActivationFactoryKeepsARankThreeInputShape

`NN::make_gelu` given the per-sample shape [4, 4, 3] returns a layer that reports [4, 4, 3]
and 48 features. A factory that reduced its input to a width would report [48].

### ALinearFactoryRefusesARankThreeInputAndNamesFlatten

`NN::make_linear` given [4, 4, 3] throws with a message containing the context, "[4, 4, 3]"
and "flatten". Given [48], the same object builds.

## modules/core/autograd/tests/test_layout.cpp

Suites: `Unit_Layout`.

Tests that every value carries a tensor layout and that the rebuilding passes keep it.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### EveryValueIsNHWCUnlessSaidOtherwise

CV-3 fixes one native layout. A value that never mentions layout is NHWC, so a graph built by
code that predates this field is uniform by construction and not by accident.

### FinaliseAcceptsAUniformGraph

`finalise` does not throw on a graph of one MATMUL over a [2, 4] input and a [4, 3] parameter.

### FinaliseRejectsAMixedGraphAndNamesTheValue

The failure this check exists to catch is a layer emitting an NCHW value into an NHWC network.
Nothing downstream would throw: the shapes still line up, every kernel still runs, and the
model learns from transposed data.

### LayoutConvertIsAllowedToChangeIt

K_LAYOUT_CONVERT is the one exemption, and the reason the check runs per node and not as a single
sweep over every value.

### DeclareLikeInheritsTheSourceValuesLayout

Fourteen tasks still to come each emit a value and carry its layout forward by hand.
`declare_like` is the one place that does it, so a forgotten carry cannot happen.

### Crossed

Both rebuilding passes copy the value table entry by entry, so that copy is what the `Crossed`
graph tests, and not the pass's own subject. The chain is deliberately plain for the same
reason: a fusion candidate or a precision decision in it would only obscure which field went
missing.

### FusionCarriesTheLayoutTagAcrossItsRebuild

`fuse()` rewrites the node list and copies the value table. A field it forgets to copy is reset
to `add_value`'s default, which is NHWC. A dropped tag is therefore a wrong layout, not a
missing one, and every shape check downstream still agrees with it.

The cost of a dropped tag shows in `lower_layout_convert` in `modules/core/graph/source/lowering/shape.cpp`, which
decides from these two tags whether each side is read or written through a transpose. A genuine
NCHW -> NHWC convert that reads as NHWC -> NHWC applies no transpose and copies transposed data
straight through.

### CastInsertionCarriesTheLayoutTagAcrossItsRebuild

Cast insertion is the other rebuilding pass and could make the same omission. Trainer runs both
passes unconditionally, so either one is enough to lose the tag on the default path.

### ARebuiltGraphStillProvesUniform

The consequence is worth stating separately. With every tag reset to one value, `finalise()`
passes for the wrong reason. A graph that is uniform because it was flattened cannot be told
from one that was uniform to begin with, so the proof has to be re-run against tags that
survived.

### TheThreeRulesThatEmitZerosCannotBeReachedWithAnNCHWOperand

The test checks why the zeros exposure is latent and not live, and does not leave that to a
comment. It is the premise the `zeros_like()` change rests on, and it is a premise about other
emitters' guards, which is the kind that stops being true without anyone editing this file.

The two `EXPECT_THROW` lines on `slice_channels` and `channel_affine` cover the rank-4 sites,
`slice_channels_backward` and `channel_affine_backward`. CV-3 makes their forward emitters
refuse an NCHW operand outright, so their dy can never carry that tag.

The `reshape` to `rows` covers the rank-2 site, `sum_rows_backward`. Reshape carries an NCHW
tag into rank-2 land with no guard at all, so unlike the two rank-4 sites the emitters let this
chain be built.

The last `EXPECT_THROW(graph.finalise(), Error)` shows what blocks that chain: the far end.
`differentiate()` only seeds a loss produced by K_NLL_LOSS, whose one-hot operand is a leaf and so
always NHWC, and nothing but K_LAYOUT_CONVERT may cross back, which is rank-4 only. `finalise()`
rejects that node before differentiation ever runs, so the rule is never reached with an NCHW
gradient. The guard is the pre-existing proof in `finalise()`, not a check in any emitter, and
that is why it is worth a test and not a sentence.

### ZerosLikeCarriesTheTagAndPlainZerosCannot

The three backward rules now depend on this contract. K_ZERO reads nothing, so plain `zeros()` has
no operand to inherit from and its NHWC tag is a default, not an answer. That is harmless for
the K_ZERO node itself, which the per-node proof skips for having no inputs. The consumer that
reads those zeros beside a real operand is the node that would be seen crossing layouts.

### FinaliseStillPropagatesRequiresGrad

The rename must not quietly drop the behaviour the old name described. The assertion is the
same one `test_ir.cpp` made.

## modules/layers/nn/tests/test_leaky_relu.cpp

Suites: `Unit_LeakyReLU`.

Tests the LeakyReLU layer's trace, shape inference and slope range.

This file needs no device. `NN::LeakyReLU` in `modules/layers/nn/include/SushiAI/nn/leaky_relu.hpp` holds a
slope and no parameter, and the slope must lie strictly inside (0, 1). The section on
`test_ops.cpp`, under LeakyReLUDifferentiatesAtItsOutput, says why a slope of 0 or below is
excluded: the gradient reads the layer's output, which is a substitution only for a positive
slope.

Three `static_assert` lines at file scope require `LeakyReLU` to be a `Module`, not to be a
`ModuleContainer`, and to list no parameter field in `fields()`.

### is_batch_of

`is_batch_of` reports whether one shape is another with a single batch axis in front. It
compares every per-sample axis, so it serves a feature vector and an image alike.

### is_slope_rejected

`is_slope_rejected` traces a layer of the given slope over a [2, 4] input and reports whether
the trace throws `Error`. It uses `Test::capture_error` from `tests/common/test_throws.hpp`,
so an exception of another type does not count as a rejection.

### TheDefaultSlopeIsOneHundredth

A default-constructed layer has slope 0.01.

### ForwardEmitsOneLeakyReLUNodeCarryingTheSlope

The trace of a layer with slope 0.2 adds exactly one node, a K_LEAKY_RELU with the input as its
one operand and the result as its one output. Attribute slot 0 of the node holds 0.2.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

A [2, 4] input traces to a [2, 4] value of the input's element type.

### ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

A [2, 8, 6, 3] input traces to a value of the same shape tagged NHWC.

### ForwardCarriesTheLayoutOfItsInput

The input is converted to NCHW first. The result is tagged NCHW and has the converted value's
shape. The activation is elementwise and reads no axis, so it accepts either layout and
passes the tag on, where the spatial layers refuse NCHW.

### OutputShapeAgreesWithForwardForAFeatureVector

`output_shape([7])` is [7], and the traced shape of a [5, 7] input is a batch of it.

### OutputShapeAgreesWithForwardForAnImage

`output_shape([8, 6, 4])` is [8, 6, 4], and the traced shape of a [3, 8, 6, 4] input is a
batch of it.

### ForwardRejectsASlopeOfZero

A slope of exactly 0 is rejected. The range is open at this end.

### ForwardRejectsASlopeOfOne

A slope of exactly 1 is rejected. The range is open at this end too.

### ForwardRejectsANegativeSlope

A slope of -0.01 is rejected.

### ForwardRejectsASlopeAboveOne

A slope of 1.5 is rejected.

### ForwardRejectsANaNSlope

A NaN slope is rejected. NaN compares false against both bounds, so a check written as "throw
when the slope is at or below 0 or at or above 1" lets it through.

### ForwardAcceptsSlopesJustInsideTheOpenRange

The smallest positive normal float32 and 1 minus the float32 epsilon are both accepted. Each
lies inside the range and still differs from its bound after narrowing to float32.

### OutputShapeRejectsASlopeOutsideTheOpenRange

For the five slopes the forward cases reject, `output_shape([4])` throws with a message
containing "LeakyReLU" and "slope". A factory calls `output_shape()` and not `forward()`, so
this is the check that refuses a slope when a model file is read.

### ForwardNamesTheLayerAndTheSlopeItRefuses

`forward()` with slope 1.5 throws with a message containing "LeakyReLU" and "slope".

### ARejectedSlopeEmitsNothing

After `forward()` throws for a slope of 1, the node count and the value count are unchanged.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.

## modules/layers/nn/tests/test_linked_rows.cpp

Suites: `Unit_LinkedRows`, `Unit_LinkedRowsDeathTest`.

Tests `NN::Detail::read_row_above`, `NN::LinkedNodeEntry` rows of `NN::NodeRegistry` and
`NN::BuiltRows`. The file registers a linked type `follower` whose factory reads a row number
under the key `names`, fails unless that row holds an embedding and returns an `Identity`
node.

### ARowReferenceTakesThreeForms

With three rows above, `{"names": 0}` reads as row 0, `{"names": -1}` as row 2 and
`{"names": {"node": 1}}` as row 1.

### ARowReferenceThatDoesNotFitIsRefused

Seven references fail with `InvalidDescription`: row 3 of three ("layer 3", "'names'", "row 3",
"not above"), -2 ("row -2"), -1 with no row above ("0 rows are above"), a string ("row number",
"string"), an object with a `port` key ("'port'"), an empty object ("'node'") and an absent key
("'names'", "absent").

### ALinkedTypeHasOneRowAmongAllKinds

`find_linked("follower")` returns a row, `find_node("follower")` is null and `linked()` holds
that one row. Adding a linked row named `concat` or `relu`, or a `NodeEntry` named `follower`,
fails with `InvalidDescription`, the name and "already".

### BuildModelHandsALinkedRowTheRowsAbove

`NN::build_model` over an embedding row, a `layernorm` row and a `follower` row naming row 0
returns a model of three nodes. `NN::find_embedding` returns a layer for node 0 and null for
node 1. The same model with `"names": -1` fails with `InvalidDescription`, "layer 2
'follower'", "row 1" and "no embedding".

### ALinkedRowBuiltAloneSeesNoRow

`NN::make_node` for `follower` with `{"names": 0}` fails with `InvalidDescription`, "layer 0
'follower'" and "not above". For the unknown type `missing` it fails with a message containing
`'missing'` and `'follower'`.

### BuiltRowsCountsTheNodesOfItsGraph

A `BuiltRows` over a `ModelGraph` of one node has the size 1 and `node(0)` is the graph's node
0. A default `BuiltRows` has the size 0.

### ARowOutsideTheViewIsFatal

`node(0)` on a default `BuiltRows` stops the process with a report containing
"BuiltRows::node" and "row 0".

## modules/core/graph/tests/test_log_softmax_ops.cpp

Suites: `Unit_LogSoftmaxOps`.

Tests `GraphBuilder::log_softmax_backward`.

### TheGradientWritesItsIdentityAndKeepsTheShape

Over a [4, 3] float64 gradient and the `log_softmax` of a [4, 3] float64 input, the call is ok
and emits one LOG_SOFTMAX_BACKWARD node that reads the gradient and then the log-probabilities.
The result has the shape [4, 3], the type float64 and the lifetime `Activation`, and the graph
has no refusal.

### TheGradientRefusesOperandsThatDoNotFit

Four calls fail with `InvalidDescription`: rank-3 operands ("rank-2 log-probabilities"), int32
operands ("half, float32 or float64 log-probabilities"), a [4, 5] gradient ("[4, 5]") and a
float64 gradient under float32 log-probabilities ("a gradient like them"). After each the node
count is unchanged and `has_refusal()` is true.

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

## test_loss_scaling.cpp

Suites: `Unit_LossScaling`.

Tests the dynamic loss-scaling rule on LossScaleState, with no device involved.

This file tests the dynamic-scaling rule with no device in the measurement.

LossScaleState is split out of LossScaler so that this file can exist. "The scale doubles after
exactly N clean steps" is a property of the rule, not of a kernel. Asserting it through a
training run would make every one of these cases cost a compile and a replay while proving less.

The control matters as much as the case. A growth test that only ran N clean steps and saw a
doubling would pass against an implementation that doubled after every step, so each growth
assertion in this file is paired with the N-1 run that must not double.

### TheScaleStartsWhereItWasAsked

A state made with the initial scale 65536 reports that scale, its inverse 1 / 65536, no
skipped last step and 0 skipped steps.

### AnOverflowHalvesTheScaleAndSkipsTheStep

From the initial scale 1024, `observe(true)` gives the scale 512. The last step counts as
skipped, and the state reports 1 skipped step, 1 backoff and 0 growths.

### TheScaleDoublesAfterExactlyTheGrowthInterval

From the initial scale 1024 with a growth interval of 4, four clean steps give the scale
2048, 1 growth and no skipped last step.

### OneStepShortOfTheIntervalTheScaleDoesNotMove

This case is the control for TheScaleDoublesAfterExactlyTheGrowthInterval. Without it, "the
scale doubled after four clean steps" is equally true of an implementation that doubles after
one.

### AnOverflowRestartsTheRunOfCleanSteps

Three clean steps since the overflow is one short of the interval, so nothing has grown back.
The scale is still the backed-off one.

### GrowthStopsAtTheCeilingWithoutCountingAGrowth

With an interval of 1, the initial scale 1024 and the maximum 2048, one clean step gives 2048.
A second leaves the scale at 2048 and the growth count at 1.

### BackoffStopsAtTheFloorButStillSkipsTheStep

At the floor the scale cannot go lower. The step is still garbage and must still be thrown
away, or an overflowed gradient reaches the weights the moment the backoff bottoms out.

### ANonDyadicConstantIsRefusedByName

Every constant is a power of two so that scaling and unscaling change only the exponent field.
A scale of 3 would work and would quietly make every gradient in the model carry a rounding
error, so it is refused.

### IncoherentBoundsAreRefused

`LossScaleState::make` fails with `InvalidDescription` for an initial scale of 65536 above a
maximum of 1024 ("outside the bounds") and for a growth interval of 0
("growth interval must be positive").

### EveryReachableScaleHasAnExactReciprocal

The loss-scaling design is arranged around this property, and the test asserts it on the rule
itself. Whatever sequence of backoffs and growths the machine takes, the scale is a power of
two, so the reciprocal is exact and the round trip is lossless.

## modules/core/graph/tests/test_lowering_table.cpp

Suites: `Unit_LoweringTable`.

Tests the table that maps each operation identity to its lowering. The lowering is eight domain
tables under `modules/core/graph/include/SushiAI/graph/lowering/`, which `all.hpp` joins into `K_LOWERINGS`. The
identities that have no row are listed in `K_UNLOWERED`, in `unlowered.hpp`.

The file needs no device. It reads the constants and the constexpr helpers of `all.hpp`. The
`static_assert`s in `all.hpp` prove the same counts at build time. These cases name the identity
or the domain whose count is wrong, so a change to one table reads as one line in the report.

### EveryIdentityIsLoweredOrListedAsUnlowered

Each entry of `Ops::K_ALL` has exactly one row, or it is listed in `K_UNLOWERED` and has none.

### TheUnloweredIdentitiesHaveNoRow

`K_UNLOWERED` holds four identities, and none of them has a row.

### NoOperationHasTwoRows

No identity appears in two rows of `K_LOWERINGS`.

### TheCombinedTableCoversTheOperationTable

Every identity is accounted for, and the rows plus the unlowered identities add up to the
operation table's size.

### EachDomainTableHasItsRowCount

The domain tables hold 17, 14, 6, 3, 4, 15, 6 and 6 rows, in the order elementwise, activation,
linalg, loss, shape, spatial, norm and fused. The joined table holds 71.

## modules/layers/nn/tests/test_max_pool2d.cpp

Suites: `Unit_MaxPool2d`.

Tests the 2-D max pooling layer's trace, shape inference and rejections.

This file needs no device. `NN::MaxPool2d` in `modules/layers/nn/include/SushiAI/nn/max_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter. `GraphBuilder::maxpool2d` returns the pooled value and
an INT64 argmax together; the layer returns the first.

Two `static_assert` lines at file scope require `MaxPool2d` to be a `Module` and not a
`ModuleContainer`.

### make_pool

`make_pool` builds a layer from a kernel and a stride per axis and an optional top and left
padding, which default to 0.

### traced_shape

`traced_shape` traces a layer over a batch of the given per-sample shape and returns the
result's shape.

### with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

### ForwardHalvesTheSpatialExtentsOfAnEvenInput

A 2x2 stride-2 pool traces a [2, 8, 6, 3] input to [2, 4, 3, 3].

### ForwardTagsItsResultNHWC

The traced result is rank 4 and tagged NHWC.

### ForwardReturnsThePooledValueNotTheArgmax

The result has the input's element type, and that type is not INT64. A layer that returned
the second result of the node would hand the next layer the index buffer.

### OutputShapeOfAnUnpaddedSquareWindow

`output_shape([8, 6, 3])` of a 2x2 stride-2 pool is [4, 3, 3].

### OutputShapeAgreesWithForwardForASquareWindow

For the same pool and sample, the traced shape of a batch of two equals `output_shape()` with
the batch extent in front.

### OutputShapeAgreesWithForwardForAPaddedAnisotropicWindow

A 3x2 window with stride 2 by 1 and a top padding of 1 maps [5, 7, 4] to [3, 6, 4], and the
traced shape agrees.

### OutputShapeAgreesWithForwardWhenTheStrideLeavesARemainder

A 3x3 stride-2 window maps [8, 10, 1] to [3, 4, 1], and the traced shape agrees. Neither
extent divides evenly, so the case covers the rounding down.

### OutputShapeAgreesWithForwardForADilatedAnisotropicWindow

A 2x2 unit-stride window with dilations 2 by 3 maps [7, 9, 2] to [5, 6, 2], and the traced
shape agrees.

### ForwardRejectsAnInputThatIsNotRankFour

`forward()` throws on a [2, 16] input.

### ForwardRejectsAWindowThatDoesNotFit

`forward()` throws for a 5x5 window over a [2, 4, 4, 3] input.

### ForwardRejectsAnNCHWInput

A [2, 8, 8, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### ForwardNamesTheLayoutOfAnNCHWInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

### ForwardRejectsAZeroStride

`forward()` with a vertical stride of 0 throws with a message containing "MaxPool2d" and
"stride 0,2".

### ForwardRejectsANegativePadding

`forward()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

### OutputShapeRejectsAZeroStride

`output_shape()` throws when the vertical stride is 0.

### OutputShapeNamesAZeroStride

The message for the same layer contains "MaxPool2d" and "stride 0,2", so it reports the
stride and not a window that does not fit.

### OutputShapeRejectsANegativePadding

`output_shape()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

### OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

### OutputShapeRejectsABatchedShape

`output_shape([2, 8, 8, 3])` throws.

### OutputShapeRejectsAWindowTallerThanTheInput

A 5x2 window over [4, 8, 3] throws. The window fits the width, so the height check refuses.

### OutputShapeRejectsAWindowWiderThanTheInput

A 2x5 window over [8, 4, 3] throws. The window fits the height, so the width check refuses.

### TrailingRowKeepsTheExtentOfAStrideOnePool

A 2x2 pool of stride 1 over [13, 13, 4] reports [12, 12, 4]. With a trailing pad of 1 on both
axes it reports [13, 13, 4], and a trace over a batch of 2 has that shape behind the batch.

### TrailingPadAtStrideTwoGivesSevenFromThirteen

A 2x2 pool of stride 2 with a trailing pad of 1 on both axes reports [7, 7, 4] over
[13, 13, 4], and its trace over a batch of 1 is [1, 7, 7, 4].

### TrailingPadGivesTwoHundredEightFromFourHundredSixteen

The same pool reports [208, 208, 16] over [416, 416, 16].

### TrailingPadAppliesToOneAxisOnly

A 2x2 pool of stride 1 with `trailing_pad_h` 1 alone reports [3, 2, 2] over [3, 3, 2].

### TrailingPadLetsAWindowFitAShortInput

A 2x2 pool of stride 1 throws `Error` from `output_shape` over [1, 4, 3]. With a trailing pad
of 1 on both axes it reports [1, 4, 3] and traces to [1, 1, 4, 3].

### OutputShapeRejectsANegativeTrailingPad

With `trailing_pad_w` -1, `output_shape` raises an error holding "MaxPool2d",
"non-negative trailing padding" and "0,-1".

### ForwardRejectsANegativeTrailingPad

With `trailing_pad_h` -2, `forward` raises an error holding "MaxPool2d",
"non-negative trailing padding" and "-2,0", and the graph's node count does not change.

### OutputShapeRejectsATrailingPadThatOverflows

With `trailing_pad_h` at the largest `int64_t`, `output_shape` raises an error holding
"MaxPool2d" and "within 64 bits".

### OutputShapeRejectsATrailingPadThatLeavesAWindowNoTap

A 2x2 pool with `trailing_pad_h` 2 raises from `output_shape` an error holding "MaxPool2d",
"every window a tap inside the input" and "2,0".

### ForwardRejectsATrailingPadThatLeavesAWindowNoTap

A 2x2 pool of stride 2 with `trailing_pad_w` 3 raises from `forward` an error holding
"MaxPool2d", "every window a tap inside the input" and "0,3", and the graph's node count does
not change.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace.

## modules/core/graph/tests/test_memory_plan.cpp

Suites: `Unit_MemoryPlan`.

Tests that the memory planner never shares a byte between two values live together.

This file needs no device: no engine, no allocator, no SYCL. The memory planner is a pure
function of the IR, which is the reason the IR is a host-lane structure, and every case is a
synthetic graph and a table of offsets.

The contract is one sentence: two values whose live ranges intersect never share a byte, and
two whose live ranges do not are free to. The first half is checked exhaustively over every
pair. The liveness it is checked against is computed in the test file by a breadth-first search,
`descendants()`, and not by asking the planner what it thinks. A planner checked against its own
reachability code would agree with itself whatever it did.

The individual cases are for the second half, and the diamond in
TwoUnorderedBranchesNeverShareBytes is the one that matters. It is the shape where a planner
that overlaps disjoint [first_use, last_use] intervals on a linearised schedule gets a wrong
answer that no test written from the linear model would catch.

The cases fall into six groups, in file order:

- degenerate graphs
- the two shapes that decide whether the pass is right
- lifetimes
- aliasing
- mixed dtypes
- knobs, refusals and malformed input

### AnEmptyGraphAsksForNothing

The plan of an empty graph has 0 values, no arena and 0 arena bytes.

### ASingleActivationGetsOneArenaAndOffsetZero

For one ReLU over an input, the plan has one arena of the `Activation` lifetime with one
buffer and 128 bytes. The input is not packed. The ReLU's value is packed at offset 0 with
128 bytes.

### AGraphOfOnlyLeavesPacksNothing

The plan of a graph of one input and one parameter has no arena and 0 arena bytes.

### AChainPingPongsBetweenTwoBuffers

The graph is x -> a -> b -> c -> d. Nothing branches, so `a` is finished before `c` starts and
`b` before `d`. Four activations need exactly two buffers, which is the most any planner can do
with a chain.

### TwoUnorderedBranchesNeverShareBytes

This is the case the planner is judged by. Linearised, the two branches have disjoint
[first, last] intervals: a1 dies at node 2 and b1 is born at node 3. An interval planner
overlaps them and is wrong. Nothing in the graph orders node 2 before node 3 and the runtime may
run the branches concurrently, so the sharing would corrupt the result and also serialise the
parallelism it was meant to leave free.

### OnlyTheNonPersistentClassesArePacked

The graph holds all six lifetimes. The four persistent ones live for the whole run, so packing
them would save nothing and would merge the dependency-tracker shards that matter most. Only
ACTIVATION and SCRATCH reach an arena, and they reach different ones.

### AValueNothingWritesKeepsItsOwnAllocation

An ACTIVATION with no producer is written from outside the step and never rewritten, so its
bytes are live for the whole run whatever its class says. Handing them away would survive
exactly one replay.

### AnAccumulateChainIsOneBufferWithOneLiveRange

K_ACCUMULATE's result is its accumulator's storage. Planning the two separately would be silently
wrong and not merely wasteful: the planner would think the accumulator died at the K_ACCUMULATE
node and hand its bytes to `sink`, which is produced by the node still reading them.

### EveryStartIsAWholeNumberOfItsOwnElements

Alignment is turned down to 1 so that the element-width floor is the only thing left doing the
work. A 20-byte float32 buffer at offset 0 leaves the next free byte at 20, which is not a whole
number of float64 elements. The float64 buffer therefore has to start at 24, or its
storage_offset would not be representable at all.

### ReuseCanBeSwitchedOffAndSaysSo

With `reuse` false over a chain of three ReLUs, the plan reports reuse disabled, a non-empty
`reuse_note` and 384 arena bytes.

### TooManyNodesDeclinesOverlapRatherThanApproximatingIt

The refusal is the point. A planner that quietly fell back to overlapping linear intervals
would produce a plan that is wrong in the way TwoUnorderedBranchesNeverShareBytes describes, and
would report a saving for it.

### APinnedValueKeepsItsOwnAllocation

With the first of two chained ReLU values pinned, that value is not packed and the second is.

### PinningAValueThatDoesNotExistThrows

`MemoryPlan::build` with the value 99 pinned throws `Error`.

### ZeroAlignmentThrows

`MemoryPlan::build` with an alignment of 0 throws `Error`.

### AGraphThatIsNotInTopologicalOrderThrows

GraphBuilder cannot produce this graph, because it always emits a producer before anything can
read it. The whole pass rests on the property all the same, so the test checks it. A closure
built over a back edge would silently miss ordering and the plan would overlap things it must
not.

### ThePlanIsDeterministicForTheSameGraph

The offsets are byte-identical across two builds. A plan that shuffled would move every tensor
and take the library's bit-for-bit replay guarantee with it.

### TheTwoResultsOfOneNodeAreNeverGivenTheSameBytes

`FusedForm::GEMMBiasGELU` writes two values from one node: its result and the pre-activation
its epilogue spills. They have the same shape, they are born at the same instant, and the
planner packs by liveness. "One died before the other was written" is the mistake available to
it, and taking it would have the GELU computed from its own output with nothing to show for it.

Nothing special prevents it, which is why it is worth pinning. `precedes()` asks whether every
use of the earlier buffer strictly reaches the later one's first write, and a node does not
reach itself. The planner therefore refuses both directions and the two values conflict. That is
an inference from an existing rule and not a rule of its own, so a future rewrite of
`precedes()` could take it away silently.

## modules/layers/train/tests/test_metrics.cpp

Suites: `Unit_Metrics`.

Tests `Train::Metrics`, `Train::describe_metric_value` and `Train::only_output`.

### MetricsAreKeptInOrderAndFoundByName

A new `Metrics` is empty. After `accuracy` 0.75 and `perplexity` 12.5 are added, `all()` holds
them in that order, `find("perplexity")` has the value 12.5 and `find("recall")` is null.
`value_or` returns 0.75 for `accuracy` and the fallback -1 for `recall`. After `clear()` the
value is empty and `find("accuracy")` is null.

### AnEmptyNameADuplicateAndAnOverflowAreRefused

`add` fails with `InvalidDescription` for an empty name ("non-empty name") and for a second `a`
("'a'", "twice"), and `a` keeps the value 1. After `K_METRIC_CAPACITY` metrics the next `add`
fails with "at most 8" and `'i'`, and the size stays `K_METRIC_CAPACITY`.

### AFractionPrintsAsAPercentage

`describe_metric_value` gives "92.40%" for the fraction 0.9240 and "12.5" for the plain number
12.5.

### OnlyOutputReturnsTheOneTensor

`only_output` over a view with one output returns the address of that output's tensor.

### OnlyOutputRefusesAnyOtherCount

`only_output` fails with `InvalidDescription` for no output ("published 0"), for two outputs
("published 2") and for one output whose tensor is null ("'output'", "no tensor").

## modules/layers/nn/tests/test_mish.cpp

Suites: `Unit_Mish`.

Tests `NN::Mish` in `modules/layers/nn/include/SushiAI/nn/mish.hpp`, which has no parameter and
no attribute.

Three `static_assert` lines at the top of the file hold that `Mish` is a `Module`, is not a
`ModuleContainer`, and that `Mish::fields()` is an empty tuple.

### ForwardEmitsSoftplusTanhThenMulReadingTheInput

`forward` adds three nodes in the order SOFTPLUS, TANH, MUL. SOFTPLUS reads the input, TANH
reads the SOFTPLUS result, and MUL reads the input and then the TANH result and writes the
value `forward` returns.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

Over a [2, 4] input the result has the shape [2, 4] and the input's element type.

### ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

Over a [2, 8, 6, 3] input the result has that shape and the layout `TensorLayout::NHWC`.

### ForwardCarriesTheLayoutOfItsInput

Over a value converted to NCHW with `layout_convert`, the result is tagged NCHW and has the
converted value's shape.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## modules/layers/nn/tests/test_model_graph.cpp

Suites: `Unit_ModelGraph`, `Unit_ModelGraphDeathTest`.

Tests ModelGraph: what add and output refuse, what a trace emits, and the state names.

This file needs no device. `NN::ModelGraph` in `modules/layers/nn/include/SushiAI/nn/model_graph.hpp` holds
named nodes in the order they were added. `add` takes a name, a layer or an `NN::Node` and the
`NodeRef`s the node reads, and returns the `NodeRef` of the new node. `output` publishes a ref
under a name. `trace` returns the published values as an `NN::ModelOutputs`, and `forward`
returns the one published value. `add`, `output`, `port`, `output_ports` and `trace` return a
`Status` or a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelGraph` is a `TraceModule`,
is not a `Module`, and is detected by `Detail::HasStateWalk`.

### OpList

`OpList` is a vector of the operation of every node of a graph, in order.

### TwoPortNode

`TwoPortNode` is an `NN::Node` with one input and two ports. `forward` appends the ReLU and
then the GELU of its input. The one-argument constructor names the ports "low" and "high", and
the three-argument one takes the two names.

### BranchingModel

`BranchingModel` holds a `ModelGraph` and the refs `stem`, `up` and `mix` of three of its
nodes.

### build_branching_model

`build_branching_model` builds five nodes over an [8, 8, 3] sample and publishes nothing:
"stem", a same-padded `Conv2d(3, 4)` of kernel 3 and stride 2 on the input; "down", a
`Conv2d(4, 8)` of the same window on the stem; "up", an `Upsample2d{2}` on that; "merge", a
`Concat` of up and stem; and "mix", a pointwise `Conv2d(12, 4)` on the merge.

### ops_of

`ops_of` returns the `OpList` of a graph.

### parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields under a prefix as
"name:extents" strings.

### add_relu_declared_as

`add_relu_declared_as` adds a `LayerNode` over a `LayerAdapter<ReLU>` constructed with a given
shape. The node declares that shape on its port whatever its input is.

### RefsNameTheInputAndTheNodesInTheOrderAdded

`model.input()` reports `is_input()` and port 0. The refs of stem, up and mix report the nodes
0, 2 and 4, mix reports port 0, stem and up compare unequal, and `size()` is 5.

### ShapeOfReturnsTheDeclaredPerSampleShapes

`shape_of` is [8, 8, 3] for the input, [4, 4, 4] for stem, [4, 4, 8] for up and [4, 4, 4] for
mix. Node 3, the merge, declares [4, 4, 12] on its port.

### BranchAndMergeTracesTheNodeSequenceOfTheHandWrittenTrace

The model's `forward` under an evaluation trace is compared with the same five layers traced
by hand through a `GraphBuilder` in a second graph. The two graphs hold the same operations in
the same order, and both results are [2, 4, 4, 4]. The sequence holds one K_UPSAMPLE2D and one
K_CONCAT_CHANNELS.

### MergeEmitsOneConcatChannelsReadingBothBranches

With stem, up and mix published, the traced graph holds one K_CONCAT_CHANNELS node. Its inputs
are the published value of up and then that of stem, and its output is [2, 4, 4, 12].

### TraceReturnsEveryPublishedOutputByName

With "features" and "skip" published in that order, `trace` returns two entries in that order.
`at` returns a different id for each name, and both values are [2, 4, 4, 4].

### ForwardReturnsTheSinglePublishedOutput

A model of one ReLU node whose result is published traces one K_RELU node. That node reads the
input, and its output is the id `forward` returned.

### ForwardFailsWhenTwoOutputsArePublished

With two outputs published, `forward` returns its input id and records `InvalidDescription` on
the trace with a message containing "exactly one model output", "features" and "skip". The
graph is marked refused.

### OutputShapeIsTheShapeOfTheSinglePublishedOutput

`output_shape([8, 8, 3])` is an empty `Shape` while nothing is published, [4, 4, 8] once up is
the one published output, and an empty `Shape` again once a second output is published.

### AGraphIsANodeOfAnotherGraph

A `ModelGraph` holding a `Linear{6, 8}` named "dense" is added to an outer graph as the node
"inner", with a ReLU after it. `shape_of` of the nested node is [8], the one output is
[4, 8], and the outer walk yields "inner.dense.weight:[8, 6]" and "inner.dense.bias:[8]".

### ANestedGraphThatPublishesTwoOutputsFailsTheOuterTrace

An inner graph that publishes "first" and "second" is added as a node. The outer `trace` fails
with `InvalidDescription` and a message containing "node 'inner'", "exactly one model output",
"first" and "second".

### ASequentialAndALayerStackAreNodesOfAGraph

A `Sequential{Linear{6, 8}, ReLU}` is added as "head" and a `LayerStack` of a `Linear{8, 4}`
and a GELU as "tail". `shape_of` is [8] and [4], the output is [4, 4], and the walk yields
"head.0.weight:[8, 6]", "head.0.bias:[8]", "tail.0.weight:[4, 8]" and "tail.0.bias:[4]".

### TheInputCanBePublishedAsAnOutput

A model with no node that publishes its input returns the input id from `forward` and emits no
node.

### WalkNamesStateByNodeNameInTheOrderAdded

After a trace of the branching model, the walk under "model" yields "model.stem.weight:[4, 3,
3, 3]", "model.down.weight:[8, 3, 3, 4]" and "model.mix.weight:[4, 1, 1, 12]", and
`NN::buffer_count` is 0. The upsample and the concat contribute no name.

### ATrainingTraceReachesAModalNodeAndTheWalkNamesItsBuffers

A model of one `BatchNorm2d(3)` node named "bn", traced under a training trace, emits one
K_BATCHNORM2D node and leaves two updates in the list. The parameter walk under "model" yields
"model.bn.weight:[3]" and "model.bn.bias:[3]", and `for_each_buffer` yields
"model.bn.running_mean" and "model.bn.running_var".

### AChainNamedByIndexYieldsTheNamesOfALayerStack

A model whose nodes are named "0", "1" and "2" holds a `Linear{6, 8}`, a GELU and a
`Linear{8, 4}`. A `LayerStack` holds the same three layers. After each is traced into its own
graph, the two parameter walks are equal, the two graphs hold the same operation sequence, and
the names are "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]" and "2.bias:[4]".

### NestsInASequentialUnderItsIndex

A `Sequential<ModelGraph, ReLU>` traces to a [4, 8] value whose last node is a K_RELU. Its walk
yields "0.dense.weight:[8, 6]" and "0.dense.bias:[8]".

### AddRejectsADuplicateName

A second node named "act" fails with `InvalidDescription` and a message containing "already
has a node named 'act'" and "index 0". `size()` stays 1.

### AddRejectsAnEmptyName

The name "" fails with "needs a name" and `size()` stays 0.

### AddRejectsANameWithADot

The name "a.conv" fails with "'a.conv'" and "contains '.'" through the layer overload, and
"b.act" fails the same way through the `NN::Node` overload. `size()` stays 0.

### AddReportsARefusedNameBeforeTheLayerSeesItsSource

A `Conv2d` added under a taken name, or under an empty one, on a [6] source fails with the
message about the name. The convolution would refuse a [6] source, and that failure is not the
one returned. `size()` stays 1.

### AddReturnsTheFailureOfALayerThatRefusesItsSource

`add("stem", Conv2d::same_padded(3, 4, 3, 2), input)` on a [6] sample fails with a message
containing "node 'stem'", "Conv2d(3, 4)" and "[6]", and `size()` stays 0.

### AddRejectsAnUnknownSource

A ref to node 1 of another graph, used as the source in a graph of one node, fails with a
message containing "node 'late'", "source node 1" and "holds 1 nodes". The `NN::Node` overload
fails with "node 'later'" and "source node 1". `size()` stays 1.

### AddRejectsASourcePortTheNodeDoesNotHave

A ref to port 1 of a two-port node of another graph, used on a graph whose node 0 has one
port, fails with a message containing "node 'late'", "'only' has 1 ports" and "port 1".

### PortSelectsANamedPortThatALaterNodeReads

`port(pair, "high")` returns a ref to node 0, port 1, and `port(high, "low")` returns a ref
equal to `pair`. A ReLU added on `high` and published traces three nodes. Node 1 is the GELU,
and node 2 reads node 1's output.

### PortRejectsAnUnknownNameAndTheInput

`port(pair, "middle")` fails with a message containing "node 'pair'", "'middle'" and "low,
high". `port(input, "low")` fails with "the model input has no port named 'low'".

### OutputPortsPublishesEveryPortUnderItsOwnName

`output_ports(pair)` publishes two outputs. `at("low")` is the output of graph node 0 and
`at("high")` that of graph node 1.

### OutputPortsPublishesNothingWhenAPortNameIsTaken

With "high" already published, `output_ports(pair)` fails with "port 'high' of node 'pair'",
and `output_ports(input)` fails with "the model input has no ports to publish". A trace then
returns one output, so the failed call did not publish "low" either.

### OutputRejectsATakenNameAnEmptyNameAndAnUnknownSource

A second `output("echo", ...)` fails with "already publishes an output named 'echo'". The name
"" fails with "needs a name". A ref to a node of another graph fails with "output 'far'",
"source node 0" and "holds 0 nodes".

### TraceRejectsADeclaredShapeThatDisagreesWithTheTracedOne

A node that declares [5] and traces a ReLU of a [4, 6] value fails the trace with
`InvalidDescription` and a message containing "node 'narrow'", "port 'output'", "[5]" and
"[4, 6]".

### TraceRejectsAnInputOfAnotherSampleShape

A model over a [6] sample traced with a [4, 7] value fails with "the model input", "[6]" and
"[4, 7]". Traced with a [6] value, which has no batch axis, it fails with "the model input"
and "[6]". The graph gains no node.

### TraceRejectsAModelThatPublishesNoOutput

A model with one node and no published output fails the trace with "publishes no output", and
the graph gains no node.

### AddRejectsASourceCountTheNodeDoesNotRead

A `Concat` built over two shapes and added with one source fails with a message containing
"node 'merge'", "reads 2 values" and "given is 1". A `TwoPortNode` added with two sources fails
with "node 'split'", "reads 1 values" and "given is 2". `size()` stays 0.

### AMovedGraphKeepsItsRefsAndLeavesAnEmptyUsableGraph

A graph move-constructed from the branching model has `size()` 5, resolves the ref `up` to
[4, 4, 8] and accepts `output("features", mix)`. The graph moved from has `size()` 0. A ReLU
added to it and published traces one node whose output is the published value.

### AccessorsAreFatalOutOfRange

The case is in the suite `Unit_ModelGraphDeathTest`. On an empty graph, `shape_of` of a ref to
another graph's node stops the process with "shape_of" and "source node 0", and `node(0)`
stops it with "ModelGraph::node" and "node 0".

### AddIsFatalOnANullNode

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a null `unique_ptr<NN::Node>`
stops the process with a report containing "null node" and "'ghost'".

### AddIsFatalOnANodeWithTwoPortsOfOneName

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a `TwoPortNode` whose ports
are both named "low" stops the process with a report containing "'pair'" and "two ports named
'low'".

### TraceIncrementalHandsEachLayerThePerSampleShapeItReads

A model of the sample shape [5, 6] holds one `SliceProbe`. `trace_incremental` in a carrying
trace over a [1, 1, 6] slice hands the probe the shape [5, 6] and adds one GELU node, and the
output "y" has the shape [1, 1, 6].

### TraceStillChecksThePerSampleShapeTheIncrementalWalkSkips

`trace` of the same model over the [1, 1, 6] slice fails with `InvalidDescription`, "[5, 6]"
and "[1, 1, 6]".

### TraceIncrementalNamesTheNodeThatHasNoIncrementalForm

With a `ReLU` named "act" after the probe, `trace_incremental` fails with `Unsupported`,
"node 'act'" and "no incremental form".

### TraceIncrementalNeedsACarryingTrace

`trace_incremental` in an evaluation trace fails with `InvalidDescription` and
"keeps nothing between runs", and adds no node.

### ANestedGraphTracesItsIncrementalForm

With the probe's graph added as one node of an outer graph, `trace_incremental` of the outer
graph hands the probe [5, 6] and adds one GELU node.

## modules/layers/nn/tests/test_model_outputs.cpp

Suites: `Unit_ModelOutputs`, `Unit_ModelOutputsDeathTest`.

Tests ModelOutputs: publishing named values and reading them back.

This file needs no device. `NN::ModelOutputs` in `modules/layers/nn/include/SushiAI/nn/model_outputs.hpp` is
the set of named `ValueID`s one trace of a model publishes. `add()` returns a `Status`, and
`at()` and `only()` return a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelOutputs` is final,
copy-constructible and nothrow move-constructible.

### boxes_and_scores

`boxes_and_scores` returns a set holding "boxes" as value 4 and "scores" as value 9, added in
that order.

### add_without_value

`add_without_value` adds "boxes" with `Graph::K_NO_VALUE`. It is the statement of the death
test.

### StartsEmpty

A new set's `all()` is empty.

### AllKeepsTheOrderOfAdd

After adding "scores", "boxes" and "features", `all()` holds those three entries in that order
with their ids 9, 4 and 0. The order is the order of the calls, not of the names or the ids.

### AddRejectsADuplicateName

A second `add("boxes", ...)` returns `InvalidDescription` with a message containing "'boxes'"
and "boxes, scores". The set still holds two entries and `at("boxes")` is still 4.

### AddRejectsAnEmptyName

`add("", ...)` returns `InvalidDescription` with "needs a name" and adds nothing.

### AddStopsOnNoValue

The case is in the suite `Unit_ModelOutputsDeathTest`. `add("boxes", K_NO_VALUE)` stops the
process with a report containing "'boxes'" and "no value".

### AtReturnsTheNamedValue

`at("boxes")` is 4 and `at("scores")` is 9.

### AtDoesNotMatchAPrefix

`at("box")` and `at("boxes.")` both fail with `InvalidDescription` and the name that was asked
for.

### AtOnAMissingNameListsTheNames

`at("masks")` fails with a message containing "'masks'" and "boxes, scores".

### AtOnAnEmptySetSaysNone

`at("boxes")` on an empty set fails with a message containing "'boxes'" and "none".

### OnlyReturnsTheSingleValue

`only()` on a set of one entry returns that entry's id, 12.

### OnlyFailsOnZero

`only()` on an empty set fails with a message containing "exactly one", "has 0" and "none".

### OnlyFailsOnTwo

`only()` on a set of two fails with a message containing "exactly one", "has 2" and "boxes,
scores".

## modules/core/config/tests/test_model_writer.cpp

Suites: `Unit_ModelWriter`.

Tests `Config::write_model`. Three models are parsed from text in the file. The chain model has
three rows and states neither `from` nor `outputs`. The full model has a `defaults` block, an
attribute of every JSON kind, every form of a `from` entry and every form of an `outputs`
entry. The task model has a `task` with a value of every JSON kind. A round trip parses the
written text and compares the input shape and type, each layer's type, attributes and sources,
the outputs and the task, with a real compared bit for bit. The two example cases locate the
repository through `SA_TEST_FIXTURE_DIR` and report themselves skipped when it is not defined.

### WritesTheInputElementTypeOnlyWhenItIsInt32

The text of a float32 model does not contain "dtype". With `input_dtype` set to int32 the text
contains `"dtype": "int32"` and reads back as int32. With float64 the write fails with
`InvalidDescription` and "float64".

### RoundTripsTheMNISTExample

The model loaded from `applications/examples/mnist_cnn/model.json` round-trips.

### RoundTripsTheYOLO26Example

The model loaded from `applications/examples/yolo26/model.json` round-trips.

### RoundTripsEveryAttributeKindAndEveryFormOfFromAndOutputs

The full model round-trips.

### WritesTheSameTextFromWhatItWrote

The text written from the full model, parsed and written again, equals the first text.

### LeavesOutTheFromAndTheOutputsTheParserFillsIn

The chain model is written as `format_version`, `input` and three rows that hold a `type` and
their attributes, with no `from` and no `outputs`. The whole text is compared.

### WritesTypeThenFromThenTheAttributesAndTheOutputsLast

A model whose rows state `type` after their attributes is written with `type` first, then
`from`, then the attributes, and with `outputs` after `layers`. The whole text is compared.

### WritesADefaultIntoEveryRowAndNoDefaultsBlock

The text of the full model has no `defaults` key. Its first `conv` row carries `bn_epsilon`
0.001 and `bn_momentum` 0.03 after its own keys, and the `leaky_relu` row carries `slope`
0.001.

### RoundTripsAPortOfTheModelInputAtRowZero

The chain model whose row 0 reads the port `image` of `K_MODEL_INPUT` round-trips.

### AnEmptyOutputsListReadsBackAsTheLastRowPublishedWhole

With `outputs` cleared the text has no `outputs` key and reads back with one output: node 2,
with an empty port and an empty name.

### RefusesAModelWithoutLayers

A model with no layers fails with `InvalidDescription` and "no layers".

### RefusesAnInputShapeAModelFileCannotState

An empty input shape fails with "input shape" and "no extents". A shape whose first extent is
the largest int64 and whose second is 2 fails with "input shape" and "extent 1".

### RefusesAttributesThatAreNotAnObject

Null attributes on row 1 fail with `InvalidDescription`, "layer 1", `'relu'` and "null".

### RefusesAnAttributeNamedAsARowKey

An attribute named `type` on row 1 fails with "layer 1" and `'type'`. An attribute named `from`
on row 2 fails with "layer 2" and `'from'`.

### RefusesARowThatReadsNothingOrARowNotAboveIt

Row 1 with no source fails with "layer 1" and "reads nothing". Row 1 reading row 1, row 0
reading row 2 and row 2 reading row -2 each fail with the layer, the row and "not above".

### RefusesAPortOfTheModelInputBelowRowZero

Row 2 reading the port `image` of `K_MODEL_INPUT` fails with `InvalidDescription`, "layer 2",
`'image'` and "row 0".

### RefusesAnOutputAModelFileCannotState

An output at row 3 of three fails with "output 0", "row 3" and "0 to 2". A second output at
row -1 fails with "output 1" and "row -1". A named output without a port fails with "output
0", `'predictions'` and "without a port".

### ForwardsTheRefusalOfAnAttributeJSONCannotSpell

An infinite `rate` on row 2 fails with `InvalidDescription` and a message containing "key
'layers'", "entry 2", "key 'rate'" and "infinite".

### RoundTripsATaskOfEveryValueKind

The task model, whose `task` is an object, round-trips.

### WritesTheTaskAfterTheOutputsWithItsKeysInOrder

In the text of the task model the `outputs` key comes before the `task` key. The task reads
back with the keys `type`, `classes`, `anchors`, `heads`, `suppression`, in that order.

### LeavesOutANullTaskAndReadsItBackNull

The chain model has a null task, its text has no `task` key, and the task reads back null.

### RefusesATaskThatIsNotAnObjectWithAStringType

Four tasks fail with `InvalidDescription` and "task": an array ("an array", "object"), an
object without `type` ("no 'type'"), a `type` of 3 ("'type'", "a whole number") and a `type`
of "" ("'type'", "empty string").

## modules/layers/nn/tests/test_module.cpp

Suites: `Unit_Module`.

Tests Parameter across graphs and the state walk over layers, containers and fields.

This file needs no device. The layer library is host-lane, so the parameter walk, which decides
whether the optimizer sees a layer at all, is testable without hardware.

The file holds six groups: the walk over the library's own layers and containers, the
tutorial's custom layer, the run-time arm of the walk, the per-sample output shape of `Linear`
and the four activations of `modules/layers/nn/include/SushiAI/nn/layers.hpp`, a parameter traced into more than
one graph, and the state walk over parameters, buffers and children.

### AnUntracedLayerYieldsNoParameters

Parameters gain a shape and an id when `forward()` first runs, so the walk over an untraced
model is legitimately empty and not full of blanks.

### ALinearLayerNamesItsWeightAndBias

After a trace, the walk of `Linear{4, 3}` lists `weight` [3, 4] and `bias` [3], and
`parameter_elements` is 15.

### ABiasFreeLayerSimplyHasNoBiasEntry

The walk of `Linear{4, 3, false}` lists `weight` [3, 4] alone.

### SequentialNamesParametersByLayerIndex

The naming follows PyTorch's convention: the index is the layer's position, so a parameterless
activation leaves a gap and "2.weight" still names the third layer. A checkpoint diff against a
reference stays a diff.

### NestedContainersCompoundTheirIndices

The walk of a `Sequential` holding a `Sequential` of a `Linear{4, 4}` and a `ReLU`, then a
`Linear{4, 2}`, lists `0.0.weight` [4, 4], `0.0.bias` [4], `1.weight` [2, 4] and `1.bias`
[2].

### TracingTwiceDoesNotDuplicateParameters

The second trace emits nodes but declares no new parameters, so the value count grows by
exactly the two intermediates.

### The tutorial's custom layer

`MyOffset` is the custom layer of `docs/getting_started/TUTORIAL.md` §6, compiled.

A tutorial whose code sample does not compile is worse than no tutorial, and prose cannot be
made to fail a build. So the sample lives in this file as well, and the two must be kept
identical: if you edit one, edit the other.

TheTutorialsCustomLayerTracesAndIsWalkable and TheTutorialIsRightThatMulWouldNotBroadcast also
pin the two claims §6 makes about the sample: that bias_add is required and is not a stylistic
choice, and that the two ways of getting `fields()` wrong fail differently.

### ParametersAreDeclaredAsGradientRequiringLeaves

After a trace of `Linear{4, 3}`, its weight and bias require a gradient, the weight has the
`Parameter` lifetime, and the input does not require a gradient.

### RolesAndFansAreRecordedForTheInitializer

The weight of `Linear{4, 3}` has the role `Weight`, a fan-in of 4 and a fan-out of 3, and the
bias has the role `Bias`.

### LinearRejectsAnInputWhoseInnerExtentIsWrong

`Linear{4, 3}` over a [2, 7] input throws `Error`.

### TheTutorialsCustomLayerTracesAndIsWalkable

The test checks the role the tutorial gives the parameter, because the role decides the
initialiser: a BIAS starts at zero, a WEIGHT gets a fan-scaled draw.

### TheTutorialIsRightThatMulWouldNotBroadcast

§6 says bias_add is required and not merely preferred, because `mul` goes through
`require_same` and this builder has no implicit broadcast anywhere. That is the whole reason the
sample reads the way it does, so the test asserts it and does not leave it as a remark.

### The run-time arm of the walk

`WalkableStub` exercises the third arm of the walk: a layer whose parameters are enumerated at
run time.

The stub is deliberately not a LayerStack. This file tests the walk, and the walk's contract is
`walk_parameters(prefix, ParameterVisitor)`. Anything that satisfies it is enumerated through
it, whether or not it is one of the library's own types. Testing the walk against LayerStack
would tie its test to a type that arrived two commits later.

### AWalkableLeafIsEnumeratedByItsOwnWalk

The walk of `WalkableStub` lists `weight` [2, 3] and `bias` [2], with a parameter count of 2
and 8 elements.

### AContainerPrefixesAWalkableChildTheSameWayItPrefixesAnyOther

The two arms compose: the container supplies "0.", the walkable child supplies the leaf names,
and nothing about the join changed.

### TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt

`modules/layers/nn/include/SushiAI/nn/layers.hpp` asserts this for Linear and Sequential at build time. The test
restates it so that the property is visible in the test that covers the walk, and so that it
names the detector and not the arm's consequences.

### Per-sample output shapes

Every layer type has `Shape output_shape(const Shape&) const`, which maps the per-sample shape
that enters the layer to the one that leaves it. A layer factory asks the layer for that shape
and does not restate the rule. The six cases below hold `Linear`, `ReLU`, `GELU`, `Tanh` and
`Sigmoid` to it, and compare it with what `forward()` traces.

### traced_shape

The file-local `traced_shape` traces a layer over a [2, features] input and returns the
result's shape.

### LinearReportsItsOutFeaturesAsThePerSampleOutputShape

`Linear{4, 3}` maps [4] to [3].

### LinearsOutputShapeAgreesWithItsTrace

The traced shape of `Linear{4, 3}` over a batch of two is [2, n], where n is the one extent
of `output_shape([4])`.

### LinearRefusesAPerSampleShapeOfTheWrongRank

`output_shape([4, 2, 1])` throws with a message containing "Linear(4, 3)", "[4, 2, 1]" and
"[4]". The first extent equals `in_features`, so only the rank check can refuse this shape.

### LinearRefusesAPerSampleShapeOfTheWrongWidth

`output_shape([7])` throws with a message containing "Linear(4, 3)", "[7]" and "[4]".

### AnActivationReportsItsInputShapeAsItsOutputShape

For the shapes [5] and [4, 4, 3], each of the four activations returns the shape it was given.
The rank-3 shape is the one a convolution hands on.

### AnActivationsOutputShapeAgreesWithItsTrace

The traced shape of each activation over a [2, 5] input is [2, n], where n is the one extent
of its `output_shape([5])`.

### A parameter across graphs

The cases from AParameterTracedIntoTwoGraphsHasALeafInEach to ARoleDeclareFillsByItsRole cover
`NN::Parameter` in `modules/layers/nn/include/SushiAI/nn/module.hpp`. A parameter holds a `Graph::StateID` and
no graph. `declare()` adds a leaf to the graph of the builder it is given and binds the leaf to
the state there, so one parameter has one leaf in each graph it was traced into. `id_in(graph)`
returns that graph's leaf.

The file-local `K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is
stored and read back unchanged.

### declare_weight

`declare_weight` declares a [2, 3] parameter of role `WEIGHT` named "weight" through a given
builder and returns its leaf.

### AParameterTracedIntoTwoGraphsHasALeafInEach

The first graph holds a padding input, so the parameter's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, `state_leaf` of the
parameter's state returns the same in each graph, and the second leaf has the lifetime
`PARAMETER`.

### AParameterHasNoLeafInAGraphItWasNotTracedInto

Before any `declare()`, `id_in` is `K_NO_VALUE` and `state()` is `K_NO_STATE`. After one
`declare()` the state is set, and `id_in` of a graph that was never traced into is still
`K_NO_VALUE`.

### ASecondDeclareInOneGraphReturnsTheFirstLeaf

A second `declare()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

### ADeclareWithAnotherShapeIsRefused

After a [2, 3] declaration, a `declare()` with [4, 3] throws with a message containing
"'weight'", "[2, 3]" and "[4, 3]", through the same builder and through a builder over a second
graph. The second graph holds no value and the parameter's shape stays [2, 3].

### ADeclareWithAnotherElementTypeIsRefused

After a FLOAT32 declaration, a `declare()` with FLOAT64 throws with a message containing
"'weight'" and both type names, in the same graph and in a second one. The second graph holds
no value and the parameter's type stays FLOAT32.

### ALaterDeclareKeepsTheFirstDescription

A parameter declared as a `SCALE` with a constant fill of 1.5 and fans of 4 is declared into a
second graph as a `WEIGHT` with a fan-scaled fill and fans of 9. Its fill, role and both fans
are still the first ones, and it has a leaf in the second graph.

### ACopiedLayerHasANewStateID

A copy of a traced `Linear` has a weight whose state is `K_NO_STATE` and whose `id_in(graph)`
is `K_NO_VALUE`, and `declared()` is true. Once the copy is traced, its weight has a leaf in the
graph and a state that differs from the original's, and its weight and bias leaves differ from
the original's.

### AMovedLayerKeepsItsStateID

A `Linear` move-constructed from a traced one has the same weight state and the same weight
leaf in the graph. Tracing it again adds two values, the two activations, and no parameter
leaf.

### ARoleDeclareRecordsTheFillOfItsRole

A `declare()` that states a role and no fill records a fan-scaled fill for a `WEIGHT` and a
constant fill of 0.0 for a `BIAS`.

### AnInitDeclareRecordsTheStatedFillAndTheStatedRole

A `declare()` that states both keeps both: a `BIAS` with a constant fill of 1.5 and a fan-in
of 4, and a `WEIGHT` with a fan-scaled fill. The bias's leaf requires a gradient.

### ANormalFillKeepsItsDeviation

`ParameterInit::normal(0.02)` has the kind `Normal`, the deviation 0.02 and the constant 0.
`fan_scaled()` has the kind `FanScaled` and the deviation 0. `constant_value(1.5)` has the
kind `Constant`, the constant 1.5 and the deviation 0.

### AnEmbeddingRoleWithoutAFillIsFanScaled

A parameter declared with the role `Embedding` and no fill reports that role and an init of
the kind `FanScaled`.

### ALinearDeclaresTheFillItStates

After a trace, the weight of a default `Linear{3, 4}` has an init of the kind `FanScaled`.
With `weight_init` set to `normal(0.02)` the weight's init is `Normal` with the deviation
0.02, its role is still `Weight`, and the bias's init is `Constant`.

### ARoleDeclareFillsByItsRole

The three roles give three fills: fan-scaled for `WEIGHT`, a constant 0.0 for `BIAS` and a
constant 1.0 for `SCALE`.

### The state walk

The cases from WalkStateYieldsTheThreeFieldKindsInDeclarationOrder on cover `NN::walk_state`
and the two walks derived from it. A `fields()` list holds three kinds of entry: `NN::field`
for a `Parameter`, `NN::buffer` for a `Buffer` and `NN::child` for a member that is itself
walked. `walk_state` hands each parameter and each defined buffer to an `NN::StateVisitor`.
`for_each_parameter` and `for_each_buffer` each yield one kind.

### StatePair

`StatePair` lists a parameter "scale" and a buffer "mean". `describe()` declares the parameter
as a [4] `SCALE` and defines the buffer as [4] zeros.

### MixedState

`MixedState` lists, in this order, a buffer "count", a `StatePair` child "pair", a parameter
"offset" and a buffer "unset". `describe()` defines "count" as a [1] INT64 buffer, describes
the pair and declares "offset" as a [4] `BIAS`. It leaves "unset" undefined.

### walk_all

`walk_all` collects what `walk_state` yields under a prefix as "parameter name:extents" and
"buffer name:extents" strings.

### buffer_names

`buffer_names` collects the names `NN::for_each_buffer` yields.

### WalkStateYieldsTheThreeFieldKindsInDeclarationOrder

The walk of a described `MixedState` yields "buffer count:[1]", "parameter pair.scale:[4]",
"buffer pair.mean:[4]" and "parameter offset:[4]". Parameters and buffers interleave in the
order `fields()` lists them.

### WalkStateJoinsItsPrefixToEveryName

Under the prefix "model.3" the same four entries are named "model.3.count",
"model.3.pair.scale", "model.3.pair.mean" and "model.3.offset".

### ForEachParameterNeverYieldsABuffer

`for_each_parameter` yields "pair.scale:[4]" and "offset:[4]". `parameter_count` is 2 and
`parameter_elements` is 8.

### ForEachBufferNeverYieldsAParameter

`for_each_buffer` yields "count" and "pair.mean", and `buffer_count` is 2.

### AnUndefinedBufferIsSkipped

Before `describe()` no buffer is yielded. After it two are. Once `define()` is called on
"unset", the walk yields "count", "pair.mean" and "unset".

### ADefinedBufferIsWalkedWithoutALeaf

The buffer "count" has no leaf in the graph, `id_in` is `K_NO_VALUE`, and it is one of the two
buffers counted.

### SelfWalkingStub

`SelfWalkingStub` has a member `walk_state(prefix, visitor)` and no `fields()`. It yields its
buffer "level" before its parameter "gain".

### StubPair

`StubPair` lists a `SelfWalkingStub` as the child "own" and a `WalkableStub` as the child
"legacy".

### ATypesOwnWalkStateIsTheWalk

`Detail::HasStateWalk` is true for `SelfWalkingStub` and false for `Linear`. The stub's walk
under "head" yields "buffer head.level:[3]" and then "parameter head.gain:[3]", the order its
member function states. `for_each_parameter` yields "gain:[3]" and `for_each_buffer` yields
"level".

### AChildFieldNamesEveryKindOfWalkableChild

The walk of a `StubPair` yields "buffer own.level:[3]", "parameter own.gain:[3]", "parameter
legacy.weight:[2, 3]" and "parameter legacy.bias:[2]". A `child` entry prefixes a type with its
own `walk_state` and a type with `walk_parameters` alike.

### AWalkParametersTypeYieldsNoBuffer

The walk of a `WalkableStub` yields its weight and its bias as parameters, and `buffer_count`
is 0.

### ALayerAdapterYieldsItsLayersParametersUnderThePrefix

A traced `LayerAdapter<Linear>` walked under the prefix "2" yields "parameter 2.weight:[3, 4]"
and "parameter 2.bias:[3]", and `buffer_count` is 0.

### TwoFieldsWithOneNameAreDetected

`Detail::has_unique_names` is checked in four `static_assert` lines. It is true for
`MixedState::fields()` and for an empty tuple. It is false for two `field` entries both named
"weight", and for a `buffer` and a `child` both named "pair". At run time it is true for
`Linear::fields()`. `walk_state` asserts the same predicate.

### ASequentialOfModulesKeepsItsBuilderForward

Three `static_assert` lines hold that `Sequential<Linear, ReLU>` is a `Module`, that it stays
one when nested in another `Sequential`, and that its `forward` over a `GraphBuilder` returns
a `Graph::ValueID`. A nested model of `Linear{4, 8}`, `ReLU` and `Linear{8, 3}` traces to
[2, 3] through the builder and walks as "0.0.weight:[8, 4]", "0.0.bias:[8]", "1.weight:[3, 8]"
and "1.bias:[3]".

## modules/layers/nn/tests/test_module_list.cpp

Suites: `Unit_ModuleList`, `Unit_ModuleListDeathTest`.

Tests ModuleList: its storage, its index names and its place in the state walk.

This file needs no device. `NN::ModuleList<M>` in `modules/layers/nn/include/SushiAI/nn/module_list.hpp` holds
children of one type whose count is known at run time.

Three `static_assert` lines at the top of the file hold that `ModuleList<Linear>` is a
`ModuleContainer` and that neither `Detail::HasStateWalk` nor `Detail::HasParameterWalk`
detects it, so the walk reaches it through `visit_children`.

### Stage

`Stage` lists a `ModuleList<Linear>` under the child name "blocks" and a `Linear{4, 2}` under
"head".

### Counter

`Counter` holds one `NN::Buffer` listed as "count", so a list of it carries state that is not
a parameter.

### parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields as "name:extents" strings.

### read_past_the_end

`read_past_the_end` builds a list of two layers and reads `list[2]`. It is the statement of
the death test.

### ANewListIsEmpty

A new list has size 0, `visit_children` calls its visitor zero times, and `begin()` equals
`end()`.

### PushBackAppendsInOrder

After two `push_back` calls the size is 2, element 0 has 8 output features and element 1
has 3. Indexing a const reference to the list reads element 1's 8 input features.

### IterationFollowsIndexOrder

A range-for over three layers reads the output widths 8, 3 and 1.

### VisitChildrenPairsEachChildWithItsIndex

`visit_children` over two layers passes the indices 0 and 1 with the addresses of `list[0]` and
`list[1]`.

### TheWalkNamesChildrenByIndex

Two traced layers, the second without a bias, walk as "0.weight:[8, 4]", "0.bias:[8]" and
"1.weight:[3, 8]", and `parameter_count` is 3.

### AListUnderAChildFieldIsNamedByFieldThenIndex

A traced `Stage` with two blocks walks as "blocks.0.weight", "blocks.0.bias",
"blocks.1.weight", "blocks.1.bias", "head.weight" and "head.bias", each with its extents.

### TheWalkNamesBuffersByIndex

Of three `Counter` children, 0 and 2 have their buffer defined. `for_each_buffer` yields
"0.count" and "2.count": child 1 yields nothing and still keeps its index. `parameter_count`
is 0.

### GrowingTheListKeepsATracedChildsStateIdentity

The first child is traced, then 64 more layers are appended, which moves a one-element
vector's storage at least once. The first child's weight keeps its state and its leaf in the
graph, and its bias keeps its leaf in the graph.

### AnIndexPastTheEndIsFatal

The case is in the suite `Unit_ModuleListDeathTest`. `list[2]` on a list of two stops the
process with a report containing "index 2" and "holds 2 children".

## modules/layers/nn/tests/test_node.cpp

Suites: `Unit_Node`, `Unit_NodeDeathTest`.

Tests ModuleNode, LayerNode and the buffer check LayerAdapter makes at compile time.

This file needs no device. `NN::Node` in `modules/layers/nn/include/SushiAI/nn/node.hpp` is the seam a
`ModelGraph` holds its rows through: a node declares its ports, reads a span of values and
appends its results. `NN::ModuleNode<L>` in `modules/layers/nn/include/SushiAI/nn/module_node.hpp` wraps a
layer of any of the three forms, and `NN::LayerNode` in `modules/layers/nn/include/SushiAI/nn/layer_node.hpp`
wraps an `NN::Layer`. Both return a `Status` from `forward`, and an `Error` a layer throws
comes back as a failure.

### BufferedStub

`BufferedStub` lists one `Buffer` as "running" and defines it as [4] zeros in its builder
`forward`.

### BufferedParentStub

`BufferedParentStub` lists a `BufferedStub` as the child "inner".

### ModalStub

`ModalStub` has the modal pair, counts the calls of each form and returns its input.

### TraceStub

`TraceStub` has a trace `forward` that records the mode of each trace it is given.

### ShapelessStub

`ShapelessStub` has `fields()` and a builder `forward` and no `output_shape`.

### RefusingStub

`RefusingStub` records an `InvalidDescription` failure on the trace and returns its input.

### RefusingThenThrowingStub

`RefusingThenThrowingStub` records an `OutOfMemory` failure on the trace and then throws an
`Error` with a different message.

### node_of

`node_of` calls `ModuleNode<L>::create` over a layer and an input shape, expects success and
returns the node.

### span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

### state_names

`state_names` collects the name of every parameter and buffer a node's `walk_state` yields
under a prefix.

### make_test_conv2d

`make_test_conv2d` calls `NN::make_conv2d` with 5 filters and a 3x3 kernel, the layer a
`conv2d` row of a model file builds.

### The two predicates at compile time

`LayerAdapter` asserts that `Detail::HoldsBufferField` is false for its layer. The predicate is
true for `BufferedStub`, for `BufferedParentStub`, for `Sequential<Linear, BufferedStub>` and
for `ModuleList<BufferedStub>`, and false for `Linear` and `Sequential<Linear, ReLU>`.

`ModuleNode` asserts that `Detail::HasOutputShapeOfInput` is true for its layer. The predicate
is true for `Linear` and `TraceStub` and false for `ShapelessStub`.

### ModuleNodeDeclaresOnePortWithTheLayersOutputShape

The node of `Linear{6, 8}` over [6] has one port named "output" of shape [8].

### ALayerNodeOfEitherKindReadsOneValue

`input_count()` is 1 for a `ModuleNode<Linear>` and for a `LayerNode`.

### ModuleNodeCreateReturnsTheFailureOfAShapeTheLayerRefuses

`ModuleNode<Linear>::create(Linear{6, 8}, [5])` returns `InvalidDescription` with a message
containing "Linear(6, 8)", "[5]" and "[6]". `Linear::output_shape` throws for that shape, and
`create` returns the throw as a failure.

### ModuleNodeForwardReturnsTheFailureOfALayerThatThrows

`forward` over a [2, 5] value returns `InvalidDescription` with "Linear(6, 8)" and "[2, 5]",
and `outputs` stays empty.

### ModuleNodeForwardReturnsTheFailureALayerRecorded

The node of `RefusingStub` returns the failure the stub recorded on the trace, and `outputs`
stays empty although the stub returned an id.

### ModuleNodeForwardReturnsTheRecordedFailureBeforeALaterThrow

The node of `RefusingThenThrowingStub` returns the `OutOfMemory` failure with "the stub's
recorded cause". The `Error` thrown after it is not what the caller sees.

### ModuleNodeForwardAppendsTheLayersResult

`outputs` holds one id before the call and two after it. The first is untouched and the second
is a value of shape [2, 8].

### ModuleNodeTracesAModalLayerInTheModeOfTheTrace

Under a training trace the stub counts one training call and no evaluation call. After an
evaluation trace it counts one of each.

### ModuleNodeHandsTheTraceToATraceModule

A training trace and then an evaluation trace leave `TraceStub::modes` as Training,
Evaluation, and `outputs` as the input id twice.

### ModuleNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("stem", ...)` of the node of `Linear{6, 8}` yields "stem.weight" and
"stem.bias".

### ModuleNodeWalksABufferOfTheLayer

After a trace, `walk_state("bn", ...)` of the node of `BufferedStub` yields "bn.running".

### ModuleNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

### LayerNodeReportsTheShapeOfTheLayerItWraps

A `LayerNode` over the conv2d layer built for [8, 8, 2] has one port named "output" whose
shape is the layer's `output_shape()`, [6, 6, 5].

### LayerNodeForwardAppendsAValueShapedLikeItsPort

`forward` over a [2, 8, 8, 2] value appends one id of shape [2, 6, 6, 5].

### LayerNodeForwardReturnsTheFailureOfALayerThatThrows

A `LayerNode` over `LayerAdapter<Linear>` given a [2, 5] value returns `InvalidDescription`
with "Linear(6, 8)" and "[2, 5]", and `outputs` stays empty.

### LayerNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("0", ...)` yields "0.weight" and "0.bias".

### LayerNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

### LayerNodeRejectsNoInput

`forward` given an empty span returns `InvalidDescription` with "input count is 1" and "given
is 0".

### LayerNodeOverANullLayerIsFatal

The case is in the suite `Unit_NodeDeathTest`. Constructing a `LayerNode` from a null pointer
stops the process with a report containing "LayerNode" and "null layer".

## modules/layers/nn/tests/test_node_registry.cpp

Suites: `Unit_NodeRegistry`.

Tests `NN::NodeRegistry`, `NN::register_layers`, `NN::make_node` and `NN::build_model` over
the rows `register_layers` adds. The file needs no device. `make_node` looks a type up among
the registry's node rows and then among its layer rows, and wraps a layer in an
`NN::LayerNode`. `build_model` turns a `Config::ModelSpec` into an `NN::ModelGraph`. Each
returns a `Result`. `MinimalCase`, `TraceSummary`, `span_of`, `make_minimal` and
`trace_summary` come from `tests/common/test_utils.hpp`.

### K_NORM

`K_NORM` is `NormOptions{1e-3, 0.03}`, the epsilon and the momentum the defaults of
`K_BRANCHING_MODEL` state.

### MinimalCase

`MinimalCase` holds a node registry key, a row's attributes as JSON, the per-sample shapes the
row reads and the per-sample shape of the node's one port.

### K_MINIMAL_CASES

`K_MINIMAL_CASES` has one `MinimalCase` per node row of `registered()`, in row order:
batchnorm2d, concat, add, embedding, layernorm and dropout.

### LayerCase

`LayerCase` holds a layer registry key, a row's attributes as JSON and the per-sample shape
the row reads.

### K_LAYER_CASES

`K_LAYER_CASES` has one `LayerCase` per layer row of `registered()`, in row order.

### K_INPUT_JOINING_MODEL

`K_INPUT_JOINING_MODEL` is a version 2 model over [8, 8, 3] with three rows: a `conv2d` of 4
filters and kernel 1, a `silu`, and a `concat` whose `from` is `["input", -1]`.

### K_BRANCHING_MODEL

`K_BRANCHING_MODEL` is a version 2 model over [16, 16, 3] with a `defaults` block that gives
every `batchnorm2d` row `bn_epsilon` 0.001 and `bn_momentum` 0.03. Its seven rows are a
`conv2d` of 8 filters, kernel 3 and stride 2; a `conv2d` of 16 filters with the same window;
an `upsample` of scale 2; a `concat` whose `from` is `[-1, 0]`; a `conv2d` of 8 filters and
kernel 1; a `batchnorm2d`; and a `silu`.

### TraceSummary

`TraceSummary` holds the operation list of one traced graph, each published output as "name
[extents]" and each piece of state as "name [extents] parameter" or "name [extents] buffer".

### span_of

`span_of` views a vector of shapes as the `NN::ShapeSpan` of one row.

### make_minimal

`make_minimal` calls `NN::make_node` for a `MinimalCase` with given attributes and the context
"layer 0 '<type>'".

### trace_summary

`trace_summary` traces a model under a training trace over a batch of 2 samples and returns
its `TraceSummary`.

### registered

`registered` returns a registry that `register_layers` filled.

### build_branching_model_by_hand

`build_branching_model_by_hand` builds the graph `K_BRANCHING_MODEL` describes through
`ModelGraph::add`, with the nodes named "0" to "6", and publishes the last node through
`output_ports`.

### named_output

`named_output` returns an `OutputSpec` that publishes the port `NN::K_SINGLE_PORT_NAME` of a
row under a name.

### RegisterLayersAddsTheRowsInOrder

`register_layers` gives the layer rows linear, gelu, relu, tanh, sigmoid, leaky_relu,
unflatten, conv2d, maxpool2d, avgpool2d, global_avgpool, flatten, upsample, silu, mish and
slice_channels, and the node rows batchnorm2d, concat, add, embedding, layernorm and dropout,
each list in that order.

### ATypeAddedTwiceIsRefused

Adding a node row named "concat", and one named "relu", each fail with `InvalidDescription`,
"the type '<name>'" and "already". The node and layer row counts do not change.

### AnEmptyRegistryKnowsNoType

`make_node` of "silu" over an empty registry fails with `InvalidDescription`, "layer 0",
"the type 'silu'" and "does not have", and its message ends with
"the node types are: ; the layer types are: ".

### EveryRowHasAMinimalCase

`K_MINIMAL_CASES` has as many entries as `registered()` has node rows, and entry `i` names
the type of row `i`.

### EachRowBuildsFromAMinimalObject

For every case, `make_node` returns a node with one port named `NN::K_SINGLE_PORT_NAME` whose
shape is the one the case lists.

### EachRowRejectsAnUnknownKey

For every case, the attributes `{"no_such_key": 1}` fail with `InvalidDescription` and a
message containing the row's context and "'no_such_key'".

### AConcatRowNamesItselfWhenItsInputsDisagree

A `concat` row over [4, 4, 2] and [8, 4, 2] fails with a message containing "layer 7
'concat'", "[8, 4, 2]" and "H and W".

### ANodeTypeBuildsItsOwnNode

`make_node("concat", ...)` over [4, 4, 2] and [4, 4, 3] returns an `NN::Concat` and not a
`LayerNode`.

### ALayerTypeIsWrappedInALayerNode

`make_node("upsample", ...)` with scale 2 over [4, 4, 2] returns a `LayerNode` with one port
named `K_SINGLE_PORT_NAME` of shape [8, 8, 2]. The case then walks `K_LAYER_CASES`, which has
one entry per layer row of the registry in order. Every type builds a `LayerNode`, and every
type given `{"no_such_key": 1}` fails with "'no_such_key'", which shows the layer's own
factory ran.

### ALayerTypeRefusesTwoInputs

A `relu` row over two shapes fails with a message containing "layer 3 'relu'", "reads 2
values" and "reads one".

### ALayerFactorysRefusalComesBackAsAStatus

A `linear` row with no attribute fails with a message containing "layer 4 'linear'" and
"'units'". A `linear` row over [8, 8, 2] fails with the context, "[8, 8, 2]" and "'flatten'".
The layer factories throw, and `make_node` returns each throw as an `InvalidDescription`
failure.

### AMaxPoolRowReadsItsTrailingPadding

Over [13, 13, 2] with `kernel` 2 and `stride` 1, the maxpool2d row gives a port of
[13, 13, 2] for `trailing_padding` 1, [13, 12, 2] for [1, 0] and [12, 12, 2] without the key.

### AnAvgPoolRowRefusesTrailingPadding

The avgpool2d row with `trailing_padding` fails with `InvalidDescription`,
"layer 0 'avgpool2d'" and "'trailing_padding'".

### AMaxPoolRowRefusesANegativeTrailingPadding

The maxpool2d row with `trailing_padding` -1 fails with `InvalidDescription`,
"layer 0 'maxpool2d'", "non-negative trailing padding" and "-1,-1".

### ASliceChannelsRowReadsItsOffsetAndCount

Over [4, 4, 64], the slice_channels row with `offset` 32 and `count` 32 gives a port of
[4, 4, 32]. With `offset` 48 and `count` 32 it fails with `InvalidDescription`,
"layer 0 'slice_channels'", "offset=48" and "count=32", and without `count` with "count".

### AnUnknownTypeListsBothTables

`make_node("conv3d", ...)` fails with a message containing "layer 5 'conv3d'", "the type
'conv3d'", "does not have", "node types" and "layer types". The message also contains every
type of the registry's node rows and of its layer rows in quotes.

### AFileAndAHandBuiltGraphAgree

`build_model` of `K_BRANCHING_MODEL` and `build_branching_model_by_hand` both hold seven
nodes. Their training traces have equal operation lists, equal outputs and equal state
records. The one output is "output [2, 8, 8, 8]". The state holds 7 records, from
"model.0.weight [8, 3, 3, 3] parameter" to "model.5.running_var [8] buffer".

### ARowReadsItsSourcesInFileOrder

In the model built from `K_BRANCHING_MODEL`, node 3 declares [8, 8, 24]: row 2's 16 channels
and then row 0's 8. Node 2 declares [8, 8, 16] and node 1 [4, 4, 16].

### OutputsArePublishedUnderTheirStatedNames

With the outputs set to row 4 as "mixed" and row 0 as "stem", the trace publishes "mixed
[2, 8, 8, 8]" and "stem [2, 8, 8, 8]" in that order.

### AnOutputWithoutANameKeepsThePortsName

An output of row 1 with an empty name is published as "output [2, 4, 4, 16]".

### ASourceMayNameAPort

A source that names the port `K_SINGLE_PORT_NAME` of the row above builds. A source that names
"no_such_port" fails with "layer 1 'conv2d'" and "'no_such_port'".

### ALaterRowMayReadTheModelInput

Row 2 of `K_INPUT_JOINING_MODEL` parses with `Config::K_MODEL_INPUT` as its first source. The
built node 2 declares [8, 8, 7]: the input's 3 channels and then row 1's 4.

### ASourceCannotNameAPortOfTheModelInput

A source that is the model input and names a port fails with a message containing "layer 2
'concat'", "the model input has no port" and "'output'".

### ARefusedRowNamesItsIndexAndType

Row 4 with `filters` 0 fails with a message containing "layer 4 'conv2d'",
"positive channel counts" and "out=0".

### ARowThatReadsARowNotAboveItIsRefused

The specs here are changed after parsing, so they hold what the parser would have refused. A
source of row 3 on row 1 fails with "layer 1 'conv2d'", "row 3" and "not above". A source of -2
on row 2 fails with "layer 2 'upsample'" and "row -2". A row with no source fails with "layer 2
'upsample'" and "no source".

### AnEmptyOutputListPublishesTheLastRowWhole

A spec whose `outputs` were cleared builds a model whose `output_shape` is [8, 8, 8], the same
as that of the model built from the parsed spec.

### AWrongOutputListIsRefused

A spec with no layer fails with "no layers". An output of row 7 fails with "output 0", "row 7"
and "rows 0 to 6". A named output with an empty port fails with "output 0", "'mixed'" and
"without a port". An output of the port "no_such_port" fails with "output 0" and
"'no_such_port'". Two outputs named "mixed" fail with "output 1" and "'mixed'".

## modules/layers/train/tests/test_objective.cpp

Suites: `Unit_Objective`, `Unit_ObjectiveScore`.

Tests the objective seam: the leaves it declares, the loss it emits, the resulting IR, and what
`feed` and `measure` do with a batch.

The `Unit_Objective` cases need no device. Declaring leaves and emitting a loss are IR-to-IR,
so that part of the seam is testable without an Engine. The `Unit_ObjectiveScore` fixture is
an `EngineTest`, because `feed` and `measure` read tensors.

The test that carries this file is BuildGraphAndBuildClassifierProduceTheSameIR. The objective
seam's whole promise is that it changed no numbers. The way to check that is not "the loss still
falls". It is that the two entry points emit the same graph, node for node, in the same order. A
graph comparison fails on a reordering that a loss comparison would absorb.

### BuildGraphAndBuildClassifierProduceTheSameIR

`build_classifier` and `build_training_graph` with a `CrossEntropyObjective`, each over its
own model of the same widths, give graphs with the same value count, node count and
description, and specs with the same `inputs`, `targets`, `logits` and `loss` ids.

### TheTargetLeafIsDeclaredWhereItAlwaysWas

The target leaf is declared between the input leaf and the model's own trace. Value ids are
positions, so declaring it anywhere else renumbers the whole graph. Every regression test that
pins a node count or a byte-for-byte result would then compare two different graphs while
appearing to compare one.

### CrossEntropyRequiresTheLabelsRole

The training graph under `CrossEntropyObjective(4, 3)` requires one role: `K_ROLE_LABELS`
with an empty sample shape, `Int32` and 3 classes.

### LeavesKeepEveryRequiredRoleInOrder

`ObjectiveLeaves` starts with no required role. After `require` of labels, boxes and labels
again it lists the three in that order, the second with the sample shape [8, 4], and still
has no leaf.

### AccuracyIsReadFromTheLabelsRole

`measure` over the fixture's logits and the labels 1, 0, 0 reports one metric, `accuracy`,
with `is_fraction` set and the value 2/3.

### ABatchWithoutTheLabelsRoleIsRefused

`measure` over a batch without targets fails with `InvalidDescription` and "'labels'", and
adds no metric.

### AccuracyReadsOneOutput

`measure` over a step with two outputs fails with `InvalidDescription`, "one model output"
and "published 2".

### FeedWritesTheOneHotRowsOfTheLabels

`feed` of the labels 1, 0, 0 over 2 classes writes 0, 1, 1, 0, 1, 0 into the tensor bound to
the `targets` leaf.

### AnObjectiveFedByItsOwnStepRefusesABatch

`feed` of the base class fails with `Unsupported`, "'step_fed'" and "fed by its own step".
Its `measure` succeeds and adds no metric.

### ALeafWithoutATensorIsNamed

`find_leaf_tensor` fails with `InvalidDescription` for a leaf that was not declared
("no leaf 'collocation'") and for a declared leaf the pool has no tensor for ("no tensor",
"'targets'").

### TheObjectiveOwnsTheOutputShapeCheck

What shape an output must be is the objective's question, since a detector's is not a
classifier's. The message therefore comes from the objective and not from the graph builder.

### LeavesRefuseADuplicateRoleAndNameAnAbsentOne

Adding the leaf `targets` a second time raises an error holding "'targets'" and "twice".
`at("collocation")` raises one holding "'collocation'" and "'targets'". `has` is true for
`targets` and false for `collocation`, and the set holds one leaf.

### CrossEntropyReadsTheOnlyOutput

The model of `MLP` with 6, 8 and 3 units publishes one output, under the name
`NN::K_SINGLE_PORT_NAME`, which is "output". The case asserts that the outputs hold that one
entry, that the loss node is an `NLL_LOSS` reading a `LOG_SOFTMAX`, and that the `LOG_SOFTMAX`
reads the value published under that name.

### CrossEntropyRefusesTwoOutputs

The model of `Test::build_two_output_model`, from `tests/common/test_utils.hpp`, publishes
"features" and "logits". The case asserts that
`build_training_graph` returns `StatusCode::InvalidDescription` and that the message names both
outputs.

### AnEmptyLeafSetSaysSoRatherThanListingNothing

`at("targets")` on an empty leaf set raises an error holding "'targets'" and "(none)".

### TheFactoryBuildsCrossEntropyOverTheContext

`make_cross_entropy` for a batch of 4 and 3 classes gives an objective named "cross_entropy"
whose `targets` leaf has the shape [4, 3].

### CrossEntropyTakesNoKeysAtAll

Cross-entropy has no hyperparameters, so a key written onto it is an error and not a line whose
author believed it did something.

### CrossEntropyNeedsTheClassCountOfTheData

`make_cross_entropy` for a context without a class count fails with `InvalidDescription`,
"objective", "class count" and "0 classes".

## modules/layers/train/tests/test_objective_context.cpp

Suites: `Unit_ObjectiveContext`.

Tests `Train::list_output_shapes` and `Train::find_output_shape`. No case needs an engine.

### ListsEveryOutputWithItsExtentsInPublishedOrder

For a `ModelOutputs` holding `coarse` of shape [2, 4, 4, 14] and `fine` of shape [2, 8, 8, 14],
the list has two entries with those names and shapes, in that order.

### AnEmptyOutputSetListsNothing

An empty `ModelOutputs` over an empty graph gives an empty list.

### AProbeTraceGivesTheShapesWithoutAnEngine

For a `ModelGraph` of two `Linear` nodes, 6 to 5 and 5 to 3, that publishes both,
`Train::build_probe_graph` over [4, 6] gives a graph whose listed outputs are `hidden` of shape
[4, 5] and `logits` of shape [4, 3].

### FindsAnOutputByItsName

`find_output_shape` returns the entry named `fine` with its shape, and null for `medium`.

## modules/layers/train/tests/test_objective_registry.cpp

Suites: `Unit_ObjectiveRegistry`.

Tests `Train::ObjectiveRegistry`, `Train::register_objectives` and `Train::make_objective`. The
file's stand-in factory returns a `CrossEntropyObjective` when the context holds a task with a
`type`, one output named `logits`, the input sample [6], a `weight` attribute and a registry
that holds the row `stand_in`. Otherwise it fails with `Unsupported` and names the field.

### TheLayersTierAddsItsFiveObjectives

After `register_objectives` the registry holds five rows in the order `cross_entropy`,
`binary_cross_entropy`, `multi_label`, `mean_squared_error`, `sum`. `find("cross_entropy")`
returns a row and `find("anchor")` is null.

### ATypeIsHeldOnce

A second `add` of `stand_in` fails with `InvalidDescription`, `'stand_in'` and "already
registered", and the registry holds one row.

### TheFactoryReceivesTheContextAndItsRegistry

`make_objective` for `stand_in` with the attribute `weight` returns an objective named
"cross_entropy".

### AnUnknownTypeListsWhatTheRegistryHolds

`make_objective` for `yolo` fails with `InvalidDescription` and a message containing `'yolo'`,
`'cross_entropy'` and `'stand_in'`.

### TheFailureOfAFactoryIsReturned

`make_objective` for `stand_in` with empty attributes and the label "the objective" fails with
`Unsupported`, "the objective" and "other attributes".

## modules/layers/train/tests/test_one_hot.cpp

Suites: `Unit_OneHot`.

Tests `Train::fill_one_hot`. The fixture needs an engine. The labels are an int32 tensor of one
extent, and the rows hold 7 in every element before a fill.

### EveryRowHoldsOneInItsLabelsColumn

The labels 2, 0, 1, 2 fill a [4, 3] float32 tensor with 1 in each row's label column and 0
elsewhere.

### AFloat64LeafReceivesTheSameRows

The same labels fill a [4, 3] float64 tensor with the same rows.

### ASecondFillReplacesTheFirst

After a fill with the labels 0, 0 and a second with 1, 0, a [2, 2] tensor holds 0, 1, 1, 0.

### ALabelOutsideTheClassesIsRefusedAndNothingIsWritten

Over three classes the labels 0, 3 fail with `Incompatible`, "row 1", "is 3" and "3 classes",
and the labels -1, 0 fail with "row 0", "is -1" and "3 classes". The rows still hold six 7s.

### AnotherLabelCountIsRefusedWithBothShapes

Three labels for a [4, 3] tensor fail with `Incompatible`, "[3]" and "[4, 3]".

### LabelsOfAnotherElementTypeAreRefused

Float32 labels fail with `Incompatible`, "float32" and "int32".

### RowsThatAreNotRealAreRefused

An int32 [2, 3] tensor of rows fails with `Incompatible`, "int32", "float32" and "float64".

## modules/core/core/tests/test_ops.cpp

Suites: `Unit_Ops`.

Tests that the operation table is complete and that the reverse lookup resolves names.

This file needs no device. The uniqueness proof already runs as a static_assert inside
`modules/core/core/include/SushiAI/core/ops.hpp`. These tests cover what that assert cannot: that the table is
complete, and that the reverse lookup behaves.

### EveryIdentityHashesDistinctly

No two entries of `Ops::K_ALL` have the same `OpID` value.

### EveryNameCarriesTheLibraryPrefix

The prefix keeps a SushiAI identity from colliding with SushiBLAS's own (blas.lvl3.gemm and
friends) in the profiler's report.

The test calls `rfind(prefix, 0)` and not `starts_with()`, which is C++20. `rfind` searches
backwards from position 0, so it can only match at position 0, which is the same question asked
with the vocabulary C++17 has.

### NameOfRoundTripsEveryIdentity

`Ops::name_of` returns the name of every entry of `Ops::K_ALL` for its identity.

### NameOfReportsUnknownForAForeignIdentity

`Ops::name_of(OpID(0xDEADBEEF))` is "sushiai.unknown".

### TheTableContainsEveryDeclaredIdentity

A new operation declared but left out of K_ALL would escape the uniqueness static_assert
entirely, so the test checks membership by name.

### EverySpatialIdentityIsListedInAll

K_ALL is what `all_ids_unique()` proves over and what `name_of()` searches. An identity declared
but left out of K_ALL is not covered by the collision proof and prints as "sushiai.unknown" in
every profile. Both failures are silent.

### ThereIsNoConv2dIdentity

CV-1 says convolution is K_IM2COL + K_MATMUL in the IR. A CONV2D identity would be a second, opaque
spelling of the same thing, and the first person to add one will do it by analogy and not by
reading the spec.

### LeakyReLUDifferentiatesAtItsOutput

For alpha > 0, leaky_relu(x) > 0 exactly when x > 0. Reading the output in place of the input is
therefore a bit-identical substitution, not an approximation, and it is the same argument the
table already records for K_RELU. It is what makes the fold free: an ordinary use count licenses
it with no activation-specific knowledge in the fusion pass.

### EveryElementwiseAndSelectionIdentityIsListedInAll

`Ops::name_of` resolves K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_LESS_EQUAL, K_SELECT, K_SILU,
K_SILU_BACKWARD, K_SOFTPLUS and K_SOFTPLUS_BACKWARD to a name other than "sushiai.unknown". The
reason is the one EverySpatialIdentityIsListedInAll gives.

### TheComparisonAndTheActivationGradientsAreNotFlaggedDifferentiable

`differentiable` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU, and
false for K_LESS_EQUAL, K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD.

### TheGradientBarriersAreTheFourIdentitiesThatPassNoGradient

`Ops::is_gradient_barrier` is true for K_ZERO, K_CONSTANT, K_LESS_EQUAL and K_STOP_GRADIENT. Over every
row of `Ops::K_ALL` it equals the row's `blocks_gradient` flag, and exactly four rows carry the
flag. A fifth barrier therefore fails the count until the list in this test names it.

### NoGradientBarrierIsFlaggedDifferentiable

No row of `Ops::K_ALL` has both `differentiable` and `blocks_gradient` set.

### LeafCastAndTheGradientOperationsAreNotGradientBarriers

`is_gradient_barrier` is false for K_LEAF, K_CAST, K_DYNAMIC_SCALE, K_LOG_SOFTMAX, K_RELU_BACKWARD,
K_SILU_BACKWARD, K_SOFTPLUS_BACKWARD, K_MAXPOOL2D_BACKWARD, K_BATCHNORM2D_BACKWARD,
K_BATCHNORM_UPDATE_STATS, K_BATCHNORM_FOLD, K_GEMM_BIAS, K_EXP_SUB_SCALE and K_ADAMW_UPDATE, and for the
foreign identity 0xDEADBEEF. These either pass a gradient or throw when one reaches them.
Marking one as a barrier would turn that throw into a gradient that is dropped without a
message.

### SiLUDifferentiatesAtItsInput

The `Ops::K_ACTIVATIONS` row of K_SILU names K_SILU_BACKWARD as its gradient and
`GradientSource::Input` as its source, and `activation_survives_fusion(K_SILU)` is false.

### SoftplusDifferentiatesAtItsInput

The row of K_SOFTPLUS names K_SOFTPLUS_BACKWARD and `GradientSource::Input`, and
`activation_survives_fusion(K_SOFTPLUS)` is false.

### IsActivationBackwardAcceptsOnlyTheTablesGradientColumn

For every row of `Ops::K_ACTIVATIONS`, `is_activation_backward` is true for the row's `backward`
identity and false for its `forward` one. Over `Ops::K_ALL` it accepts as many identities as the
table has rows.

### BatchNormFoldIsListedAndClaimsNoRule

`Ops::name_of(K_BATCHNORM_FOLD)` is "sushiai.batchnorm_fold", and the row has `differentiable`
false and `blocks_gradient` false. `test_backward.cpp` holds what differentiation does when a
gradient reaches the node.

## modules/core/optim/tests/test_parameter_decay.cpp

Suites: `Unit_ParameterDecay`.

Tests `ParameterSlot::decays` and the `decay_all` option of SGD and AdamW. The fixture needs an
engine and makes two slots of three float32 elements, each starting at 2 with a zero gradient:
`weight`, which decays, and `bias`, which does not. Both optimizers use the rate 0.5 and the
weight decay 0.5, so a decayed parameter is 1.5 after one step. Values are compared within
1e-6.

### ASlotDecaysUnlessItSaysOtherwise

A default `ParameterSlot` has `decays` true.

### SGDDecaysOnlyTheSlotsThatTakeIt

For the kernels `Fused` and `Separate`, one SGD step leaves the weight at 1.5 and the bias
at 2.

### AdamWDecaysOnlyTheSlotsThatTakeIt

For the kernels `Fused` and `Separate`, one AdamW step leaves the weight at 1.5 and the bias
at 2.

### DecayAllDecaysEverySlot

With `decay_all` true, one step of SGD and one step of AdamW each leave both parameters at 1.5.

### TheKeyDecayAllReachesTheOptimizer

For the types `sgd` and `adamw`, an optimizer from `make_optimizer` without the key `decay_all`
leaves the weight at 1.5 and the bias at 2 after one step. With `"decay_all": true` it leaves
both at 1.5.

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

## modules/vision/training/tests/test_pixel_targets.cpp

Suites: `Unit_VisionTrainingPixelTargets`.

Tests `validate_pixel_labels` in
`modules/vision/training/include/SushiAI/vision/training/pixel_labels.hpp`,
`fill_pixel_targets` in
`modules/vision/training/include/SushiAI/vision/training/pixel_targets.hpp` and
`fill_dice_targets` in
`modules/vision/training/include/SushiAI/vision/training/dice_targets.hpp`. The fixture is an
`EngineTest`. The masks hold 6 pixels over 3 classes with the ignore index 255. Before a fill
every index leaf holds -7 and every real leaf -7.0, and reals are compared within 1e-7.

### TheLabelCheckNamesThePixelOutOfBounds

`validate_pixel_labels` passes a mask of classes and 255. It fails with `Incompatible` for a
3 at pixel 3 ("holds 3 at pixel 3", "below 3", "255"), a -1 at pixel 4
("holds -1 at pixel 4"), an expected count of 7 ("6 pixels", "expects 7") and a `Float32` mask
("mask is").

### AKeptPixelIndexesItsClassAndSharesTheWeight

For the mask 0, 2, 255, 1, 1, 255, `fill_pixel_targets` writes the indices 0, 5, 6, 10, 13, 15
and the weights 0.25, 0.25, 0, 0.25, 0.25, 0.

### AMaskWithNoKeptPixelGivesZeroWeights

For a mask of six 255s, the indices are 0, 3, 6, 9, 12, 15 and the six weights are 0.

### ARefusedPixelFillChangesNothing

`fill_pixel_targets` fails with `Incompatible` for a mask with a 9 at pixel 4
("holds 9 at pixel 4"), for a weights leaf of five elements ("6 pixel indices",
"5 pixel weights") and for a `Float32` indices leaf ("pixel indices are"). Afterwards the
indices still hold -7 and the weights -7.0.

### TheDiceLeavesCountTheKeptPixelsOfEachClass

For the mask 0, 2, 255, 0, 2, 0, `fill_dice_targets` writes `score_indices` 0, 5, 6, 9, 14,
15, `class_indices` 0, 2, 0, 0, 2, 0, `kept` 1, 1, 0, 1, 1, 1, `class_pixels` 3, 0, 2 and
`class_weights` 0.5, 0, 0.5.

### ARefusedDiceFillChangesNothing

`fill_dice_targets` fails with `Incompatible` for a mask with a 3 at pixel 4
("holds 3 at pixel 4"), for a `class_weights` leaf of two elements ("'class_weights'",
"2 elements") and for a `Float32` `class_indices` leaf ("'class_indices'", "int32").
Afterwards `score_indices` and `class_indices` still hold -7, and `kept` and `class_pixels`
-7.0.

## modules/core/autograd/tests/test_precision.cpp

Suites: `Unit_Precision`, `Unit_CastInsertion`.

Tests the mixed-precision policy and where the cast insertion pass places its casts.

This file needs no device. The precision policy is a table of judgements and the cast pass is a
pure function of the IR, as the fusion pass and the memory planner are.

The cases pin placement, not survival. A mixed-precision graph that runs proves almost nothing:
a missing cast throws in SushiBLAS, so "it ran" only rules out the loud half of the failure
space. The quiet half, a cast in the wrong place, leaves a graph that trains and learns worse.
So every case is one of these:

- the whole-graph invariant (every node's operands agree with the type it computes in), asserted
  by walking the graph and not by spot checks,
- a control (the same graph under the fp32 policy must get zero casts, computed and not
  short-circuited), or
- one of the two places the master-weight arrangement can silently break: a parameter retyped in
  place, or a gradient that reaches its fp32 buffer without being widened.

The cases fall into five groups, in file order:

- the policy itself
- the control: the fp32 policy changes nothing
- the invariant
- master weights, and the direction that goes silently wrong
- what the pass promises about ids, reuse and repetition

### Traced

The `Traced` graph is built by hand and not through NN::Sequential, so that the test can name
every value it wants to assert about. It is also the smallest graph that still contains all four
interesting shapes: a parameter read by both passes, a reduction, a loss, and a gradient buffer.

### TheUniformPolicyAnswersTheGraphsOwnTypeToEverything

For every entry of `Ops::K_ALL` and each of `Half`, `Float32` and `Float64`,
`UniformPrecisionPolicy::choose` returns the type it was asked about. The policy's name is
"fp32".

### OnlyTheProductAndWhatAGEMMFoldsIntoItNarrow

The `narrows` list states the rule set as a table, so that adding an operation to the narrowing
set is a visible edit in this test and not a silent behaviour change. Everything absent from it
stays in the graph's own precision, which is the safe answer for an operation nobody has thought
about yet.

The K_LEAKY_RELU entries are in `narrows` because K_LEAKY_RELU joined `Ops::K_ACTIVATIONS` with
`GradientSource::Output`. It narrows by the same substitution argument as K_RELU.

The `stays` list opens with the reductions and anything that sums many terms. fp16 carries about
11 bits of significand and SushiBLAS's own reduction header documents a 2048 ceiling for HALF,
so these are the operations that go quietly wrong.

K_DYNAMIC_SCALE is in `stays` because it carries the gradient loss scale, the one number in the
graph that exists to keep values out of fp16's subnormals. Narrowing the node that applies it
would overflow the very quantity it protects; see `modules/core/optim/include/SushiAI/optim/loss_scaler.hpp`.

K_GELU, K_GELU_BACKWARD and K_GEMM_BIAS_GELU, the one fused form that folds K_GELU, are in `stays`. The
form is a GEMM and rule 1 would otherwise take it. Narrowing it puts the tanh approximation's
`x*x*x` into fp16, which overflows at |x| ~ 40.3. Rule 5 in
`modules/core/graph/include/SushiAI/graph/precision.hpp` says so, and also explains why that leaves rules 2 and 5
consistent and not merely quiet.

The twenty spatial and CV identities are in `stays` because none of them is a GEMM epilogue or
an activation this policy folds. Each gets the same safe default as everything else nobody has
special-cased: it stays at the graph's own precision.

K_SILU, K_SILU_BACKWARD, K_SOFTPLUS and K_SOFTPLUS_BACKWARD are in `stays`. K_SILU and K_SOFTPLUS read
their input, so neither they nor their gradients narrow. K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN,
K_LESS_EQUAL, K_SELECT, K_STOP_GRADIENT and K_CONSTANT are in `stays` under the same default.

K_BATCHNORM_FOLD is in `stays`. It takes the square root of a variance plus epsilon.

The closing `EXPECT_EQ` on the two list sizes holds because every identity is accounted for by
`narrows` or by `stays`. A new operation cannot join the enum without someone deciding which
list it belongs to.

### TheDepthwiseIdentitiesKeepTheirDeclaredTypeUnderMixedPrecision

For DEPTHWISE_CONV2D and its two backward identities, `narrows_under_mixed_precision` is
false and `MixedPrecisionPolicy::choose` returns `Float32` for `Float32` and `Float64` for
`Float64`.

### GELUStaysWideEvenThoughACatalogEntryNowFoldsIt

Rule 5 used to rest on two arguments, and one of them is gone. `FusedForm::GEMMBiasGELU`
folds GELU by spilling the pre-activation, so "nothing folds it, therefore nothing forces its
precision" is no longer a reason for anything. The surviving argument is the arithmetic: the
tanh approximation computes `x*x*x`, which overflows fp16 at |x| about 40.3. A unit test cannot
assert that without a device.

The first three assertions check what a unit test can: that the conclusion did not quietly
follow the dead premise. The catalog can fold GELU and the policy still keeps it wide, which is
the pair of facts rule 5 in `modules/core/graph/include/SushiAI/graph/precision.hpp` now asserts.

`narrows_under_mixed_precision(K_GEMM_BIAS_GELU)` is false because the fused form has to go with
GELU. Otherwise the cast pass would narrow a node whose epilogue is the very expression rule 5
refuses to put in fp16.

### ChoosingIsIdempotentForEveryOperation

`modules/core/graph/include/SushiAI/graph/precision.hpp` states this contract, and it is what makes the invariant
checkable after the pass has run and not only while it runs. Without it,
`find_dtype_mismatches` would report a graph the pass had just finished making correct.

### AMixedPolicyNeverWidensAndNeverNarrowsPastItsLowType

The checks on the fp64 and fp16 graphs say that an fp64 graph keeps fp64 masters and still gets
fp16 products, and that an fp16 graph is already at the floor and is left alone, not "narrowed"
to something wider.

The `to_single` policy narrows to fp32. That is a legal thing to ask for, and it is what an fp64
model would use.

### ANonFloatLowTypeIsRefusedAtConstruction

`MixedPrecisionPolicy{DType::Int32}` and `MixedPrecisionPolicy{DType::Complex32}` throw
`Error`, and `MixedPrecisionPolicy{DType::Half}` does not.

### EveryChoiceCarriesAReasonSomebodyCanRead

For every entry of `Ops::K_ALL`, the choice of `MixedPrecisionPolicy` over `Float32` has a
non-empty `reason`.

### TheUniformPolicyInsertsNoCasts

This is the control the mixed cases are read against. If it ever shows casts, the pass is
inserting them unconditionally and every "the mixed policy inserted N casts" assertion in the
later Unit_CastInsertion cases measures nothing.

### NoPolicyMeansTheUniformPolicy

`insert_casts` without options reports 0 casts and leaves the node count unchanged.

### EveryNodesOperandsAgreeWithTheTypeItComputesIn

This assertion is worth more than a dozen spot checks. It walks the whole graph and states the
rule, so it keeps holding as operations are added.

### TheInvariantSurvivesTheFusionPassThatRunsAfterIt

Fusion rewrites nodes and re-pairs operands, so it is where a correctly cast graph could stop
being one, and where a master weight could get swallowed.
`modules/core/graph/include/SushiAI/graph/cast_insertion.hpp` argues the ordering, and this test is the assertion
behind the argument.

### ADeliberatelyBrokenGraphIsReportedRatherThanAccepted

`find_dtype_mismatches` is the test file's own instrument, so it has to be shown to fail on a
graph that is wrong. Otherwise every "no mismatches" in the earlier cases could come from a
checker that never looks.

`mismatches.size()` is 3 because the graph is an uncast fp32 matmul under a mixed policy: two
operands and one result, all three float32 where the policy says half.

### AParameterKeepsItsTypeAndGetsADerivedLowPrecisionCopy

The fp16 value the GEMM reads is a different value from the parameter. A cast produces it, and
it lives with the step's scratch and not for the whole run.

### AGradientIsWidenedBeforeItReachesItsBuffer

The test covers the backward direction. A matmul computing in fp16 produces an fp16 gradient,
and if that reached the fp32 gradient buffer uncast, SushiBLAS would throw. If the buffer had
been retyped to fp16 instead, nothing would throw and the optimizer would accumulate in fp16,
which is the failure master weights exist to prevent.

The loop over `gradients[0]` and `gradients[2]` covers the weights only. A backward K_MATMUL,
which computes low, produces the gradients of w1 and w2. K_SUM_ROWS, which does not, produces
those of b1 and b2, so only the weights need widening.

The loop over `gradients[1]` and `gradients[3]` covers the biases. Their gradients come out of
a reduction that stayed wide, so nothing was widened into them and nothing needed to be.

### AccumulatingGradientContributionsHappensWide

A parameter read twice gets two gradient contributions and an K_ACCUMULATE to sum them. Two things
have to hold at once. The sum runs in the wide type, because it is a sum. Its result and its
first operand agree, because the lowering makes the result alias the accumulator's buffer, so a
disagreement there is a wrong-width write and not merely a wrong number.

### AnObservedValueKeepsItsTypeAndAnUnobservedOneDoesNot

This case covers the half of the pinning rule that a lifetime class does not. The logits are an
ordinary activation. The training loop reads them back by id for accuracy, and a caller who
forgets to say so gets an fp16 buffer where their host-side argmax expects floats.

### AValueReadTwiceIsConvertedOnce

w2 is read by the forward matmul and again by the backward one. Two conversions would be two
kernels and two buffers for one number, and the reuse is safe because the IR is
single-assignment.

### ValueIDsSurviveThePass

After `insert_casts` with a `MixedPrecisionPolicy`, the graph has more values than before,
and every value that existed before keeps its shape and its lifetime under the same id.

### RunningThePassTwiceAddsNothingTheSecondTime

The first `insert_casts` reports at least one cast. A second run with the same options
reports 0 and leaves the node count and the value count unchanged.

### TheReportNamesEveryConversionAndItsDirection

The report's policy is "mixed". Its count equals the number of CAST nodes, the difference
between `nodes_after` and `nodes_before`, and the sum of `count_into(Half)` and
`count_into(Float32)`, both of which are above 0. `retyped_values` is above 0, and
`to_string` holds "mixed" and "casts inserted".

### AGraphThatIsNotInTopologicalOrderIsRefused

`fuse()` and `MemoryPlan::build()` carry the same guard for the same reason. The pass emits a
conversion immediately before its consumer, which is only a valid position if producers already
come first.

### Pooled

K_MAXPOOL2D is the first operation in this IR whose results are not all one element type: it
writes the pooled tensor and, beside it, the INT64 argmax its backward reads. The cast pass
derives one precision per node from `outputs[0]`, which makes the argmax look like a value in
the wrong type and not like a value with no precision to choose.

`x` is a parameter and not an input for the reason
`tests/regression/test_argmax_lifetime.cpp` gives: K_MAXPOOL2D_BACKWARD is only emitted
when something upstream of the pool needs a gradient.

### AnIndexBufferHasNoPrecisionToChoose

The pass forced every operand and every result of a node to one dtype, derived from the node's
first result. For K_MAXPOOL2D that is FLOAT32 from the pooled tensor, so the INT64 argmax read as
a mismatch and the pass tried to convert it. `GraphBuilder::cast` refuses anything but
real-float to real-float by design, so the whole build threw.

The defect does not depend on the policy. `wanted` comes from `outputs[0]`, not from the
policy's `choose()`, so the uniform policy reaches it too.

### AMixedPolicyLeavesTheIndexBufferAloneToo

`insert_casts` with a `MixedPrecisionPolicy` over the pooled trace does not throw. The pool's
index value stays `Int64` and no CAST node reads or writes it.

### APinnedIndexBufferIsNotWidenedIntoItsOwnBuffer

This is the loud half of the same defect. An unpinned argmax is retyped in place and the
mismatch disappears silently. A pinned one keeps its id and its INT64 buffer, so the pass
reaches for the widening cast, and K_CAST is real-float to real-float by design.

### ABufferKeepsItsDeclaredTypeAndItsStateLeafAcrossThePass

The graph is a K_MATMUL and a K_BIAS_ADD whose bias is a buffer. The weight and the buffer are
bound to the states 31 and 32 and the graph is marked refused before `insert_casts` runs under
the mixed policy. The report counts at least one cast and the K_BIAS_ADD's result is HALF, so
the pass did rewrite the graph. The buffer is still FLOAT32 with the lifetime `BUFFER` and no
gradient. Both `state_leaf` lookups, the whole `state_leaves()` table and `has_refusal()` are
what they were, and `find_dtype_mismatches` returns nothing.

### BatchNormFoldStaysInTheMasterTypeUnderTheMixedPolicy

The graph is one `batchnorm_fold` over two parameters and two buffers. `insert_casts` under
the mixed policy reports zero casts and the graph holds no K_CAST node. The four operands and
the two results are all FLOAT32.

## modules/core/autograd/tests/test_provenance.cpp

Suites: `Unit_Provenance`.

Tests the forward origin that differentiation records for every node it appends.

This file needs no device. `Autograd::differentiate()` fills a `Provenance` while it appends
backward nodes, so every case differentiates a host-side graph and reads the record back.
`modules/core/eager/README.md`, under `autograd/provenance.hpp`, states the origin rule and
what the partitioner needs it for.

### TwoLinearGraph

`TwoLinearGraph` is a two-layer perceptron with a cross-entropy loss: an [8, 6] input, a 6-to-4
linear layer, a K_RELU, a 4-to-3 linear layer, then K_LOG_SOFTMAX and K_NLL_LOSS. It keeps the ids the
cases name: the four parameters, the result of the first K_MATMUL and the loss.

### differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. It passes no seed factor and hands the `Provenance` pointer through unchanged,
which lets one case pass null.

### nodes_of

`nodes_of` returns the ids of the nodes that run one operation, starting at a given node index.
A case passes `forward_node_count` as the start, so the search covers the appended nodes only.

### EveryBackwardNodeHasAForwardOrigin

`forward_node_count` equals the node count taken before differentiation, `origin` holds one
entry per appended node, and every entry is below `forward_node_count`. An origin at or above
that count would name a backward node, which no segment of the forward trace holds.

### ARuleEmitsUnderTheNodeItDifferentiates

The node that produces the first weight's gradient is an appended node, and its origin is the
K_MATMUL that produces `first_product`. The case asserts the operation and the node id, so an
origin that named another K_MATMUL of the graph fails it.

### AFanOutAccumulateBelongsToTheProducer

`shared` is a K_RELU result that a K_TANH and a K_SIGMOID both read, so its gradient has two
contributors. The backward nodes hold exactly one K_ACCUMULATE, and its origin is the K_RELU that
produces `shared`. `x` is read once, so no closing sum is emitted and the count stays at one.

### AParameterReadTwiceAccumulatesUnderItsLeaf

The parameter `x` is read by a K_RELU and a K_TANH. The node that produces its gradient is an
K_ACCUMULATE, and the origin of that node is the K_LEAF of `x`.

`targets` is declared before `x` so that the K_LEAF of `x` is not node 0. The reverse walk ends at
node 0, so with `x` there an origin taken from the walk's last position would equal the K_LEAF's
id and the `EXPECT_NE(origin, 0u)` line could not tell the two apart.

### AReusedProvenanceHoldsOnlyTheLatestRun

One `Provenance` is filled by a first differentiation and then passed to a second one over a
`TwoLinearGraph`. Its `forward_node_count` and `origin` equal those of a fresh `Provenance`
filled from an identical graph. A `differentiate()` that appended to `origin` and did not reset
it would leave the first run's entries in front.

### ARequestedValueWithoutAProducerIsRefused

`x` is added with `add_value` and so has no K_LEAF node, and it is read twice, so the closing sum
of its gradient needs an origin that does not exist. The call throws, and the message contains
"provenance" and "producer node".

### PassingNoProvenanceChangesNothing

Two identical graphs are differentiated, one with a `Provenance` and one with null. The returned
gradient ids are equal, the node counts are equal, and every node has the same operation,
inputs and outputs. Recording is therefore an observation of the transform and not a branch in
it.

## modules/core/optim/tests/test_rate_schedule.cpp

Suites: `Unit_RateSchedule`.

Tests `Optim::RateSchedule`, the variant of `StepSchedule` and `CosineSchedule`: its section
reader and writer, `rate_at`, `base_rate_of`, `validate_rate_schedule` and
`Optim::read_rate_schedule`. The step section has the base rate 0.5 and one step at 3 of scale
0.5. The cosine section has the base rate 0.25, a warm-up of 2, a total of 10 and the final
rate 0.125. A rate is compared with a tolerance of 0.

### ASectionIsReadByTheTypeItStates

The step section reads as a `StepSchedule` with the base rate 0.5. The cosine section reads as
a `CosineSchedule` with `total_iterations` 10 and the final rate 0.125.

### ASectionWithoutATypeIsAStepSchedule

`{"base_rate": 0.5}` reads as a `StepSchedule`.

### AnUnknownTypeIsRefusedWithTheTypesThereAre

The type `linear` fails with `InvalidDescription`, "schedule 'type' is 'linear'", `'step'` and
`'cosine'`. A `type` of 3 fails with "schedule 'type' must be a string", and the number 7 in
place of the section with "schedule must be an object".

### AKeyOfAnotherTypeIsRefused

`steps` in a cosine section fails with "states the key 'steps'", and `final_rate` in a step
section with "states the key 'final_rate'".

### TheRateIsThatOfTheScheduleHeld

`rate_at` of the step schedule is 0.5 at iteration 2 and 0.25 at iteration 3. `rate_at` of the
cosine schedule is 0.125 at iterations 1 and 10. `base_rate_of` is 0.5 and 0.25.

### AWrittenSectionStatesItsTypeAndReadsBack

`write_rate_schedule_section` of the cosine schedule gives a section whose `type` is "cosine".
It reads back as a `CosineSchedule` with `warmup_iterations` 2.

### ARunReadsACosineScheduleUnderTheSameRules

`read_rate_schedule` returns no schedule for a null section beside an optimizer that states
`learning_rate`, and a `CosineSchedule` for the cosine section beside `{}`. The cosine section
beside a `learning_rate` fails with `InvalidDescription` and "stated twice". A cosine section
without a base rate fails with "'base_rate' must be stated and above zero".

### ARunReadsAStepScheduleAsAStepSchedule

`read_rate_schedule` of the step section beside `{}` returns a `StepSchedule` with the base
rate 0.5.

### AHeldScheduleIsCheckedByItsOwnValidation

`validate_rate_schedule` fails with `InvalidDescription`, "2 steps" and "1 scales" for a held
`StepSchedule` of two steps and one scale. It is ok for a held `CosineSchedule` with the base
rate 0.5 and a total of 10.

## modules/core/eager/tests/test_region_signature.cpp

Suites: `Unit_RegionSignature`.

Tests the signature that `signature_of` computes for one region of a partition.

This file needs no device. `Eager::signature_of` reads a graph, a partition and one identity per
value, all of which a case builds on the host. `modules/core/eager/README.md`, under
`region_signature.hpp`, lists what the hash covers and gives the reason for each operand rule.

Most cases build two graphs that differ in one property and compare the two signatures. Where
the operation list is the same in both, the case asserts that first, so the difference it then
asserts is a difference in the hash alone.

### Region keys and identities

`K_BODY` is key 1, the one region of a hand-built partition, and `K_READER` is key 2, the region
that reads from it in a two-region partition. `K_PARAMETER_IDENTITY` and `K_EXTERNAL_IDENTITY`
are the default identities of the parameter and of the value `K_READER` reads from outside.
They are stated once so that a case names the one it changes.

### Traced

`Traced` holds a graph, its partition and one identity per value id, which are the three things
`Eager::signature_of` reads.

### signature_of

The file-local `signature_of` has three overloads. The first takes a `Traced` and a region key
and calls `Eager::signature_of` with the identity vector as a span. The other two take a
`Recipe` or an `ExternalRecipe`, build its graph and return the signature of `K_BODY` or of
`K_READER`.

### Recipe

`Recipe` describes `scale(combine(relu(x), tanh(x)), alpha)` over a parameter `x`, and each of
its fields is one property a case varies: the shape, the element type, the K_SCALE factor, K_SUB in
place of K_ADD, the operand order of the combine, and the identity of `x`. `build(Recipe)` puts
the four operations under `K_BODY` and the K_LEAF under no region, and gives every value but `x`
identity zero.

### node_of

`node_of` returns the first node of a graph that runs a given operation. The attribute and
layout cases use it to reach the node they edit after `build` returns.

### assign_regions

`assign_regions` gives a `Traced` its partition from one key per non-K_LEAF node, in node order,
and a region list. A K_LEAF gets `K_NO_REGION`. It also resets every identity to zero, so a
builder that calls it sets afterwards only the identities its case is about.

### ExternalRecipe

`ExternalRecipe` describes a graph whose `K_READER` region holds one K_SCALE that reads a value
the region does not produce. Its fields choose whether that value is the parameter `x` or the
K_RELU result of `K_BODY`, whether `K_BODY` appears in the region list, and which identity the
value has.

### ProbeOperands

`ProbeOperands` names four operand orders for a hand-built three-operand K_ADD in `K_READER`.
`build(ProbeOperands)` leaves every identity at zero, and the one in-region operand is result 0
of the region's node 0. Both orders of a pair therefore feed the hash the same words, and the
two signatures can differ only if the kind of each operand enters the hash.

### build_reader_of_result

`build_reader_of_result` adds a K_RELU node by hand with two results and a K_TANH that reads the
result at the given index, both under `K_BODY`. The K_RELU is added through `Graph::add_node`
because `GraphBuilder::relu` returns one result.

### with_identity

`with_identity` records an identity for one value and returns the value id, so that
`ScopedPerceptron` can assign an identity in the expression that declares a leaf.

### info_of

`info_of` returns the `RegionInfo` of a key, or null when the partition does not list the key.

### role_identity

`role_identity` hashes a region's name, its direction, how many nodes of the region follow the
producing node, and the index of the value among that node's results. Counting from the
region's end keeps the identity of a region's last result the same when nodes are added before
it.

### identify_produced_values

`identify_produced_values` gives every value that a region's node writes its `role_identity`.
In the tests it stands in for the caller of `signature_of`, which owns the storage a boundary
value is bound to; `modules/core/eager/README.md` says why only the caller can supply that
identity.

### ScopedPerceptron

`ScopedPerceptron` traces three scopes, "stem", "body" and "head", then differentiates and
partitions the graph with `ScopePartitioner`. Its argument is the number of linear layers in
"stem". Leaves get fixed identities and produced values get role identities. `key_of` returns
the key of a region by name and direction, and `stem_output` is the value "body" reads from
"stem".

### TheSameTraceGivesTheSameSignature

Two graphs built from the default `Recipe` have equal hashes and compare equal. The operation
list is K_RELU, K_TANH, K_ADD, K_SCALE, which is the region's nodes in order with the K_LEAF left out.

### AChangedAttributeChangesTheHash

A K_SCALE factor of 3.0 in place of 2.0 leaves the operation list equal and makes the hash
differ.

### AChangedLastAttributeSlotChangesTheHash

The case writes 1.0 into slot `Attributes::K_CAPACITY - 1` of the K_SCALE node and expects another
hash. It covers the last slot because a hash that read a fixed number of leading slots would
agree with the other attribute case and miss this one.

### AChangedLayoutChangesTheHash

The K_RELU result is retagged NCHW in one of two otherwise equal graphs, and the hashes differ.
Shape and element type are the same in both, so the layout is the only input that moved.

### AChangedShapeChangesTheHash

A shape of [4, 5] in place of [4, 3] leaves the operation list equal and makes the hash differ.

### AChangedDTypeChangesTheHash

FLOAT64 in place of FLOAT32 leaves the operation list equal and makes the hash differ.

### AChangedOpChangesTheHash

K_SUB in place of K_ADD makes both the operation list and the hash differ. It is the one
`Recipe` case in which the two lists are unequal.

### ADifferentParameterChangesTheHash

The same structure over a parameter with identity `K_PARAMETER_IDENTITY + 1` has the same
operation list and another hash. The parameter is external to the region, so its identity is
the only thing the hash holds about it.

### AnEarlierRegionGrowingDoesNotChangeALaterRegion

`ScopedPerceptron(1)` and `ScopedPerceptron(2)` differ by one linear layer in "stem". The three
`ASSERT` lines at the top establish that the case tests what it claims: the value "body" reads
has a different id in the two graphs, the same identity in both, and the forward and backward
regions of "body" have different keys.

For each direction, the signature of "body#0" has a non-empty operation list and is equal in
the two graphs, by hash and by `operator==`. The forward signature of "stem#0" differs between
them, which shows that the two graphs are not equal throughout.

### ARewiredOperandChangesTheHash

`add(relu(x), tanh(x))` and `add(tanh(x), relu(x))` have the same operation list and different
hashes. Both operands are produced inside the region, so the case covers the in-region operand
position.

### AReadOfAnotherResultChangesTheHash

A K_TANH that reads result 1 of a two-result node hashes apart from one that reads result 0. The
producer's index in the region is the same in both, so the result index is what separates them.

### ABoundaryValueOfAnotherIdentityChangesTheHash

The K_SCALE in `K_READER` reads the K_RELU result of `K_BODY`. With identity
`K_EXTERNAL_IDENTITY + 1` for that value, the operation list is equal and the hash differs.

### ALeafAndABoundaryValueOfOneIdentityHashAlike

The reader's signature is equal, by hash and by `operator==`, whether its operand is the
parameter `x` or the result of another region, as long as both carry `K_EXTERNAL_IDENTITY`.
Where an external operand comes from in the graph does not enter the hash.

### AnOperandFromAnUnlistedRegionHashesAsExternal

The same graph is signed with `K_BODY` present in the region list and with it absent, and the
two signatures of `K_READER` are equal. An operand whose producer has a key other than the
region's is external whether or not the partition lists that key.

### AnOperandInsideTheRegionHashesApartFromOneOutside

The probe node reads one result of `K_READER` and the result of `K_BODY` twice, in the two
orders `InsideThenOutside` and `OutsideThenInside`. The operation lists are equal and the
hashes differ.

### ALeafOperandHashesApartFromOneInsideTheRegion

The same pair of orders with the leaf `x` in place of the `K_BODY` result,
`InsideThenLeaves` and `LeavesThenInside`, also gives equal operation lists and different
hashes.

### EqualHashesWithDifferentOpsCompareUnequal

Three signatures are written by hand with hash 42. The one listing K_ADD and the one listing K_SUB
compare unequal under `==` and `!=`, and two that both list K_ADD compare equal. A real collision
cannot be produced on demand, so the case constructs one.

### ARegionThePartitionDoesNotListThrows

`signature_of` throws for key 6 and for `K_NO_REGION`. Each message names the key, "Region 6"
and "Region 0", and contains "not listed".

### APartitionOfAnotherNodeCountThrows

One entry is removed from `region_of_node`, which leaves four entries for a five-node graph.
The message contains "assigns 4 nodes" and "holds 5".

### AnExternalValueWithoutAnIdentityThrows

The identity vector is cleared, so the id of `x` lies beyond its end. The message contains
"External value 0" and "no identity". The call throws and does not read past the span.

## modules/language/registration/tests/test_registration.cpp

Suites: `Unit_LanguageRegistration`.

Tests the three registration entry points of the language module, `Language::register_nodes`,
`Language::register_datasets` and `Language::register_objectives`, and that model files using
their rows build.

### RegisterNodesAddsTheSixBlockRows

`register_nodes` adds five node rows in the order "position_embedding",
"causal_self_attention", "feed_forward", "transformer_block", "moe_feed_forward", and one linked
row, "language_head". A second call fails with `InvalidDescription` and a message holding
"'position_embedding'" and "already".

### RegisterDatasetsAddsTokensAndPrompt

After `register_datasets` the registry has two rows, and `find` returns a row for "tokens" and
for "prompt".

### RegisterObjectivesAddsNextTokenAndExpertBalance

After `register_objectives` the registry has two rows, and `find` returns a row for "next_token"
and for "expert_balance".

### TheNextTokenRowBuildsBesideTheRowsOfTrain

With the rows of `Train::register_objectives` and of the language module in one registry,
`Train::make_objective` builds "next_token" over an output of the shape [2, 3, 5] and five
classes, and the objective's name is "next_token". A second `Language::register_objectives`
fails with "'next_token' is already registered".

### ADecoderModelFileBuilds

A model of an embedding, a position embedding, a transformer block and a language head tied to
row 0 builds into four nodes, and `output_shape([8])` is [8, 32]. A model whose head is tied to
row 1, the position embedding, fails with `InvalidDescription`; the message holds
"layer 2 'language_head'", "'tie'", "row 1" and "no 'embedding' row".

### ADecoderWithExpertsPublishesWhatTheBalanceRowReads

A decoder with a four-expert transformer block and a two-expert `moe_feed_forward` row
publishes five outputs. Traced over a [2, 8] `Int32` input, the logits have the shape
[2, 8, 32], "balance2" has [1, 1] and "usage3" has [1, 2]. `Blocks::K_BALANCE_PORT` equals
`Training::K_BALANCE_OUTPUT_PREFIX`, `Blocks::K_USAGE_PORT` equals
`Training::K_USAGE_OUTPUT_PREFIX`, and a "sum" objective of a "next_token" term and an
"expert_balance" term builds over the published shapes.

## modules/vision/registration/tests/test_registration.cpp

Suites: `Unit_VisionRegistration`, `Unit_VisionRegistrationData`.

Tests the four registration entry points of the vision module and what builds from their rows.
The YOLO26n cases read `applications/examples/yolo26/model.json` and
`tests/fixtures/yolo26/MANIFEST.json` and report themselves skipped when the repository root is
not found. `Unit_VisionRegistrationData` is an engine fixture.

### RegisterNodesAddsTheEightBlocks

`Vision::register_nodes` adds the node rows "conv", "bottleneck", "c3k", "c3k2", "sppf",
"c2psa", "detect" and "segment_head", in that order, and no layer row.

### RegisterDatasetsAddsMNISTAndReader

`Vision::register_datasets` adds "mnist" and "reader", in that order. The "mnist" row has a
factory, and the row of `Config::K_READER_DATA_TYPE` has `Vision::Datasets::make_reader_dataset`.

### ARegistryWithoutTheBlocksRefusesConv

`NN::build_model` over the rows of `NN::register_layers` alone fails for a model with a "conv"
row. The code is `InvalidDescription`, and the message holds "layer 0 'conv'", "the type 'conv'"
and "does not have".

### EveryRowHasAMinimalCase

The file's minimal cases, followed by its detect case and its segment head case, have the
types of the registered node rows in the same order and the same count.

### SegmentHeadBuildsOnePortAtTheUpsampledSize

The "segment_head" row builds a node with one port of the shape its case expects.

### DetectBuildsFourPortsPerLevel

The "detect" row over its case's levels builds a node with four ports per level.

### EachRowBuildsFromAMinimalObject

Each of the other rows builds, from its case's attributes, a node with one port named
`NN::K_SINGLE_PORT_NAME` and of the shape the case expects.

### EachRowRejectsAnUnknownKey

Each row given the key "no_such_key" fails with `InvalidDescription`, and the message holds
"layer 0 '<type>'" and "'no_such_key'". The same holds for "detect".

### TheYOLO26FileHoldsItsTwentyFourRows

The model file parses to 24 rows of the types the case lists in order, from "conv" to
"detect", with the input shape [640, 640, 3]. The four `concat` rows read rows 11 and 6, 14 and
4, 17 and 13, and 20 and 10, and the detect row reads rows 16, 19 and 22. Row 0 has a
`bn_epsilon` of 0.001, row 9 a `bn_momentum` of 0.03, and row 11 has no `bn_epsilon`. The one
output is row 23 with an empty port.

### TheYOLO26FileBuildsWhole

The model file builds over the composed registry. Its trace publishes the detect row's twelve
outputs in port order, and its state is the 594 records the manifest lists for rows 0 to 23,
without the "model." prefix.

### YOLO26RowsTwoAndFourAddTheirBottlenecksInput

Rows 2 and 4 of the file have "shortcut" true, and the first five rows trace to exactly two ADD
nodes.

### YOLO26RowsZeroToNineMatchTheManifest

Rows 0 to 9 build and yield the 165 records the manifest lists under model.0 to model.9, and
the traced output is "output [2, 20, 20, 256]".

### YOLO26RowsBuildToTheReferenceShapes

With a `silu` row in place of row 10, the first 22 rows build, and each row's one port has the
per-sample shape `K_YOLO_ROW_SHAPES` holds for it.

### YOLO26NeckTracesToTheReferenceLevels

With a `silu` row in place of row 10, rows 16 and 19 trace to the height, width and channel
count the manifest records for feature levels 0 and 1, and each level's stride times its height
is the input height. Row 21 traces to [2, 20, 20, 384]. The state is the manifest's records for
rows 0 to 9 followed by its 145 records for rows 11 to 21.

### RegistersTheAnchorAndThePixelObjectives

After `Train::register_objectives`, `Vision::register_objectives` adds three rows: "anchor",
"pixel_cross_entropy" and "dice". Row 0 stays "cross_entropy". A second call fails with
`InvalidDescription`, "'anchor'" and "already registered".

### TheAnchorRowReadsTheTaskOfTheModel

`Train::make_objective` of "anchor" over an empty section fails with `InvalidDescription`, and
the message holds "the objective block" and "'task'".

### ThePixelRowsBuildOverOneLogitMap

"pixel_cross_entropy" and "dice" each build an objective over one [2, 4, 4, 3] output with an
`ignore_index` of 255, and the objective's name is the row's type.

### RegistersTheDarknetImporter

`Vision::register_importers` adds one row of the format "darknet". A second call fails with
`InvalidDescription`, "'darknet'" and "already registered".

### TheDarknetRowRefusesACallWithoutFiles

`IO::import_model` of "darknet" with no file fails with `InvalidDescription`, and the message
holds "'darknet'", ".cfg" and "0 files".

### TheReaderRowBuildsADataSection

`Data::make_dataset` of the reader type with the reader "coco_json", the task "detection" and
an absent root fails with `InvalidDescription`. The message holds "the data block 'reader'",
"'coco_json'" and "'yolo_txt'".

## modules/core/autograd/tests/test_rematerialise.cpp

Suites: `Unit_Rematerialise`.

Tests the structure of the graph that the K_IM2COL rematerialisation pass rewrites.

This file needs no device. The K_IM2COL rematerialisation pass is an IR-to-IR rewrite, so what it
produces is a graph that can be inspected node by node.

The unit tier proves the structure: that a clone appears, that it reads the same operands and
carries the same attributes as the node it copies, that it is inserted ahead of its consumers
and not after them, and that an inference graph is left alone. It cannot prove the numbers are
unchanged. Integration_Rematerialise in `tests/integration/test_rematerialise.cpp`
does that by running the same graph twice.

The brief's three cases are all in this file, with two deviations that the library forced:

1. `Autograd::differentiate()` refuses any root that is not a scalar produced by K_NLL_LOSS
   (`modules/core/autograd/source/differentiator.cpp` explains why every other root would silently return empty
   gradients). The brief's graphs stopped at a K_MATMUL, so each is carried on to a loss. The
   shapes, geometry and expectations are the brief's.
2. A "backward consumer" cannot be identified by node index, because the pass runs when the
   backward nodes are already in the graph and no marker separates the phases. The rule the pass
   applies is "every consumer after the first", which is the same set for every graph
   differentiation produces and is well-defined for one it did not produce.

### TheColumnBufferStopsBeingAnActivation

This case checks the number CV-10 is about. ResNet-18's first block at batch 128 holds
`128*32*32*64*9*4` bytes, about 302 MB, in this one buffer, and there are twenty convolutions.
Cloning the gather next to its backward consumer drops the lifetime to SCRATCH, so one buffer is
live at a time and not twenty.

### TheCloneComputesTheSameThingAsTheOriginal

The operand and attribute checks guard against a clone that read a different operand or carried
different attributes. Such a clone would produce a plausible column matrix and a wrong weight
gradient. Both are checked because the geometry lives in the attributes and is easy to drop when
copying a node.

The check on `copy` expects the same description and the same layout. The layout is a
graph-wide invariant, and a clone that dropped it would fail `finalise()` on a graph that is
numerically correct.

### TheCloneIsInsertedAheadOfEveryConsumerItServes

The failure this case catches is not a wrong number. It is a graph whose insertion order stopped
being a topological order, with a recomputation appended after the GEMM that reads it.
`MemoryPlan::build()` rests on that order and says so by name, and a lowering that got past it
would record a GEMM against a buffer nothing had written yet.

### AnIm2colWithNoBackwardConsumerIsLeftAlone

This is the inference case. Nothing was differentiated, so there is no second consumer and
cloning would be a free slowdown.

### AForwardOnlyConvolutionIsLeftAlone

This is the other half of the inference case, and the one that would break if the pass cloned
for every consumer and not for every consumer after the first. The column matrix has exactly one
reader, the GEMM.

### RunningItTwiceAddsNothingTheSecondTime

Once the rewiring is done the clone has one consumer of its own, so the pass has no reason to
rematerialise its own output. If it did, the pass would not terminate on a graph with three
consumers.

## modules/core/core/tests/test_result.cpp

Suites: `Unit_Result`.

Tests Result, its reference form and SA_TRY_ASSIGN.

This file needs no device. `Result<T>` and `SA_TRY_ASSIGN` are in
`modules/core/core/include/SushiAI/core/result.hpp`. A `Result<T>` holds a value or a failed `Status`, and
`Result<T&>` holds a reference to the caller's object or a failed `Status`.

Nine `static_assert` lines at the top of the file hold the type's shape. Neither form is
default-constructible. Construction of a `Result<int>` from an `int` and from a `Status`, and
its move, are `noexcept`. A `Result<int>` is at most two pointers and a `Result<int&>` is
exactly two. Neither `Result<int&>` nor `Result<const int&>` is constructible from an rvalue
`int`, so a reference form cannot refer to a temporary.

### description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

### checked_extent

`checked_extent` returns its extent as a `Result<int>`, or an `InvalidDescription` failure from
`SA_FAIL_IF` when the extent is not positive.

### doubled_extent

`doubled_extent` binds `checked_extent`'s value through `SA_TRY_ASSIGN` and returns twice it.

### boxed_extent

`boxed_extent` returns a `std::unique_ptr<int>` holding the checked extent, a move-only value.

### unboxed_extent

`unboxed_extent` binds `boxed_extent`'s pointer through `SA_TRY_ASSIGN` and returns the
integer it owns.

### forward_temporary_status

`forward_temporary_status` runs `SA_TRY(checked_extent(extent).status())`, which reads a
`Status` reference into a temporary `Result`, and sets a flag after it.

### slot_at

`slot_at` returns a `Result<int&>` to one of two slots, or an `InvalidDescription` failure for
an index outside 0 and 1.

### bump_slot

`bump_slot` binds `int& slot` from `slot_at` through `SA_TRY_ASSIGN` and increments it.

### HoldsAValueWithASuccessStatus

`Result<int>(7)` has a value, an ok status and the value 7. Writing 9 through `value()` is
read back through a const reference.

### HoldsAFailureAndNoValue

A result built from the failure "rank must be 2" has no value. `SA_EXPECT_FAILURE` accepts
both the result and its `status()`.

### ReturnStatementsConvertFromAValueAndFromAStatus

`checked_extent(5)` yields 5 through `SA_ASSERT_OK_AND_ASSIGN`, and `checked_extent(-1)` is an
`InvalidDescription` failure containing "extent must be positive" and "got -1". The function
returns an `int` on one path and the `Status` of `SA_FAIL_IF` on the other.

### SuccessPathCreatesNoFailureBlock

A made, a copied and a moved `Result<int>` and one `SA_TRY_ASSIGN` through `doubled_extent`
sum to 13 and leave `live_failure_count()` where it was.

### RvalueValueMovesAMoveOnlyValueOut

`std::move(result).value()` on a `Result<std::unique_ptr<int>>` returns a pointer that is not
null and owns 11.

### MovedFromValueResultStillHoldsAMovedFromValue

After a `Result<std::unique_ptr<int>>` is moved from, the target owns 2. The source still
reports a value and an ok status, and its pointer is null.

### MovedFromFailureResultKeepsItsFailure

After a failed `Result<int>` is moved from, the source and the target both report
`InvalidDescription` and "layout must be NHWC", and the source has no value.

### CopyAndAssignmentCarryTheHeldState

Assigning a failed result to one that holds 1 leaves no value and the failure "no such key".
Assigning `Result<int>(8)` to it afterwards leaves a value, an ok status and 8.

### TryAssignBindsTheValueOrReturnsTheFailure

`doubled_extent(21)` is 42. `doubled_extent(0)` is the failure of `checked_extent`, with
"got 0" in its message.

### TryAssignMovesFromATemporaryResult

`unboxed_extent(6)` is 6, which requires `SA_TRY_ASSIGN` to move the `unique_ptr` out of the
temporary result. `unboxed_extent(-6)` is the failure with "got -6".

### ReferenceFormRefersToTheCallersObject

`slot_at(slots, 1)` has a value whose address is `&slots[1]`, and writing 25 through it
changes the array. A `Result<const int&>` built from a const `int` returns that object's
address.

### TryReadsAStatusReferenceIntoATemporaryResult

`forward_temporary_status(-4)` returns the failure with "extent must be positive" and
"got -4" and leaves its flag false. `forward_temporary_status(4)` succeeds and sets it.
`live_failure_count()` is at its start after each call.

### MovedFromReferenceResultKeepsItsState

A `Result<int&>` to slot 0 is copied and moved from. The source, the copy and the target all
refer to `&slots[0]`. A failed reference result moved from keeps the failure "slot 2 is not
one of 2" in both source and target. Assigning the failed one over a result to slot 1 gives
the failure, and assigning the successful one over that gives `&slots[0]` and an ok status.

### ReferenceFormHoldsAFailure

`slot_at(slots, 2)` has no value and the `InvalidDescription` failure "slot 2 is not one of 2".

### TryAssignBindsAReferenceFromTheReferenceForm

`bump_slot(slots, 0)` succeeds and slot 0 goes from 10 to 11, so the reference `SA_TRY_ASSIGN`
bound is the array's element. `bump_slot(slots, 5)` fails with "slot 5" and slot 1 stays 20.

### ValueOnAFailureIsFatalAndPrintsTheFailure

`value()` on a failed `Result<int>` and on a failed `Result<int&>` stops the process. The
report contains "Result::value() was called on a failure" and the failure's own message.

### BuildingFromASuccessStatusIsFatal

`Result<int>(Status())` and `Result<int&>(Status())` stop the process with "a Result was built
from a success Status".

## modules/layers/train/tests/test_role_match.cpp

Suites: `Unit_RoleMatch`.

Tests `Train::match_roles`. The dataset is a local `Published` class named "published" that
publishes the roles it is built with and holds no batch. The cases use two roles: `labels`,
one int32 per sample over a class count, and `boxes`, a float32 [n, 4] value per sample.

### ADatasetThatPublishesEveryRoleIsAccepted

A dataset that publishes `labels` over 4 classes and `boxes` of 8 matches a step that requires
`boxes` of 8 and then `labels` over 4.

### AStepThatReadsNoRoleAcceptsAnyDataset

An empty list of required roles matches a dataset that publishes no role and one that
publishes `labels`.

### AMissingRoleIsNamedWithTheDatasetAndWhatItPublishes

Requiring `boxes` of a dataset that publishes only `labels` fails with `InvalidDescription`
and a message containing `'published'`, `'boxes'`, "float32 [8, 4] per sample" and "publishes
'labels'". Requiring `labels` of a dataset that publishes nothing fails with "(none)".

### AnotherShapeIsRefusedWithBothShapes

`boxes` of 8 against a dataset that publishes `boxes` of 16 fails with `Incompatible`, "as
float32 [16, 4] per sample" and "compiled for float32 [8, 4] per sample".

### AnotherElementTypeIsRefusedWithBothTypes

Int32 `labels` against float32 `labels` fails with `Incompatible`, "as 4 classes of float32 []
per sample" and "compiled for 4 classes of int32 [] per sample".

### AnotherClassCountIsRefusedWithBothCounts

`labels` over 4 classes against `labels` over 5 fails with `Incompatible`, "as 5 classes of
int32 [] per sample" and "compiled for 4 classes of int32 [] per sample".

### ARequiredClassCountOfZeroAcceptsAnyCount

`labels` required over 0 classes matches a dataset that publishes `labels` over 5.

### TheFirstRoleThatDoesNotMatchIsReported

Requiring `boxes` and then `labels` over 4 of a dataset that publishes `labels` over 5 fails
with `InvalidDescription` and a message naming `'boxes'`.

## modules/core/autograd/tests/test_rule_table.cpp

Suites: `Unit_RuleTable`.

Tests that every domain's backward rules reach the combined table exactly once.

### EveryDomainsRulesReachTheCombinedTable

`modules/core/autograd/include/SushiAI/autograd/rules/all.hpp` concatenates the domains. A domain left out of the
concatenation compiles and never differentiates. The count is the cheapest thing that notices.

### NoOperationHasTwoRules

Two entries for one identity is not a compile error, and the first one silently wins. Which
gradient runs would then depend on the order in which two domains happened to be concatenated.

### TheRulesThatExistedBeforeTheSplitStillDo

The split is a move, not a rewrite. These fourteen rules were the table before it, and every
one must still resolve.

### NoGradientBarrierHasARule

For every row of `Ops::K_ALL` with `blocks_gradient` set, `Autograd::has_rule` and
`Autograd::has_backward_rule` are both false. The loop counts the rows it checked and expects
4, so it cannot pass by checking none.

### TheComparisonIsABarrierWithoutARule

`Ops::K_LESS_EQUAL` has `differentiable` false and `blocks_gradient` true, and
`Autograd::has_rule` is false for it.

### EveryActivationRowHasARuleForItsForwardIdentityOnly

For every row of `Ops::K_ACTIVATIONS`, `Autograd::has_rule` is true for the `forward` identity
and false for the `backward` one. `Autograd::K_ACTIVATION_RULES` has as many entries as
`Ops::K_ACTIVATIONS` has rows.

### BatchNormFoldHasNoRuleAndBatchNormKeepsIts

`Autograd::has_rule` is true for K_BATCHNORM2D and false for K_BATCHNORM_FOLD, `has_backward_rule`
is false for K_BATCHNORM_FOLD, and `Autograd::K_NORM_RULES` has two entries.

### LayerNormHasItsRuleAndItsGradientHasNone

`has_rule` is true for LAYER_NORM. LAYER_NORM_BACKWARD is not differentiable and has no rule.

### LogSoftmaxHasItsRuleAndItsGradientHasNone

`K_LOSS_RULES` has two rows. LOG_SOFTMAX is differentiable and has a rule.
LOG_SOFTMAX_BACKWARD is not differentiable and has none.

### TheCausalMaskHasItsRule

`K_LINALG_RULES` has six rows and `has_rule` is true for CAUSAL_MASK.

### DropoutHasTheRuleOfTheRandomDomain

`K_RANDOM_RULES` has one row. DROPOUT is differentiable and has a rule.

### TheGatherAndTheScatterAddDifferentiateIntoEachOther

GATHER_ROWS and SCATTER_ADD_ROWS are both differentiable and both have a rule.

### TheDepthwiseForwardHasItsRuleAndItsGradientsHaveNone

`K_SPATIAL_RULES` has nine rows and `has_rule` is true for DEPTHWISE_CONV2D. Its two backward
identities are not differentiable and have no rule.

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

## modules/core/optim/tests/test_run_schedule.cpp

Suites: `Unit_RunSchedule`.

Tests `Optim::read_rate_schedule`, which takes the schedule section and the optimizer section
of a training file. The schedule section of the cases is a step schedule with the base rate
0.00261, a warm-up of 1000 and one step at 400000 of scale 0.1.

### ANullSectionYieldsNoSchedule

A null schedule section beside an optimizer that states `learning_rate` returns no schedule.

### AScheduleIsReadWhenTheOptimizerStatesNoRate

The schedule section beside `{"momentum": 0.9}` returns a `StepSchedule` with the base rate
0.00261, a warm-up of 1000 and one step at 400000.

### ARateStatedTwiceIsRefused

The schedule section beside `{"learning_rate": 0.00261}` fails with `InvalidDescription` and a
message containing "stated twice", `'learning_rate'` and `'base_rate'`.

### AScheduleWithoutABaseRateIsRefused

`{"type": "step"}` fails with `InvalidDescription`, `'base_rate'` and "above zero".

### ABaseRateNotAboveZeroIsRefused

The base rates 0 and -0.01 each fail with `InvalidDescription`, `'base_rate'` and "above zero".

### WhatTheSectionReaderRefusesIsRefused

A section with the key `gamma` fails with `InvalidDescription`, "schedule" and `'gamma'`. A
string in place of the section fails with "schedule" and "object".

## modules/core/optim/tests/test_run_state.cpp

Suites: `Unit_OptimizerRunState`, `Unit_LossScaleStateRestore`.

Tests `Optimizer::run_state`, `restore_scalars` and `check_state_owner`, and
`Optim::LossScaleState::restore`. The `Unit_OptimizerRunState` fixture needs an engine and
makes the slots `weight` and `bias`, each three float32 zeros with a gradient of ones.

### SGDListsItsMomentumBuffersAndItsStepCount

After one step of an SGD with momentum 0.9, `type_name()` is "sgd" and the run state lists the
tensors `weight.momentum` and `bias.momentum`, the first of three elements. Its one scalar is
`step_count`, a `uint64_t` of 1.

### SGDWithoutMomentumListsNoTensor

The run state of an SGD with momentum 0 has no tensor and one scalar whose value is 0.

### AdamWListsBothMomentsOfEachSlot

`type_name()` is "adamw" and the run state lists `weight.first_moment`, `weight.second_moment`,
`bias.first_moment` and `bias.second_moment`, in that order. The first listed tensor has the
data pointer of `first_moment(0)` and the fourth that of `second_moment(1)`.

### TheStepCountIsRestored

After `restore_scalars` with a step count of 41, `step_count()` is 41, and it is 42 after one
step.

### AnotherOwnerIsRefusedWithBothTypes

On an SGD, `check_state_owner("sgd")` is ok and `check_state_owner("adamw")` fails with
`Incompatible`, `'adamw'` and `'sgd'`.

### AScalarThatDoesNotFitIsRefusedByName

`restore_scalars` on an SGD fails with `Incompatible` four times: for no scalar ("does not
hold", "'step_count'", "'sgd'"), for an added `loss_scale` ("'loss_scale'", "'sgd'", "does not
keep"), for `step_count` given twice ("more than once") and for a `step_count` that is a real
("a real number where a count is kept"). The step count stays 0.

### TakesASavedScaleAndRefusesOneOutOfBounds

`restore(4096.0, 1999)` is ok and sets the scale to 4096 and the count of good steps to 1999.
One `observe(false)` then gives the scale 8192 and the count 0. `restore` fails with
`Incompatible` for the scale 3000 ("power of two"), the scale 0.5 ("bounds") and a count of
2000 ("growth interval"), and the scale stays 8192.

## modules/core/eager/tests/test_scope_partitioner.cpp

Suites: `Unit_ScopePartitioner`.

Tests the region keys ScopePartitioner gives the nodes of a differentiated graph.

This file needs no device. `ScopePartitioner::assign` reads a graph, the segments of its forward
trace and the provenance of its backward nodes, and returns a `Partition`.
`modules/core/eager/README.md`, under `scope_partitioner.hpp` and `partition.hpp`, states the
key rule the cases check: with `m` segments, key `k` forward, key `2m - k` backward, and `2m` for
the optimizer.

The cases fall into three groups, in file order:

- keys and region lists of traced and differentiated steps
- graphs built by hand to reach a K_LEAF origin or a graph with no backward node
- inputs that `assign` refuses

### Step

`Step` holds one traced, differentiated and partitioned training step: the graph, the segments,
the provenance, the parameters, one gradient per parameter and the partition.

### as_span

`as_span` returns a segment vector as the span `assign` takes.

### finish

`finish` seals the trace, differentiates the loss against `step.parameters` with a `Provenance`
and partitions the result. Every step builder ends with it, so the four steps differ only in
what they trace.

### linear

`linear` emits a K_MATMUL and a K_BIAS_ADD over a new weight and bias and appends both to the
parameter list. The parameter order is therefore weight, bias per layer, which is how a case
indexes `gradients`.

### unscoped_step

`unscoped_step` builds two linear layers and a loss with no scope open, which gives one segment.

### three_segment_step

`three_segment_step` builds scopes "first" and "second" with one linear layer and an activation
each, and a scope "loss". It has three segments, six leaves and seven other forward nodes.

### shared_weight_step

`shared_weight_step` declares one weight in a scope "weights", multiplies by it in "first" and
again in "second", and closes with "loss". It has four segments, and the first holds only a
K_LEAF.

### repeated_block_step

`repeated_block_step` opens a scope "block" twice in a loop, with one linear layer each time,
then "loss". The input is declared before the first scope, so the trace opens with an unnamed
segment that holds only a K_LEAF.

### TwoReLUGraph

`TwoReLUGraph` holds a graph built without differentiation: a parameter, a K_RELU in a segment
"first" and a K_RELU of that in a segment "second". `build_two_relu_graph` fills it, seals the
trace and sets `forward_node_count`, and leaves `origin` empty. A case then appends backward
nodes and writes their origins by hand, which reaches arrangements `differentiate()` does not
emit.

### is_leaf

`is_leaf` reports whether a node of a graph is a K_LEAF.

### segment_number

`segment_number` returns the number, counted from 1, of the segment that holds a forward node,
or 0 when none does. It is computed from the segment ranges and not from the partition, so a
forward key is compared against something the partitioner did not write.

### backward_keys_of

`backward_keys_of` returns the keys of the backward nodes whose origin is a non-K_LEAF node of
one segment. It skips K_LEAF origins because those take their key from their operands.

### AnUnscopedStepIsOneRegion

With one segment, every non-K_LEAF node of the forward and backward graph has key 1 and the
optimizer key is 2. The region list has one entry, named "#0" and marked forward: the empty
scope name followed by its occurrence.

### ForwardKeysFollowTraceOrder

Every non-K_LEAF forward node of `three_segment_step` has the number of its segment as its key.
The closing count of seven checked nodes fails if the loop skips nodes it should compare.

### BackwardKeysMirrorForwardKeys

With three segments the optimizer key is 6, and the backward nodes whose origins lie in segments
1, 2 and 3 have keys 5, 4 and 3. Each segment must have at least one such node, so an empty
result cannot satisfy the loop.

The region list has five entries with keys 1 to 5, named "first#0", "second#0", "loss#0",
"second#0", "first#0". The first three are forward and the last two backward.

### TheLastSegmentSharesItsRegionWithItsBackward

The K_NLL_LOSS node has key 3, and so does every backward node whose origin lies in the third
segment. The region list holds exactly one entry with key 3, and its direction is forward. A
list with a second, backward entry under the same key would describe one region twice.

### EveryOperandComesFromALowerOrEqualKey

For each of the four steps, every node's key is at least the key of the node that produces each
of its operands. Operands with no producer, or whose producer has no region, are skipped. The
check runs over every node of the graph and not over chosen ones, and each step must contribute
at least one comparison. `modules/core/eager/README.md` gives this as the condition for a
valid partition under `RegionKey`.

### AWeightGradientLandsInItsOwnLayersBackwardRegion

The node that produces the second layer's weight gradient has key 4, the backward key of the
second segment. The case then asserts that this node reads a value produced under key 1. If a
change to the differentiation rules removed that read, the first assertion would hold for a
graph that no longer contains the case, and the message on the last assertion says so.

### ASharedParametersAccumulateSitsAboveBothContributors

The gradient of the shared weight is produced by an K_ACCUMULATE whose origin is a K_LEAF. Its two
operands are produced under keys 5 and 6, the backward keys of "second" and "first" among four
segments, and the K_ACCUMULATE has key 6. Key 7, the backward key of the segment that holds the
weight's K_LEAF, is not used for it.

### ALeafOriginNodeTakesItsHighestOperandKeyInEitherOrder

Two K_ADD nodes are appended to a `TwoReLUGraph` with the K_LEAF as their origin. One reads the
key 2 value first and the key 1 value second, the other the reverse, and both get key 2. A rule
that took the key of the first or of the last operand would give one of them key 1.

### RegionsListOnlyKeysThatHoldANode

`shared_weight_step` has four segments and optimizer key 8. Its region list has five entries
with keys 2 to 6. Key 1 belongs to the K_LEAF-only segment "weights" and key 7 to its backward
mirror, no node carries either, and neither is listed.

### ARepeatedScopeIsNamedByItsOccurrence

`repeated_block_step` has four segments and optimizer key 8. The region list has keys 2 to 6,
named "block#0", "block#1", "loss#0", "block#1", "block#0", so the backward regions carry the
names of the forward regions they mirror.

### AGraphWithNoBackwardNodeHasForwardRegionsOnly

A `TwoReLUGraph` with an empty `origin` gives its two K_RELU nodes keys 1 and 2 and optimizer
key 4. The region list holds "first#0" and "second#0", both forward. The optimizer key is `2m`
whether or not any backward node exists.

### LeavesBelongToNoRegion

In `three_segment_step` every K_LEAF has `K_NO_REGION` and every other node has a key. The count
of six leaves, two inputs and four parameters, shows that the loop reached the K_LEAF branch.

### SegmentsThatDoNotCoverTheForwardGraphThrow

Four malformed segment lists are refused, each with its own message: a list missing its last
segment ("The segments cover nodes 0 to", "forward graph has"), a list with a gap before segment
1 ("Segment 1 ('second')", "contiguous"), a list with an empty segment inserted ("Segment 1
('empty')", "holds no node"), and an empty list ("at least one segment").

### AProvenanceOfAnotherGraphThrows

A provenance with one origin too few is refused with a message containing "The provenance
describes" and "backward nodes". One whose last origin equals `forward_node_count`, the first
id that is not a forward node, is refused with "has origin" and "forward nodes".

### ABackwardNodeThatReadsAHigherKeyThrows

Two K_RELU nodes are appended to a `TwoReLUGraph`, the second reading the first, with origins in
segments 1 and 2. The formula gives them keys 3 and 2, so the reader has the lower key. `assign`
throws, and the message contains "has region key 2" and "whose key is 3". The partitioner
therefore checks the ordering it produces and does not assume the provenance is consistent with
the graph.

### ALeafOriginNodeThatReadsOnlyLeavesThrows

An K_ADD of two parameters is appended with a K_LEAF as its origin. Neither operand's producer has
a region, so there is no key to take. The message contains "as its origin" and "reads only
leaves".

## modules/core/eager/tests/test_scope_trace.cpp

Suites: `Unit_ScopeTrace`.

Tests how ScopeTrace cuts a trace into segments and which calls it rejects.

This file needs no device. A `ScopeTrace` reads the node count of a graph, so every case
appends nodes to a host-side graph, opens and closes scopes around them and inspects what
`seal()` returns. `modules/core/eager/README.md`, under `segment.hpp` and `scope_trace.hpp`,
defines a segment and the names a scope may have.

The cases fall into two groups, in file order: the segments a trace yields, and the calls
`open()`, `close()` and `seal()` reject.

### trace_nodes

`trace_nodes` appends a given number of K_ADD nodes to a graph, each adding one input value to
itself. The input is added as a value with no K_LEAF node, so the node count rises by exactly the
number asked for and a case can state every range as a literal.

### segment_is

`segment_is` compares one segment with an expected name, occurrence, begin and end. On a
mismatch it prints all four fields of both, so a failure shows which field differs.

### NodesOutsideAnyScopeFormOneUnnamedSegment

Three nodes traced with no scope open give one segment: empty name, occurrence 0, nodes 0 to 3.

### AScopeSplitsTheTraceIntoThreeSegments

One node, then two in a scope "body", then three give three segments. The unnamed run after the
scope has occurrence 1, because the unnamed run before it was occurrence 0.

### ANestedScopeIsNamedByItsPath

Inside "outer" and "inner" the depth is 2 and the segment is named "outer/inner". The nodes of
"outer" before and after it form two segments named "outer", with occurrences 0 and 1.

### AScopeOpenedTwiceGetsRisingOccurrences

A scope "cell" opened three times in a loop, with two nodes each time, gives three segments
named "cell" with occurrences 0, 1 and 2 and consecutive ranges.

### AnEmptyScopeLeavesNoSegment

A scope "cell" that closes with no node traced leaves no segment. The next "cell", which holds
one node, has occurrence 0, so the dropped scope did not use up an occurrence.

### SealWithAScopeStillOpenThrows

`seal()` throws while "outer" and "inner" are open, and the message contains "seal()", "still
open" and the path "outer/inner". The depth is still 2 afterwards.

The case then traces one more node, closes both scopes and seals. The result is one segment
"outer/inner" over nodes 0 to 2, which shows the failed `seal()` did not cut the running
segment.

### CloseWithNoScopeOpenThrows

`close()` throws on a new trace, and again after one matched `open()` and `close()`. Both
messages contain "close()" and "no scope open".

### SegmentsCoverEveryNodeExactlyOnce

The trace mixes unscoped nodes, a scope opened twice, a nested scope and an empty one. Each
segment begins where the previous one ended and holds at least one node, the first begins at 0,
and the last ends at the graph's node count.

### OpenRejectsAnEmptyName

`open("")` throws with a message containing "open()" and "empty scope name", and the depth
stays 0.

### OpenRejectsANameWithASlash

`open("outer/inner")` throws. The message contains the name and "may not contain", and the
depth stays 0.

### OpenRejectsANameWithAHash

`open("cell#0")` throws. The message contains the name and "may not contain", and the depth
stays 0.

### ARejectedNameDoesNotSplitTheRunningSegment

One node is traced, `open("a/b")` throws, and one more node is traced. `seal()` returns one
unnamed segment over nodes 0 to 2. A trace that ended the running segment before it validated
the name would return two.

### ARejectedCloseDoesNotSplitTheRunningSegment

The same arrangement with a `close()` that throws for having no scope open also returns one
unnamed segment over nodes 0 to 2.

### ASecondSealThrows

A second `seal()` throws with a message containing "seal()" and "sealed trace".

### OpenAndCloseAfterSealThrow

After `seal()`, `open("late")` and `close()` both throw with "sealed trace" in the message, and
the depth stays 0.

## test_seam.cpp

Suites: `Unit_Seam`, `Unit_SeamEngine`.

Tests the seam in `modules/core/core/include/SushiAI/core/seam.hpp`, the one place where an exception of SushiBLAS
or of a standard type, or an `Error` SushiRuntime returned, becomes a `Status`. `guard` and
`guard_value` run a call and return its Status or its value. `status_of_current_exception`
chooses the code from the type of the exception and from the `SeamPhase` the call was in:
`Record`, `Allocate` or `Execute`. `status_of_error` chooses it from the `ErrorCode`.

The `Unit_Seam` cases need no device and build each exception and each `Error` they map by hand. The
`Unit_SeamEngine` fixture is an `EngineTest`: its cases record work on an engine through
`record_and_run`.

### GuardReturnsTheStatusOfACallThatDoesNotThrow

A success passes through `guard`, and a failure Status returns with its code and its message.

### GuardValueReturnsTheValueOfACallThatDoesNotThrow

`guard_value` returns the value of a call that does not throw, as an ok `Result`.

### GuardValueReturnsTheStatusOfWhatTheCallThrew

A `std::bad_alloc` thrown by the call becomes a failed `Result` of code `OutOfMemory`.

### UnsupportedExecutionIsUnsupportedThoughItIsAnInvalidArgument

`SushiBLAS::UnsupportedExecution` derives from `std::invalid_argument`, and it maps to
`Unsupported`, not `InvalidDescription`. The message is kept.

### InvalidArgumentIsInvalidDescription

A plain `std::invalid_argument` maps to `InvalidDescription`, and its text is kept.

### PlainRuntimeErrorTakesItsCodeFromThePhase

A plain `std::runtime_error` maps to `OutOfMemory` when the call was allocating, and to
`DeviceFailed` when it was recording or executing.

### RuntimeFailureThatTheUserCanActOnMapsByItsCode

A SushiRuntime `Error` that carries a code the user can act on keeps it: `OutOfMemory` for a
missing USM allocation, `DeviceFailed` for a device fault or an unplugged device, and `Unsupported`
for a missing fp16 path.

### RuntimeFailureWithTheOkCodeTakesItsCodeFromThePhase

A SushiRuntime `Error` that carries the `Ok` code takes its code from the phase, as a plain
`std::runtime_error` does.

### RuntimeFailureThatReportsADefectIsFatal

A SushiRuntime `Error` that reports a defect stops the process rather than returning a Status.
The case checks five defect codes with `SA_EXPECT_FATAL`: `InvalidGraph`, `CompileFailed`,
`InvalidAccess`, `InvalidState` and `DanglingResource`.


### ErrorOfAHostUnitIsFatalWithItsText

A `SushiAI::Error` thrown while recording stops the process, and the report carries its text.

### BadAllocIsOutOfMemoryInEveryPhase

`std::bad_alloc` maps to `OutOfMemory` in each of the three phases.

### AnyOtherStandardExceptionIsDeviceFailed

Any other standard exception maps to `DeviceFailed` and keeps its text.

### ExceptionOfAnUnknownTypeIsDeviceFailed

An exception that derives from no standard type maps to `DeviceFailed`.

### RecordAndRunRunsWhatASuccessfulRecordingRecorded

A recording that sets a tensor of ones to 2.0 and returns success leaves every element at
2.0 after `record_and_run`.

### RecordAndRunDiscardsWhatAFailedRecordingRecorded

A recording that sets a tensor to 5.0 and then throws `std::invalid_argument` makes
`record_and_run` fail with `InvalidDescription` and the exception's message. A following
recording sets a second tensor to 2.0; that tensor holds 2.0 and the first still holds 1.0.

### RecordAndRunDiscardsWhenTheRecordingReturnsAFailure

A recording that sets a tensor to 5.0 and returns an `Unsupported` status makes
`record_and_run` fail with that status and "no host path". After `run_recorded` the tensor
still holds 1.0.

## modules/core/config/tests/test_section_reader.cpp

Suites: `Unit_SectionReader`.

Tests `Config::SectionReader` and the free readers of
`modules/core/config/include/SushiAI/config/section_reader.hpp`. The reader of every case is
named "schedule" and the accepted keys are `type`, `base_rate` and `steps`.

### NamesAMemberAnEntryAndAList

`member_of("schedule", "steps")` is "schedule 'steps'", `entry_of` of that with 2 is "schedule
'steps' entry 2", and `list_quoted` of the three keys is "'type', 'base_rate', 'steps'" and of
an empty span "". `SectionReader::member("type")` is "schedule 'type'".

### CheckObjectRefusesAnotherKind

`check_object` is ok for `{}`. For a string it fails with `InvalidDescription` and "schedule
must be an object, but it is a string".

### CheckKeysNamesTheKeyAndListsWhatIsAccepted

`check_keys` is ok for a section with `steps` and `type`. For a section with `gamma` it fails
with "schedule states the key 'gamma', which it does not have; it accepts 'type', 'base_rate',
'steps'".

### CheckTypeAcceptsAnAbsentKeyAndRefusesAnotherType

`check_type("type", "step", "a step schedule")` is ok when `type` is absent and when it is
"step". For "cosine" it fails with "schedule 'type' is 'cosine'; a step schedule states
'step'", and for 3 with "schedule 'type' must be a string, but it is a whole number".

### FindReturnsTheMemberOrNamesTheAbsentKey

Over `{"base_rate": 0.5}`, `has` is true for `base_rate` and false for `steps`. `find` returns
a `Kind::Real` member for `base_rate` and fails for `steps` with "schedule has no 'steps'".

### FindOfChecksTheKind

Over `{"steps": 3}`, `find_of("steps", Kind::Int)` is ok. Asked for `Kind::Array` it fails with
"schedule 'steps' must be an array, but it is a whole number", and for the absent `type` with
"schedule has no 'type'".

### TheNumberReadersCheckTheKind

`read_real` returns 7.0 for the whole number 7 and 0.25 for the real 0.25, and `read_whole`
returns 7 for 7. `read_real` of a string fails with "rate must be a number, but it is a
string", and `read_whole` of 0.25 with "count must be a whole number, but it is a number".

### ReadFloat32RefusesWhatFloat32CannotHold

`read_float32` returns 0.5f for 0.5. For 1e39 it fails with `InvalidDescription`, "scale is"
and "which float32 does not hold as a finite value", and for a string with "scale must be a
number, but it is a string".

### TheStringAndArrayReadersCheckTheKind

`read_string` returns "step" for a string and `read_array` returns three entries for
`[1, 2, 3]`. `read_string` of an array fails with "type must be a string, but it is an array",
and `read_array` of a string with "steps must be an array, but it is a string".

## modules/vision/segment/tests/test_segment_confusion_matrix.cpp

Suites: `Unit_SegmentConfusionMatrix`.

Tests `Vision::Segment::ConfusionMatrix` in
`modules/vision/segment/include/SushiAI/vision/segment/confusion_matrix.hpp`. Every `add` call
passes the ignore index 255.

### ANewMatrixCountsNothing

A matrix made for 3 classes reports 3 classes, 0 pixels and a count of 0 at (2, 1).

### AddCountsByTruthAndPrediction

For the predictions 0, 1, 1, 2, 2, 2 against the truth 0, 0, 1, 2, 1, 2, `count(truth,
predicted)` is 1 at (0, 0), (0, 1), (1, 1) and (1, 2), 2 at (2, 2) and 0 at (1, 0).
`truth_pixels(1)` is 2, `predicted_pixels(2)` is 3 and `pixel_count` is 6.

### AnIgnoredPixelIsNotCounted

Of three pixels, two have the truth 255, one of them with the prediction 7. The matrix counts
one pixel, at (0, 0), and `predicted_pixels(1)` is 0.

### CountsAccumulateOverBatches

The same three pixels added twice give 2 at (0, 0), (0, 1) and (1, 1) and a pixel count of 6.

### MergeAddsAndClearEmpties

Two matrices each count the predictions 1, 1 against the truth 0, 1. After `first.merge(second)`
and `second.clear()`, the first holds 2 at (0, 1) and at (1, 1) and the second counts 0 pixels.

### MakeRefusesAClassCountOutsideItsBounds

`ConfusionMatrix::make(0)` fails with `InvalidDescription`, "ConfusionMatrix" and "0 classes".
`make(K_MAX_CONFUSION_CLASSES + 1)` fails with "4097 classes" and "4096".

### AddRefusesWithoutCounting

Over 2 classes, `add` fails with `Incompatible` for spans of two and three pixels
("2 predicted", "3 true"), a truth of 2 at pixel 1 ("true class of pixel 1", "is 2",
"below 2", "255") and a prediction of -1 at pixel 1 ("predicted class of pixel 1", "is -1").
The pixel count stays 0 after the three calls.

### MergeRefusesAnotherClassCount

Merging a matrix of 3 classes into one of 2 fails with `Incompatible`, "2 classes" and "of 3".

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

## modules/vision/training/tests/test_segmentation_metrics.cpp

Suites: `Unit_VisionTrainingSegmentationMetrics`.

Tests `measure_segmentation` in
`modules/vision/training/include/SushiAI/vision/training/segmentation_metrics.hpp`. The
fixture is an `EngineTest`. A step holds one output of `Float32` logits of [1, 2, 2, 3] and a
batch whose `mask` role is `Int32` of [1, 2, 2], with the ignore index 255.

### TheMetricsEqualTheHandCount

The logits pick the classes 0, 1, 0, 1 against the mask 0, 2, 255, 1. `measure_segmentation`
reports two metrics: `mean_iou` 0.5 and `pixel_accuracy` 2/3, each within 1e-15. The
`mean_iou` metric has `is_fraction` set.

### AStepThatDoesNotFitIsRefused

`measure_segmentation` fails with `InvalidDescription` for a step of two outputs
("published 2") and for a batch without a mask ("'mask'"). It fails with `Incompatible` for a
mask with a 3 at pixel 1 ("true class of pixel 1", "is 3") and for extents of 4 classes over
logits of 3 ("the logits"). No metric is reported by the four calls.

## modules/core/core/tests/test_shape.cpp

Suites: `Unit_Shape`.

Tests Shape's extents, indexing, validation, equality and hash.

This file needs no device. Shape is host-lane vocabulary and must stay testable without a
runtime, which is what keeps the unit tier fast.

### DefaultIsAScalarWithOneElement

A default `Shape` has rank 0 and one element, and prints as "[]".

### ExtentsAreStoredOutermostFirst

`Shape{64, 784}` has rank 2, `dim(0)` 64, `dim(1)` 784 and 64 * 784 elements, and prints as
"[64, 784]".

### NegativeIndicesCountFromTheInnermost

On `Shape{2, 3, 4}`, `dim(-1)` is 4, `dim(-2)` is 3 and `dim(-3)` is 2.

### OutOfRangeDimensionThrows

On `Shape{2, 3}`, `dim(2)` and `dim(-3)` throw `Error`.

### NonPositiveExtentIsRejectedAtConstruction

`Shape({2, 0})` and `Shape({-1})` throw `Error`.

### RankBeyondTheMaximumIsRejected

A list of nine extents, more than `K_MAX_RANK`, throws `Error`.

### BuildsFromExtentsKnownOnlyAtRunTime

A `Shape` built from a span holding 64 and 784 equals `Shape{64, 784}` and prints as
"[64, 784]".

### BuildsTheScalarFromAnEmptyExtentSpan

A `Shape` built from an empty span has rank 0 and one element.

### RuntimeExtentsGetTheSameValidationAsAListDoes

A span holding 4 and 0 throws `Error`, and so does a span of `K_MAX_RANK` + 1 ones.

### EqualityComparesRankAndEveryExtent

[4, 5] equals [4, 5] and differs from [5, 4], from [4, 5, 1], and [4] differs from the
scalar.

### HashSeparatesShapesThatDifferOnlyByRank

[4, 1] and [4] hold the same element count and the same leading extent. A hash that folded only
the extents would collide them, and the tensor pool keys its constant operands on this hash.

### WorksAsAnUnorderedMapKey

An `unordered_map` keyed by `Shape`, written twice under [2, 2] and once under [4], holds two
entries with the values 2 and 7.

## modules/layers/nn/tests/test_silu.cpp

Suites: `Unit_SiLU`.

Tests the SiLU layer's trace, shape inference and empty state.

This file needs no device. `NN::SiLU` in `modules/layers/nn/include/SushiAI/nn/silu.hpp` has no parameter and no
attribute.

Three `static_assert` lines at the top of the file hold that `SiLU` is a `Module`, is not a
`ModuleContainer`, and that `SiLU::fields()` is an empty tuple.

### ForwardEmitsOneSiLUNodeReadingTheInput

`forward` adds one node. Its operation is `Ops::K_SILU`, its one input is the value passed in
and its one output is the value returned.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

Over a [2, 4] input the result has the shape [2, 4] and the input's element type.

### ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

Over a [2, 8, 6, 3] input the result has that shape and the layout `TensorLayout::NHWC`.

### ForwardCarriesTheLayoutOfItsInput

Over a value converted to NCHW with `layout_convert`, the result is tagged NCHW and has the
converted value's shape.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## modules/core/autograd/tests/test_spatial_shapes.cpp

Suites: `Unit_SpatialShapes`.

Tests the shapes the builder infers for spatial operations and the layouts it refuses.

### Im2ColProducesTheColumnMatrixAGEMMExpects

[N,H,W,C] with a 3x3 stride-1 pad-1 window over 5x5 gives 5x5 windows per image. Each row is
one window's kernel_h * kernel_w * C taps with channels innermost, so a sub-group reads a
contiguous run (CV-3).

### Im2ColHonoursStrideAndDilation

A dilated 3x3 covers 5 pixels, so an unpadded 7 yields 3 windows, and stride 2 over the same
input yields 2. Both numbers are wrong under the naive (H - K)/S + 1 that forgets dilation.

### Im2ColRejectsAWindowLargerThanThePaddedInput

The oversized window produces a zero extent, which Shape refuses anyway. The message a caller
gets should still name the geometry, not the Shape invariant it happened to trip.

### Im2ColRejectsANonSpatialInput

`im2col` over a [4, 8] input throws `Error`.

### Col2ImReturnsToTheImageItWasAskedFor

col2im cannot derive its output extents. ConvGeometry carries pad_top and pad_left but no
pad_bottom or pad_right, so an asymmetric padding would be silently mis-shaped. The caller
states the image shape instead. It always has it, because it held the value im2col consumed.

### Col2ImRejectsAColumnMatrixThatDoesNotMatchTheImage

`col2im` of the column matrix of a [2, 5, 5, 3] image with a padded 3x3 window, asked for the
image shape [2, 6, 6, 3], throws `Error`.

### Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward

A 3x3 stride-1 window with padding only on the bottom and right of a 13x13 input still yields
13x13 windows, which is not what the symmetric formula (13-3)/1+1 = 11 would say. The forward
pass (`im2col_with_extents`) states 13x13 outright. The backward rule must be able to state the
same numbers back, because it is the one holding them, and must not re-derive them from
ConvGeometry alone.

The geometry leaves pad_top and pad_left at 0. Padding was applied on the bottom and right
only, which ConvGeometry cannot express and does not need to.

### Col2ImDerivingStillRejectsThatSameAsymmetricPair

This case is the point of having `col2im_with_extents` as a twin. The deriving col2im assumes
symmetric padding and is not a substitute for it. Feeding it the exact cols/image pair that
`col2im_with_extents` accepted in Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward must
still throw, which proves the twin is needed.

### Im2ColRejectsAnNCHWInput

im2col reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it would read channels
as height and produce a plausible but silently wrong shape, where it should throw.

### MaxPoolProducesAValueAndItsArgmaxTogether

The indices are a second result of the node, not a second output tensor the caller allocates.
The backward rule needs exactly the indices this forward produced, and a pair returned together
cannot be mismatched the way two separate calls could.

### MaxPoolHandlesYOLOTinysAsymmetricThirteenToThirteen

YOLOv3-tiny's last pool is 2x2 stride 1 over 13x13, padded only on the bottom and right, staying
13x13. ConvGeometry cannot express that padding, so the caller states the extents. This case
proves the explicit overload is needed.

### AvgPoolProducesOnlyAValue

avgpool2d records no indices: its backward divides by a constant window area and needs nothing
the forward recorded (CV-6).

### UpsampleMultipliesTheSpatialExtentsOnly

`upsample2d` of [2, 3, 5, 7] by 2 and 2 gives [2, 6, 10, 7].

### UpsampleRejectsANonPositiveScale

`upsample2d` with a height scale of 0 throws `Error`.

### MaxPoolRejectsAnNCHWInput

maxpool2d reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it must throw and
must not pool the channel axis.

### AvgPoolRejectsAnNCHWInput

`avgpool2d` over a value converted to NCHW throws `Error`.

### UpsampleRejectsAnNCHWInput

`upsample2d` over a value converted to NCHW throws `Error`.

### SpatialMeanReducesToNAndC

The result is the [N,C] that the numeric Probe's loss needs, which is why every convolutional
gradient test in this plan ends with this call.

### SpatialMeanRejectsAnNCHWInput

`spatial_mean` over a value converted to NCHW throws `Error`.

### ConcatSumsTheChannelAxisAndNothingElse

`concat_channels` of [2, 8, 8, 16] and [2, 8, 8, 32] gives [2, 8, 8, 48].

### ConcatRejectsOperandsThatDisagreeOffTheChannelAxis

`concat_channels` of [2, 8, 8, 16] and [2, 4, 8, 16] throws `Error`.

### ConcatRejectsAnNCHWOperand

`concat_channels` of two values converted to NCHW throws `Error`.

### SliceTakesAChannelRangeAndRefusesToOverrun

Over [2, 8, 8, 48], `slice_channels(x, 16, 32)` gives [2, 8, 8, 32] and
`slice_channels(x, 32, 32)` throws `Error`.

### SliceRejectsAnNCHWInput

`slice_channels` over a value converted to NCHW throws `Error`.

### ChannelAffineKeepsTheShapeAndChecksTheParameters

`channel_affine` of [2, 8, 8, 16] with a [16] scale and a [16] shift gives [2, 8, 8, 16].
With an [8] scale it throws `Error`.

### ChannelAffineRejectsAnNCHWInput

`channel_affine` over a value converted to NCHW throws `Error`.

### BatchNorm2dKeepsItsShapeAndDeclaresThreePerChannelStatistics

`GraphBuilder::batchnorm2d` over a [2, 8, 8, 16] FLOAT64 input returns four values: the output,
of the input's shape, and `saved_mean`, `saved_inv_std` and `batch_variance`, each [16]. They
are the four outputs of one K_BATCHNORM2D node in that order, and attribute 0 of the node is the
epsilon 1e-5. Each of the four is FLOAT64 with the lifetime `ACTIVATION`. A gamma of [8]
throws.

A non-positive epsilon is a division by the standard deviation itself, which is exactly zero
for a constant channel.

### BatchNorm2dRejectsAnNCHWInput

The layout check runs before any extent is read, so the operand that is wrong is the one the
message names.

### BatchNormFoldYieldsAPerChannelScaleAndShift

`GraphBuilder::batchnorm_fold` over a [16] gamma, beta, running mean and running variance adds
one node. Its operation is `Ops::K_BATCHNORM_FOLD`, its inputs are those four values in that
order, its outputs are `fold.scale` and `fold.shift`, and attribute 0 is the epsilon 1e-3.
Both results are [16] FLOAT64 values of lifetime `ACTIVATION`, the operands' type. A
`channel_affine` of a [2, 8, 8, 16] input with the two results keeps that shape.

### BatchNormFoldRejectsAnOperandThatDoesNotMatchGamma

Five calls throw with "batchnorm_fold" and the operand at fault: a [8] mean ("mean", "[16]",
"[8]"), a [8] beta ("beta"), a FLOAT64 variance among FLOAT32 operands ("variance", "float32",
"float64"), a [4, 4] gamma ("rank-1") and four INT64 operands ("int64"). The graph's node count
and value count are unchanged after all five.

### BatchNormFoldRejectsANonPositiveEpsilon

An epsilon of 0.0 and one of -1e-3 each throw with a message containing "batchnorm_fold",
"epsilon" and "positive". An epsilon of 1e-3 does not throw.

### SumChannelsFoldsEverythingButTheChannelAxis

`sum_channels` of [2, 8, 8, 16] gives [16], and over the same value converted to NCHW it
throws `Error`.

### ReshapeRefusesToChangeTheElementCount

A reshape has one failure mode. It moves no data, so a mismatched count is not a resize; it is
a read past the end.

### Convolution, window extents and upsampled extents

The cases from here cover three additions to `modules/core/graph/include/SushiAI/graph/builder.hpp`.
`GraphBuilder::conv2d` emits a convolution as a node sequence: a K_RESHAPE of the rank-4 weight
to [C_out, taps], an K_IM2COL of the input, a K_MATMUL of the two with its second operand
transposed, an optional K_BIAS_ADD, and a K_RESHAPE of the rows to [N, out_h, out_w, C_out].
`Graph::window_extents` computes the output extents of a window and validates its geometry,
and `conv2d`, `im2col`, `maxpool2d` and `avgpool2d` all derive their extents through it.
`Graph::upsampled_extents` does the same for K_UPSAMPLE2D.

### op_names_from

`op_names_from` returns the operation names of a graph's nodes from a given index on. A case
records the node count after it has declared its operands and passes that count, so the list
holds only what the emitter under test added.

### square_window

`square_window` returns a unit-stride geometry with a square kernel and the same padding on
both axes.

### window

`window` returns a unit-stride unpadded geometry with the given kernel height and width.

### Conv2dReturnsAnNHWCValueWithOneChannelPerFilter

A [2, 5, 5, 3] input, an [8, 3, 3, 3] weight and an [8] bias under a padded 3x3 window give a
[2, 5, 5, 8] value tagged NHWC.

### Conv2dDividesTheOutputExtentsByItsStride

A padded 3x3 window with stride 2 over a [2, 8, 6, 3] input gives [2, 4, 3, 4]. The call
passes `K_NO_VALUE` as the bias.

### Conv2dWithABiasEmitsIm2colMatmulAndBiasAddBetweenTwoReshapes

The nodes the call adds are, in order, "sushiai.reshape", "sushiai.im2col", "sushiai.matmul",
"sushiai.bias_add" and "sushiai.reshape". The sequence holds no operation of its own, which is
decision CV-1: `test_ops.cpp` asserts under ThereIsNoConv2dIdentity that no CONV2D identity
exists.

### Conv2dWithoutABiasEmitsNoBiasAdd

With `K_NO_VALUE` as the bias the same call adds four nodes: reshape, im2col, matmul, reshape.

### Conv2dContractsTheColumnMatrixAgainstTheTransposedFilterMatrix

The third node added is a K_MATMUL. Its first operand has shape [2 * 5 * 5, 3 * 3 * 3], one row
per window, and its second has shape [8, 3 * 3 * 3], one row per filter. Flag 0 is false and
flag 1 is true, so the filter matrix is the transposed operand and the product has one column
per filter.

### Conv2dPointwiseEmitsNoIm2col

A 1x1 unit-stride unpadded window adds reshape, reshape, matmul, reshape: the input is
reshaped to one row per pixel and no K_IM2COL is emitted. The result is [2, 5, 5, 8].

### Conv2dStridedOneByOneKernelStillEmitsIm2col

A 1x1 window with stride 2 over a [2, 6, 6, 3] input adds reshape, im2col, matmul, reshape and
gives [2, 3, 3, 8]. A stride above one selects pixels, which a reshape cannot do, so the
shortcut of the previous case must not apply.

### Conv2dTakesANonSquareKernelInHeightThenWidthOrder

An [8, 1, 3, 3] weight under a 1x3 window over a [2, 5, 5, 3] input gives [2, 5, 3, 8]. Axis
1 of the weight is the kernel height and axis 2 the kernel width.

### Conv2dRejectsAWeightWhoseKernelExtentsDifferFromTheWindow

An [8, 3, 1, 3] weight under the same 1x3 window throws. It holds as many elements as
[8, 1, 3, 3], so the reshape that follows accepts it and only the check in `conv2d` refuses
the transposed kernel.

### Conv2dRejectsAWeightWhoseInputChannelsDifferFromTheImage

An [8, 3, 3, 4] weight over a three-channel input throws with a message containing "conv2d".
The reshape rejects the same weight by element count in a message that does not name the
convolution, so the case requires the name.

### Conv2dRejectsAFlattenedWeight

An [8, 27] weight throws. The weight enters `conv2d` as rank 4 and is flattened inside it.

### Conv2dRejectsARankThreeInput

A [5, 5, 3] input, which lacks the batch axis, throws.

### Conv2dPointwiseRejectsAnNCHWInput

A [2, 5, 5, 3] input is converted to NCHW and paired with an [8, 1, 1, 5] weight, whose
input-channel extent matches the last axis of the converted shape. The pointwise path skips
K_IM2COL, which is the emitter that checks the layout, so `conv2d` has to check it itself. The
call throws.

### Conv2dRejectsABiasWhoseExtentDiffersFromTheFilterCount

A [4] bias with eight filters throws.

### Conv2dRejectsAWindowLargerThanThePaddedInput

A 5x5 unpadded window over a [1, 2, 2, 3] input throws.

### Conv2dEmitsNothingWhenItRejectsItsOperands

Three calls throw: an oversized window, a weight with four input channels over a
three-channel input, and a [4] bias with eight filters. The node count afterwards equals the
count taken before the calls. `conv2d` validates every operand before it emits its first
reshape, so a refused call leaves no orphan node in the graph.

### Conv2dBuildsAGraphThatFinaliseAccepts

A biased 3x3 convolution feeds a pointwise one, and `Graph::finalise()` does not throw. Both
paths therefore leave every value with a layout tag the per-node proof accepts.

### Conv2dDifferentiatesToAWeightAndABiasGradient

The convolution's result goes through `spatial_mean`, `log_softmax` and `nll_loss`.
`Autograd::differentiate` returns two gradients other than `K_NO_VALUE`, of shapes [8, 3, 3, 3]
and [8]. The weight gradient has the rank-4 shape of the parameter and not the [8, 27] shape
of the reshaped operand the K_MATMUL read.

### WindowExtentsPadBothSidesOfAnAxisByTheLeadingPad

A 3x5 window with a top pad of 1 and a left pad of 2 over a 7 by 9 input yields 7 by 9. The
extents are unchanged only if each pad is applied to both sides of its axis.

### WindowExtentsHonourStrideAndDilationPerAxis

A 3x3 window with stride 2 by 3 and a height dilation of 2 over an 11 by 16 input yields 4 by
5. The two axes differ in stride and in dilation, so a value taken from the wrong axis changes
one of the two numbers.

### WindowExtentsOfAWindowThatExactlyFitsAreOneByOne

A 5x5 window with padding 1 over a 3 by 3 input yields 1 by 1. The padded input is exactly as
large as the window.

### WindowExtentsRejectANonPositiveKernel

A kernel of 0 by 3 and one of 3 by -2 throw. Each message contains "positive kernel" and the
kernel as given, "kernel 0x3" and "kernel 3x-2".

### WindowExtentsRejectANonPositiveStrideAndNameIt

A vertical stride of 0 and a horizontal stride of -1 throw with "positive stride" and the
pair, "stride 0,1" and "stride 1,-1". A zero stride is a division by zero in the extent
formula, and the message names the stride and not a window that does not fit.

### WindowExtentsRejectANonPositiveDilation

A horizontal dilation of 0 throws with "positive dilation" and "dilation 1,0".

### WindowExtentsRejectANegativePadding

A padding of -1 on both axes throws with "padding -1,-1", and a left pad of -1 alone throws
with "padding 0,-1".

### WindowExtentsRejectAWindowThatDoesNotFit

A 5x5 window over a 4 by 6 input throws with a message containing "5x5 window" and "4x6
input". The window fits the width and not the height.

### UpsampledExtentsScaleEachAxisByItsOwnFactor

A 3 by 5 input with scales 2 and 4 yields 6 by 20.

### UpsampledExtentsRejectAScaleBelowOne

Scales of 0 by 2 and of 2 by -1 throw with "at least one" and the pair, "0x2" and "2x-1".

### UpsampledExtentsRejectAProductBeyondSixtyFourBits

With a scale of 2^62, an extent of 5 on the height axis throws with "fit in 64 bits", and so
does an extent of 5 on the width axis. Each call overflows one axis and leaves the other at
1, so both products are checked.

### UpsampledExtentsAcceptTheLargestProductThatFits

Half of the largest int64, rounded down, scaled by 2 yields the largest int64 minus 1. The
overflow check therefore refuses nothing that 64 bits hold.

### UpsampleRejectsAScaledExtentBeyondSixtyFourBits

`GraphBuilder::upsample2d` with a horizontal scale of 2^62 over a width of 5 throws with "fit
in 64 bits", and the node count is unchanged.

### TheDerivingEmittersRejectANegativePadding

`im2col`, `conv2d`, `maxpool2d` and `avgpool2d` are each called with a padding of -1, and each
message contains "padding". These are the four emitters that derive their extents from the
geometry.

### Col2ImRejectsANegativePadding

`col2im` with a padding of -1 throws with a message containing "padding".

### Col2ImRejectsAnImageShapeThatIsNotRankFour

`col2im` asked for the image shape [5, 5, 3] throws with a message containing "col2im" and
"rank-4".

### TheDerivingEmittersNameTheLayoutBeforeTheWindow

A [1, 8, 8, 2] input is converted to NCHW, which leaves 2 on the axis an NHWC reader takes
for the height, and a 3x3 window does not fit that. `im2col`, `maxpool2d` and `avgpool2d` each
throw a message containing their own name and "NHWC". A caller who passed the wrong layout is
told so, and is not told that the window is too large.

## modules/core/core/tests/test_status.cpp

Suites: `Unit_Status`.

Tests Status, its macros, the fatal family and the assertions of test_status.hpp.

This file needs no device. `Status` and the `SA_FAIL_IF`, `SA_TRY` and `SA_ANNOTATE` macros are
in `modules/core/core/include/SushiAI/core/status.hpp`, and the fatal checks are in
`modules/core/core/include/SushiAI/core/check.hpp`. The assertions `SA_EXPECT_OK`, `SA_EXPECT_FAILURE` and
`SA_EXPECT_FATAL` come from `tests/common/test_status.hpp`. `SA_EXPECT_FATAL` is a death test:
the statement runs in a child process and the needles are matched against what it wrote to
stderr.

Six `static_assert` lines at the top of the file hold that a `Status` is the size of one
pointer and that its default construction, copy, move and both assignments are `noexcept`.

Several cases read `Detail::live_failure_count()`. A failed `Status` owns one failure block and
a successful one owns none, so a count that did not move means the code under test allocated
nothing for a failure.

### counted

`counted` returns its value and increments a counter. A case passes it as a format argument to
find out whether a macro evaluated its arguments.

### failure_of

`failure_of` builds a failure of a given code and message through `Status::from_message`.

### check_extent

`check_extent` fails with `StatusCode::InvalidDescription` through `SA_FAIL_IF` when its extent
is not positive. The message's argument goes through `counted`.

### forward_extent

`forward_extent` calls `check_extent` under `SA_TRY` and sets a flag after it, so a case can
tell whether control went past the macro.

### FatalRecord

`FatalRecord` holds the call count, file, line and message of the last report the recording
handler received.

### fatal_record

`fatal_record` returns the one static `FatalRecord` of the file.

### record_fatal

`record_fatal` is a `FatalHandler` that stores its report in `fatal_record()`.

### announce_fatal

`announce_fatal` is a `FatalHandler` that writes "recorded: " and the message to stderr, where
a death test reads it.

### stop_when

`stop_when` runs `SA_FATAL_IF` on its condition with the message "extent {} is refused" and
returns the line number of that check. The extent goes through `counted`.

### stop_with_throwing_composer

`stop_with_throwing_composer` calls `Detail::fail_fast_composed` with a composer that throws
`std::runtime_error`.

### stop_under_announcer

`stop_under_announcer` installs `announce_fatal` and then runs `SA_FATAL("stop {}", 3)`.

### assert_positive

`assert_positive` runs `SA_ASSERT(value > 0)`. It is compiled only when `NDEBUG` is not
defined.

### DefaultIsSuccessWithNoMessage

A default-constructed `Status` reports `is_ok()`, the code `StatusCode::Ok` and an empty
message, and `SA_EXPECT_OK` accepts it.

### EveryCodeHasALowerCaseName

`status_code_name` returns "ok", "invalid_description", "file_not_found", "invalid_file",
"incompatible", "unsupported", "out_of_memory" and "device_failed" for the eight codes, and
"unknown" for the value 200, which is no enumerator.

### FromMessageCarriesTheCodeAndTheText

A status built from `StatusCode::InvalidFile` and "the header is 3 bytes short" is not ok and
returns that code and that text.

### SuccessPathCreatesNoFailureBlock

The case constructs, copies, moves, copy-assigns and move-assigns successful statuses and
calls `annotated` on one. The annotation's composer never runs, both results are ok, and
`live_failure_count()` is what it was before.

### FailIfFormatsOnlyWhenTheConditionHolds

`check_extent(4)` succeeds with zero calls to `counted` and no failure block. `check_extent(-2)`
fails with `InvalidDescription` and a message containing "extent must be positive" and
"got -2", after one call to `counted`. The count of live blocks is back at its start once the
temporary is gone.

### CopiesShareOneBlockAndTheLastOneFreesIt

A failure, a copy of it and a status assigned from the copy leave `live_failure_count()` one
above its start, and the third status's `message().data()` is the first one's pointer. After
the scope closes the count is back at its start.

### MovedFromStatusKeepsItsFailure

A failed status is moved from twice, once by construction and once by assignment. The source
and both targets still report `Unsupported` and "no kernel for int64".

### AssignmentReplacesTheHeldFailure

Assigning a second failure to a status that holds one leaves one live block and the second
failure's code and message. A self-assignment through a const reference keeps the failure.
Assigning `Status()` makes the status ok and frees the block.

### AnnotatedPrefixesTheMessageAndKeepsTheCode

`SA_ANNOTATE(inner, "Conv2d({}, {})", 3, 16)` over the failure "stride must be positive"
returns a status with the same code and the message "Conv2d(3, 16): stride must be positive".
The inner status's message is unchanged.

### ThrowingComposerYieldsTheStaticOutOfMemoryFailure

`Status::from_message` with a composer that throws returns a failure of code `OutOfMemory`
whose message contains "message could not be built". `live_failure_count()` does not move, so
that failure is not a heap block.

### ThrowingAnnotationKeepsTheOriginalFailure

`annotated` with a composer that throws returns the code `InvalidFile` and the message "wrong
magic" of the status it was called on.

### TryForwardsAFailureAndPassesASuccess

`forward_extent(0)` returns `check_extent`'s failure, with "got 0" in the message, and leaves
its flag false. `forward_extent(5)` succeeds and sets the flag.

### FromMessageWithTheOkCodeIsFatal

`Status::from_message(StatusCode::Ok, ...)` stops the process with a report containing
"Status::from_message needs a failure code" and "got ok".

### StatusNamesRejectsSuccessAWrongCodeAndAMissingNeedle

`Test::status_names` is called directly. Over the failure `InvalidFile` "wrong magic 0x1F" it
is true for the right code with both needles and with no needle. It is false for a successful
status, for the code `FileNotFound` and for the needle "wrong version". `Test::status_is_ok` is
false for the failure.

### DeathPatternEscapesAndJoinsItsNeedles

`Test::death_pattern` is called directly. One plain needle is returned as it is. Two needles
are joined with `.*`, and each of the characters `. [ ] ( ) * { } | ^ $ + ? \` is escaped with
a backslash.

### FatalHandlerIsReplacedByARecorderAndRestored

`set_fatal_handler(&record_fatal)` returns a handler that is not null. One call of
`Detail::report_fatal("layer.cpp", 42, "a sealed trace was reused")` reaches the recorder
once, with that file, line and message. Restoring the original returns `&record_fatal`, and
setting the original again returns the original.

### NullFatalHandlerInstallsTheDefaultOne

`set_fatal_handler(nullptr)` returns the handler that was installed, `&record_fatal`. The
handler in place after it is not null and is not the recorder.

### FatalIfPassesAndFormatsNothingWhenTheConditionIsFalse

`stop_when(false, ...)` returns and never calls `counted`.

### FatalIfPrintsTheMessageThenTheFileAndTheLineAndStops

`stop_when(true, ...)` stops the process with a report containing "sushiai: fatal: extent 7 is
refused" followed by "test_status.cpp:", the line `stop_when` returned and ")". The case calls
`stop_when(false, ...)` first to learn that line.

### FatalCheckWhoseComposerThrowsReportsTheFixedMessage

`stop_with_throwing_composer` stops the process with a report containing "sushiai: fatal:",
"its message could not be composed" and "test_status.cpp:".

### ReplacedFatalHandlerRunsBeforeTheStop

`stop_under_announcer` stops the process and its stderr contains "recorded: stop 3", the line
`announce_fatal` writes.

### AssertChecksInDebugBuildsOnly

With `NDEBUG` defined, `SA_ASSERT(counted(evaluations, 0) == 1)` does not evaluate its
condition: the counter stays at zero. Without it, `assert_positive(1)` returns and
`assert_positive(0)` stops the process with "assertion failed: value > 0".

## modules/core/graph/tests/test_step_recorder.cpp

Suites: `Unit_StepRecorder`, `Unit_StepRecorderDeathTest`.

Tests that StepRecorderList drives its members as one recorder, in the order added.

This file needs no device. `Graph::StepRecorder` and `Graph::StepRecorderList` are in
`modules/core/graph/include/SushiAI/graph/step_recorder.hpp`. The list is itself a `StepRecorder` and holds
references to its members; it does not own them. `add` returns a Status, and each case checks
its call with `SA_EXPECT_OK`.

### JournalRecorder

`JournalRecorder` appends "<name>.record", "<name>.before" or "<name>.after" to a journal the
test owns, and `record_into_step()` returns the task count given at construction.

### RecordOnlyRecorder

`RecordOnlyRecorder` overrides `record_into_step()` alone and returns 3. Its `before_replay()`
and `after_replay()` are the interface's own.

### in_order_added

`in_order_added` returns the three journal entries that recorders named a, b and c write for
one call.

### EmptyListRecordsNothing

An empty list reports `empty()`, returns 0 from `record_into_step()` and does not throw from
either replay hook.

### AddMakesTheListNonEmpty

After one `add` the list is not empty and the journal is still empty, so `add` calls nothing
on the member.

### RecordSumsTheMemberCounts

Members that report 2, 0 and 5 tasks give `record_into_step()` 7.

### RecordCallsMembersInTheOrderAdded

`record_into_step()` over a, b and c journals "a.record", "b.record", "c.record".

### OrderFollowsAddNotConstruction

With b added before a, the journal is "b.record", "a.record", although a was constructed
first.

### BeforeReplayCallsMembersInTheOrderAdded

`before_replay()` over a, b and c journals "a.before", "b.before", "c.before".

### AfterReplayCallsMembersInTheOrderAdded

`after_replay()` over a, b and c journals "a.after", "b.after", "c.after". The members are not
called in reverse.

### MemberWithoutReplayHooksIsDriven

A list of a `RecordOnlyRecorder` and a journal recorder of 4 tasks records 7. After both replay
hooks the journal holds "a.record", "a.before" and "a.after" only.

### ListIsDrivenThroughTheInterface

Through an `StepRecorder*`, a list of a and b records 5 tasks and journals the record calls,
then the before calls, then the after calls, each as a then b.

### NestedListKeepsItsPlaceInTheOrder

An outer list of a, an inner list holding b, and c records 7 tasks and journals "a.record",
"b.record", "c.record".

### AddRejectsARecorderAlreadyHeld

A second `add` of the same recorder is a programmer's misuse, so it stops the process with a report
containing "StepRecorderList::add" and "already". The case runs as a death test in
`Unit_StepRecorderDeathTest`. The parent process then checks that the list still records that
member once: 2 tasks and one journal entry.

### AddRejectsTheListItself

`list.add(list)` stops the process with a report containing "StepRecorderList::add" and "itself".
It is a death test in `Unit_StepRecorderDeathTest`, and the list stays empty.

## modules/core/optim/tests/test_step_schedule.cpp

Suites: `Unit_StepSchedule`.

Tests `Optim::StepSchedule::rate_at` and `Optim::validate_step_schedule`. Every case starts from
the base rate 0.001, and the warm-up cases use 1000 warm-up updates at the default power 4. A
rate is compared with a value the test computes in double, within 1e-15 plus 1e-12 times the
expected value.

### ConstantPolicyKeepsBaseRate

With no warm-up and no step, the rate at iterations 0, 1 and 1000000 is the base rate.

### WarmupRisesAsPowerOfIterationFraction

During the warm-up the rate is the base rate times (iteration / 1000) to the power 4. The case
checks iterations 0, 1, 500 and 999, and the rate at 0 is 0.

### WarmupEndsAtItsLengthWithTheBaseRate

At iterations 1000 and 1001 the rate is the base rate.

### WarmupPowerIsTheExponentOfTheRamp

With `warmup_power` 1 the rate at iteration 250 is a quarter of the base rate. With
`warmup_power` 2 the rate at iteration 500 is a quarter of the base rate.

### StepAppliesAtItsListedIteration

With steps at 400 and 450, each of scale 0.1, the rate is the base rate at 399, a tenth of it
from 400 to 449 and a hundredth of it at 450 and 451.

### WarmupReplacesThePolicyUntilItsLength

With a step at 500 of scale 0.1 inside the warm-up, the rates at 500 and 999 are the plain
warm-up values. At 1000 the rate is a tenth of the base rate.

### StepAtFirstIterationPastWarmupApplies

With a step at 1000 of scale 0.5, the rate at 999 is the warm-up value and the rate at 1000 is
half the base rate.

### ValidateAcceptsEmptyAndAscendingSchedules

`validate_step_schedule` is ok for a default `StepSchedule` and for a warm-up schedule with
steps at 400 and 450 and two scales.

### ValidateRefusesDescendingSteps

Steps listed as 450 then 400 fail with `InvalidDescription`.

### ValidateRefusesRepeatedStep

Steps listed as 400 and 400 fail with `InvalidDescription`.

### ValidateRefusesScalesLongerThanSteps

One step with two scales fails with `InvalidDescription`.

### ValidateRefusesStepsLongerThanScales

Two steps with one scale fail with `InvalidDescription`.

### ValidateRefusesNegativeWarmup

A `warmup_iterations` of -1 fails with `InvalidDescription`.

### ValidateRefusesNegativeWarmupPower

A `warmup_power` of -1 fails with `InvalidDescription`.

## modules/core/optim/tests/test_step_schedule_section.cpp

Suites: `Unit_StepScheduleSection`.

Tests `Optim::read_step_schedule_section` and `Optim::write_step_schedule_section`. The full
section states `type`, `base_rate`, `warmup_iterations`, `warmup_power`, `steps` and `scales`.
A round trip writes a schedule to a section, the section to JSON text, parses the text and
reads it, and compares every field exactly.

### TheDefaultsRoundTripThroughTheirSection

A default `StepSchedule` round-trips.

### AWarmupAndTwoStepsRoundTripThroughTheirSection

A schedule with the base rate 0.00261, a warm-up of 1000 at the power 2.5, steps at 400000 and
450000 and the scales 0.1 and 0.3 round-trips.

### AFullSectionRoundTripsThroughItsSchedule

The full section, read to a schedule and written back, gives the same JSON text as the section
itself.

### ReadsEveryFieldOfASection

The full section reads as the base rate 0.00261, a warm-up of 1000, the power 4, the steps
400000 and 450000 and the scales 0.1 and 0.1.

### AKeyLeftOutKeepsTheDefaultOfItsField

`{}` and `{"type": "step"}` each read as a default `StepSchedule`. `{"base_rate": 0.01}` reads
as a default schedule whose base rate is 0.01.

### WritesTheTypeAndEveryKeyInOrder

The written section has the keys `type`, `base_rate`, `warmup_iterations`, `warmup_power`,
`steps`, `scales`, in that order. `type` is `K_STEP_SCHEDULE_TYPE`, `warmup_iterations` is 1000
and the second step is 450000.

### ReadsAWholeNumberAsARateAPowerAndAScale

A section with `base_rate` 1, `warmup_power` 2 and the scale 3 reads as 1.0, 2.0 and 3.0.

### RefusesASectionThatIsNotAnObject

An array and a null value each fail with `InvalidDescription`, "schedule" and "an object".

### RefusesAnotherType

The type `cosine` fails with "schedule 'type'", `'cosine'` and `'step'`. A `type` of 1 fails
with "schedule 'type'" and "a string".

### RefusesAnUnknownKey

The key `burn_in` fails with `InvalidDescription` and a message containing "schedule",
`'burn_in'`, "'type', 'base_rate'" and `'scales'`.

### RefusesAValueOfAnotherKind

Five sections fail with `InvalidDescription`: a string `base_rate` ("schedule 'base_rate'", "a
number"), a `warmup_iterations` of 1.5 ("a whole number"), a `steps` of 100 ("an array"), a
step of 20.5 ("schedule 'steps' entry 1", "a whole number") and a scale of "half" ("schedule
'scales' entry 0", "a number").

### RefusesTheSchedulesTheScheduleCheckRefuses

The reader fails with `InvalidDescription` and "schedule" for two steps without scales ("2
steps", "0 scales"), for steps 200 then 100 ("strictly ascend"), for a warm-up of -1 ("-1",
"negative") and for a warm-up power of -2 ("-2", "negative").

### WriterRefusesANumberThatIsNotFinite

The writer fails with `InvalidDescription` and "finite" for an infinite base rate ("schedule
'base_rate'"), an infinite warm-up power ("schedule 'warmup_power'") and a NaN second scale
("schedule 'scales' entry 1").

### WriterRefusesTheSchedulesTheScheduleCheckRefuses

The writer fails with `InvalidDescription` and "schedule" for two steps with one scale ("2
steps", "1 scales") and for steps 450000 then 400000 ("strictly ascend").

## modules/layers/train/tests/test_steps.cpp

Suites: `Unit_Steps`.

Tests the builders that trace a model into a training, probe or evaluation graph, and the batch
shape function.

This file needs no device. `build_training_graph`, `build_probe_graph`, `build_evaluation_graph`
and `batch_shape` are declared in `modules/layers/train/include/SushiAI/train/steps.hpp`, and `Train::TrainingGraph`
is declared in `modules/layers/train/include/SushiAI/train/training_graph.hpp`. The constants `K_BATCH` (4),
`K_FEATURES` (6), `K_HIDDEN` (8), `K_CLASSES` (3), `K_CHANNELS` (3) and `K_SIDE` (2) set every
shape in the file. Each case reads the graph a builder returns, through its nodes, its leaves,
its pinned values, its buffer updates or its status.

### K_MLP_TRAINING_OPS

`K_MLP_TRAINING_OPS` lists the operation of each of the 17 nodes of the MLP's training graph
under `CrossEntropyObjective`, in node order.

### parameter_states

`parameter_states` returns the `state()` of every parameter of a model, in the order of
`NN::for_each_parameter`.

### build_two_output_model

`build_two_output_model` is `Test::build_two_output_model` from `tests/common/test_utils.hpp`,
which this file brings in with a using-declaration. It builds a Linear named "body", a GELU named
"activation" and a Linear named "head" over [6], and publishes "features" on the GELU and
"logits" on the head.

### NamedOutputObjective

`NamedOutputObjective` is a `Train::Objective` that declares one leaf, "targets", of
[batch, classes]. Its `loss()` is the cross-entropy of the output it was given by name against
that leaf. `score()` returns no metric, and `name()` returns "named_output".

### ATrainingGraphOfAnMLPHasTheNodesBuildGraphGave

The training graph of the MLP under cross-entropy has 17 nodes. Their operations, in node order,
equal `K_MLP_TRAINING_OPS`.

### ATrainingGraphListsEveryParameterWithItsLeafInThatGraph

The training graph lists four parameters, in the order of the parameter walk, and each entry
carries the walk's name. Each entry's value is the parameter's `id_in` in the built graph, and
neither its value nor its gradient is `K_NO_VALUE`.

### ATrainingGraphListsEveryParameterWithItsRole

The training graph of the two-layer MLP lists four parameters with the roles `Weight`,
`Bias`, `Weight`, `Bias`.

### WeightDecayAppliesToWeightsOnly

`takes_weight_decay` is true for `Weight` and false for `Bias`, `Scale` and `Embedding`. A
default `ParameterBinding` has the role `Weight`.

### ABatchNormModelYieldsTwoUpdatesAndPinsTheirSources

A `BatchNorm2d`, `Flatten` and `Linear` model over [4, 2, 2, 3] yields two buffer updates, and
the source of each update is pinned.

### ATrainingGraphPinsTheLossAndTheOutputs

For the same model, the loss and every value in `outputs` are pinned.

### AProbeGraphHasBatchStatisticsAndNoUpdate

The probe graph of a `BatchNorm2d` over [4, 2, 2, 3] has one `BATCHNORM2D` node, no
`BATCHNORM_FOLD` node and no buffer leaf, and it publishes one output.

### AnEvaluationGraphFoldsTheStoredStatistics

The evaluation graph of the same layer has one `BATCHNORM_FOLD` node, two buffer leaves and no
`BATCHNORM2D` node.

### OneModelTracedThreeTimesKeepsOneStateIDPerParameter

After the training graph, the probe graph and the evaluation graph are built from one
batch-norm classifier, its four parameters have the same state identities as after the training
graph alone, in walk order. Each state has a leaf in all three graphs.

### AModelGraphWithTwoOutputsLeavesLogitsUnset

`NamedOutputObjective("logits")` reads the model of `build_two_output_model`. The training graph
has `spec.logits` set to `K_NO_VALUE`, two entries in `outputs` and a set `spec.loss`.

### ARefusingLayerFailsTheBuilderWithItsStatus

`build_probe_graph` of a `BatchNorm2d(3)` over [1, 1, 1, 3] returns `StatusCode::InvalidDescription`,
with a message containing "BatchNorm2d(3)" and "more than one value per channel".

### BatchShapePrependsTheBatch

`batch_shape(16, [28, 28, 1])` is [16, 28, 28, 1]. A batch of 0 returns
`StatusCode::InvalidDescription`, with a message containing "batch" and "positive".

### AnEvaluationGraphAndAProbeGraphFeedTheModelInputAlone

The steps `build_probe_graph` and `build_evaluation_graph` trace list no fed leaf and no
carried value.

### ACarryingGraphFoldsTheStoredStatisticsAndUpdatesNoBuffer

For a `BatchNorm2d`, `build_carrying_graph` gives the same operation names as
`build_evaluation_graph`: no BATCHNORM2D node, two buffer leaves, no fed leaf and no carried
value.

### AnIncrementalGraphNamesTheNodeThatHasNoIncrementalForm

`build_incremental_graph` of a model whose one node is a `Linear` named "dense" fails with
`Unsupported`, "node 'dense'" and "no incremental form".

## modules/layers/train/tests/test_sum_objective.cpp

Suites: `Unit_SumObjective`.

Tests `Train::SumObjective` and its factory `Train::make_sum_objective`, reached through the
`sum` row of the registry. Most cases sum two `CrossEntropyObjective` terms over 4 rows of 3
classes. Two local objectives stand in for the others: `StepFedObjective`, which keeps the
base's `is_fed_by_batch`, and `ReportingObjective`, whose `measure` adds the two metrics it was
made with.

### MakeKeepsItsTermsInOrder

A sum of the weights 1 and 0.5 holds two terms with those weights in order. The second term's
objective is named "cross_entropy", the sum is named "sum" and is fed by the batch, and
`leaf_prefix(1)` is "term1.".

### MakeRefusesNoTermAndAWeightItCannotUse

`make` fails with `InvalidDescription` for no term ("at least one term"), a second weight of 0
("Term 1", "'cross_entropy'", "above zero"), a first weight of -0.5 ("Term 0"), a second weight
of infinity ("Term 1", "finite") and a first weight of NaN ("Term 0").

### MakeRefusesATermItsOwnStepFeeds

`StepFedObjective` reports false for `is_fed_by_batch` and `CrossEntropyObjective` reports
true. A sum whose second term is a `StepFedObjective` fails with `InvalidDescription`, "Term
1", `'step_fed'` and "fed by its own step".

### EachTermDeclaresItsLeavesUnderItsOwnPrefix

`declare` adds two leaves, `term0.targets` and `term1.targets`, with different ids, and no leaf
named `targets`. The second has the shape [4, 3]. Two targets are required and both have the
role `K_ROLE_LABELS`.

### TheLossAddsTheWeightedLossesInTheGraph

For the weights 1 and 0.5 the loss has the shape [1] and the graph holds two NLL_LOSS nodes,
one ADD and one SCALE whose factor is 0.5. The last node of the graph is the ADD and it writes
the loss.

### ATrainingGraphDifferentiatesTheSum

`Train::build_training_graph` over one `Linear` layer and a sum of the weights 2 and 0.5 gives
a loss of one element, `spec.targets` equal to `K_NO_VALUE` and both prefixed leaves. The two
parameters each have a gradient.

### MeasureKeepsTheFirstOfTwoMetricsOfOneName

With a first term that reports `accuracy` 0.25 and `precision`, and a second that reports
`recall` 0.75 and `accuracy` 1, `measure` gives three metrics in the order `accuracy`,
`precision`, `recall`. The accuracy is 0.25 and the recall is 0.75.

### TheFactoryBuildsEachTermThroughTheRegistry

A `terms` list of two `cross_entropy` entries, the second with the weight 0.5, gives a
`SumObjective` of two terms with the weights 1 and 0.5 whose first objective is named
"cross_entropy".

### ATermMayBeASum

A `terms` list holding a `cross_entropy` entry and a `sum` entry with one `cross_entropy` term
declares two leaves, `term0.targets` and `term1.term0.targets`.

### TheFactoryRefusesASectionItCannotRead

Nine sections fail with `InvalidDescription`: no `terms` ("the objective block", "'terms'"), a
`gamma` key, `terms` as an object ("an array"), an empty list ("at least one term"), an entry
that is a number ("'terms' entry 0", "an object"), an entry without `type`, a `weight` of
"half" ("entry 0 'weight'", "a number"), the type `yolo` ("'yolo'", "'cross_entropy'",
"'sum'") and a `smoothing` key in a term ("'terms' entry 0", "'smoothing'").

### TheFactoryNeedsTheRegistryOfItsContext

`make_sum_objective` called with a context that holds no registry fails with
`InvalidDescription`, "the objective block" and "registry".

## modules/core/data/tests/test_target.cpp

Suites: `Unit_DataTargetLookup`, `Unit_DataTarget`.

Tests `Data::find_target`, `Data::allocate_batch`, `Data::prepare_batch`,
`Data::describe_input` and the labels `Data::SyntheticClassification` publishes. The
`Unit_DataTarget` fixture needs an engine. The two roles of the cases are `labels`, one int32
per sample over 5 classes, and `boxes`, a float32 [3, 4] value per sample.

### ADescriptionIsFoundByItsRole

`find_target` over the two descriptions returns the address of the first for `K_ROLE_LABELS`
and of the second for `boxes`. It returns null for `mask` and for any role over an empty list.

### ABatchTargetIsFoundByItsRole

`find_target` over two `BatchTarget` values returns the address of each by its role and null
for `mask`.

### AllocateBatchYieldsOneTensorPerDescriptionInOrder

`allocate_batch` for 8 samples of a float32 [6] input gives [8, 6] float32 inputs and two
targets: `labels` as an int32 [8] tensor and `boxes` as a float32 [8, 3, 4] tensor.

### AllocateBatchWithNoDescriptionYieldsTheInputsAlone

With no description and a [4, 4, 1] sample, the inputs have rank 4 and the batch has no target.

### AllocateBatchGivesTheInputItsStatedElementType

For an int32 [7] input and 3 samples the inputs are an int32 [3, 7] tensor, and the `boxes`
target is float32.

### PrepareBatchKeepsABatchThatHoldsTheLayoutAlone

A second `prepare_batch` with the same batch size, input and roles leaves the data pointer of
the inputs unchanged and their type int32.

### PrepareBatchReplacesAnInputOfAnotherElementType

After `prepare_batch` with a float32 [7] input, a second call with an int32 [7] input leaves
int32 inputs of rank 2 whose second extent is 7.

### ADatasetStatesTheElementTypeOfItsInput

`SyntheticClassification` reports float32 for `sample_dtype()`, and `describe_input` of it
gives float32 and a sample shape of its feature count. `describe_input` of a
`MaterialisedDataset` built over an int32 [7] input gives int32 and [7].

### SyntheticClassificationPublishesItsLabelsInRotation

A synthetic set of two batches of 4 samples, 5 features and 3 classes publishes one target:
`K_ROLE_LABELS`, an empty sample shape, int32, 3 classes. Its sample shape is [5]. The labels
of batch 0 are 0, 1, 2, 0 and those of batch 1 are 1, 2, 0, 1.

## modules/language/text/tests/test_text_bpe_trainer.cpp

Suites: `Unit_TextBpeTrainer`.

Tests `Language::Text::train_bpe` in
`modules/language/text/include/SushiAI/language/text/bpe_trainer.hpp`. Unless a case says
otherwise the options state a vocabulary size and no special name, so the first merged token
has the id 257. The generated corpora are drawn with the seed 20261011.

### ItLearnsTheHandWorkedMergesOfATinyCorpus

Training on "abababc" to a size of 300 learns two merges, (97, 98) then (257, 257). The
vocabulary has 259 tokens and token 258 stands for "abab".

### PairsOfOneCountMergeBySmallerLeftThenRightID

Training on "ab ab cd cd" learns three merges in the order (32, 99), (97, 98), (257, 100).

### NoMergeCrossesTheStartOfABlankRun

Training on "a b a b a b" learns two merges, (32, 98) then (32, 97).

### TheSameCorpusGivesTheSameVocabulary

Two runs to a size of 400 over one corpus of 20000 bytes drawn from "abcdef", the blank and
the newline learn at least one merge and the same merges in the same order.

### ARunStoppedEarlyLearnsTheFirstMergesOfALongerRun

Over a corpus of 4000 bytes drawn from "abc" and the blank, a run to a size of 270 learns at
least one and at most 13 merges, and each equals the merge at the same index of a run to a
size of 290.

### ItStopsAtTheSizeAskedFor

Training on "abababc" to a size of 258 learns one merge, (97, 98).

### APairThatOccursOnceIsNotMerged

Training on "abcdef" to a size of 300 gives a vocabulary of 257 tokens, and so does training
on the empty text.

### SpecialTokensPrecedeTheMergedTokens

With the special name "<pad>", training on "abababc" learns two merges, the second of which
is (258, 258), and token 259 stands for "abab".

### ItRefusesASizeOutOfBounds

`train_bpe` fails with `InvalidDescription` for a size of 256 ("train_bpe", "256 tokens", "257
to"), a size of `K_MAX_VOCABULARY_SIZE + 1` ("train_bpe", "asked for") and an empty special
name ("special token 1").

## modules/language/text/tests/test_text_codec.cpp

Suites: `Unit_TextCodec`.

Tests `Language::Text::encode_text` and `Language::Text::decode_tokens` in
`modules/language/text/include/SushiAI/language/text/codec.hpp`. A round trip encodes a text,
checks that every id is in the vocabulary and is not a special token, and checks that the ids
decode to the same bytes.

### RandomBytesSurviveARoundTrip

With a vocabulary of 400 tokens trained on 20000 drawn bytes, the round trip holds for 64
texts of random bytes of up to 1999 bytes each, for the training corpus, for the empty text
and for eight bytes that are not valid UTF-8.

### ATextLongerThanOneChunkSurvivesARoundTrip

The vocabulary has the merge (97, 97) and seven merges that each join the token before with
itself. A text of 5000 'a' bytes makes the round trip and encodes to 21 ids: the first is 264,
the twentieth is 263 and the last is 259.

### MergesAreAppliedInTheOrderTheyWereLearned

With the merges (97, 98) and (257, 257), "abababc" encodes to 258, 257, 99. With the merges
(98, 99) and (97, 98), "abc ab" encodes to 97, 257, 32, 258.

### ATrainedCorpusEncodesToItsTrainedTokens

With the vocabulary trained on "ab ab cd cd" to a size of 300, that text encodes to 258, 32,
258, 259, 259.

### EncodeAppendsToTheIDsItIsGiven

Encoding "A" into a vector that holds 7 leaves 7, 65.

### ASpecialTokenDecodesToNoByte

Decoding the ids 72, `K_END_OF_TEXT`, 105 into a string that holds ">" leaves ">Hi".

### DecodeRefusesAnIDOutsideTheVocabulary

Over a vocabulary of 257 tokens, `decode_tokens` fails with `InvalidDescription` for the ids
72, 9999, 105 ("decode_tokens", "9999", "index 1", "[0, 257)") and for the id -3 ("-3", "index
0").

## modules/language/text/tests/test_text_stream_encoder.cpp

Suites: `Unit_TextStreamEncoder`.

Tests `Language::Text::encode_stream` in
`modules/language/text/include/SushiAI/language/text/stream_encoder.hpp` against
`encode_text`.

### EveryPieceSizeGivesTheIDsOfTheWholeText

The text is 40 repetitions of two lines with runs of blanks, a tab and a carriage return, then
a word of 600 'x' bytes and " the end". With a vocabulary of 300 tokens trained on it,
`encode_stream` gives the ids of `encode_text` for pieces of 1, 7, 64 and 1000 bytes and of
`K_STREAM_PIECE_BYTES`.

### AnEmptyStreamAppendsNothing

`encode_stream` over an empty stream leaves a vector that holds 7 unchanged. Over a stream of
"ab" it then leaves 7, 'a', 'b'.

## modules/language/text/tests/test_text_vocabulary.cpp

Suites: `Unit_TextVocabulary`.

Tests `Language::Text::Vocabulary` in
`modules/language/text/include/SushiAI/language/text/vocabulary.hpp`. A merge is written here
as (left, right).

### AVocabularyWithoutMergesHoldsTheBytesAndEndOfText

A vocabulary made of no merges has 257 tokens and a `first_merged()` of 257. The ids 0 and 255
are of the kind `Byte`, id 65 stands for "A" and id 255 for the byte 0xFF. `K_END_OF_TEXT` is
of the kind `Special` and stands for no byte, and `find_special(K_END_OF_TEXT_NAME)` is 256.
`contains` is true for 256 and false for 257 and -1.

### MergedTokensTakeIDsInTheOrderOfTheirMerges

With the merges (97, 98) and (257, 99) the vocabulary has 259 tokens, id 257 is of the kind
`Merged` and stands for "ab", and id 258 stands for "abc". `find_merge` returns 257 for
(97, 98) and 258 for (257, 99), and no value for (98, 97) and (-1, 97). `merges()` holds two
merges, the second of which is (257, 99).

### SpecialTokensFollowEndOfText

With one merge and the special names "<pad>" and "<mask>" the vocabulary has 260 tokens.
`find_special` returns 257 and 258 for the two names and no value for "<absent>".
`first_merged()` is 259, id 259 stands for "ab", and `special_names()` holds three names, the
first of which is `K_END_OF_TEXT_NAME`.

### MakeRefusesAMergeOfATokenItMayNotJoin

`Vocabulary::make` fails with `InvalidDescription` for the merge (97, 257) at index 0
("Vocabulary", "merge 0", "97 and 257"), (259, 97) at index 1 ("merge 1", "259 and 97"),
(`K_END_OF_TEXT`, 97) ("merge 0", "256 and 97") and (-1, 97) ("merge 0", "-1 and 97").

### MakeRefusesAPairNamedTwice

Two merges of (97, 98) fail with `InvalidDescription` and a message containing "merge 1", "97
and 98" and "earlier merge".

### MakeRefusesATokenAboveTheByteLimit

Eight merges that each double the token before build a vocabulary whose id 264 stands for 256
'a' bytes. A ninth such merge fails with `InvalidDescription` and a message containing "merge
8", "512 bytes" and "at most 256".

### MakeRefusesASpecialNameOutOfBounds

`Vocabulary::make` fails with `InvalidDescription` for an empty name ("special token 1", "0
bytes"), a name of 256 bytes ("256 bytes", "1 to 255"), "<pad>" given twice ("'<pad>'", "named
twice"), `K_END_OF_TEXT_NAME` given as a name ("named twice") and 256 names ("257 special
tokens", "at most 256").

### AnIDOutsideTheVocabularyStopsTheProcess

Over a vocabulary of 257 tokens, `bytes_of(257)` stops the process with a report containing
"Vocabulary", "257" and "[0, 257)", and `kind_of(-1)` with "Vocabulary" and "-1".

## modules/language/text/tests/test_text_vocabulary_file.cpp

Suites: `Unit_TextVocabularyFile`.

Tests `write_vocabulary_file`, `read_vocabulary_file` and `parse_vocabulary_file` in
`modules/language/text/include/SushiAI/language/text/vocabulary_file.hpp`. Most cases start
from a hand-written file of 49 bytes: a header of 24 bytes with the magic "SAVB" and version
1, a special section of 17 bytes that holds the name "<end_of_text>", and the one merge
(97, 98) in 8 bytes.

### TheWriterProducesTheDocumentedBytes

`write_vocabulary_file` of a vocabulary with the one merge (97, 98) writes exactly those 49
bytes.

### AVocabularySurvivesItsFile

A vocabulary of three merges and the special names "<pad>" and "<mask>" is written and read
back. The read vocabulary has the same size, the same special names and the same three
merges, and its id 261 stands for " abc".

### ATruncatedFileIsRefusedAtEveryLength

For every length from 0 to 48 bytes, `parse_vocabulary_file` over that prefix fails with
`InvalidFile` and a message containing "read_vocabulary_file" and "ends at byte". A prefix of
10 bytes names "ends at byte 10" and "header of 24 bytes"; one of 30 names "ends at byte 30"
and "declares 49 bytes".

### BytesAfterTheDeclaredEndAreRefused

With three bytes appended, the parse fails with `InvalidFile` and a message containing "3
bytes follow" and "byte 49".

### AnotherMagicOrVersionIsRefused

The parse fails with `InvalidFile` for a first byte of 'X' ("'SAVB'", "no vocabulary file"),
for a version of 2 ("byte 4", "format version 2", "versions 1 to 1") and for a version of 0
("format version 0").

### AHeaderCountOutOfBoundsNamesItsOffset

The parse fails with `InvalidFile` for a byte-token count of 257 ("byte 8", "257 byte
tokens"), a special-token count of 0 ("byte 12", "0 special tokens") and a merge count whose
last byte is 0x7F ("byte 16", "merges").

### ASpecialTokenPastItsSectionNamesItsOffset

The parse fails with `InvalidFile` for a name length of 14 ("special token 0", "byte 28", "14
bytes", "ends at byte 41"), a name length of 12 ("byte 28", "'<end_of_text'") and a name
length of 0 ("byte 24", "0 bytes").

### AMergeOfALaterTokenIsRefused

With byte 42 set to 1 the merge reads as (353, 98) and the parse fails with
`InvalidDescription` and a message containing "merge 0" and "353 and 98". With byte 44 set to
0xFF the parse fails with `InvalidFile` and a message containing "merge 0" and "byte 41".

### TheReaderNamesThePathOfAFileItRefuses

`read_vocabulary_file` fails with `FileNotFound` for a path that does not exist ("absent.savb",
"read_vocabulary_file") and with `InvalidFile` for a file cut to 45 bytes ("cut.savb", "ends
at byte 45", "declares 49 bytes").

## modules/core/data/tests/test_threaded_loader.cpp

Suites: `Unit_ThreadedLoader`, `Unit_Epoch`.

Tests `Data::ThreadedLoader` and `Data::begin_epoch`. The loader wraps a local
`RecordingDataset` of 12 batches. Each batch it reads carries one target whose role is
"<epoch>:<index>" and holds no tensor. A `Probe` records the threads that called `read`, how
far ahead of the caller a read started and the epoch last begun. The cases start worker
threads and need no engine.

### HandsTheBatchesOverInIndexOrder

With 3 workers and a prefetch of 4, reading indices 0 to 11 returns the batch named "0:<index>"
each time. At least one and at most three threads called `read`, and the test's own thread is
not among them.

### ReadsNoFurtherAheadThanThePrefetchCount

With 4 workers and a prefetch of 3, no read started more than 3 indices ahead of the index the
test had asked for.

### AFailedReadIsTheStatusOfItsIndexAlone

With a dataset whose batch 5 fails, reading index 5 fails with `InvalidFile`, "batch 5" and
"unreadable". Every other index from 0 to 11 returns its batch.

### AnIndexOutOfOrderIsRead

Reading the indices 7, 8, 2, 3, 11, 0 in that order returns the batch of each index.

### BeginEpochReachesTheDatasetAndDropsWhatWasReadAhead

After reads of indices 0 and 1, `Data::begin_epoch(*loader, 3)` sets the dataset's epoch to 3.
Reading indices 0 to 11 then returns the batches named "3:<index>".

### TheDestructorJoinsWithBatchesReadAhead

A loader of 3 workers and a prefetch of 4 is destroyed after `begin_epoch` and before any read
by the test. The case ends with `SUCCEED()` and asserts nothing else after the loader is made.

### StatesWhatItsDatasetStates

The loader reports the batch size 1, 12 batches, the name "recording", the sample type float32
and no target.

### MakeRefusesNoDatasetAndAnOptionBelowOne

`ThreadedLoader::make` fails with `InvalidDescription` for a null dataset ("no dataset"), for 0
workers ("0 threads") and for a prefetch of 0 ("0 batches ahead").

### ADatasetThatIsNotEpochAwareIsLeftAlone

`Data::begin_epoch` on a dataset that does not implement `EpochAware` returns, and the
dataset's batch count is still 1.

## modules/layers/nn/tests/test_throwing_bridge.cpp

Suites: `Unit_ThrowingBridge`, `Unit_ThrowingBridgeDeathTest`.

Tests bridge_throwing: a value passes through and a thrown Error becomes a Status.

This file needs no device. `NN::Detail::bridge_throwing` in
`modules/layers/nn/include/SushiAI/nn/throwing_bridge.hpp` runs a callable that may throw and returns its value
as a `Result`.

### refuse_a_shape

`refuse_a_shape` throws an `Error` through `SA_THROW_IF` with the message "Linear(6, 8) was
given [5]".

### refuse_an_allocation

`refuse_an_allocation` throws `std::bad_alloc`.

### throw_a_logic_error

`throw_a_logic_error` throws `std::logic_error("an index was out of range")`, which is neither
an `Error` nor `std::bad_alloc`.

### bridge_a_logic_error

`bridge_a_logic_error` passes `throw_a_logic_error` to `bridge_throwing`, as the statement of
a death test.

### AValuePassesThrough

A callable that returns 7 gives an ok `Result<int>` holding 7.

### AMoveOnlyValuePassesThrough

A callable that returns `std::make_unique<int>(7)` gives an ok result whose pointer is not
null and owns 7.

### AnErrorBecomesInvalidDescriptionWithItsMessage

Bridging `refuse_a_shape` gives an `InvalidDescription` failure whose message contains
"Linear(6, 8)" and "[5]".

### ARefusedAllocationBecomesOutOfMemory

Bridging `refuse_an_allocation` gives an `OutOfMemory` failure whose message contains
"allocation was refused".

### AnotherExceptionIsFatal

The case is in the suite `Unit_ThrowingBridgeDeathTest`. `bridge_a_logic_error` stops the
process with a report containing "other than Error" and "an index was out of range".

## modules/layers/nn/tests/test_trace.cpp

Suites: `Unit_Trace`, `Unit_TraceDeathTest`.

Tests Trace, the buffer-update sinks, the three layer forms and trace_layer.

This file needs no device. `NN::Trace` in `modules/layers/nn/include/SushiAI/nn/trace.hpp` carries a
`Graph::GraphBuilder`, a `TraceMode` and the sink that receives buffer updates. A layer has one
of three forms: `Module`, whose `forward` takes a builder; `TraceModule`, whose `forward` takes
a `Trace`; and `ModalModule`, which states `trace_training` and `trace_evaluation` and no
`forward`. `NN::trace_layer` calls whichever form a layer has.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### ModalStub

`ModalStub` has the modal pair and an empty `fields()`. `trace_training` emits one ReLU and
`trace_evaluation` returns its input, and each counts its calls.

### CompositeStub

`CompositeStub` lists a `ModalStub` as the child "inner". Its `forward` takes a `Trace` and
passes it to the child through `NN::trace_layer`.

### ByValueTraceStub

`ByValueTraceStub` has a `forward` that takes its `Trace` by value.

### TrainingOnlyStub

`TrainingOnlyStub` states `trace_training` and no `trace_evaluation`.

### ModalWithForwardStub

`ModalWithForwardStub` states the modal pair and a builder `forward` together.

### BothForwardsStub

`BothForwardsStub` has a builder `forward` and a trace `forward` and counts the calls of each.

### FieldlessStub

`FieldlessStub` has the modal pair and no `fields()`.

### RefusingStub

`RefusingStub` has a trace `forward` that records an `InvalidDescription` failure on the trace
through `Trace::fail` and returns its input.

### failure_saying

`failure_saying` returns an `InvalidDescription` failure with a given message.

### defined_buffer

`defined_buffer` returns a `Buffer` defined as [3] with the starting value 0.0 that no graph
has read.

### update_undefined_buffer

`update_undefined_buffer` requests an update of a buffer on which `define()` was never called.
It is the statement of a death test.

### update_from

`update_from` requests an update whose source is a given id, in a graph that holds one value.
It is the statement of two death tests.

### parameter_names

`parameter_names` collects the names `NN::for_each_parameter` yields, in walk order.

### The layer forms at compile time

The `static_assert` lines after the helpers hold which form each type has. `Linear` is a
`Module` only, `ModalStub` a `ModalModule` only and `CompositeStub` a `TraceModule` only, and
all three are `Traceable`. A `Sequential` of `Module`s is both a `Module` and a `TraceModule`.
A `Sequential` that holds a modal layer or a trace-only composite, at any depth, is a
`TraceModule` and not a `Module`, and is still a `ModuleContainer`. `BothForwardsStub` is a
`Module` and a `TraceModule`.

`Detail::HasHalfModalPair` detects `TrainingOnlyStub` and `Detail::HasModalMemberAndForward`
detects `ModalWithForwardStub`; neither detects `ModalStub` or `Linear`. `TrainingOnlyStub`,
`FieldlessStub`, `ByValueTraceStub` and `int` are not `Traceable`.

`Trace` is final. It is not copy-constructible, move-constructible or default-constructible,
and a `GraphBuilder` does not convert to it. `NN::BufferUpdateSink` has a virtual destructor.

### AFactoryStatesItsModeAndKeepsItsBuilder

`Trace::training(builder, list)` reports `TraceMode::Training` and `Trace::evaluation(builder)`
reports `TraceMode::Evaluation`. `builder()` of each returns the address of the builder it was
made from.

### UpdateUnderAnEvaluationTraceFails

`update()` on an evaluation trace returns `InvalidDescription` with a message containing
"evaluation trace".

### ANewListIsEmpty

A new `BufferUpdateList` has no update and no source.

### EachUpdateAddsOneUpdateAndOneSource

Two `update()` calls on a training trace leave two entries in `updates()` and two in
`sources()`. Entry 0 holds the first buffer's state, the source x, the momentum 0.1 and the
source scale 1.0. Entry 1 holds the second buffer's state, the source y, 0.25 and 1.5.
`sources()` is x then y.

### AnUpdateAddsNothingToTheGraph

An `update()` leaves the graph's value count unchanged and the buffer without a leaf in it.

### ARecordedTargetSurvivesAMoveOfItsBuffer

After the buffer of a recorded update is move-constructed into another, the update's `target`
equals the moved buffer's state and is not `K_NO_STATE`.

### TheDiscardingSinkAcceptsAnUpdateAndIsOneObject

A trace made with `NN::discard_updates()` is in Training mode, its `update()` returns ok and
the graph gains no value. Two calls of `discard_updates()` return the same address.

### AnUpdateOfAnUndefinedBufferIsFatal

The case is in the suite `Unit_TraceDeathTest`. `update_undefined_buffer` stops the process
with "define() has not given a shape".

### AnUpdateOfABufferThatWasNeverReadIsRecorded

An update of a defined buffer that no graph has read is recorded with the buffer's state as
its target. The graph gains no value and the buffer has no leaf in it.

### ANewTraceHoldsNoFailure

A new trace's `status()` is ok and the graph has no refusal.

### FailKeepsTheFirstFailureAndMarksTheGraphRefused

After `fail` with "the first reason" and then "the second reason", `status()` carries the
first message and does not contain "second". `graph.has_refusal()` is true.

### ValueOrFailReturnsAHeldValue

`value_or_fail(Result<ValueID>(y), x)` returns y and leaves `status()` ok.

### ValueOrFailRecordsAFailureAndReturnsTheFallback

`value_or_fail` over a failed result returns the fallback x, and `status()` then carries that
failure.

### FailGivenASuccessIsFatal

The case is in the suite `Unit_TraceDeathTest`. `trace.fail(Status())` stops the process with
"Trace::fail was given a success".

### SequentialThrowsTheFailureAChildRecorded

`Sequential{RefusingStub, ModalStub}::forward` over a trace throws with a message containing
"RefusingStub refuses". The second child's `trace_evaluation` is never called.

### SequentialFoldsAnInputShapeThroughItsChildren

`output_shape([4])` of `Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` is [3].
`output_shape([5])` throws with a message containing "Linear(4, 8)" and "[5]".

### AnUpdateWithoutASourceIsFatal

The case is in the suite `Unit_TraceDeathTest`. An update whose source is `K_NO_VALUE` stops the
process with "a buffer update reads value".

### AnUpdateFromAValueOutsideTheGraphIsFatal

The case is in the suite `Unit_TraceDeathTest`. An update whose source is value 7, in a graph
of one value, stops the process with a report containing "reads value 7" and "holds 1 values".

### TraceLayerCallsTheTrainingFormInTrainingMode

Under a training trace, `trace_layer` calls `ModalStub::trace_training` once and
`trace_evaluation` never. The result differs from the input, because the training form emits a
ReLU.

### TraceLayerCallsTheEvaluationFormInEvaluationMode

Under an evaluation trace, `trace_layer` calls `trace_evaluation` once and `trace_training`
never, and returns its input.

### TraceLayerTracesAModuleThroughTheBuilder

`trace_layer` on a `Linear{4, 3}` and `Linear::forward` on a second one, each in its own graph,
return the same id and leave the same value count. The traced result has the shape [2, 3].

### TraceLayerPassesTheTraceToATraceModule

`CompositeStub` is traced once under an evaluation trace and twice under a training trace. Its
child counts one evaluation call and two training calls.

### TraceLayerPrefersTheTraceForwardOverTheBuilderForward

On `BothForwardsStub`, `trace_layer` calls the trace `forward` once and the builder `forward`
never.

### ANestedSequentialTracesItsModalChildInTheTracesMode

A `ModalStub` inside a `Sequential` inside a `Sequential` counts one evaluation call after an
evaluation trace of the outer container, and one training call after a training trace of it.

### SequentialTracesAModalChildInTheTracesMode

`Sequential{Linear{4, 3}, ModalStub}` traced under each mode calls the stub's matching form
once and yields a [2, 3] value both times. Its parameter walk yields "0.weight" and "0.bias".

### SequentialOfModulesTracesTheSameThroughBothForwards

`Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` traced through the trace `forward` in one graph
and through the builder `forward` in another returns the same id, leaves the same value count
and walks the same parameter names.

### AnEvaluationTraceRefusesAFedLeafAndACarriedValue

An evaluation trace is not carrying. `feed` fails with `InvalidDescription`, "'position'" and
"carrying trace"; `carried_leaf` with "'kept'" and "carrying trace"; `carry` with
"carrying trace". No node is added.

### ATrainingTraceRefusesAFedLeaf

A training trace is not carrying, and `feed` fails with `InvalidDescription` and
"'position'".

### ACarryingTraceIsAnEvaluationTraceThatRefusesABufferUpdate

A carrying trace reports `is_carrying` and the mode `TraceMode::Evaluation`. `update` fails
with `InvalidDescription` and "evaluation trace", and no carried value is recorded.

### AFedLeafIsDeclaredOncePerName

Two `feed` calls for "position" return the same value and add one node between them; `feed`
for "mask" returns another. The value is an `Int32` input leaf. The fed list holds "position"
then "mask", `find("mask")` returns the mask's value and `find("absent")` returns
`K_NO_VALUE`.

### AFedLeafDeclaredAgainAsAnotherLeafIsRefused

After "mask" is fed as [1, 4] `Float32`, feeding it as [1, 5] fails with
`InvalidDescription`, "'mask'", "[1, 4]" and "[1, 5]", and as `Float64` with "'mask'". The
fed list still holds one leaf.

### CarryRecordsTheLeafItsNextValueAndItsInitialValue

`carried_leaf` declares an input leaf that is not in the fed list. After
`carry(leaf, next, 0.5)` the carried list holds one entry with that leaf, that next value and
the initial value 0.5, and `sources` holds the next value.

### CarryRefusesANextValueOfAnotherShape

`carry` of a [4, 3] leaf with a [3, 4] next value fails with `InvalidDescription`, "[4, 3]"
and "[3, 4]", and records nothing.

### CarryingAValueThatIsNoCarriedLeafIsFatal

`carry` of a value that `carried_leaf` did not declare is fatal with "did not declare".

## modules/core/graph/tests/test_traced.cpp

Suites: `Unit_Traced`.

Tests Traced and the refusal it carries through a chain of emitters.

This file needs no device. `Graph::Traced` in `modules/core/graph/include/SushiAI/graph/traced.hpp` holds a
`ValueID` or a failed `Status`. An emitter that returns one takes its operands as `Traced`
too, so a call can be nested inside another and the first refusal travels to the outermost
result.

Six `static_assert` lines at the top of the file hold that a `Traced` is default-constructible
and constructible from a `ValueID` and from a `Status` without throwing, that both conversions
are implicit, and that its size is two pointers.

The cases up to BuildingFromASuccessStatusIsFatal use `StandInBuilder` and involve no graph.
The last four use a `Graph::GraphBuilder` and read `Graph::has_refusal()`.

### TraceLog

`TraceLog` counts the values the stand-in builder emitted and the refusals it recorded.

### StandInBuilder

`StandInBuilder` has two members. `emit()` returns the next id, and `refuse()` counts one
refusal and returns the failure it was given. `SA_REFUSE_IF` is run over it in `emit_below`.

### emit_below

`emit_below` binds its operand through `SA_TRY_ASSIGN`, refuses through `SA_REFUSE_IF` when
the id is above a limit, and otherwise emits one value.

### emit_over

`emit_over` binds its operand through `SA_TRY_ASSIGN` and emits one value. It has no condition
of its own, so it accepts the absent operand.

### forward_traced

`forward_traced` runs `SA_TRY` on a `Traced` and sets a flag after it.

### declare_below

`declare_below` runs `SA_REFUSE_IF` over a `Graph::GraphBuilder` when an extent is above a
limit, and otherwise declares one input of that extent.

### description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

### DefaultHoldsTheAbsentValueAndNoFailure

A default-constructed `Traced` has a value, an ok status and the id `Graph::K_NO_VALUE`.

### HoldsAnIDWithASuccessStatus

`Traced(ValueID{12})` has a value, is ok and returns 12.

### HoldsAFailureAndTheAbsentID

A `Traced` built from a failure has no value and reports the failure's code and message.

### SuccessPathCreatesNoFailureBlock

Two chained emitters over an absent operand, a copy and a move yield the id 1, record no
refusal and leave `live_failure_count()` where it was.

### MovedFromTracedKeepsItsState

A failed `Traced` that was moved from still reports its failure, as does the target. A
`Traced` holding 4 that was move-assigned from still returns 4, as does the target.

### AssignmentReplacesAFailureWithAnID

Assigning `ValueID{3}` to a failed `Traced` leaves a value and the id 3. Assigning a failure to
it afterwards leaves that failure.

### AnAbsentOperandPassesThroughAnEmitter

`emit_over(builder, Traced())` succeeds with the id 0, and the log shows one emitted value.

### RefusalTravelsThroughAChainOfTwoCallsAndEmitsNothing

`emit_below` given the id 9 and the limit 4 fails with "operand 9 is above the limit 4",
records one refusal and emits nothing. That result is then passed through `emit_below` with
the limit 100, nested inside `emit_over`. The outer result carries the same message, and its
`message().data()` is the first result's pointer, so the failure was forwarded and not rebuilt.
The log still shows one refusal and no emitted value.

### NestedCallsEmitWhenNothingIsRefused

`emit_over(builder, emit_below(builder, ValueID{2}, 4))` yields the id 1 with two values
emitted and no refusal.

### TryForwardsATracedFailureAsAStatus

`forward_traced` over a failed `Traced` returns the failure and leaves its flag false. Over
`ValueID{1}` it succeeds and sets the flag.

### ValueOnAFailureIsFatalAndPrintsTheFailure

`value()` on a failed `Traced` stops the process with a report containing "Traced::value() was
called on a failure" and the failure's message.

### BuildingFromASuccessStatusIsFatal

`Traced(Status())` stops the process with "a Traced was built from a success Status".

### RefuseMarksTheGraphAndReturnsTheFailure

`GraphBuilder::refuse` returns the failure it was given, with the same message pointer, and
`graph.has_refusal()` goes from false to true.

### RefuseIfOverAGraphBuilderMarksTheGraphOnlyOnARefusal

`declare_below(builder, 4, 8)` declares value 0 and leaves the graph without a refusal.
`declare_below(builder, 9, 8)` fails with "extent 9 is above the limit 8", adds no value and
marks the graph.

### ObserveMarksTheGraphForAFailedStatusOnly

`GraphBuilder::observe(Status())` is ok and leaves the graph unmarked. `observe` of a failed
status returns that failure and marks the graph.

### ObserveMarksTheGraphForAFailedResultOnly

`observe(Result<int>(7))` returns a result holding 7 and leaves the graph unmarked. `observe`
of a failed `Result<int>` returns the failure and marks the graph.

## modules/core/config/tests/test_train_config.cpp

Suites: `Unit_TrainConfig`.

Tests `Config::parse_train`, `Config::check_start` and `Config::read_run_seed`. Most cases
parse a file that states `sushiai_train`, `model`, `objective`, `data` and an `sgd` optimizer,
followed by the members the case adds. The last four cases parse a file whose `data` block the
case supplies.

### AFileWithoutAScheduleHoldsTheNullValue

Without a `schedule` member, `spec.schedule` is null, the optimizer type is "sgd" and `epochs`
is 1.

### TheScheduleIsCarriedAsTheFileStatesIt

A `schedule` object with `type`, `base_rate` and a key named `unread_here` is kept as an object
of three keys. Its `type` is "step", its `base_rate` is a real and it has `unread_here`.

### AScheduleOfAnotherKindIsCarriedUnchecked

A `schedule` that is the string "later" parses, and `spec.schedule` has the kind `String`.

### AKeyBesideTheScheduleIsStillRefused

A member named `schedules` raises an error naming `'schedules'`.

### AFileWithoutWeightsHoldsAnEmptyPath

Without a `weights` member, `spec.weights` is empty.

### TheWeightsPathResolvesAgainstTheFile

Parsed with the directory `runs`, `"weights": "start.sushi"` gives `runs/start.sushi` and the
model path is `runs/model.json`.

### WeightsThatAreNoStringAreRefused

`"weights": 3` raises an error naming `'weights'`.

### TheResumePathResolvesAgainstTheFile

Without a `resume` member the path is empty. Parsed with the directory `runs`,
`"resume": "last.sushi"` gives `runs/last.sushi` and leaves `weights` empty.

### AResumeThatIsNoStringIsRefused

`"resume": true` raises an error naming `'resume'`.

### WeightsAndResumeTogetherAreRefused

`check_start` is ok for a file with neither member, with `weights` alone and with `resume`
alone. With both it fails with `InvalidDescription` and a message naming `'train.json'`,
`'weights'` and `'resume'`.

### TheSeedIsZeroUnlessTheFileStatesOne

Without a `seed` member `spec.seed` is 0 and `read_run_seed` returns 0. With
`"seed": 20261011` both are 20261011.

### ASeedThatIsNoIntegerIsRefused

The seeds "seven" and 0.5 each raise an error naming `'seed'`.

### ANegativeSeedIsRefused

A file with `"seed": -3` parses, and `read_run_seed` fails with `InvalidDescription` and a
message containing `'train.json'`, `'seed'` and "-3".

### ADataBlockWithAReaderTakesTheReaderType

A `data` block with `reader`, `task` and `root` and no `type` gets the type
`Config::K_READER_DATA_TYPE`. Its attributes keep all three keys and `reader` is "yolo_txt".

### ATypedDataBlockKeepsItsType

A `data` block with `"type": "mnist"` and `directory` gets the type "mnist". Its attributes
hold `directory` and no `type`.

### ATypeBesideAReaderStaysATypedBlock

A `data` block with `"type": "mnist"` and `reader` gets the type "mnist" and its attributes
still hold `reader`.

### ADataBlockWithNeitherKeyIsRefused

A `data` block with only `root` raises an error naming `'type'`.

## modules/language/training/tests/test_training_expert_balance_objective.cpp

Suites: `Unit_LanguageTrainingExpertBalanceObjective`,
`Unit_LanguageTrainingExpertBalanceMetrics`.

Tests `Language::Training::ExpertBalanceObjective` in
`modules/language/training/include/SushiAI/language/training/expert_balance_objective.hpp` and
its factory `make_expert_balance_objective`. The blocks the cases describe have four experts.
The fixture of the metrics suite provides an engine and allocates `Float32` usage tensors of
the shape [1, 4].

### MakeKeepsItsOutputsAndRefusesNone

An objective made of the balance "balance" and the usage "usage" of four experts returns that
name from `balances()` and one usage of four experts from `usages()`. Its name is
"expert_balance" and `is_fed_by_batch()` is true. `make` fails with `InvalidDescription` for no
balance ("at least one balance output", "0 and 1"), no usage ("1 and 0"), an empty balance
name ("no name") and a usage of 0 experts ("'usage' of 0").

### ItDeclaresNoLeafAndReadsNoRole

After `declare`, the leaves list no leaf and no required role, and the graph has no node.

### TheLossIsTheMeanOfTheBalanceOutputs

The model outputs "balance2" and "balance3" are [1, 1] values. The loss of an objective over
"balance3" alone has the shape [1] and comes from a RESHAPE node that reads that output, with
no SCALE node in the graph. The loss of an objective over both has the shape [1] and comes
from a SCALE node whose attribute is 0.5 and whose input comes from an ADD node.

### TheFactoryReadsTheOutputsByTheirNames

Over a context that publishes "output", "balance2", "usage2", "balance3" and "usage3", the
factory builds an `ExpertBalanceObjective` whose balances are "balance2" and "balance3" and
whose usages are "usage2" and "usage3", the second of four experts.

### TheFactoryRefusesWhatDoesNotFit

The factory fails with `InvalidDescription` for a `weight` key ("objective states the key
'weight'"), for a context that publishes only "output" ("begin with 'balance'", "'usage'", "0
and 0", "among its 1") and for a "balance2" output of the shape [1, 2] ("'balance2'", "one
element", "[1, 2]").

### MeasureReportsTheSmallestAndLargestShare

The step publishes "usage2" with four shares of 0.25 and "usage3" with the shares 0.125, 0.5,
0.0625 and 0.3125. `measure` writes two metrics, "expert_usage_min" of 0.0625 and
"expert_usage_max" of 0.5, each within 1e-7 and each marked as a fraction.

### MeasureRefusesAUsageItCannotRead

Over a step that publishes only "usage2" with four shares, `measure` fails with
`InvalidDescription` for an objective that reads "usage9" ("'usage9'", "did not publish") and
with `Incompatible` for one that expects eight experts in "usage2" ("the usage of the
experts", "8 values", "holds 4"). No metric is written.

## modules/language/training/tests/test_training_next_token_objective.cpp

Suites: `Unit_LanguageTrainingNextTokenObjective`.

Tests `Language::Training::NextTokenObjective` in
`modules/language/training/include/SushiAI/language/training/next_token_objective.hpp` and its
factory `make_next_token_objective`. Unless a case says otherwise the batch has two rows of
three positions over a vocabulary of five, and the factory's context publishes one output
"out" of the shape [2, 3, 5].

### MakeKeepsItsExtentsAndRefusesNone

`batch()`, `length()` and `vocabulary()` return 2, 3 and 5, and the name is "next_token".
`make` fails with `InvalidDescription` and "at least 1" for a batch of 0 and for a vocabulary
of 0.

### MakeRefusesWhatInt32CannotIndex

`make(64, 1024, 50257)` fails with `InvalidDescription` and a message containing "int32 index"
and "64 * 1024 * 50257". `make(32, 256, 8192)` and `make(1, 1, 2147483647)` are ok, and
`make(1, 2, 2147483647)` fails with "int32 index".

### ItDeclaresAnIndexLeafAndReadsLabels

`declare` adds one leaf, "targets", an `Int32` value of the shape [6]. It requires one role,
`K_ROLE_LABELS`, with the sample shape [3], the type `Int32` and 0 classes.

### TheLossGathersOneLogProbabilityPerPosition

`Train::build_training_graph` over an `NN::Embedding` of five rows and five columns, with an
`Int32` input of the shape [2, 3], gives a loss of one element and `Int32` targets. The graph
holds no NLL_LOSS, one LOG_SOFTMAX, two GATHER_ROWS, two SCATTER_ADD_ROWS and one
LOG_SOFTMAX_BACKWARD node. The one parameter has a gradient and the role `Embedding`.

### TheFactoryReadsTheModelsOutput

The objective the factory builds from an empty section declares a "targets" leaf of the shape
[6] and has the name "next_token".

### TheFactoryRefusesWhatDoesNotFit

The factory fails with `InvalidDescription` for a `smoothing` key ("objective states the key
'smoothing'"), a second published output ("one [N, T, V] output", "publishes 2"), an output of
the shape [2, 5] ("[2, 3, V] logits", "'out'"), an input sample of [4, 4, 3] ("[T] token ids
per sample") and a context of six classes ("6 token ids", "scores 5").

### ABatchAloneFeedsTheObjective

`is_fed_by_batch()` is true.

### MeasureReportsThePerplexityOfTheLoss

Over a step whose loss is ln 8, `measure` writes one metric, "perplexity", that is not a
fraction and equals 8 within a relative 1e-12.

### TheOutputKeyNamesTheLogitsAmongSeveral

Over a context that also publishes "balance" of the shape [1, 1], the section `{"output":
"out"}` builds an objective. Its loss over model outputs "balance" and "out" has one element,
and the graph holds one LOG_SOFTMAX node.

### TheFactoryRefusesAnOutputItCannotFind

Over the same two outputs, the factory fails with `InvalidDescription` for an empty section
("publishes 2", "name the logits with 'output'"), `"output": "logits"` ("the output 'logits'",
"none of that name among its 2"), `"output": 3` ("'output' names a published output"),
`"output": ""` ("'output'", "is empty") and `"output": "balance"` ("[2, 3, V] logits",
"'balance'", "[1, 1]").

## modules/language/training/tests/test_training_next_token_targets.cpp

Suites: `Unit_LanguageTrainingNextTokenTargets`, `Unit_LanguageTrainingPerplexity`.

Tests `Language::Training::fill_next_token_targets` in
`modules/language/training/include/SushiAI/language/training/next_token_targets.hpp`, and
`Language::Training::perplexity` and `Language::Training::next_token_metrics`. The fixture of
the first suite provides an engine, `Int32` labels of the shape [2, 3] and a leaf of six
elements that starts at -7 everywhere. The vocabulary has five tokens.

### APositionIndexesItsLabelAmongEveryScore

The labels 4, 0, 2, 1, 1, 3 fill the leaf with 4, 5, 12, 16, 21, 28.

### ALabelOutsideTheVocabularyIsRefused

`fill_next_token_targets` fails with `Incompatible` for a label of 5 at position 4 ("position
4 is 5", "holds 5 tokens") and for a label of -1 at position 2 ("position 2 is -1"). Every
element of the leaf is still -7.

### AFillThatDoesNotFitIsRefused

The fill fails with `Incompatible` for labels of the shape [2, 2] ("holds 4 labels", "expects
6"), for `Float32` labels ("float32", "int32"), for a vocabulary of 2147483647 ("6 positions",
"cannot be indexed") and for a vocabulary of 0 ("cannot be indexed").

### ThePerplexityIsTheExponentialOfTheLoss

`perplexity(0)` is exactly 1, `perplexity(ln 64)` is 64 within a relative 1e-12 and
`perplexity(1e6)` is infinite. The case is in the suite `Unit_LanguageTrainingPerplexity`.

### TheMetricsAreTheLossAndThePerplexity

`next_token_metrics(ln 8)` returns "loss" with exactly ln 8 and "perplexity" with 8 within a
relative 1e-12, neither marked as a fraction. The case is in the suite
`Unit_LanguageTrainingPerplexity`.

## modules/vision/training/tests/test_truth_from_roles.cpp

Suites: `Unit_TruthRoles`, `Unit_TruthFromRoles`.

Tests `describe_truth_roles` and `truth_from_roles` in
`modules/vision/training/include/SushiAI/vision/training/truth_from_roles.hpp`. The
`Unit_TruthFromRoles` fixture is an `EngineTest`. Its batch holds two images of three slots
over 5 classes: the slots 0, 2 and 5 are in use with the classes 4, 2 and 1, and the padding
slots hold zeros.

### DescribesBoxesClassesAndValidOverTheSlots

`describe_truth_roles(3, 5)` gives three roles: `K_ROLE_BOXES` of [3, `K_TRUTH_BOX_VALUES`],
`Float32` and 0 classes; `K_ROLE_CLASSES` of [3], `Int32` and 5 classes; `K_ROLE_VALID` of
[3], `Float32` and 0 classes.

### DescribesAClassCountOfZero

`describe_truth_roles(1, 0)` succeeds and its classes role states 0 classes.

### RefusesNoSlotAndANegativeClassCount

`describe_truth_roles` fails with `InvalidDescription` for 0 slots ("object slot", "got 0")
and for -1 classes ("classes", "got -1").

### CopiesTheCornersWidensTheClassesAndMakesTheFlagsBytes

`truth_from_roles` gives a truth of batch 2 and 3 boxes per image whose `boxes` equal the
corners written, whose `labels` are 4, 0, 2, 0, 0, 1 and whose `valid` bytes are 1, 0, 1, 0,
0, 1.

### FindsTheRolesInAnotherOrder

With the first and third target of the batch swapped, the truth has the same boxes and valid
bytes.

### KeepsTheStorageOfATruthBetweenBatchesOfOneSize

After a second `truth_from_roles` into the same truth, `boxes`, `labels` and `valid` have the
data pointers of the first call, and the first label is the 3 written in between.

### RefusesABatchThatLacksARole

Without any one of the three roles the call fails with `InvalidDescription`, "no role '<role>'"
and "it carries". A batch with no target fails with "no role 'boxes'" and "(none)".

### RefusesARoleOfAnotherElementTypeOrRank

The call fails with `Incompatible` for `Float32` classes ("'classes'", "int32",
"float32 [2, 3]"), for `Int32` valid flags ("'valid'", "float32", "int32 [2, 3]") and for
boxes of sample shape [12] ("'boxes'", "rank 3", "[2, 12]").

### RefusesABoxOfAnotherWidthAndRolesOfOtherSlots

The call fails with `Incompatible` for boxes of three values ("'boxes'", "[2, 3, 3]",
"4 values"), for classes of four slots ("'classes'", "[2, 4]", "[2, 3, 4]") and for valid
flags of two slots ("'valid'", "[2, 2]", "[2, 3, 4]").

### RefusesAFlagThatIsNeitherZeroNorOneAndLeavesTheTruth

After a first copy, a valid flag of 0.5 fails with `InvalidDescription`, "'valid'", "0.5" and
"slot 1 of image 1", and the truth's first label stays 4 although the batch now holds 1. A
flag of NaN fails with `InvalidDescription`, "'valid'" and "slot 1 of image 1".

## modules/layers/nn/tests/test_unflatten.cpp

Suites: `Unit_Unflatten`.

Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.

This file needs no device. `NN::Unflatten` in `modules/layers/nn/include/SushiAI/nn/unflatten.hpp` holds a
height, a width and a channel count and no parameter. It reshapes an
[N, height * width * channels] value to [N, height, width, channels]. Images enter a model
flat, as [N, features], so this layer is what gives a convolution its rank-4 input.

### per_sample

`per_sample` returns the shape of a graph value with the batch axis removed.

### SatisfiesTheModuleContract

Three `static_assert` lines require `Unflatten` to be a `Module`, not to be a
`ModuleContainer`, and to have no parameter walk of its own, so the walk reaches it through
`fields()`.

### TracesTheImageShape

`Unflatten{4, 5, 3}` traces a FLOAT64 [2, 60] input to a FLOAT64 [2, 4, 5, 3] value.

### TagsTheResultNHWC

The result is rank 4 and tagged NHWC, and `finalise()` accepts the graph.

### EmitsOneReshapeNode

The trace adds exactly one node. The producer of the result is a K_RESHAPE whose one operand is
the input.

### OutputShapeAgreesWithTheTraceForAnRGBImage

`output_shape()` of the input's per-sample shape equals the per-sample shape of the traced
result, and `output_shape([60])` is [4, 5, 3].

### OutputShapeAgreesWithTheTraceForAGreyImage

The same two assertions hold for `Unflatten{28, 28, 1}` over a [7, 784] input, the MNIST
geometry: `output_shape([784])` is [28, 28, 1].

### ForwardRejectsAZeroExtent

`Unflatten{4, 0, 3}` throws with a message containing "Unflatten needs positive extents". The
element-count check would also throw for this layer, in a message that does not name the
extents, so the case requires the layer's own message.

### ForwardRejectsANegativeExtent

`Unflatten{-4, -5, 3}` throws the same message. The two negative extents multiply to a
positive count of 60, which matches the input, so no later check refuses the layer by count;
`Shape`'s constructor would throw, in a message that does not name the layer.

### ForwardRejectsAnInputThatIsNotRankTwo

A [2, 60, 1, 1] input throws. Its second extent is 60 and it holds 120 elements, so only the
rank check can refuse it.

### ForwardRejectsAMismatchedElementCount

A [3, 40] input to `Unflatten{4, 5, 3}` throws with a message containing "Unflatten(4, 5, 3)"
and "inner extent is 60". `reshape` would also reject the input, in a message that does not
name the layer, so the case requires the layer's own.

### ForwardRejectsAnNCHWInput

The layout tag of a [2, 60] input is set to NCHW by hand, and `forward()` throws.

### OutputShapeRejectsAZeroExtent

`Unflatten{4, 5, 0}.output_shape([60])` throws with a message containing "Unflatten needs
positive extents".

### OutputShapeRejectsANegativeExtent

`Unflatten{-4, -5, 3}.output_shape([60])` throws the same message.

### OutputShapeRejectsAShapeThatIsNotRankOne

`output_shape([60, 1])` throws.

### OutputShapeRejectsAMismatchedElementCount

`output_shape([59])` throws.

### HasNoParameters

`parameter_count` is 0 before and after a trace and `parameter_elements` is 0. The trace
raises the value count by one, the reshaped value, so it declares no parameter.

## modules/layers/nn/tests/test_upsample2d.cpp

Suites: `Unit_Upsample2d`.

Tests the shape, layout and rejections of the Upsample2d layer.

This file needs no device. `NN::Upsample2d` in `modules/layers/nn/include/SushiAI/nn/upsample2d.hpp` holds one
integer scale, 2 by default, and no parameter. It repeats every pixel of an NHWC image over a
scale-by-scale block.

Two `static_assert` lines at file scope require `Upsample2d` to be a `Module` and not a
`ModuleContainer`.

### traced_per_sample_shape

`traced_per_sample_shape` traces a layer over a new input of the given batched shape and
returns the result's shape with the batch axis dropped. The forward rejection cases call it
too, since it throws whatever `forward()` throws.

### K_OVERFLOWING_SCALE

`K_OVERFLOWING_SCALE` is 2^62. Its product with an extent of 5 does not fit in 64 bits.

### TheDefaultScaleDoublesHeightAndWidth

A default layer traces a [2, 3, 5, 7] input to [2, 6, 10, 7].

### TheScaleFieldSetsTheFactorOnBothSpatialAxes

`Upsample2d{3}` traces a [1, 4, 2, 5] input to [1, 12, 6, 5].

### TheResultIsTaggedNHWC

The traced result is rank 4 and tagged NHWC.

### ForwardEmitsOneUpsampleNodeCarryingTheScale

The trace of `Upsample2d{3}` adds exactly one node. The producer of the result is an
K_UPSAMPLE2D whose one operand is the input, and attribute slots 0 and 1 both hold 3.

### OutputShapeScalesHeightAndWidthOnly

A default layer maps [3, 5, 7] to [6, 10, 7], and `Upsample2d{4}` maps [2, 1, 9] to [8, 4, 9].
The channel extent is unchanged in both.

### OutputShapeAgreesWithTheTracedShape

For scales 2, 3 and 1, `output_shape()` equals the traced shape without its batch axis. A
scale of 1 is the smallest the layer accepts and leaves the shape as it is.

### OutputShapeRejectsAZeroScale

`Upsample2d{0}.output_shape([3, 5, 7])` throws with a message containing "Upsample2d" and
"[3, 5, 7] with scale=0".

### OutputShapeRejectsANegativeScale

`Upsample2d{-2}.output_shape([3, 5, 7])` throws with a message containing "Upsample2d" and
"[3, 5, 7] with scale=-2".

### OutputShapeRejectsARankOnePerSampleShape

`output_shape([16])` throws with a message containing "Upsample2d", the expected form
"[H, W, C]" and "[16]".

### OutputShapeRejectsAScaledHeightThatOverflows

With `K_OVERFLOWING_SCALE`, `output_shape([5, 1, 7])` throws with a message containing
"Upsample2d" and "fit in 64 bits". The width is 1, so the height is the product that
overflows.

### OutputShapeRejectsARankFourShape

`output_shape([2, 3, 5, 7])` throws. The shape still carries its batch axis.

### ForwardRejectsAZeroScale

`forward()` of `Upsample2d{0}` over a [1, 2, 2, 3] input throws with a message containing
"Upsample2d" and "[2, 2, 3] with scale=0".

### ForwardRejectsARankTwoInput

`forward()` over a [2, 12] input throws with a message containing "Upsample2d", "rank-4" and
"[2, 12]".

### ForwardRejectsAScaledWidthThatOverflows

With `K_OVERFLOWING_SCALE`, `forward()` over a [1, 1, 5, 3] input throws with a message
containing "Upsample2d" and "fit in 64 bits". The height is 1, so the width is the product
that overflows.

### ForwardRejectsAnNCHWInput

A [1, 3, 2, 2] input is converted to NCHW, and `forward()` throws with a message containing
"upsample2d" and "NHWC". The lower-case name is the builder's: `GraphBuilder::upsample2d`
raises this message.

### TheParameterWalkVisitsNothing

After a trace the walk visits nothing and `parameter_count` is 0.

## modules/core/version/tests/test_version.cpp

Suites: `Unit_Version`.

Tests that the runtime version agrees with the compile-time macros.

### MatchesEncodedMacro

`SUSHIAI_VERSION` equals `SUSHIAI_VERSION_ENCODE` of the major, minor and patch macros.

### RuntimeStringMatchesCompileTimeMacro

`version_string()` equals `SUSHIAI_VERSION_STRING`.
