# Design completion

**Status:** Open — written 2026-11-15; jobs run in parallel, one build and one test run close it

## Rulings

The owner delegated every engineering decision on 2026-11-15. Each ruling below is written into its topic in the commit that carries the work.

| Topic | Ruling |
| --- | --- |
| PROFILING | The text follows the SushiProfiler v0.4.1 pin; tests take the names PLAN of 2026_10_26_PROFILER_ADOPTION gives |
| MODEL_IMPORT D1 | `inference` takes the MOT17 frame as one image; no new argument |
| MODEL_IMPORT D3 | New module `modules/layers/interchange`: ONNX and Caffe message tables, axis translation, constant folding. Operator tables and importers stay in `vision/importers` |
| MODEL_IMPORT D4 | The first Caffe model is LeNet on MNIST; the oracle is OpenCV's `readNetFromCaffe`, run outside the repository |
| MODEL_IMPORT D5 | The first ONNX operator set is what YOLOX and Ultralytics exports use |
| GRAPH O1 | The new SushiBLAS row-index class is `RowIndexOps` in module `operations/gather` |
| GRAPH O3 | G4 is in scope; its timings are taken after the final build |
| GRAPH O4 | B9, B10, G11, G12 stay out; backlog entries |
| LANGUAGE 6 | T 128, dim 256, 4 blocks, 4 heads as model-file keys; `decisions` dataset; options through the same encoder, K from the model file; a lengths leaf masks attention and pooling; `output` key and `type_mask` role on the existing objectives; one temperature per head in the model file, ECE over 15 bins, bound 0.05 on the synthetic set; masked-token pretraining is a later phase; new module `modules/language/decision_models` |
| DATASETS | The 0.1.0 segmentation example is a segment head over the YOLOv7-tiny body; no reference run for segmentation objectives; bilinear upsample is in (G7); `LabelSet` holds several polygons per object (RW-79) |

## Shared files

No worker edits a shared registration file: a module or test `CMakeLists.txt`, `cmake/*Layers.cmake`, `modules/vision/registration/source/registration.cpp`, `modules/language/registration/source/registration.cpp`, `modules/core/ops/include/SushiAI/ops/ops.hpp`, `builder.hpp`/`builder.cpp`. A worker lists the lines it needs there in its report; the orchestrator applies them. The jobs that must change `ops.hpp` and the builder run as one chain (job SA-OPS).

## Jobs

| Job | Files | Waits on | Status |
| --- | --- | --- | --- |
| PR1 Profiling audit: test names, header comments, design and SPEC/PLAN text to v0.4.1 | tests/core/profile, tests/core/graph/test_compiled_step.*, tests/layers/train/test_trainer/test_zones.*, tests/applications/programs/source/test_profile_session.cpp, compiled_step.hpp, ir.hpp, 2026_10_26_PROFILER_ADOPTION | - | done |
| MI5A SafetensorsReader | modules/layers/io safetensors_reader.*, tests/layers/io/test_safetensors_reader.* | - | done |
| MI5B Name map and `import weights` | io name_map.*, applications/programs/source/import.cpp, tests/layers/io/test_name_map.* | MI5A | done |
| MI91 Inflate and deflated zip entries | io inflate.*, zip_archive.cpp, tests/layers/io/test_inflate.*, test_zip_archive.cpp | - | done |
| MI92 NpyReader | io npy_reader.*, tests/layers/io/test_npy_reader.* | - | done |
| MI93 `.npz` weights | io npz_weights.cpp, import.cpp | MI91, MI92, MI5B | done |
| MI71 ProtobufWire | io protobuf_wire.*, tests | - | done |
| MI81 ProtobufText | io protobuf_text.*, tests | - | done |
| MI72 interchange module: ONNX graph reader | modules/layers/interchange onnx_graph.*, tests/layers/interchange | MI71 | done |
| MI73 Axis translation | interchange axis_translation.*, tests | - | done |
| MI74 Constant folding | interchange constant_folding.*, tests | MI72 | done |
| MI82 Caffe net reader | interchange caffe_net.*, tests | MI71, MI81 | done |
| MI61 Focus block | vision/blocks focus.*, tests | - | done |
| MI62 Decoupled head | vision/blocks decoupled_head.*, tests | - | done |
| MI63 YOLOX task and decode | vision/detect/yolox, tests | - | done |
| MI75 ONNX operator table and importer | vision/importers onnx_*, import.cpp | MI72, MI73, MI74, MI5B | done |
| MI83 Caffe operator table and importer | vision/importers caffe_* | MI82, MI73 | done |
| MI5C Layout permutation in weight import; public nn state listing without an engine | io weight_import.*, nn state listing, programs model_state_entries.cpp | MI5B | done |
| MI64 YOLOX-X model file, map, large tests, export script | applications/examples/yolox_x, tools/reference/export_yolox_reference.py, large tests | MI61, MI62, MI63, MI5C | done |
| MI76 ONNX YOLOX parity and Caffe LeNet parity (large) | tests/vision/importers test_onnx_yolox, test_caffe_parity, tools/reference/export_caffe_reference.py | MI75, MI83, MI64 | done |
| SB-P1 B7, B8 thread and host-access tests | sushiblas tensor docs and tests | - | done |
| SB-B1 Positioned random | sushiblas operations/random | - | done |
| SB-B4 Bilinear upsample forward and backward | sushiblas spatial, spatial_kernels | - | done |
| SB-B5 argmax_along | sushiblas reductions | - | done |
| SB-B2 RowIndexOps in operations/gather, take/put along, benchmark | sushiblas operations/gather, kernels/index_kernels | - | done |
| SB-B3 Broadcast seam and BroadcastOps | sushiblas functor_map, operations/broadcast | - | done |
| G8 Finite HALF mask fill | lowering/linalg.cpp triangle_affine, tests | - | done |
| G9 Recorder-only step test | tests/core/graph/test_compiled_step.cpp | PR1 | done |
| G6 Repeated and unnamed index cases | tests/core/autograd/rules/test_linalg.cpp | - | done |
| SUP Move free_task and anchor_task onto suppression_section | vision/detect free_task.cpp, anchor_task.cpp, tests | MI63 | done |
| SA-OPS G2, G7, G10, G3, G5 ops, builder, lowering, rules | ops.hpp, builder.*, lowering/*, autograd rules/*, tests | SB-B4, SB-B5, SB-B2, SB-B3, G8, G6 | done |
| G1 Positioned dropout mask, A11 dropout training form | lowering/random.*, nn/dropout.hpp, tests | SB-B1 | done |
| G4 Five lowering rows on broadcast | lowering linalg, norm, loss | SA-OPS | done |
| A12 Language and vision objectives on row ops; segment head mode key; device segmentation metric | language/training, vision/training, nn/upsample2d.hpp, segment_head.cpp | SA-OPS | done |
| L6a Key-padding mask, sequence pool, text encoder | language/blocks | - | done |
| L6b Option head, decision heads | language/blocks | L6a | done |
| L6c `output` key and `type_mask` on objectives | layers/train objectives | - | done |
| L6d Decisions dataset | language/datasets | - | done |
| L6e Temperature scaling, ECE | language/training | - | done |
| L6f decision_models module, end-to-end test | modules/language/decision_models, tests | L6b, L6c, L6d, L6e | done |
| GAP1 RW-122, RW-111, RW-76/84 core and train fixes | failure/status.*, train/trainer.cpp, feed.cpp, objective.*, sum objective | L6c | done |
| GAP2 RW-89, RW-68, RW-79, RW-54, RW-143 vision fixes; segment overfit test file | vision/training, vision/datasets, vision/imaging | A12 | done |
| GAP3 RW-98, RW-90 language fixes | nn/embedding.cpp, language/training, language/datasets | L6d | done |
| GAP4 RW-136/137/138 Darknet import keys, RW-74 exporter v2 | vision/importers darknet_*, fit_pipeline.hpp, tools/reference | - | done |
| GAP5 RW-21, RW-22 eager buffers and shared gradients | core/eager | - | done |
| GAP4B RW-136 darknet pipeline row (jitter, hue, saturation, exposure); RW-138 darknet_letterbox fit; float32_value in other writers | config, datasets, imaging, importers | GAP4 | done |
| NORM `input.normalise` (channel order, scale, mean, std) in the model file, applied with the fit; yolox_x model.json uses ByteTrack values | core/config model_config, vision/datasets fit, applications/examples/yolox_x | GAP4B | done |
| DVSPLIT One file per detection report row in applications/programs (anchor, anchor-free, yolox) sharing one batch-report helper; README stale sentence | applications/programs/source/domains_vision*.cpp | MI64 | done |
| TDOUBLE InMemoryObjective gains scripted feed, metric names and call counts; replace FixedAccuracyObjective, NamedMetricsObjective, SkippingObjective; leaf guard checks every element, not any byte; fix test_sum_objective.md stale sentence; replica path uses the guard | layers/train in-memory objective, leaf_writes.*, tests/layers/train, training_replica_model.cpp | GAP1 | done |
| MULTIIN Named extra model inputs fed from batch roles; objective `target` role key; decisions dataset roles choice/yes_no/number; remove DecisionInputsObjective and DecisionTermObjective | core/config model_config, layers/train training graph and objective selection, language/datasets decision_dataset, language/decision_models | L6f, TDOUBLE | done |
| NNROWS nn rows mul, maximum, sub, div (broadcasting via G3 ops), softmax, transpose; conv2d groups; maxpool asymmetric pads; wire them into the Caffe and ONNX tables; move CHW->HWC inner-product weight permutation into interchange axis_translation shared by Caffe and ONNX | layers/nn rows, interchange axis_translation, vision/importers caffe_op_eltwise/softmax/inner_product, onnx gemm | MI75, SA-OPS | done |
| HOSTEVAL HostEvaluator rows for div, maximum, sqrt, log, pow, broadcast ops, sum_along, take_along, put_along, argmax_rows; write_tensor/read_tensor INT64 (replace local write_int64 helpers); move shared gradient Probe and expect_close into tests/support and split rules/test_all.cpp by family | core/graph host evaluator, tensor values support, tests/core/autograd/rules | SA-OPS | done |
| G4B log_softmax out-of-place sub; ops.hpp K_ ids for the four G4 literals; split lowering/linalg.cpp into matmul, row-index and mask units | core/ops ops.hpp, core/graph lowering linalg/loss/norm | G4, HOSTEVAL | done |
| SBPORT SushiBLAS work moved to feature/graph_capabilities from main in a scratch worktree; registration, refactors, checkers, commits | sushiblas worktree | SB-B1..B5 | done |
| A12D Objective hook publishing extra observed values; measure_segmentation on device via argmax_rows; drop the 4096-class ConfusionMatrix limit | layers/train objective/steps, vision/segment, vision/training | MULTIIN | done |
| MULTIINB revert role_match extent-of-1 widening, declare dense objective role shape from the target output width; copy_extra_inputs helper shared by Trainer and replica feed; sa inference and evaluation callers feed extra inputs | layers/train role_match, objectives, extra_inputs, trainer; applications/programs inference | MULTIIN, A12D | done |
| NNROWSB graph K_TRANSPOSE op (builder, lowering via a SushiBLAS permute copy or existing kernel, autograd rule, host row, tests); Transpose and non-last-axis Softmax use it, permute_axes identity-matmul goes; MaxPool2d one signed end pad per axis replacing trailing_pad/end_shortfall; ONNX Mul general binary row | core/ops, core/graph builder+lowering, core/autograd, layers/nn transpose/softmax/max_pool2d/trailing_window, vision/importers onnx_op_mul | NNROWS | done |
| SBSTACK SushiBLAS one branch feature/sushiblas_stack = main + global_handle_runtime (11) + graph_capabilities (3), ours ported to GlobalHandle | sushiblas worktree sb_wt | SBPORT | done |
| GAP2B Objective::declare_metric_values receives the loss result (a LossTrace with term ids/leaves) so no objective keeps mutable state; FreeLossTerms own header; free loop calls match_input; free_detection_loss_host moves to tests/support | layers/train objective, sum/dice/pixel/free objectives, vision/training free_* | GAP2 | done |
| INT Registration, CMake, design, manual, changelog, backlog | shared files | all | done |
| BUILD sa build, sb build, sa test, sb test; fix | - | INT | todo |
