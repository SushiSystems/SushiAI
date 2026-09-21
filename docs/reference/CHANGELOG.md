# Changelog

All notable changes to SushiAI are documented here.

## Unreleased

- 2026-09-22 — build: Took `sushicore` from PyPI instead of a sibling checkout (`cli/pyproject.toml`).
- 2026-09-16 — docs: Standardized source comments and file headers across core and source trees (`include/`, `src/`).
- 2026-09-16 — docs: Restructured documentation hierarchy and added repository hygiene tools (`docs/`, `tools/`).

- 2026-09-07 — docs: Renamed the hub CLI command from ss to hub (`docs/`).
- 2026-08-09 — graph: Validated tensor layout uniformity on entry to graph lowering (`Lowering::record`, `lowering.cpp`).
- 2026-08-08 — cli: Added out-of-tree package consumer test suite to sa test (`project.py`, `tests/package/`).
- 2026-08-08 — build: Added configure-time check verifying the umbrella header includes every public header (`src/CMakeLists.txt`, `SushiAI.h`).
- 2026-08-08 — graph: Added IM2COL activation rematerialisation pass demoting column matrix to scratch storage (`rematerialise.hpp`).
- 2026-08-08 — test: Pinned MAXPOOL2D argmax lifetime preservation against memory planner buffer overlap (`test_argmax_lifetime.cpp`).
- 2026-08-08 — autograd: Added gradient rule and lowering for LEAKY_RELU activation (`rules/activation.cpp`, `lowering.cpp`).
- 2026-08-08 — autograd: Added gradient rules for batch normalisation and channel affine operations (`rules/norm.cpp`).
- 2026-08-08 — autograd: Added gradient rules for channel concatenation and channel slicing (`rules/shape.cpp`).
- 2026-08-08 — autograd: Added gradient rules for 2D pooling, 2D upsampling, and spatial mean operations (`rules/spatial.cpp`).
- 2026-08-08 — autograd: Added backward differentiation rules for IM2COL and COL2IM convolution primitives (`rules/spatial.cpp`).
- 2026-08-08 — autograd: Added backward rules for RESHAPE and LAYOUT_CONVERT operations (`rules/shape.cpp`).
- 2026-08-08 — autograd: Split backward differentiation rules into one translation unit per domain (`rules/`, `differentiator.cpp`).
- 2026-08-08 — graph: Renamed Graph::propagate_requires_grad to Graph::finalise (`ir.hpp`, `differentiator.cpp`).
- 2026-08-08 — graph: Added shape inference emitters for pooling, upsample, channels, and normalisation (`graph_builder.cpp`).
- 2026-08-08 — graph: Stored convolution output window extents in attributes rather than re-deriving them (`ir.hpp`).
- 2026-08-08 — graph: Corrected emitted operation count reporting for batchnorm and channel concat (`lowering.cpp`).
- 2026-08-08 — graph: Inherited gradient layout operand in activation backward shape declaration (`graph_builder.cpp`).
- 2026-08-08 — graph: Added rank validation before extent division in spatial mean backward lowering (`lowering.cpp`).
- 2026-08-08 — graph: Unified LeakyReLU slope range validation across forward and backward paths (`graph_builder.cpp`).
- 2026-08-08 — autograd: Rejected gradient propagation on saved BatchNorm running statistics (`rules/norm.cpp`).
- 2026-08-08 — test: Replaced wildcard test names with literal test identifiers in backward coverage ledger (`test_backward.cpp`).
- 2026-08-07 — graph: Added tensor layout invariants to IR and declared convolutional op identities (`ir.hpp`, `ops.hpp`).
- 2026-08-07 — cli: Added train and inference subcommands for JSON-configured workflows (`train.py`, `inference.py`).
- 2026-08-07 — apps: Added config-driven training and inference application drivers (`apps/train.cpp`, `apps/inference.cpp`).
- 2026-08-07 — io: Implemented SACP little-endian checkpoint format serialization and deserialization (`checkpoint.hpp`).
- 2026-08-07 — test: Verified checkpoint exchange equivalence between Sequential and LayerStack models (`test_checkpoint.cpp`).
- 2026-08-07 — train: Added IObjective interface and objective registry for non-classification loss tasks (`objective.hpp`).
- 2026-08-07 — data: Added concentric rings synthetic classification dataset and registry (`rings.hpp`).
- 2026-08-07 — config: Added typed configuration parser tree behind private JSON parser implementation (`value.hpp`).
- 2026-08-07 — nn: Added runtime dynamic LayerStack matching Sequential parameter enumeration (`layer_stack.hpp`).
- 2026-08-07 — examples: Converted heavy MLP example into JSON configuration files (`examples/heavy_mlp/`).
- 2026-08-04 — docs: Added step-by-step tutorial walking from checkout to trained model execution (`TUTORIAL.md`).
- 2026-08-04 — graph: Fused GELU into GEMM epilogue with pre-activation spill buffer allocation (`fusion.cpp`).
- 2026-08-04 — cli: Added sa doxygen command and committed documentation configuration (`docs/Doxyfile`, `mainpage.md`).
- 2026-08-03 — train: Added dynamic loss scaling with overflow backoff for mixed-precision training (`loss_scaler.hpp`).
- 2026-08-03 — graph: Added mixed-precision policy and automatic cast insertion pass (`precision.hpp`).
- 2026-08-03 — graph: Added graph compilation pass fusing linear and activation operations into GEMM epilogues (`fusion.cpp`).
- 2026-08-03 — graph: Added static memory planner packing activations into per-lifetime scratch arenas (`memory_planner.cpp`).
- 2026-08-03 — cli: Added sa demo mlp command running end-to-end MLP training showcase (`demo.py`).
- 2026-08-03 — build: Aligned codebase and interface headers to C++17 standard across SushiStack (`CMakeLists.txt`).
- 2026-08-03 — graph: Shifted log-softmax by row-wise maximum instead of batch maximum for numerical stability (`lowering.cpp`).
- 2026-08-03 — autograd: Implemented graph-native autograd engine and reverse-mode tape differentiation (`differentiator.cpp`).
- 2026-08-03 — train: Added Trainer coordinating forward execution, backward pass, and optimizer updates (`trainer.cpp`).
- 2026-08-03 — train: Rejected non-floating-point logit types in accuracy computation (`trainer.cpp`).
- 2026-08-03 — graph: Recorded ZERO node buffer fill into replay execution step (`lowering.cpp`).
- 2026-08-03 — graph: Handled idempotent identical buffer aliasing in tensor pool (`tensor_pool.cpp`).
- 2026-08-03 — autograd: Rejected unsupported loss roots during backward pass differentiation (`differentiator.cpp`).
- 2026-08-03 — graph: Zeroed and flushed gradient buffers at materialisation time (`lowering.cpp`).
- 2026-08-03 — core: Added checked integer multiplication preventing extent calculation overflow (`tensor.cpp`).
- 2026-08-02 — cli: Added sa CLI package supporting build, test, run, and environment management (`cli/sushiai/`).
- 2026-08-02 — build: Added D3 layered CMake dependency resolution for SushiRuntime and SushiBLAS (`cmake/`).
- 2026-08-02 — core: Created SushiAI C++ static library skeleton and version interface (`version.hpp`).
- 2026-08-02 — cli: Fixed C compiler binary discovery in bundled toolchain (`services/project.py`).
- 2026-08-02 — build: Added path existence checks for SUSHISTACK_HOME sibling resolution (`Runtime.cmake`, `BLAS.cmake`).
