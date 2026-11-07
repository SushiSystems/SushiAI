# tests/core/eager/test_region_program.cpp


Suites: `Small_RegionProgram`, `Medium_RegionProgramFatal`.

Tests the graph, the plan and the reports `compile_region` makes for one region.

This file needs no device. `modules/core/eager/README.md`, under `region_program.hpp`, describes
the program.

## TheGraphHoldsTheRegionsNodesBehindOneLeafPerImport

The region "activation#0" reads the linear layer's result and computes the step's result. With
fusion off, the import is produced by a K_LEAF node of the program and the export by another
op that reads the import. The case does not fix the node order or the node count.

## ValuesKeepTheIdsOfTheStep

A value has the same id in the program and in the step, so the storage list of the step addresses
both.

## AnImportAnotherRegionComputesIsAnInputOfTheProgram

The linear layer's result is an activation in the step and has `Lifetime::Input` in the program
that imports it.

## AParameterImportKeepsItsLifetime

A leaf of the step keeps the lifetime it was declared with.

## NoSharedValueIsPackedIntoAnArena

Over every region of a differentiated three-segment step, no import and no export is packed. A
packed export would be overwritten by a later value of the same region while another region
still reads it.

## AChainInOneRegionFolds

With no scope, K_MATMUL, K_BIAS_ADD and K_RELU fold into one `GEMMBiasReLU` and the program
holds no K_RELU.

## AChainCutByAScopeFoldsOnlyUpToTheBoundary

With a scope between the layer and the activation, the first region folds a `GEMMBias`, no
region holds a `GEMMBiasReLU`, and the second region keeps its K_RELU. This is the price of a
boundary the design names.

## SwitchingFusionOffLeavesTheNodesAsTraced

`ProgramOptions::fusion` reaches the pass.

## AKeyThatNamesNoRegionIsFatal

`K_NO_REGION` is the key of the leaves and compiles to nothing.

## AZeroAlignmentComesBackAsAStatus

A `ProgramOptions::plan` with an alignment of zero reaches the memory planner, and
`compile_region` returns its `InvalidDescription`, "alignment must be positive", instead of
stopping the process.

## ANonFloatCastComesBackAsAStatus

An `InMemoryPrecisionPolicy` scripted to ask int32 of the three operations of the region
reaches the cast pass, and `compile_region` returns its `InvalidDescription`, "real floating types".
