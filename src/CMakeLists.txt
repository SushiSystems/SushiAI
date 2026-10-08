# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

cmake_minimum_required(VERSION 3.20)

# --- Define SushiAI Library ---
# Every source belongs in this ONE variable: it feeds add_library() AND
# sushiai_apply_sycl() below, so a file added straight to add_library() would
# compile without -fsycl on the intel-llvm path and fail to link the moment it
# touches a SushiBLAS kernel.
set(SUSHIAI_SOURCES
    version.cpp

    config/value.cpp
    config/model_config.cpp
    config/train_config.cpp

    core/allocation.cpp
    core/check.cpp
    core/seam.cpp
    core/status.cpp

    graph/builder.cpp
    graph/precision.cpp
    graph/cast_insertion.cpp
    graph/fusion.cpp
    graph/rematerialise.cpp
    graph/memory_plan.cpp
    graph/arena.cpp
    graph/tensor_pool.cpp
    graph/fusion_selector.cpp
    graph/lowering.cpp
    graph/lowering/activation.cpp
    graph/lowering/context.cpp
    graph/lowering/elementwise.cpp
    graph/lowering/fused.cpp
    graph/lowering/linalg.cpp
    graph/lowering/loss.cpp
    graph/lowering/norm.cpp
    graph/lowering/shape.cpp
    graph/lowering/spatial.cpp
    graph/compiled_step.cpp

    autograd/differentiator.cpp
    autograd/tape.cpp
    autograd/rules/elementwise.cpp
    autograd/rules/linalg.cpp
    autograd/rules/activation.cpp
    autograd/rules/loss.cpp
    autograd/rules/shape.cpp
    autograd/rules/spatial.cpp
    autograd/rules/norm.cpp
    autograd/gradcheck.cpp

    nn/layers.cpp
    nn/unflatten.cpp
    nn/conv2d.cpp
    nn/max_pool2d.cpp
    nn/avg_pool2d.cpp
    nn/global_avg_pool.cpp
    nn/flatten.cpp
    nn/leaky_relu.cpp
    nn/upsample2d.cpp
    nn/layer_stack.cpp
    nn/layer_registry.cpp
    nn/init.cpp
    nn/batch_norm2d.cpp
    nn/concat.cpp
    nn/identity.cpp
    nn/layer_node.cpp
    nn/model_graph.cpp
    nn/model_outputs.cpp
    nn/node_registry.cpp
    nn/silu.cpp
    nn/trace.cpp
    nn/blocks/block_attributes.cpp
    nn/blocks/bottleneck.cpp
    nn/blocks/c3k.cpp
    nn/blocks/c3k2.cpp
    nn/blocks/conv_norm.cpp
    nn/blocks/sppf.cpp

    optim/optimizer.cpp
    optim/step_scalars.cpp
    optim/loss_scaler.cpp
    optim/sgd.cpp
    optim/adamw.cpp
    optim/optimizer_registry.cpp

    data/dataset.cpp

    detect/anchors.cpp
    detect/assigner.cpp
    detect/assigner_config.cpp
    detect/box_coding.cpp
    detect/box_format.cpp
    detect/detections_nms.cpp
    detect/detections_topk.cpp
    detect/ground_truth.cpp
    detect/iou.cpp
    detect/level_rows.cpp
    detect/target_leaves.cpp
    detect/top_k.cpp
    detect/anchor_leaves.cpp
    detect/average_precision.cpp

    imaging/affine.cpp
    imaging/boxes.cpp
    imaging/flip.cpp
    imaging/hsv.cpp
    imaging/image.cpp
    imaging/labels.cpp
    imaging/letterbox.cpp
    imaging/mosaic.cpp
    imaging/pipeline.cpp
    imaging/random.cpp
    imaging/raw_cache.cpp
    imaging/resize.cpp
    imaging/bilinear.cpp
    imaging/dataset_partners.cpp
    imaging/gray.cpp
    imaging/label_set.cpp
    imaging/polygon.cpp
    imaging/recent_partners.cpp
    imaging/rectangle.cpp

    eager/scope_trace.cpp
    eager/scope_partitioner.cpp
    eager/region_signature.cpp

    io/checkpoint_format.cpp

    train/objective.cpp
    train/trainer.cpp
)

add_library(sushiai STATIC ${SUSHIAI_SOURCES})

# A namespaced alias so an in-tree consumer (add_subdirectory, D3's
# TARGET-guard/sibling modes) and an out-of-tree one (find_package) spell the
# dependency identically - mirrors SushiBLAS::sushiblas exactly (SB-6).
add_library(SushiAI::sushiai ALIAS sushiai)

# --- Include Directories ---
target_include_directories(sushiai PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# --- Compile Policy ---
# Warnings, architecture targeting and sanitizers, PRIVATE to this tree.
sushiai_apply_build_policy(sushiai)

# --- Link Libraries ---
# PUBLIC so consumers (examples, tests, and SushiAI's own future autograd
# core) inherit both siblings' include paths and, transitively, SushiRuntime's
# `sushiruntime_deterministic_fp` floating-point contract - SushiAI does not
# need (or own) a copy of that guarantee itself.
target_link_libraries(sushiai PUBLIC
    SushiRuntime::SushiRuntime
    SushiBLAS::sushiblas
)

# --- The config layer's JSON parser ---
# The name `nlohmann` appears in no public header: src/config/value.cpp is the
# only translation unit that includes it. That is the containment
# SushiRuntime::span and SushiBLAS::Core::format apply to std::span and
# std::format, and it is what lets cmake/SushiAIConfig.cmake.in stay free of a
# find_dependency() and the out-of-tree consumer in tests/package/ never see
# this dependency at all.
#
# The include directories are pulled on as VALUES rather than by linking the
# target, and that is required rather than stylistic. `sushiai` is STATIC, so a
# static archive carries none of its dependencies - CMake therefore records
# even a PRIVATE link as $<LINK_ONLY:...> in the exported
# INTERFACE_LINK_LIBRARIES, and a consumer that ran find_package(SushiAI)
# without also finding nlohmann_json would fail to configure on a name it
# should never have heard. Measured, not assumed: the export set carried
# $<LINK_ONLY:nlohmann_json::nlohmann_json> until this was written this way.
#
# It is safe to drop rather than propagate because nlohmann_json is an
# INTERFACE (header-only) target: it contributes include directories and
# definitions and there is no archive for anyone to link. This is the same
# export-set-safe shape sushiai_apply_build_policy() uses for the warning,
# architecture and sanitizer targets - see docs/ARCHITECTURE.md §6.
#
# Declared in cli/sushistack.deps.toml rather than a vcpkg.json manifest; that
# file's header says why a manifest would break this build.
find_package(nlohmann_json CONFIG REQUIRED)
target_include_directories(sushiai PRIVATE
    "$<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>"
)
target_compile_definitions(sushiai PRIVATE
    "$<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_COMPILE_DEFINITIONS>"
)

# --- Properties ---
# Restated on the target rather than left to CMAKE_CXX_STANDARD alone, so an
# add_subdirectory consumer at a different standard still compiles this library
# at the one its ABI was decided under - see the root CMakeLists.txt comment.
set_target_properties(sushiai PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# SYCL device-compilation policy (see cmake/SushiAISyclTarget.cmake). The
# lowering, optimizer, dataset and trainer sources all reach SushiBLAS and
# therefore SYCL; -fsycl on the intel-llvm path is applied PUBLIC so examples
# and tests inherit it from linking sushiai rather than reapplying it.
sushiai_apply_sycl(sushiai SOURCES ${SUSHIAI_SOURCES})

# --- Install & export (SB-6 packaging shape, from day one) ---
install(TARGETS sushiai
    EXPORT SushiAITargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# --- The umbrella must name every installed header --------------------------
#
# install(DIRECTORY) above ships all of include/, so a header is installed
# whether or not anybody remembered to add it to SushiAI/SushiAI.h. Those are
# two different failures and only one of them is loud:
#
#   * referenced but not installed - the consumer's compile fails, and
#     `sa test --suite package` catches it.
#   * installed but never referenced - nothing fails anywhere. A consumer that
#     includes the umbrella (the documented entry point) simply never compiles
#     that header, so it is free to be broken, and no test in this repo or in
#     the package suite can say so.
#
# The second is not hypothetical: autograd/tape.hpp, autograd/rule.hpp,
# autograd/rules/all.hpp and graph/rematerialise.hpp were all missing from the
# umbrella and were found by reading, not by a failing test. The package suite
# cannot close that gap by construction - it can only compile what the umbrella
# names - so the completeness of the list is checked here instead, at the same
# place the public surface is declared.
#
# Configure time rather than a GTest case because the question is answered by
# globbing a directory, which CMake can do and a compiled test cannot without
# having the answer generated for it first. There are no exclusions: every
# header under include/SushiAI/ is public, and the day one is not, it belongs
# somewhere other than the installed include tree.
file(GLOB_RECURSE SUSHIAI_PUBLIC_HEADERS CONFIGURE_DEPENDS
    RELATIVE "${PROJECT_SOURCE_DIR}/include/SushiAI"
    "${PROJECT_SOURCE_DIR}/include/SushiAI/*.hpp"
)

file(READ "${PROJECT_SOURCE_DIR}/include/SushiAI/SushiAI.h" SUSHIAI_UMBRELLA_TEXT)

set(SUSHIAI_UNREFERENCED_HEADERS "")
foreach(header IN LISTS SUSHIAI_PUBLIC_HEADERS)
    string(FIND "${SUSHIAI_UMBRELLA_TEXT}" "\"${header}\"" found)
    if(found EQUAL -1)
        list(APPEND SUSHIAI_UNREFERENCED_HEADERS "${header}")
    endif()
endforeach()

if(SUSHIAI_UNREFERENCED_HEADERS)
    string(REPLACE ";" "\n    " SUSHIAI_UNREFERENCED_REPORT
        "${SUSHIAI_UNREFERENCED_HEADERS}")
    message(FATAL_ERROR
        "These public headers are installed but not included by "
        "include/SushiAI/SushiAI.h:\n    ${SUSHIAI_UNREFERENCED_REPORT}\n"
        "A consumer that includes the umbrella - the documented entry point - "
        "would never compile them, and nothing else in this repo would notice. "
        "Add each to the umbrella, or move it out of the installed include "
        "tree if it is not public.")
endif()

install(EXPORT SushiAITargets
    FILE SushiAITargets.cmake
    NAMESPACE SushiAI::
    DESTINATION "${SA_INSTALL_CMAKEDIR}"
)

# The same targets, usable from a build tree without installing.
export(EXPORT SushiAITargets
    FILE "${PROJECT_BINARY_DIR}/SushiAITargets.cmake"
    NAMESPACE SushiAI::
)
