# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# ============================================================================
# SushiAI test pipeline
#
# Three GTest tiers, distinguished by the suite-name prefix and nothing else:
# `Unit_*` is device-free host-lane work (shapes, graph topology, the
# differentiation transform, the parameter walk, the gradient checker's own
# arithmetic); `Integration_*` runs the real Engine through both siblings;
# `Regression_*` pins a guarantee whose failure is silent, so a future change
# cannot quietly take it away.
#
# THE PREFIX IS THE ONLY THING THAT ASSIGNS A CTEST LABEL to a GTest case. A
# suite named without one is discovered by none of the gtest_discover_tests
# blocks below, so it never runs under `sa test` and never runs in CI - it is
# invisible rather than failing. Adding a fourth tier means a fourth block here
# AND a matching entry in cli/sushiai/services/project.py's Suite enum and
# _SUITE_LABEL_REGEX.
#
# One test is not a GTest case: the `sa demo mlp` smoke test at the bottom runs
# the demo binary itself, because the thing being guarded is the demo's printed
# contract and no in-process test can see that. It carries the `integration`
# label explicitly.
#
# This whole directory is behind SA_BUILD_TESTS (root CMakeLists.txt).
# ============================================================================

cmake_minimum_required(VERSION 3.22) # gtest_discover_tests(TEST_FILTER ...)

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# The tests of a module sit in its own tests/ folder and reach this binary
# through sushiai_add_module(); the files listed here cross module boundaries.
get_property(SUSHIAI_MODULE_TESTS GLOBAL PROPERTY SUSHIAI_TESTS_sushiai)

add_executable(sushiai_tests
    common/test_main.cpp
    ${SUSHIAI_MODULE_TESTS}

    unit/test_backward.cpp
    unit/test_conv2d.cpp
    unit/test_diagnostics.cpp
    unit/test_layer_registry.cpp
    unit/test_loss_roots.cpp
    unit/test_loss_scaling.cpp
    unit/test_seam.cpp
    integration/test_arena.cpp
    integration/test_max_pool_trailing.cpp
    integration/test_mish_values.cpp
    integration/test_attention_operations.cpp
    integration/test_autograd_numeric.cpp
    integration/test_buffer_update_recorder.cpp
    integration/test_checkpoint.cpp
    integration/test_engine_smoke.cpp
    integration/test_evaluation_step.cpp
    integration/test_forward.cpp
    integration/test_gather_rows.cpp
    integration/test_dropout.cpp
    integration/test_layer_norm.cpp
    integration/test_run_checkpoint.cpp
    integration/test_token_feed.cpp
    integration/test_dense_objectives.cpp
    integration/test_import_weights.cpp
    integration/test_init_state.cpp
    integration/test_lowered_operations.cpp
    integration/test_mixed_precision.cpp
    integration/test_model_state.cpp
    integration/test_registries.cpp
    integration/test_rematerialise.cpp
    integration/test_resilience.cpp
    integration/test_state_checkpoint.cpp
    integration/test_training.cpp
    regression/test_argmax_lifetime.cpp
    regression/test_compile_once.cpp
    regression/test_determinism.cpp
    regression/test_fused_optimizer.cpp
    regression/test_fusion_numerics.cpp
    regression/test_learning_curve.cpp
    regression/test_log_softmax_stability.cpp
    regression/test_loss_scaling.cpp
    regression/test_mixed_precision.cpp
    regression/test_static_memory.cpp
)

# Test support is included as <common/name.hpp> from every test folder.
target_include_directories(sushiai_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# GTest::gtest_main is deliberately absent: common/test_main.cpp supplies main()
# so the RuntimeContext is built once per binary rather than once per test.
# Linking both is a duplicate-symbol error at link time.
target_link_libraries(sushiai_tests PRIVATE
    sushiai
    GTest::gtest
)

# Where the reference tensors live; a test that needs one reports itself skipped
# when the file is absent (tests/common/reference_fixture.hpp).
target_compile_definitions(sushiai_tests PRIVATE
    SA_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

sushiai_apply_build_policy(sushiai_tests)

gtest_discover_tests(sushiai_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Unit_*"
    PROPERTIES LABELS "unit"
)
gtest_discover_tests(sushiai_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Integration_*"
    PROPERTIES LABELS "integration"
)
gtest_discover_tests(sushiai_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Regression_*"
    PROPERTIES LABELS "regression"
)

# --- The `sa demo mlp` smoke test ---
# A demo whose output nobody checks rots into a lie, and this one's output is
# quoted in docs/README.md. The run is deliberately tiny (2 steps on the
# synthetic fallback, pointed at a directory that cannot hold MNIST) so it
# costs about as much as one GTest case.
#
# The assertion is PASS_REGULAR_EXPRESSION rather than the exit code because
# the compile-count line IS the contract - and CTest ignores a test's exit code
# once that property is set, so a bare exit-0 check would be silently dropped
# anyway. `sa demo mlp`'s exit status is covered from the CLI side, in
# cli/tests/test_demo.py.
#
# The regex's trailing [^0-9] is load-bearing: CMake's regex has no multiline
# mode, so `$` would anchor to the end of the whole output rather than the end
# of the line, and without it "compiles : 12" would satisfy a bare "...: 1".
# The demo reads MNIST, so it exists only in a build with the vision domain.
if(TARGET sushiai_demo_mlp)
    add_test(NAME Integration_Demo.MLPRunsAndReportsOneCompile
        COMMAND sushiai_demo_mlp
            --epochs 1 --batches 2 --batch-size 8 --hidden 16
            --features 6 --classes 4 --mnist-dir does-not-exist
    )
    set_tests_properties(Integration_Demo.MLPRunsAndReportsOneCompile PROPERTIES
        LABELS "integration"
        PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
    )
endif()

# --- The `sushiai train` / `sushiai inference` smoke tests ---
# Two separate tests, not one chained round trip: a single test that trained
# and then evaluated would report one failure for two independent programs, and
# a broken inference would look like a broken trainer.
#
# tiny_train.json is GENERATED rather than committed, because a train.json's
# relative paths resolve against its own directory - so a committed one would
# write its checkpoint into the source tree. Generating it puts absolute paths
# into the binary tree instead. tiny_model.json and tiny_eval.json name no
# paths at all and are committed as they are.
set(SUSHIAI_TEST_MODEL_JSON "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny_model.json")
set(SUSHIAI_TEST_CHECKPOINT "${CMAKE_CURRENT_BINARY_DIR}/fixtures/tiny.sushi")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny_train.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/tiny_train.json"
    @ONLY
)

# The compile-count line IS the contract, exactly as it is for `sa demo mlp`,
# so the assertion is PASS_REGULAR_EXPRESSION rather than the exit code -
# CTest ignores a test's exit code once that property is set, so a bare exit-0
# check would be silently dropped anyway.
#
# The trailing [^0-9] is load-bearing: CMake's regex has no multiline mode, so
# `$` would anchor to the end of the whole output rather than the end of the
# line, and without it "compiles: 12" would satisfy a bare "...: 1".
add_test(NAME Integration_App.TrainRunsAndReportsOneCompile
    COMMAND sushiai_train "${CMAKE_CURRENT_BINARY_DIR}/fixtures/tiny_train.json"
)
set_tests_properties(Integration_App.TrainRunsAndReportsOneCompile PROPERTIES
    LABELS "integration"
    PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
)

# DEPENDS is a CTest ordering constraint, not a chained assertion: inference
# needs a checkpoint to exist, and the training test is what writes one. Each
# still asserts its own contract on its own output.
add_test(NAME Integration_App.InferenceLoadsACheckpointAndReportsOneCompile
    COMMAND sushiai_inference
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny_model.json"
        "${SUSHIAI_TEST_CHECKPOINT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny_eval.json"
)
set_tests_properties(Integration_App.InferenceLoadsACheckpointAndReportsOneCompile PROPERTIES
    LABELS "integration"
    DEPENDS Integration_App.TrainRunsAndReportsOneCompile
    PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
)

# --- Post-Build: Copy the runtime DLL ---
if(WIN32)
    add_custom_command(TARGET sushiai_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiai_tests>
        COMMENT "Copying runtime DLL to test directory..."
    )
endif()

# --- The vision domain's suite ---
# Its cases keep the Unit_, Integration_ and Regression_ prefixes, so the
# labels above select them too.
if(SA_BUILD_VISION)
    get_property(SUSHIAI_VISION_MODULE_TESTS GLOBAL PROPERTY SUSHIAI_TESTS_sushiai_vision)

    add_executable(sushiai_vision_tests
        common/test_main.cpp
        ${SUSHIAI_VISION_MODULE_TESTS}

        vision_support/darknet_forward.cpp
        vision_support/darknet_import.cpp
        vision_support/darknet_reference.cpp
        vision_support/darknet_training.cpp
        integration/test_anchor_training_step.cpp
        integration/test_darknet_decode.cpp
        integration/test_darknet_import.cpp
        integration/test_darknet_store.cpp
        integration/test_darknet_yolov3_forward.cpp
        integration/test_darknet_yolov3_tiny_forward.cpp
        integration/test_darknet_yolov3_tiny_layers.cpp
        integration/test_darknet_yolov4_forward.cpp
        integration/test_darknet_yolov4_tiny_forward.cpp
        integration/test_darknet_yolov7_forward.cpp
        integration/test_darknet_yolov7_tiny_forward.cpp
        integration/test_darknet_yolov3_tiny_training.cpp
        integration/test_darknet_yolov3_training.cpp
        integration/test_darknet_yolov4_tiny_training.cpp
        integration/test_darknet_yolov4_training.cpp
        integration/test_darknet_yolov7_tiny_training.cpp
        integration/test_darknet_yolov7_training.cpp
        integration/test_head_delta_loss.cpp
        integration/test_pixel_objectives.cpp
        integration/test_segment_overfit.cpp
        integration/test_yolo26_forward.cpp
    )

    target_include_directories(sushiai_vision_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

    find_package(JPEG REQUIRED)
    find_package(PNG REQUIRED)

    target_link_libraries(sushiai_vision_tests PRIVATE
        sushiai_vision
        GTest::gtest
        JPEG::JPEG
        PNG::PNG
    )

    target_compile_definitions(sushiai_vision_tests PRIVATE
        SA_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    )

    sushiai_apply_build_policy(sushiai_vision_tests)

    foreach(prefix_label IN ITEMS "Unit:unit" "Integration:integration" "Regression:regression")
        string(REPLACE ":" ";" prefix_and_label "${prefix_label}")
        list(GET prefix_and_label 0 prefix)
        list(GET prefix_and_label 1 label)
        gtest_discover_tests(sushiai_vision_tests
            DISCOVERY_MODE PRE_TEST
            TEST_FILTER "${prefix}_*"
            PROPERTIES LABELS "${label}"
        )
    endforeach()

    if(WIN32)
        add_custom_command(TARGET sushiai_vision_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:sushiai_vision_tests>
            COMMENT "Copying runtime DLL to the vision test directory..."
        )
    endif()
endif()

# --- The language domain's suite ---
# Its cases keep the Unit_, Integration_ and Regression_ prefixes, as the vision suite's do.
if(SA_BUILD_LANGUAGE)
    get_property(SUSHIAI_LANGUAGE_MODULE_TESTS GLOBAL PROPERTY SUSHIAI_TESTS_sushiai_language)

    add_executable(sushiai_language_tests
        common/test_main.cpp
        ${SUSHIAI_LANGUAGE_MODULE_TESTS}

        integration/test_attention_gradient.cpp
        integration/test_causal_attention.cpp
        integration/test_decoder_from_files.cpp
        integration/test_decoder_overfit.cpp
        integration/test_cached_generation.cpp
        integration/test_rotary_decoder_overfit.cpp
        integration/test_rotary_embedding.cpp
        integration/test_rotary_generation.cpp
        integration/test_generation.cpp
        integration/test_moe.cpp
        integration/test_next_token_objective.cpp
        integration/test_next_token_trainer.cpp
    )

    target_include_directories(sushiai_language_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

    target_link_libraries(sushiai_language_tests PRIVATE
        sushiai_language
        GTest::gtest
    )

    sushiai_apply_build_policy(sushiai_language_tests)

    foreach(prefix_label IN ITEMS "Unit:unit" "Integration:integration" "Regression:regression")
        string(REPLACE ":" ";" prefix_and_label "${prefix_label}")
        list(GET prefix_and_label 0 prefix)
        list(GET prefix_and_label 1 label)
        gtest_discover_tests(sushiai_language_tests
            DISCOVERY_MODE PRE_TEST
            TEST_FILTER "${prefix}_*"
            PROPERTIES LABELS "${label}"
        )
    endforeach()

    if(WIN32)
        add_custom_command(TARGET sushiai_language_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:sushiai_language_tests>
            COMMENT "Copying runtime DLL to the language test directory..."
        )
    endif()
endif()

# --- The programs' suite ---
# The programs are executables with no library, so the units a test can reach without a
# process are compiled here beside the test.
set(SUSHIAI_PROGRAM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../applications/programs/source")

add_executable(sushiai_program_tests
    common/test_main.cpp
    integration/test_program_runs.cpp
    "${SUSHIAI_PROGRAM_SOURCE_DIR}/training_run.cpp"
    "${SUSHIAI_PROGRAM_SOURCE_DIR}/trainer_run.cpp"
    "${SUSHIAI_PROGRAM_SOURCE_DIR}/inference_run.cpp"
    "${SUSHIAI_PROGRAM_SOURCE_DIR}/run_start.cpp"
)

target_include_directories(sushiai_program_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SUSHIAI_PROGRAM_SOURCE_DIR}")

target_link_libraries(sushiai_program_tests PRIVATE
    sushiai
    GTest::gtest
)

target_compile_definitions(sushiai_program_tests PRIVATE
    SA_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

sushiai_apply_build_policy(sushiai_program_tests)

gtest_discover_tests(sushiai_program_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Integration_*"
    PROPERTIES LABELS "integration"
)

if(WIN32)
    add_custom_command(TARGET sushiai_program_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiai_program_tests>
        COMMENT "Copying runtime DLL to the program test directory..."
    )
endif()
