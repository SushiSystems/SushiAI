# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the application programs, and the smoke tests of sushiai_train and
# sushiai_inference.

# The program sources are the library sushiai_program_shared, built once in applications/programs
# and linked here. The tests compile only their own units.
set(SUSHIAI_PROGRAM_TEST_SOURCES)
set(SUSHIAI_PROGRAM_TEST_LINKS sushiai_program_shared)
if(SA_BUILD_VISION)
    list(APPEND SUSHIAI_PROGRAM_TEST_SOURCES
        source/test_inference_program.cpp
        source/test_training_program.cpp)
    list(APPEND SUSHIAI_PROGRAM_TEST_LINKS sushiai_vision)
endif()
if(SA_BUILD_LANGUAGE)
    list(APPEND SUSHIAI_PROGRAM_TEST_LINKS sushiai_language)
endif()

# The cluster tests exist in a distributed build; any other build has none.
set(SUSHIAI_PROGRAM_TEST_PROFILED)
set(SUSHIAI_PROGRAM_TEST_PROFILED_DEPENDS)
if(SA_ENABLE_PROFILER)
    list(APPEND SUSHIAI_PROGRAM_TEST_PROFILED source/test_profile_session.cpp)
    list(APPEND SUSHIAI_PROGRAM_TEST_PROFILED_DEPENDS profile_records)
endif()

set(SUSHIAI_PROGRAM_TEST_CLUSTER)
set(SUSHIAI_PROGRAM_TEST_CLUSTER_DEPENDS)
if(SA_ENABLE_DISTRIBUTED)
    list(APPEND SUSHIAI_PROGRAM_TEST_CLUSTER
        source/test_cluster_run.cpp
        source/test_replica_fallback_warning.cpp
        source/test_cluster_training.cpp
        source/test_master_cluster.cpp
        source/test_replica_tensors.cpp
        source/test_served_kernel.cpp
        source/test_training_cluster_section.cpp
        source/test_training_replica_model.cpp
        source/test_worker_arguments.cpp)
    list(APPEND SUSHIAI_PROGRAM_TEST_CLUSTER_DEPENDS distribution)
endif()

sushiai_add_test_program(APPLICATION programs
    SOURCES
        source/test_domains.cpp
        source/test_programs_log_category.cpp
        source/test_epoch_loop.cpp
        source/test_inference_run.cpp
        source/test_registries.cpp
        source/test_run_start.cpp
        source/test_trainer_run.cpp
        source/test_training_run.cpp
        source/test_training_setup.cpp
        source/test_profile_arguments.cpp
        ${SUSHIAI_PROGRAM_TEST_PROFILED}
        ${SUSHIAI_PROGRAM_TEST_CLUSTER}
        ${SUSHIAI_PROGRAM_TEST_SOURCES}
    DEPENDS
        classifier_objective
        failure_expectations
        scratch_files
        ${SUSHIAI_PROGRAM_TEST_CLUSTER_DEPENDS}
        ${SUSHIAI_PROGRAM_TEST_PROFILED_DEPENDS}
    LINKS
        ${SUSHIAI_PROGRAM_TEST_LINKS}
        ${SA_PROFILER_LINKS})

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
set(SUSHIAI_TINY_RUN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tiny_run")
set(SUSHIAI_TINY_RUN_BINARY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/tiny_run")
set(SUSHIAI_TEST_MODEL_JSON "${SUSHIAI_TINY_RUN_SOURCE}/tiny_model.json")
set(SUSHIAI_TEST_CHECKPOINT "${SUSHIAI_TINY_RUN_BINARY}/tiny.sushi")
configure_file(
    "${SUSHIAI_TINY_RUN_SOURCE}/tiny_train.json.in"
    "${SUSHIAI_TINY_RUN_BINARY}/tiny_train.json"
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
add_test(NAME Medium_App.TrainRunsAndReportsOneCompile
    COMMAND sushiai_train "${SUSHIAI_TINY_RUN_BINARY}/tiny_train.json"
)
set_tests_properties(Medium_App.TrainRunsAndReportsOneCompile PROPERTIES
    LABELS "medium"
    TIMEOUT 300
    PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
)

# DEPENDS is a CTest ordering constraint, not a chained assertion: inference
# needs a checkpoint to exist, and the training test is what writes one. Each
# still asserts its own contract on its own output.
add_test(NAME Medium_App.InferenceLoadsACheckpointAndReportsOneCompile
    COMMAND sushiai_inference
        "${SUSHIAI_TEST_MODEL_JSON}"
        "${SUSHIAI_TEST_CHECKPOINT}"
        "${SUSHIAI_TINY_RUN_SOURCE}/tiny_eval.json"
)
set_tests_properties(Medium_App.InferenceLoadsACheckpointAndReportsOneCompile PROPERTIES
    LABELS "medium"
    TIMEOUT 300
    DEPENDS Medium_App.TrainRunsAndReportsOneCompile
    PASS_REGULAR_EXPRESSION "compiles[ \t]*:[ \t]*1[^0-9]"
)
