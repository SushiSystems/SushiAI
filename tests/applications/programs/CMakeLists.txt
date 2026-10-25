# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the application programs, and the smoke tests of sushiai_train and
# sushiai_inference.

# The programs are executables with no library, so the units a test reaches without a
# process are compiled here beside the tests.
set(SUSHIAI_PROGRAM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/applications/programs/source")

set(SUSHIAI_PROGRAM_TEST_DOMAINS)
set(SUSHIAI_PROGRAM_TEST_LINKS)
if(SA_BUILD_VISION)
    list(APPEND SUSHIAI_PROGRAM_TEST_DOMAINS "${SUSHIAI_PROGRAM_SOURCE_DIR}/domains_vision.cpp")
    list(APPEND SUSHIAI_PROGRAM_TEST_LINKS sushiai_vision)
else()
    list(APPEND SUSHIAI_PROGRAM_TEST_DOMAINS "${SUSHIAI_PROGRAM_SOURCE_DIR}/domains_vision_none.cpp")
endif()
if(SA_BUILD_LANGUAGE)
    list(APPEND SUSHIAI_PROGRAM_TEST_DOMAINS "${SUSHIAI_PROGRAM_SOURCE_DIR}/domains_language.cpp")
    list(APPEND SUSHIAI_PROGRAM_TEST_LINKS sushiai_language)
else()
    list(APPEND SUSHIAI_PROGRAM_TEST_DOMAINS
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/domains_language_none.cpp")
endif()

# The cluster units and their tests exist in a distributed build; any other
# build compiles the refusal in their place.
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
        source/test_worker_arguments.cpp
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/cluster_training.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/cluster_run.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/master_cluster.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/served_kernel.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/training_cluster_section.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/training_replica_model.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/replica_tensors.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/replica_fallback_warning.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/worker_arguments.cpp")
    list(APPEND SUSHIAI_PROGRAM_TEST_CLUSTER_DEPENDS distribution)
else()
    list(APPEND SUSHIAI_PROGRAM_TEST_CLUSTER
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/cluster_training_none.cpp")
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
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/training_run.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/epoch_loop.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/trainer_run.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/inference_run.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/registries.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/run_start.cpp"
        "${SUSHIAI_PROGRAM_SOURCE_DIR}/training_setup.cpp"
        ${SUSHIAI_PROGRAM_TEST_CLUSTER}
        ${SUSHIAI_PROGRAM_TEST_DOMAINS}
    DEPENDS
        classifier_objective
        failure_expectations
        scratch_files
        ${SUSHIAI_PROGRAM_TEST_CLUSTER_DEPENDS}
    LINKS
        ${SUSHIAI_PROGRAM_TEST_LINKS}
    INCLUDES
        "${SUSHIAI_PROGRAM_SOURCE_DIR}")

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
