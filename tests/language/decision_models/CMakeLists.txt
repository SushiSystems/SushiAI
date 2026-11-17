# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module language/decision_models; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/decision_models
    SOURCES
        test_calibrate_decision_model.cpp
        test_decision_calibration.cpp
        test_decision_model.cpp
        test_decision_predictions.cpp
        test_decision_scores.cpp
        test_decision_session.cpp
        test_decision_set.cpp
    DEPENDS
        language/decision_models
        language/registration
        model_text
        failure_expectations
        scratch_files)
