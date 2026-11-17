# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# decision_models: the synthetic decision set, the decision model the language tests train and the
# calibration step that stores its temperatures in the model file.

sushiai_add_module(NAME decision_models TIER language INTERNAL
    SOURCES
        source/calibrate_decision_model.cpp
        source/decision_calibration.cpp
        source/decision_predictions.cpp
        source/decision_run.cpp
        source/decision_scores.cpp
        source/decision_session.cpp
        source/decision_set.cpp
    PUBLIC_DEPENDS config data execution failure nn train
        language/datasets language/text
    PRIVATE_DEPENDS dtype graph optim seam shape
        language/blocks language/training
    EXTERNAL_PUBLIC SushiRuntime::core)
