# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module language/text; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/text
    SOURCES
        test_bpe_trainer.cpp
        test_codec.cpp
        test_stream_encoder.cpp
        test_vocabulary.cpp
        test_vocabulary_file.cpp
    DEPENDS
        failure_expectations)
