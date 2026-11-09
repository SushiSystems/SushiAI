# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module language/text; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/text
    SOURCES
        compile_token.cpp
        test_bpe_trainer.cpp
        test_codec.cpp
        test_stream_encoder.cpp
        test_vocabulary.cpp
        test_vocabulary_file.cpp
    DEPENDS
        failure_expectations)
