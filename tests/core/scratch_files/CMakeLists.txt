# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module scratch_files; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER scratch_files
    SOURCES
        test_scratch_directory.cpp
        test_scratch_file.cpp
    DEPENDS
        scratch_files)
