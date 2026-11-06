# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module vision/codecs; see cmake/SushiAITestProgram.cmake.

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)

sushiai_add_test_program(OWNER vision/codecs
    SOURCES
        test_decode_image.cpp
        test_decode_options.cpp
        test_image_format.cpp
        test_jpeg_decoder.cpp
        test_png_decoder.cpp
    DEPENDS
        failure_expectations
    LINKS
        JPEG::JPEG
        PNG::PNG)
