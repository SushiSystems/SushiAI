# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# codecs: the decoders that turn the bytes of an image file into an image.

# libjpeg-turbo and libpng are named by this module's sources alone; see
# NOTICE.md and sushiai_add_module().
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)

sushiai_add_module(NAME codecs TIER vision
    SOURCES
        source/decode_image.cpp
        source/image_format.cpp
        source/jpeg_decoder.cpp
        source/png_decoder.cpp
    PUBLIC_DEPENDS failure vision/imaging
    EXTERNAL_PRIVATE_LINKS JPEG::JPEG PNG::PNG SushiLog::text
    EXTERNAL_PUBLIC SushiRuntime::core)
