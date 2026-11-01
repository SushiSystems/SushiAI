# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# codecs: the decoders that turn the bytes of an image file into an image.

# libjpeg-turbo and libpng are named by this module's sources alone; see
# NOTICE.md and sushiai_add_module().
find_package(JPEG REQUIRED)
# A Debug tree links the Release libjpeg, as it links every imported package (SushiAIAbiPinning.cmake).
# vcpkg's debug build uses the debug C runtime, which this tree does not, and the heap then breaks.
if(JPEG_LIBRARY_RELEASE)
    set_property(TARGET JPEG::JPEG PROPERTY IMPORTED_LOCATION_DEBUG "${JPEG_LIBRARY_RELEASE}")
endif()
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
