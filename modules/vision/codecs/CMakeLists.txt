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
# vcpkg's app-local copy puts the Debug jpeg62.dll beside a Debug test program, and a program built against the
# Release library and runtime crashes on its heap. The test programs copy the Release DLL after it.
if(WIN32 AND JPEG_LIBRARY_RELEASE)
    get_filename_component(sushiai_jpeg_library_dir "${JPEG_LIBRARY_RELEASE}" DIRECTORY)
    file(GLOB sushiai_jpeg_dlls "${sushiai_jpeg_library_dir}/../bin/jpeg*.dll")
    set_property(GLOBAL APPEND PROPERTY SUSHIAI_RELEASE_RUNTIME_DLLS ${sushiai_jpeg_dlls})
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
