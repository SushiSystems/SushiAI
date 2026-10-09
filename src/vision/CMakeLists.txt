# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The vision domain: detection, imaging, the model blocks and the datasets of
# computer vision, built on the layers of SushiAI::sushiai.

set(SUSHIAI_VISION_SOURCES
    blocks/bottleneck.cpp
    blocks/c3k.cpp
    blocks/c3k2.cpp
    blocks/conv_norm.cpp
    blocks/sppf.cpp
    registration.cpp
    datasets/mnist.cpp
    detect/anchors.cpp
    detect/assigner.cpp
    detect/assigner_config.cpp
    detect/box_coding.cpp
    detect/box_format.cpp
    detect/detections_nms.cpp
    detect/detections_topk.cpp
    detect/ground_truth.cpp
    detect/iou.cpp
    detect/level_rows.cpp
    detect/target_leaves.cpp
    detect/top_k.cpp
    detect/anchor_leaves.cpp
    detect/average_precision.cpp
    imaging/affine.cpp
    imaging/boxes.cpp
    imaging/flip.cpp
    imaging/hsv.cpp
    imaging/image.cpp
    imaging/labels.cpp
    imaging/letterbox.cpp
    imaging/mosaic.cpp
    imaging/pipeline.cpp
    imaging/random.cpp
    imaging/raw_cache.cpp
    imaging/resize.cpp
    imaging/bilinear.cpp
    imaging/dataset_partners.cpp
    imaging/gray.cpp
    imaging/label_set.cpp
    imaging/polygon.cpp
    imaging/recent_partners.cpp
    imaging/rectangle.cpp
)

add_library(sushiai_vision STATIC ${SUSHIAI_VISION_SOURCES})
add_library(SushiAI::vision ALIAS sushiai_vision)

target_link_libraries(sushiai_vision PUBLIC sushiai)

sushiai_apply_build_policy(sushiai_vision)

set_target_properties(sushiai_vision PROPERTIES
    EXPORT_NAME vision
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

sushiai_apply_sycl(sushiai_vision SOURCES ${SUSHIAI_VISION_SOURCES})

install(TARGETS sushiai_vision
    EXPORT SushiAITargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# --- The domain's umbrella must name every header of the domain ---
file(GLOB_RECURSE SUSHIAI_VISION_HEADERS CONFIGURE_DEPENDS
    RELATIVE "${PROJECT_SOURCE_DIR}/include/SushiAI/vision"
    "${PROJECT_SOURCE_DIR}/include/SushiAI/vision/*.hpp"
)

file(READ "${PROJECT_SOURCE_DIR}/include/SushiAI/vision/vision.h" SUSHIAI_VISION_UMBRELLA_TEXT)

set(SUSHIAI_VISION_UNREFERENCED "")
foreach(header IN LISTS SUSHIAI_VISION_HEADERS)
    string(FIND "${SUSHIAI_VISION_UMBRELLA_TEXT}" "\"${header}\"" found)
    if(found EQUAL -1)
        list(APPEND SUSHIAI_VISION_UNREFERENCED "${header}")
    endif()
endforeach()

if(SUSHIAI_VISION_UNREFERENCED)
    string(REPLACE ";" "\n    " SUSHIAI_VISION_UNREFERENCED_REPORT
        "${SUSHIAI_VISION_UNREFERENCED}")
    message(FATAL_ERROR
        "These vision headers are installed but not included by "
        "include/SushiAI/vision/vision.h:\n    ${SUSHIAI_VISION_UNREFERENCED_REPORT}\n"
        "Add each to the umbrella, or move it out of the installed include tree.")
endif()
