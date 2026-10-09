# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The modules, tier by tier, and the two libraries they are archived into.
# The functions are in cmake/SushiAIModule.cmake and the tier table in
# cmake/SushiAILayers.cmake.

include(SushiAIModule)

add_subdirectory(core)
add_subdirectory(layers)

sushiai_add_library(NAME sushiai EXPORT_NAME sushiai
    UMBRELLA "${CMAKE_CURRENT_SOURCE_DIR}/layers/umbrella/include/SushiAI/SushiAI.h")

# A domain is its own library in the same export set; see
# docs/design/DOMAIN_MODULES.md.
if(SA_BUILD_VISION)
    add_subdirectory(vision)

    sushiai_add_library(NAME sushiai_vision EXPORT_NAME vision
        UMBRELLA "${CMAKE_CURRENT_SOURCE_DIR}/vision/umbrella/include/SushiAI/vision/vision.h"
        DEPENDS sushiai)
endif()

install(EXPORT SushiAITargets
    FILE SushiAITargets.cmake
    NAMESPACE SushiAI::
    DESTINATION "${SA_INSTALL_CMAKEDIR}"
)

# The same targets, usable from a build tree without installing.
export(EXPORT SushiAITargets
    FILE "${PROJECT_BINARY_DIR}/SushiAITargets.cmake"
    NAMESPACE SushiAI::
)
