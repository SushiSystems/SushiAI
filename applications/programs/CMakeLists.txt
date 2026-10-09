# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# sushiai_train and sushiai_inference. The domains a program registers are
# chosen here, at link time; no source holds an #if.

if(SA_BUILD_VISION)
    set(SUSHIAI_PROGRAM_DOMAINS source/domains_vision.cpp)
else()
    set(SUSHIAI_PROGRAM_DOMAINS source/domains_none.cpp)
endif()

set(SUSHIAI_PROGRAM_TARGETS sushiai_train sushiai_inference)

add_executable(sushiai_train
    source/train.cpp source/registries.cpp ${SUSHIAI_PROGRAM_DOMAINS})
add_executable(sushiai_inference
    source/inference.cpp source/registries.cpp ${SUSHIAI_PROGRAM_DOMAINS})

foreach(target IN LISTS SUSHIAI_PROGRAM_TARGETS)
    target_link_libraries(${target} PRIVATE sushiai)
    if(SA_BUILD_VISION)
        target_link_libraries(${target} PRIVATE sushiai_vision)
    endif()
    sushiai_apply_build_policy(${target})
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying runtime DLL beside ${target}..."
        )
    endif()
endforeach()
