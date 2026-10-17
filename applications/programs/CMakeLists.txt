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
    set(SUSHIAI_PROGRAM_DOMAINS source/domains_vision_none.cpp)
endif()
if(SA_BUILD_LANGUAGE)
    list(APPEND SUSHIAI_PROGRAM_DOMAINS source/domains_language.cpp)
else()
    list(APPEND SUSHIAI_PROGRAM_DOMAINS source/domains_language_none.cpp)
endif()

set(SUSHIAI_PROGRAM_TARGETS sushiai_train sushiai_inference sushiai_import)

# What the programs share: the registries and the runs and reports they name.
set(SUSHIAI_PROGRAM_SHARED
    source/registries.cpp source/trainer_run.cpp source/training_run.cpp
    source/inference_run.cpp source/run_start.cpp
    ${SUSHIAI_PROGRAM_DOMAINS})

add_executable(sushiai_train
    source/train.cpp ${SUSHIAI_PROGRAM_SHARED})
add_executable(sushiai_inference
    source/inference.cpp ${SUSHIAI_PROGRAM_SHARED})
add_executable(sushiai_import
    source/import.cpp ${SUSHIAI_PROGRAM_SHARED})

# The tokenizer uses no registry, so it takes none of the shared sources.
if(SA_BUILD_LANGUAGE)
    add_executable(sushiai_tokenize source/tokenize.cpp)
    list(APPEND SUSHIAI_PROGRAM_TARGETS sushiai_tokenize)
endif()

foreach(target IN LISTS SUSHIAI_PROGRAM_TARGETS)
    target_link_libraries(${target} PRIVATE sushiai)
    if(SA_BUILD_VISION)
        target_link_libraries(${target} PRIVATE sushiai_vision)
    endif()
    if(SA_BUILD_LANGUAGE)
        target_link_libraries(${target} PRIVATE sushiai_language)
    endif()
    sushiai_apply_build_policy(${target})
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::shared_library>"
                "$<TARGET_FILE:SushiLog::SushiLog>"
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying the runtime and log DLLs beside ${target}..."
        )
    endif()
endforeach()
