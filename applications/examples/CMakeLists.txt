# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The two example programs. Both read MNIST, so they link the vision domain.

set(SUSHIAI_EXAMPLE_TARGETS sushiai_example sushiai_demo_mlp)

add_executable(sushiai_example source/example.cpp)
add_executable(sushiai_demo_mlp source/demo_mlp.cpp)

foreach(target IN LISTS SUSHIAI_EXAMPLE_TARGETS)
    target_link_libraries(${target} PRIVATE sushiai sushiai_vision)
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
