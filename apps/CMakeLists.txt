# The two config-driven drivers. Separate from examples/ because these are
# shipping programs the `sushiai train` / `sushiai inference` CLI subcommands
# invoke, not teaching files - and because neither is allowed to name a layer,
# loss, dataset or optimizer type, which is a property worth keeping in its own
# directory where it can be checked in one place.
set(SUSHIAI_APP_TARGETS sushiai_train sushiai_inference)

add_executable(sushiai_train train.cpp)
add_executable(sushiai_inference inference.cpp)

foreach(target IN LISTS SUSHIAI_APP_TARGETS)
    target_link_libraries(${target} PRIVATE sushiai)
    sushiai_apply_build_policy(${target})

    # --- Post-Build: Copy the runtime DLL to the output directory ---
    # hwloc is a private implementation detail of SushiRuntime's own build and
    # never reaches consumers, so only the runtime's own DLL needs copying -
    # matching examples/CMakeLists.txt exactly.
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying runtime DLL to app directory..."
        )
    endif()
endforeach()
