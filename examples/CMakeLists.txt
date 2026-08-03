# Two executables, two different jobs. `sushiai_example` is the teaching file
# (`sa run sushiai_example`); `sushiai_demo_mlp` is the program behind the
# supported `sa demo mlp` command, whose flags and printed lines are a
# contract the test suite pins. See demo_mlp.cpp's header for the full
# argument for keeping them apart.
set(SUSHIAI_EXAMPLE_TARGETS sushiai_example sushiai_demo_mlp)

add_executable(sushiai_example example.cpp)
add_executable(sushiai_demo_mlp demo_mlp.cpp)

foreach(target IN LISTS SUSHIAI_EXAMPLE_TARGETS)
    target_link_libraries(${target} PRIVATE sushiai)
    sushiai_apply_build_policy(${target})

    # --- Post-Build: Copy the runtime DLL to the output directory ---
    # hwloc is a private implementation detail of SushiRuntime's own build (linked
    # PRIVATE via PkgConfig::HWLOC, BUILD_INTERFACE-only) and never reaches
    # consumers, so only the runtime's own DLL needs copying here - matching
    # sushiblas's own examples/tests precedent exactly.
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying runtime DLL to example directory..."
        )
    endif()
endforeach()
