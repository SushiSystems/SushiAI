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
    # Only the runtime's own DLL is copied. hwloc is linked PRIVATE by
    # SushiRuntime (PkgConfig::HWLOC, BUILD_INTERFACE-only), so it never reaches
    # a consumer's *link line* - but PRIVATE says nothing about the loader, and
    # hwloc-15.dll is in sushiruntime.dll's import table regardless. These
    # binaries get it from PATH instead: `sa` puts the vcpkg installed bin there
    # for every build/test/run subprocess (see cli/sushiai/env.py), which is why
    # nothing needs copying here. An out-of-tree consumer has no such PATH, and
    # the package suite stages hwloc explicitly for exactly that reason (see
    # _deploy_consumer_dlls).
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
                $<TARGET_FILE_DIR:${target}>
            COMMENT "Copying runtime DLL to example directory..."
        )
    endif()
endforeach()
