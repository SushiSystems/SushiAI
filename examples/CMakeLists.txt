add_executable(sushiai_example example.cpp)
target_link_libraries(sushiai_example PRIVATE sushiai)
sushiai_apply_build_policy(sushiai_example)

# --- Post-Build: Copy the runtime DLL to the output directory ---
# hwloc is a private implementation detail of SushiRuntime's own build (linked
# PRIVATE via PkgConfig::HWLOC, BUILD_INTERFACE-only) and never reaches
# consumers, so only the runtime's own DLL needs copying here - matching
# sushiblas's own examples/tests precedent exactly.
if(WIN32)
    add_custom_command(TARGET sushiai_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiai_example>
        COMMENT "Copying runtime DLL to example directory..."
    )
endif()
