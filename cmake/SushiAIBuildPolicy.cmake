# SushiAIBuildPolicy.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiAIBuildPolicy.cmake — the single entry point for this tree's compile policy.
#
# Single responsibility: compose the policy targets and expose one function
# that applies them to a target. Every target in the tree — library, examples,
# tests — goes through sushiai_apply_build_policy() and therefore compiles
# under identical settings.
#
# Named with the full SushiAI prefix (not just "BuildPolicy.cmake") because
# CMAKE_MODULE_PATH is a flat, shared namespace once D3's sibling
# add_subdirectory brings sushiruntime's and sushiblas's own cmake/ dirs onto
# the same path list: a same-named file here would shadow (or be shadowed by)
# theirs, and whichever `include(BuildPolicy)` call runs would silently load
# the wrong file's functions/targets. This exact bug was found and fixed
# during SushiBLAS's own SB-6 packaging work — the fix there was renaming
# SushiBLAS's four policy files the same way. The other three policy files
# below (and their include() calls) are prefixed for the same reason.
#
# All five targets composed here (sushiai_warnings, sushiai_perf, sushiai_debug_info,
# sushiai_sanitizers, and sushiai_device_link from SushiAISyclTarget.cmake) are PRIVATE,
# in-tree-only policy. SushiAI does not own a floating-point contract to export either: it
# receives SushiRuntime's
# `sushiruntime_deterministic_fp` transitively, through the SushiRuntime target each
# module links (SushiRuntime::core, SushiRuntime::api; see cmake/SushiAIModule.cmake).
include(SushiAICompilerWarnings)  # -> sushiai_warnings
include(SushiAIOptimization)      # -> sushiai_perf
include(SushiAIDebugInformation)  # -> sushiai_debug_info
include(SushiAISanitizers)        # -> sushiai_sanitizers

# Applies the project's in-tree compile policy to one target. Pulls flag
# VALUES off the policy targets via target_compile_options/
# target_compile_definitions/target_link_options + $<TARGET_PROPERTY:...>
# rather than linking the targets themselves - target_link_libraries would
# record the policy target's NAME in the exported interface (for a STATIC
# library, PRIVATE link deps are recorded as $<LINK_ONLY:...> even
# BUILD_INTERFACE-wrapped), which made install(EXPORT)/export(EXPORT) fail
# during SB-6 ("requires target sushiblas_warnings that is not in any export
# set"). target_compile_options/definitions/link_options never record a
# target name anywhere - only the resulting flag list - so there is nothing
# for export() to trip on.
function(sushiai_apply_build_policy target)
    target_compile_options(${target} PRIVATE
        "$<TARGET_PROPERTY:sushiai_warnings,INTERFACE_COMPILE_OPTIONS>"
        "$<TARGET_PROPERTY:sushiai_perf,INTERFACE_COMPILE_OPTIONS>"
        "$<TARGET_PROPERTY:sushiai_debug_info,INTERFACE_COMPILE_OPTIONS>"
        "$<TARGET_PROPERTY:sushiai_sanitizers,INTERFACE_COMPILE_OPTIONS>"
    )
    target_compile_definitions(${target} PRIVATE
        "$<TARGET_PROPERTY:sushiai_warnings,INTERFACE_COMPILE_DEFINITIONS>"
    )
    target_link_options(${target} PRIVATE
        "$<TARGET_PROPERTY:sushiai_sanitizers,INTERFACE_LINK_OPTIONS>"
        "$<TARGET_PROPERTY:sushiai_device_link,INTERFACE_LINK_OPTIONS>"
    )
endfunction()
