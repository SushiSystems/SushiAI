# SushiAIDistributedOptions.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# sushiai_follow_distributed_option() keeps a sibling's distributed option in step with
# SA_ENABLE_DISTRIBUTED, and never overrides a value the user chose.

# sushiai_follow_distributed_option(<option> <help>)
#
# SushiAI owns <option> when the cache has no entry for it, holds the value SushiAI wrote there
# last (kept in the internal entry SA_WROTE_<option>), or holds a default the sibling's option()
# wrote, as an older build tree does. An owned option follows SA_ENABLE_DISTRIBUTED. A value
# given with -D, which carries CMake's command-line help text, is the user's and is kept; if it is OFF while SA_ENABLE_DISTRIBUTED is ON, the configure stops, since the
# distribution module links the sibling's distributed layer and cannot build without it.
function(sushiai_follow_distributed_option option help)
    set(owned OFF)
    if(NOT DEFINED CACHE{${option}})
        set(owned ON)
    elseif(DEFINED CACHE{SA_WROTE_${option}})
        set(current OFF)
        if("$CACHE{${option}}")
            set(current ON)
        endif()
        if("${current}" STREQUAL "$CACHE{SA_WROTE_${option}}")
            set(owned ON)
        endif()
    else()
        # A value given with -D carries CMake's own help text; any other was written by the
        # sibling's option() call as its default, which an older build tree keeps, and is ours.
        get_property(set_by_hand CACHE ${option} PROPERTY HELPSTRING)
        if(NOT set_by_hand STREQUAL "No help, variable specified on the command line.")
            set(owned ON)
        endif()
    endif()

    if(owned)
        set(wanted OFF)
        if(SA_ENABLE_DISTRIBUTED)
            set(wanted ON)
        endif()
        set(${option} ${wanted} CACHE BOOL "${help}" FORCE)
        set(SA_WROTE_${option} ${wanted} CACHE INTERNAL
            "The value SushiAI last wrote to ${option}.")
        return()
    endif()

    unset(SA_WROTE_${option} CACHE)
    if(SA_ENABLE_DISTRIBUTED AND NOT "$CACHE{${option}}")
        message(FATAL_ERROR
            "SA_ENABLE_DISTRIBUTED is ON and ${option} was set OFF. The two must agree: SushiAI's "
            "distribution module links the distributed layer ${option} builds. Set ${option} ON, "
            "or SA_ENABLE_DISTRIBUTED OFF.")
    endif()
endfunction()
