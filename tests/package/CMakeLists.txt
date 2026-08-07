cmake_minimum_required(VERSION 3.20)

# An out-of-tree consumer of an INSTALLED SushiAI. Mirrors sushiblas's
# tests/package/ exactly (SB-6) - find_package, nothing else, no access to
# this source tree.
#
# Build it against an install prefix:
#
#   sa build
#   cmake --install build --prefix /tmp/sa-prefix
#   cmake -S tests/package -B /tmp/sa-consumer \
#         -DCMAKE_PREFIX_PATH="/tmp/sa-prefix;/path/to/sushiblas-install;/path/to/sushiruntime-install" \
#         -DCMAKE_CXX_COMPILER=<the same SYCL compiler>
#   cmake --build /tmp/sa-consumer
#   /tmp/sa-consumer/sushiai_package_consumer

project(SushiAIPackageConsumer LANGUAGES CXX)

find_package(SushiAI REQUIRED)

# --- Flag hygiene ------------------------------------------------------------
get_target_property(_sa_interface_options
    SushiAI::sushiai INTERFACE_COMPILE_OPTIONS)
if(NOT _sa_interface_options)
    set(_sa_interface_options "")
endif()

get_target_property(_sa_interface_links
    SushiAI::sushiai INTERFACE_LINK_LIBRARIES)
if(NOT _sa_interface_links)
    set(_sa_interface_links "")
endif()

set(_sa_forbidden_flags "-Werror" "-Wall" "-Wextra" "-pedantic"
                        "/W4" "/permissive-"
                        "-march=" "/arch:" "-fsanitize")

foreach(_sa_flag IN LISTS _sa_forbidden_flags)
    string(FIND "${_sa_interface_options}" "${_sa_flag}" _sa_found)
    if(NOT _sa_found EQUAL -1)
        message(FATAL_ERROR
            "SushiAI's exported interface leaks '${_sa_flag}' to consumers. "
            "Development flags belong on the PRIVATE policy targets; see "
            "cmake/SushiAIBuildPolicy.cmake. Full interface: ${_sa_interface_options}")
    endif()
endforeach()

foreach(_sa_leaked IN ITEMS sushiai_warnings sushiai_perf sushiai_sanitizers)
    string(FIND "${_sa_interface_links}" "${_sa_leaked}" _sa_found)
    if(NOT _sa_found EQUAL -1)
        message(FATAL_ERROR
            "SushiAI's exported interface names the in-tree policy target "
            "'${_sa_leaked}'. It should be wrapped in BUILD_INTERFACE. "
            "Full link interface: ${_sa_interface_links}")
    endif()
endforeach()

# The config layer's JSON parser is an implementation detail of
# src/config/*.cpp and must not reach here. It is easy to lose by accident:
# `sushiai` is STATIC, so a plain target_link_libraries(... PRIVATE ...) is
# recorded as $<LINK_ONLY:...> in the exported interface, and this consumer -
# which never runs find_package(nlohmann_json) - would then fail to configure
# on a name it should never have heard. src/CMakeLists.txt pulls the include
# directories on as values for exactly that reason; this is the assertion that
# it still does.
foreach(_sa_private_dep IN ITEMS nlohmann_json)
    string(FIND "${_sa_interface_links}" "${_sa_private_dep}" _sa_found)
    if(NOT _sa_found EQUAL -1)
        message(FATAL_ERROR
            "SushiAI's exported interface names '${_sa_private_dep}', which is "
            "private to src/config/. See src/CMakeLists.txt for why a STATIC "
            "library cannot simply link it PRIVATE. "
            "Full link interface: ${_sa_interface_links}")
    endif()
endforeach()

message(STATUS "SushiAI interface compile options: ${_sa_interface_options}")
message(STATUS "SushiAI interface link libraries:  ${_sa_interface_links}")

# --- The consumer ------------------------------------------------------------

add_executable(sushiai_package_consumer main.cpp)

target_link_libraries(sushiai_package_consumer PRIVATE
    SushiAI::sushiai
)

# C++17, and NOT C++20 - which is where this file deliberately stops mirroring
# sushiblas's tests/package/, so the difference is written down here rather than
# discovered.
#
# SushiBLAS keeps its own package consumer at C++20 as a live assertion that a
# C++20 consumer still compiles and links against a C++17 library. That holds
# for SushiBLAS because every one of its `SushiRuntime::span`-taking entry
# points is defined inline in a header: the std::span/shim branch is re-decided
# in the consumer's own translation unit and never has to agree with a symbol
# somebody else compiled.
#
# SushiAI is not in that position. `Autograd::differentiate`,
# `allocate_tensor`, `StepScalars::write` and the gradcheck entry points are
# out-of-line exports whose parameter type IS the span, so the branch reaches
# the mangled name. Measured, not assumed:
#
#   library (C++17)  ?differentiate@Autograd@SushiAI@@...V?$span@$$CBI@SushiRuntime@@@Z
#   consumer (C++20) ?differentiate@Autograd@SushiAI@@...V?$span@$$CBI$0?0@4@@Z
#
# Two different symbols - an unresolved external, not a warning. And it is not
# a corner: the header template `Train::build_classifier` calls
# `Autograd::differentiate`, so a C++20 consumer that trains anything at all
# hits it.
#
# Leaving this at 20 would therefore assert something untrue and pass only
# because main.cpp happens to call nothing that takes a span - a green light
# for a guarantee the library no longer offers. The honest assertion is the one
# below: a consumer at the standard the whole stack agrees on. See
# docs/INTEGRATION.md §3.
set_target_properties(sushiai_package_consumer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
