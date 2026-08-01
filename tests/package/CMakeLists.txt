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

message(STATUS "SushiAI interface compile options: ${_sa_interface_options}")
message(STATUS "SushiAI interface link libraries:  ${_sa_interface_links}")

# --- The consumer ------------------------------------------------------------

add_executable(sushiai_package_consumer main.cpp)

target_link_libraries(sushiai_package_consumer PRIVATE
    SushiAI::sushiai
)

set_target_properties(sushiai_package_consumer PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
