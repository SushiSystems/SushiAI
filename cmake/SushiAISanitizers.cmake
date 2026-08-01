# SushiAISanitizers.cmake — the project's sanitizer policy as a target.
#
# Single responsibility: define an INTERFACE target `sushiai_sanitizers`.
# Currently empty (no sanitizer is auto-enabled by build type here); it exists
# so SushiAIBuildPolicy.cmake and every target can link it unconditionally,
# and a sanitizer lane can be added later without touching every target's
# CMakeLists.txt — mirrors sushiblas/cmake/SushiBLASSanitizers.cmake's shape.

add_library(sushiai_sanitizers INTERFACE)
