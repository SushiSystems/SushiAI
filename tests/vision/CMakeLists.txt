# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The vision domain's test binary. Its cases keep the Unit_, Integration_ and
# Regression_ prefixes, so the labels of tests/CMakeLists.txt select them too.

add_executable(sushiai_vision_tests
    ../common/test_main.cpp

    test_detect_anchors.cpp
    test_detect_assigner.cpp
    test_detect_boxes.cpp
    test_detect_postprocess.cpp
    test_imaging_affine.cpp
    test_imaging_boxes.cpp
    test_imaging_flip.cpp
    test_imaging_hsv.cpp
    test_imaging_image.cpp
    test_imaging_labels.cpp
    test_imaging_letterbox.cpp
    test_imaging_mosaic.cpp
    test_imaging_pipeline.cpp
    test_imaging_random.cpp
    test_imaging_raw_cache.cpp
    test_imaging_resize.cpp
    test_detect_average_precision.cpp
    test_imaging_gray.cpp
    test_imaging_label_set.cpp
    test_imaging_partners.cpp
    test_imaging_polygon.cpp
    test_imaging_rectangle.cpp
    test_block_bottleneck.cpp
    test_block_c3k.cpp
    test_block_c3k2.cpp
    test_block_conv_norm.cpp
    test_block_sppf.cpp
    test_registration.cpp
    test_datasets_mnist.cpp
)

target_link_libraries(sushiai_vision_tests PRIVATE
    sushiai_vision
    GTest::gtest
)

target_compile_definitions(sushiai_vision_tests PRIVATE
    SA_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../fixtures"
)

sushiai_apply_build_policy(sushiai_vision_tests)

gtest_discover_tests(sushiai_vision_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Unit_*"
    PROPERTIES LABELS "unit"
)

gtest_discover_tests(sushiai_vision_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Integration_*"
    PROPERTIES LABELS "integration"
)

gtest_discover_tests(sushiai_vision_tests
    DISCOVERY_MODE PRE_TEST
    TEST_FILTER "Regression_*"
    PROPERTIES LABELS "regression"
)

if(WIN32)
    add_custom_command(TARGET sushiai_vision_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SushiRuntime::SushiRuntime>"
            $<TARGET_FILE_DIR:sushiai_vision_tests>
        COMMENT "Copying runtime DLL to the vision test directory..."
    )
endif()
