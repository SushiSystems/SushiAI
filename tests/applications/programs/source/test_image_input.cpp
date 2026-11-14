/**************************************************************************/
/* test_image_input.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_image_input.cpp
 * @brief Tests the data block that reads one image file for `sushiai inference`.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/shape/shape.hpp>
#include "image_input.hpp"

namespace
{
    using SushiAI::StatusCode;
    namespace Apps = SushiAI::Apps;
    namespace Config = SushiAI::Config;

    /** @brief Returns a model of an 8 by 6 input with three channels and a task of 3 classes. */
    Config::ModelSpec detector_of_three_classes()
    {
        Config::ModelSpec model;
        model.input_shape = SushiAI::Shape{8, 6, 3};
        model.task = Config::Value::object({"classes"}, {Config::Value::integer(3)});
        return model;
    }
} // namespace

/** @brief Tests that the extension alone decides whether a path is an image. */
TEST(Small_ImageInput, AnImageExtensionNamesAnImageAndAnyOtherDoesNot)
{
    EXPECT_TRUE(Apps::is_image_path(std::filesystem::path("data/cat.PNG")));
    EXPECT_FALSE(Apps::is_image_path(std::filesystem::path("data/eval.json")));
}

/** @brief Tests that the block names the reader, the folder, the file, a batch of one and the model. */
TEST(Small_ImageInput, TheBlockReadsTheFileForTheModelsInputAndClasses)
{
    SA_ASSERT_OK_AND_ASSIGN(const Config::DataDocument document,
                            Apps::image_data_document("pictures/cat.jpg", detector_of_three_classes()));

    const Config::Value& block = document.data.attributes;
    EXPECT_EQ(document.data.type, std::string(Config::K_READER_DATA_TYPE));
    EXPECT_EQ(block.at("reader", "block").value().as_string("reader").value(), "image_file");
    EXPECT_EQ(block.at("root", "block").value().as_string("root").value(), "pictures");
    EXPECT_EQ(block.at("file", "block").value().as_string("file").value(), "cat.jpg");
    EXPECT_EQ(block.at("batch_size", "block").value().as_int("batch").value(), 1);
    EXPECT_EQ(block.at("classes", "block").value().as_int("classes").value(), 3);
    const auto& input = block.at("input", "block").value().as_array("input").value();
    ASSERT_EQ(input.size(), 2u);
    EXPECT_EQ(input[0].as_int("height").value(), 8);
    EXPECT_EQ(input[1].as_int("width").value(), 6);
}

/** @brief Tests that a model whose task states no classes takes them from its last detect layer. */
TEST(Small_ImageInput, AModelWhoseTaskStatesNoClassesReadsThemFromItsLastLayerThatHasThem)
{
    Config::ModelSpec model = detector_of_three_classes();
    model.task = Config::Value::object({"type"}, {Config::Value::string("anchor_free_detection")});
    Config::LayerSpec head;
    head.type = "detect";
    head.attributes = Config::Value::object({"classes"}, {Config::Value::integer(7)});
    model.layers.push_back(head);

    SA_ASSERT_OK_AND_ASSIGN(const Config::DataDocument document,
                            Apps::image_data_document("pictures/cat.jpg", model));

    EXPECT_EQ(document.data.attributes.at("classes", "block").value().as_int("classes").value(), 7);
}

/** @brief Tests that a model whose task states no classes is refused. */
TEST(Small_ImageInput, AModelWithoutAClassCountIsRefused)
{
    Config::ModelSpec model = detector_of_three_classes();
    model.task = Config::Value();

    SA_EXPECT_FAILURE(Apps::image_data_document("pictures/cat.jpg", model),
                      StatusCode::InvalidDescription, "states its 'classes'");
}
