/**************************************************************************/
/* test_exported_model.cpp                                                */
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
 * @file test_exported_model.cpp
 * @brief Tests that an exported model holds its files in the order they are written.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include <SushiAI/io/exported_model.hpp>

using namespace SushiAI;

/** @brief Tests that the files keep the order they were added in, each with its own bytes. */
TEST(Small_ExportedModel, FilesKeepTheOrderTheyAreWrittenIn)
{
    IO::ExportedModel exported;
    exported.files.push_back(IO::ExportedFile{"model.onnx", {std::byte{1}}});
    exported.files.push_back(IO::ExportedFile{"model.pth", {std::byte{2}, std::byte{3}}});
    exported.notes.push_back("the training recipe is not written");

    ASSERT_EQ(exported.files.size(), 2u);
    EXPECT_EQ(exported.files[0].name, "model.onnx");
    EXPECT_EQ(exported.files[0].bytes.size(), 1u);
    EXPECT_EQ(exported.files[1].name, "model.pth");
    EXPECT_EQ(exported.files[1].bytes.size(), 2u);
    EXPECT_EQ(exported.notes.size(), 1u);
}
