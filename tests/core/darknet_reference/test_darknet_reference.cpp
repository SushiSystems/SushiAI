/**************************************************************************/
/* test_darknet_reference.cpp                                             */
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
 * @file test_darknet_reference.cpp
 * @brief Tests the readers of the files a Darknet run dumped, over files a case writes.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <support/scratch_names.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/darknet_reference/darknet_reference.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Test::decode_darknet_tensor;
    using SushiAI::Test::DarknetFile;
    using SushiAI::Test::DarknetReference;
    using SushiAI::Test::DarknetTensor;
    using SushiAI::Test::find_darknet_file;
    using SushiAI::Test::find_darknet_reference;
    using SushiAI::Test::K_REFERENCE_DATA_VARIABLE;
    using SushiAI::Test::load_darknet_tensor;

    /** @brief Holds the run folder every reference of a case lives in. */
    constexpr const char* K_MODEL = "tiny_run";

    /** @brief Sets the environment variable @p name to @p value; an empty value removes it. */
    void set_variable(const char* name, const std::string& value)
    {
#if defined(_WIN32)
        _putenv_s(name, value.c_str());
#else
        if (value.empty())
            unsetenv(name);
        else
            setenv(name, value.c_str(), 1);
#endif
    }

    /** @brief Returns the value of the environment variable @p name, if it is set. */
    std::optional<std::string> read_variable(const char* name)
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        const char* const value = std::getenv(name);
#pragma clang diagnostic pop
        if (value == nullptr)
            return std::nullopt;
        return std::string(value);
    }

    /** @brief Returns @p values as the little-endian float32 bytes of a `.bin` file. */
    std::vector<unsigned char> bytes_of(const std::vector<float>& values)
    {
        std::vector<unsigned char> bytes;
        for (const float value : values)
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int shift = 0; shift < 32; shift += 8)
                bytes.push_back(static_cast<unsigned char>((bits >> shift) & 0xFFu));
        }
        return bytes;
    }

    /** @brief Gives each case a fresh folder to hold the reference data. */
    class Medium_DarknetReference : public ::testing::Test
    {
        protected:
            /** @brief Creates the folder and its run folder. */
            void SetUp() override
            {
                root_ = std::filesystem::temp_directory_path() /
                    SushiAI::Test::unique_scratch_name("sushiai_darknet_reference_test");
                std::error_code ignored;
                std::filesystem::remove_all(root_, ignored);
                std::filesystem::create_directories(root_ / "references" / K_MODEL);
            }

            /** @brief Removes the folder. */
            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all(root_, ignored);
            }

            /** @brief Writes @p text to @p relative under the folder. */
            void write_text(const std::filesystem::path& relative, const std::string& text) const
            {
                std::ofstream out(root_ / relative, std::ios::binary);
                out << text;
            }

            /** @brief Holds the folder a case reads from. */
            std::filesystem::path root_;
    };

    /** @brief Unsets the reference variable for a case and restores it afterwards. */
    class Medium_DarknetReferenceEnvironment : public ::testing::Test
    {
        protected:
            /** @brief Saves the variable and unsets it. */
            void SetUp() override
            {
                saved_ = read_variable(K_REFERENCE_DATA_VARIABLE);
                set_variable(K_REFERENCE_DATA_VARIABLE, std::string());
            }

            /** @brief Restores the variable. */
            void TearDown() override
            {
                set_variable(K_REFERENCE_DATA_VARIABLE, saved_.value_or(std::string()));
            }

            /** @brief Holds the value the variable had before the case, if it was set. */
            std::optional<std::string> saved_;
    };
} // namespace

/** @brief Tests that a file directly in the folder is found at its path. */
TEST_F(Medium_DarknetReference, AFileInTheDirectoryIsFound)
{
    write_text("model.cfg", "[net]\n");

    const DarknetFile file = find_darknet_file(root_, "model.cfg");

    EXPECT_TRUE(file.found);
    EXPECT_EQ(file.path, root_ / "model.cfg");
}

/** @brief Tests that a missing file is not found and its reason names the file. */
TEST_F(Medium_DarknetReference, AMissingFileNamesItsPath)
{
    const DarknetFile file = find_darknet_file(root_, "absent.cfg");

    EXPECT_FALSE(file.found);
    EXPECT_NE(file.reason.find("absent.cfg"), std::string::npos);
}

/** @brief Tests that an unset variable is the reason a file is not found. */
TEST_F(Medium_DarknetReferenceEnvironment, AnUnsetVariableIsTheReason)
{
    const DarknetFile file = find_darknet_file("model.cfg");

    EXPECT_FALSE(file.found);
    EXPECT_NE(file.reason.find(K_REFERENCE_DATA_VARIABLE), std::string::npos);
}

/** @brief Tests that a reference whose `.bin` is absent is not found and names the `.bin`. */
TEST_F(Medium_DarknetReference, AReferenceNeedsBothFiles)
{
    write_text(std::filesystem::path("references") / K_MODEL / "conv.json", "{}");

    const DarknetReference reference = find_darknet_reference(root_, K_MODEL, "conv");

    EXPECT_FALSE(reference.found);
    EXPECT_NE(reference.reason.find("conv.bin"), std::string::npos);
    SA_EXPECT_FAILURE(load_darknet_tensor(reference), StatusCode::FileNotFound, "conv.bin");
}

/** @brief Tests that a CHW feature map is read and moved to HWC, extents included. */
TEST(Small_DarknetTensor, AFeatureMapIsMovedToChannelsLast)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetTensor tensor,
        decode_darknet_tensor(R"({"dtype": "float32", "byte_order": "little", "shape": [2, 1, 3],
                                  "order": "CHW"})",
                              bytes_of({0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f}), "map"));

    EXPECT_EQ(tensor.extents, (std::vector<std::int64_t>{1, 3, 2}));
    EXPECT_EQ(tensor.values, (std::vector<float>{0.0f, 10.0f, 1.0f, 11.0f, 2.0f, 12.0f}));
}

/** @brief Tests that a tensor without an order keeps its shape and its values as stored. */
TEST(Small_DarknetTensor, ATensorWithoutAnOrderIsKeptAsStored)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const DarknetTensor tensor,
        decode_darknet_tensor(R"({"dtype": "float32", "byte_order": "little", "shape": [2, 2]})",
                              bytes_of({0.5f, -0.5f, 1.5f, -1.5f}), "boxes"));

    EXPECT_EQ(tensor.extents, (std::vector<std::int64_t>{2, 2}));
    EXPECT_EQ(tensor.values, (std::vector<float>{0.5f, -0.5f, 1.5f, -1.5f}));
}

/** @brief Tests that a payload whose byte count differs from the shape is refused. */
TEST(Small_DarknetTensor, AByteCountThatDiffersFromTheShapeIsRefused)
{
    SA_EXPECT_FAILURE(
        decode_darknet_tensor(R"({"dtype": "float32", "byte_order": "little", "shape": [3]})",
                              bytes_of({1.0f, 2.0f}), "short"),
        StatusCode::InvalidFile, "bytes");
}

/** @brief Tests that a dtype other than float32 is refused and named. */
TEST(Small_DarknetTensor, ADtypeOtherThanFloat32IsRefused)
{
    SA_EXPECT_FAILURE(
        decode_darknet_tensor(R"({"dtype": "float16", "byte_order": "little", "shape": [1]})",
                              bytes_of({1.0f}), "half"),
        StatusCode::InvalidFile, "'dtype'", "float16");
}
