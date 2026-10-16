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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
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
#include <SushiAI/core/status.hpp>
#include <SushiAI/darknet_reference/darknet_reference.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using SushiAI::StatusCode;
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

    /** @brief Points the reference variable at a fresh folder and restores it afterwards. */
    class Medium_DarknetReference : public ::testing::Test
    {
        protected:
            /** @brief Creates the folder and its run folder, then points the variable at it. */
            void SetUp() override
            {
                saved_ = read_variable(K_REFERENCE_DATA_VARIABLE);
                root_ = std::filesystem::temp_directory_path() / "sushiai_darknet_reference_test";
                std::error_code ignored;
                std::filesystem::remove_all(root_, ignored);
                std::filesystem::create_directories(root_ / "references" / K_MODEL);
                set_variable(K_REFERENCE_DATA_VARIABLE, root_.string());
            }

            /** @brief Restores the variable and removes the folder. */
            void TearDown() override
            {
                set_variable(K_REFERENCE_DATA_VARIABLE, saved_.value_or(std::string()));
                std::error_code ignored;
                std::filesystem::remove_all(root_, ignored);
            }

            /** @brief Writes @p text to @p relative under the folder. */
            void write_text(const std::filesystem::path& relative, const std::string& text) const
            {
                std::ofstream out(root_ / relative, std::ios::binary);
                out << text;
            }

            /** @brief Writes @p values as little-endian float32 to @p relative under the folder. */
            void write_floats(const std::filesystem::path& relative,
                              const std::vector<float>& values) const
            {
                std::ofstream out(root_ / relative, std::ios::binary);
                for (const float value : values)
                {
                    std::uint32_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    for (int shift = 0; shift < 32; shift += 8)
                        out.put(static_cast<char>((bits >> shift) & 0xFFu));
                }
            }

            /** @brief Writes the `.json` and `.bin` of the reference @p name of K_MODEL. */
            void write_reference(const std::string& name, const std::string& description,
                                 const std::vector<float>& values) const
            {
                const std::filesystem::path folder = std::filesystem::path("references") / K_MODEL;
                write_text(folder / (name + ".json"), description);
                write_floats(folder / (name + ".bin"), values);
            }

            /** @brief Holds the folder the variable names during a case. */
            std::filesystem::path root_;

            /** @brief Holds the value the variable had before the case, if it was set. */
            std::optional<std::string> saved_;
    };
} // namespace

/** @brief Tests that a file directly in the folder is found at its path. */
TEST_F(Medium_DarknetReference, AFileInTheDirectoryIsFound)
{
    write_text("model.cfg", "[net]\n");

    const DarknetFile file = find_darknet_file("model.cfg");

    EXPECT_TRUE(file.found);
    EXPECT_EQ(file.path, root_ / "model.cfg");
}

/** @brief Tests that a missing file is not found and its reason names the file. */
TEST_F(Medium_DarknetReference, AMissingFileNamesItsPath)
{
    const DarknetFile file = find_darknet_file("absent.cfg");

    EXPECT_FALSE(file.found);
    EXPECT_NE(file.reason.find("absent.cfg"), std::string::npos);
}

/** @brief Tests that an unset variable is the reason a file is not found. */
TEST_F(Medium_DarknetReference, AnUnsetVariableIsTheReason)
{
    set_variable(K_REFERENCE_DATA_VARIABLE, std::string());

    const DarknetFile file = find_darknet_file("model.cfg");

    EXPECT_FALSE(file.found);
    EXPECT_NE(file.reason.find(K_REFERENCE_DATA_VARIABLE), std::string::npos);
}

/** @brief Tests that a reference whose `.bin` is absent is not found and names the `.bin`. */
TEST_F(Medium_DarknetReference, AReferenceNeedsBothFiles)
{
    write_text(std::filesystem::path("references") / K_MODEL / "conv.json", "{}");

    const DarknetReference reference = find_darknet_reference(K_MODEL, "conv");

    EXPECT_FALSE(reference.found);
    EXPECT_NE(reference.reason.find("conv.bin"), std::string::npos);
    SA_EXPECT_FAILURE(load_darknet_tensor(reference), StatusCode::FileNotFound, "conv.bin");
}

/** @brief Tests that a CHW feature map is read and moved to HWC, extents included. */
TEST_F(Medium_DarknetReference, AFeatureMapIsMovedToChannelsLast)
{
    write_reference("map",
                    R"({"dtype": "float32", "byte_order": "little", "shape": [2, 1, 3],
                        "order": "CHW"})",
                    {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f});

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor tensor,
                            load_darknet_tensor(find_darknet_reference(K_MODEL, "map")));

    EXPECT_EQ(tensor.extents, (std::vector<std::int64_t>{1, 3, 2}));
    EXPECT_EQ(tensor.values, (std::vector<float>{0.0f, 10.0f, 1.0f, 11.0f, 2.0f, 12.0f}));
}

/** @brief Tests that a tensor without an order keeps its shape and its values as stored. */
TEST_F(Medium_DarknetReference, ATensorWithoutAnOrderIsKeptAsStored)
{
    write_reference("boxes", R"({"dtype": "float32", "byte_order": "little", "shape": [2, 2]})",
                    {0.5f, -0.5f, 1.5f, -1.5f});

    SA_ASSERT_OK_AND_ASSIGN(const DarknetTensor tensor,
                            load_darknet_tensor(find_darknet_reference(K_MODEL, "boxes")));

    EXPECT_EQ(tensor.extents, (std::vector<std::int64_t>{2, 2}));
    EXPECT_EQ(tensor.values, (std::vector<float>{0.5f, -0.5f, 1.5f, -1.5f}));
}

/** @brief Tests that a payload whose byte count differs from the shape is refused. */
TEST_F(Medium_DarknetReference, AByteCountThatDiffersFromTheShapeIsRefused)
{
    write_reference("short", R"({"dtype": "float32", "byte_order": "little", "shape": [3]})",
                    {1.0f, 2.0f});

    SA_EXPECT_FAILURE(load_darknet_tensor(find_darknet_reference(K_MODEL, "short")),
                      StatusCode::InvalidFile, "bytes");
}

/** @brief Tests that a dtype other than float32 is refused and named. */
TEST_F(Medium_DarknetReference, ADtypeOtherThanFloat32IsRefused)
{
    write_reference("half", R"({"dtype": "float16", "byte_order": "little", "shape": [1]})",
                    {1.0f});

    SA_EXPECT_FAILURE(load_darknet_tensor(find_darknet_reference(K_MODEL, "half")),
                      StatusCode::InvalidFile, "'dtype'", "float16");
}
