/**************************************************************************/
/* test_torch_archive.cpp                                                 */
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
 * @file test_torch_archive.cpp
 * @brief Tests the PyTorch file reader on archives whose pickle and storages are written by hand.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/pickle_reader.hpp>
#include <SushiAI/io/torch_archive.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;
using namespace std::string_view_literals;

namespace
{
    /**
     * @brief Holds {'w': a half tensor of size (2, 2) and stride (1, 2) over storage "0"}, the
     *        stream torch.save writes for it: _rebuild_tensor_v2 of a persistent id, the
     *        offset, the size, the stride, requires_grad and an empty OrderedDict.
     */
    constexpr std::string_view K_TENSOR_PICKLE =
        "\x80\x02}X\x01\x00\x00\x00" "w"
        "ctorch._utils\n_rebuild_tensor_v2\n("
        "(X\x07\x00\x00\x00" "storage" "ctorch\nHalfStorage\n"
        "X\x01\x00\x00\x00" "0" "X\x03\x00\x00\x00" "cpu" "K\x04" "tQ"
        "K\x00" "K\x02K\x02\x86" "K\x01K\x02\x86" "\x89"
        "ccollections\nOrderedDict\n)R" "tR" "s."sv;

    /** @brief Holds the storage: binary16 1.0, -2.0, 0.5 and 2^-24, little-endian. */
    constexpr std::string_view K_HALF_STORAGE = "\x00\x3c\x00\xc0\x00\x38\x01\x00"sv;

    /** @brief Returns an archive of the pickle and the given storage and byte order entries. */
    std::vector<std::byte> archive_of(std::string_view storage, std::string_view order)
    {
        std::vector<SushiAI::Test::ZipFixtureEntry> entries = {
            {"a/data.pkl", std::string(K_TENSOR_PICKLE)},
            {"a/byteorder", std::string(order)},
        };
        if (!storage.empty())
            entries.push_back({"a/data/0", std::string(storage)});
        return SushiAI::Test::zip_of(entries);
    }
} // namespace

/** @brief Tests that a strided half tensor reads row-major as the float32 values it holds. */
TEST(Small_TorchArchive, ReadsAStridedHalfTensorRowMajorAsFloat32)
{
    const std::vector<std::byte> bytes = archive_of(K_HALF_STORAGE, "little");
    SA_ASSERT_OK_AND_ASSIGN(const IO::TorchArchive archive,
                            IO::TorchArchive::read(SushiAI::Test::view_of(bytes), {}, "fixture.pt"));
    const IO::PickleValue* const value = IO::find_entry(archive.root(), "w");
    ASSERT_NE(value, nullptr);
    ASSERT_TRUE(IO::TorchArchive::holds_tensor(*value));

    SA_ASSERT_OK_AND_ASSIGN(const IO::TorchTensor tensor, archive.tensor(*value, "w"));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> values, archive.read_float32(tensor, "w"));

    EXPECT_EQ(tensor.dtype, DType::Half);
    // Element [i][j] is storage[i + 2 j]: 1.0, 0.5 on the first row, -2.0, 2^-24 on the second.
    const std::vector<float> expected = {1.0f, 0.5f, -2.0f, std::ldexp(1.0f, -24)};
    EXPECT_EQ(values, expected);
}

/** @brief Tests that a file stored in big-endian byte order is refused, naming the order. */
TEST(Small_TorchArchive, RefusesAFileInBigEndianOrder)
{
    const std::vector<std::byte> bytes = archive_of(K_HALF_STORAGE, "big");

    SA_EXPECT_FAILURE(IO::TorchArchive::read(SushiAI::Test::view_of(bytes), {}, "fixture.pt"),
                      StatusCode::Unsupported, "'big'");
}

/** @brief Tests that an archive holding no data.pkl is refused as no PyTorch file. */
TEST(Small_TorchArchive, RefusesAnArchiveWithoutAPickle)
{
    const std::vector<std::byte> bytes = SushiAI::Test::zip_of({{"a/version", "3"}});

    SA_EXPECT_FAILURE(IO::TorchArchive::read(SushiAI::Test::view_of(bytes), {}, "fixture.pt"),
                      StatusCode::InvalidFile, "data.pkl");
}

/** @brief Tests that a tensor whose storage entry is missing is refused, naming the storage. */
TEST(Small_TorchArchive, RefusesATensorWhoseStorageIsMissing)
{
    const std::vector<std::byte> bytes = archive_of("", "little");
    SA_ASSERT_OK_AND_ASSIGN(const IO::TorchArchive archive,
                            IO::TorchArchive::read(SushiAI::Test::view_of(bytes), {}, "fixture.pt"));
    const IO::PickleValue* const value = IO::find_entry(archive.root(), "w");
    ASSERT_NE(value, nullptr);

    SA_EXPECT_FAILURE(archive.tensor(*value, "w"), StatusCode::InvalidFile, "'0'");
}
