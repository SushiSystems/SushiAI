/**************************************************************************/
/* test_npz_weight_source.cpp                                             */
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
 * @file test_npz_weight_source.cpp
 * @brief Tests that a .npz file, stored or deflated, reads as the two-tensor fixture model and is
 *        sniffed apart from a PyTorch zip.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/npz_weight_source.hpp>
#include <SushiAI/io/torch_weight_source.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/shape/shape.hpp>
#include "archive_bytes.hpp"
#include "weight_fixtures.hpp"

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Names the file in every message. */
    constexpr std::string_view K_ORIGIN = "fixture.npz";

    /** @brief Holds the alignment numpy.lib.format pads the preamble and the header to. */
    constexpr std::size_t K_ALIGNMENT = 64;

    /**
     * @brief Returns a version 1.0 .npy file of float32 @p values and the shape text @p shape, as
     *        `numpy.save` lays it out: the magic, the header length, the header text padded with
     *        spaces and a line end to a multiple of 64 bytes, then the little-endian data.
     */
    std::string npy_of(const std::vector<float>& values, const std::string& shape)
    {
        const std::string text =
            "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }";
        const std::size_t unpadded = 10 + text.size() + 1;
        const std::size_t padded = (unpadded + K_ALIGNMENT - 1) / K_ALIGNMENT * K_ALIGNMENT;
        const std::string header = text + std::string(padded - unpadded, ' ') + "\n";
        std::string out = std::string("\x93NUMPY\x01\x00", 8);
        SaTest::put_le(out, header.size(), 2);
        return out + header + SaTest::float_bytes(values);
    }

    /** @brief Returns the CRC-32 of @p text, the bitwise form of zlib.crc32. */
    std::uint32_t crc32_of(const std::string& text)
    {
        std::uint32_t crc = 0xFFFFFFFFu;
        for (const char byte : text)
        {
            crc ^= static_cast<unsigned char>(byte);
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ ((crc & 1u) != 0u ? 0xEDB88320u : 0u);
        }
        return ~crc;
    }

    /** @brief Returns @p text as one stored DEFLATE block: the flags byte, LEN, NLEN, the bytes. */
    std::string deflate_stored(const std::string& text)
    {
        std::string out(1, '\x01');
        SaTest::put_le(out, text.size(), 2);
        SaTest::put_le(out, ~static_cast<std::uint64_t>(text.size()) & 0xFFFFu, 2);
        return out + text;
    }

    /**
     * @brief Returns the entry of @p npy named @p name, stored or as a deflated stream.
     *
     * The Python that builds the stored form is
     * `zipfile.ZipFile(f, "w").writestr("net.fc.weight.npy", data)` over the bytes of
     * `numpy.save`; the deflated form of the same bytes is one stored DEFLATE block, which
     * `zlib.compressobj(0, zlib.DEFLATED, -15)` writes and the inflater reads like any other.
     */
    SaTest::ZipFixtureEntry entry_of(const std::string& name, const std::string& npy,
                                     bool deflated)
    {
        if (!deflated)
            return {name, npy, 0, 0, crc32_of(npy)};
        return {name, deflate_stored(npy), 8, npy.size(), crc32_of(npy)};
    }

    /** @brief Returns the npz of the two fixture tensors, `net.fc.weight` and `net.fc.bias`. */
    std::vector<std::byte> npz_fixture(bool deflated)
    {
        return SaTest::zip_of(
            {entry_of("net.fc.weight.npy", npy_of(SaTest::fixture_weight(), "(2, 3)"), deflated),
             entry_of("net.fc.bias.npy", npy_of(SaTest::fixture_bias(), "(2,)"), deflated)});
    }
} // namespace

/** @brief Tests that both forms of the npz list the fixture tensors and read their values. */
TEST(Small_NpzWeightSource, StoredAndDeflatedFilesReadTheFixtureModel)
{
    for (const bool deflated : {false, true})
    {
        IO::WeightSourceOptions options;
        options.origin = K_ORIGIN;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> source,
                                IO::open_npz_source(npz_fixture(deflated), options));

        ASSERT_EQ(source->tensors().size(), 2u);
        EXPECT_EQ(source->tensors()[0].name, "net.fc.weight");
        EXPECT_EQ(source->tensors()[1].name, "net.fc.bias");
        EXPECT_EQ(source->tensors()[0].shape, (Shape{2, 3}));
        EXPECT_EQ(source->tensors()[1].shape, (Shape{2}));
        EXPECT_EQ(source->tensors()[0].stored_type, "<f4");
        EXPECT_EQ(source->origin(), K_ORIGIN);
        SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> weight, source->read_float32(0));
        SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> bias, source->read_float32(1));
        EXPECT_EQ(weight, SaTest::fixture_weight());
        EXPECT_EQ(bias, SaTest::fixture_bias());
    }
}

/** @brief Tests that sniffing picks npz for an npz and torch for a PyTorch zip. */
TEST(Small_NpzWeightSource, SniffingPicksNpzForAnNpzAndTorchForAPth)
{
    IO::WeightSourceRegistry registry;
    ASSERT_TRUE(registry.add(IO::K_NPZ_WEIGHT_SOURCE_ROW).is_ok());
    ASSERT_TRUE(registry.add(IO::K_TORCH_WEIGHT_SOURCE_ROW).is_ok());
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;
    options.state_key = "state_dict";

    const std::vector<std::byte> stored = npz_fixture(false);
    const std::vector<std::byte> deflated = npz_fixture(true);
    const std::vector<std::byte> torch = SaTest::torch_fixture();
    EXPECT_TRUE(IO::is_npz_file(SaTest::view_of(stored)));
    EXPECT_TRUE(IO::is_npz_file(SaTest::view_of(deflated)));
    EXPECT_FALSE(IO::is_npz_file(SaTest::view_of(torch)));
    EXPECT_TRUE(IO::is_torch_file(SaTest::view_of(torch)));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> from_npz,
                            IO::open_weight_source(registry, npz_fixture(true), options));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<IO::WeightSource> from_torch,
                            IO::open_weight_source(registry, SaTest::torch_fixture(), options));
    EXPECT_EQ(from_npz->tensors()[0].stored_type, "<f4");
    EXPECT_EQ(from_torch->tensors()[0].stored_type, "float32");
    EXPECT_EQ(from_npz->tensors()[0].shape, from_torch->tensors()[0].shape);
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> npz_weight, from_npz->read_float32(0));
    SA_ASSERT_OK_AND_ASSIGN(const std::vector<float> torch_weight, from_torch->read_float32(0));
    EXPECT_EQ(npz_weight, torch_weight);
}

/** @brief Tests that one entry that is no .npy, or no entries at all, make the file no npz. */
TEST(Small_NpzWeightSource, AZipWithAnyOtherEntryOrNoEntryIsNoNpz)
{
    const std::string npy = npy_of(SaTest::fixture_bias(), "(2,)");
    const std::vector<std::byte> mixed = SaTest::zip_of(
        {entry_of("net.fc.bias.npy", npy, false), SaTest::ZipFixtureEntry{"a/data.pkl", "x"}});
    const std::vector<std::byte> empty = SaTest::zip_of({});
    const std::vector<std::byte> text = SaTest::bytes_of("hello, weights");

    EXPECT_FALSE(IO::is_npz_file(SaTest::view_of(mixed)));
    EXPECT_FALSE(IO::is_npz_file(SaTest::view_of(empty)));
    EXPECT_FALSE(IO::is_npz_file(SaTest::view_of(text)));
}

/** @brief Tests that an entry whose bytes are no .npy file is refused, naming the entry. */
TEST(Small_NpzWeightSource, RefusesAnEntryThatIsNoNpyFile)
{
    IO::WeightSourceOptions options;
    options.origin = K_ORIGIN;
    const std::vector<std::byte> bytes =
        SaTest::zip_of({SaTest::ZipFixtureEntry{"net.fc.bias.npy", "not numpy"}});

    SA_EXPECT_FAILURE(IO::open_npz_source(bytes, options), StatusCode::InvalidFile,
                      "net.fc.bias.npy");
}
