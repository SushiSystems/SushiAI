/**************************************************************************/
/* test_darknet_weights.cpp                                               */
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
 * @file test_darknet_weights.cpp
 * @brief Tests the .weights header forms, the version refusals, the vector order and the byte-count
 *        refusals of the Darknet weights reader.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <ios>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Vision::Importers::DarknetConvolution;
    using SushiAI::Vision::Importers::DarknetConvolutionShape;
    using SushiAI::Vision::Importers::DarknetWeights;
    using SushiAI::Vision::Importers::DarknetWeightsHeader;
    using SushiAI::Vision::Importers::read_darknet_header;
    using SushiAI::Vision::Importers::read_darknet_weights;

    /** @brief Appends @p value as @p width little-endian bytes, written by the test itself. */
    void put(std::string& bytes, std::uint64_t value, int width)
    {
        for (int i = 0; i < width; ++i)
            bytes.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
    }

    /** @brief Returns the IEEE 754 bit pattern of @p value. */
    std::uint32_t bits_of(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof bits);
        return bits;
    }

    /** @brief Appends @p value as its float32 bit pattern, little-endian. */
    void put_float(std::string& bytes, float value)
    {
        put(bytes, bits_of(value), 4);
    }

    /**
     * @brief Expects @p values to hold the floats @p first, @p first + 1, ... bit for bit.
     *
     * Integers up to 2^24 are exact in float32, so the comparison has a zero tolerance.
     */
    ::testing::AssertionResult holds_indices(const std::vector<float>& values, std::size_t first)
    {
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const float expected = static_cast<float>(first + i);
            if (bits_of(values[i]) != bits_of(expected))
                return ::testing::AssertionFailure()
                       << "element " << i << " is " << values[i] << ", expected " << expected;
        }
        return ::testing::AssertionSuccess();
    }

    /** @brief Returns a header of @p major, @p minor and revision 0 with a 64-bit seen count of 0. */
    std::string header_bytes(std::uint32_t major, std::uint32_t minor)
    {
        std::string bytes;
        put(bytes, major, 4);
        put(bytes, minor, 4);
        put(bytes, 0, 4);
        put(bytes, 0, 8);
        return bytes;
    }

    /**
     * @brief Returns a version 0.2.0 file: a 64-bit seen count of zero, then the floats 0 to
     *        @p float_count - 1, each equal to its own index.
     */
    std::string sample_bytes(std::size_t float_count)
    {
        std::string bytes = header_bytes(0, 2);
        for (std::size_t i = 0; i < float_count; ++i)
            put_float(bytes, static_cast<float>(i));
        return bytes;
    }

    /** @brief Reads @p bytes as a weights file in memory with the origin "memory.weights". */
    SushiAI::Result<DarknetWeights> read_bytes(
        const std::string& bytes, const std::vector<DarknetConvolutionShape>& shapes)
    {
        return read_darknet_weights(
            SushiRuntime::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()),
            "memory.weights", shapes);
    }

    /** @brief Returns the two layers of the sample: a plain one, then a normalised one. */
    std::vector<DarknetConvolutionShape> sample_shapes()
    {
        return {
            DarknetConvolutionShape{0, 2, 1, 1, false},
            DarknetConvolutionShape{2, 3, 2, 1, true},
        };
    }

} // namespace

/** @brief Tests that a version 0.2 header carries a 64-bit seen count. */
TEST(Small_DarknetWeights, ReadsAVersionZeroTwoHeaderWithA64BitSeen)
{
    std::string bytes;
    put(bytes, 0, 4);
    put(bytes, 2, 4);
    put(bytes, 5, 4);
    put(bytes, 0x0102030405060708ull, 8);
    std::istringstream stream(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeightsHeader header, read_darknet_header(stream));

    EXPECT_EQ(header.major, 0);
    EXPECT_EQ(header.minor, 2);
    EXPECT_EQ(header.revision, 5);
    EXPECT_EQ(header.seen, 0x0102030405060708ull);
    EXPECT_EQ(static_cast<std::streamoff>(stream.tellg()), 20);
}

/** @brief Tests that a version 0.1 header carries a 32-bit seen count. */
TEST(Small_DarknetWeights, ReadsAVersionZeroOneHeaderWithA32BitSeen)
{
    std::string bytes;
    put(bytes, 0, 4);
    put(bytes, 1, 4);
    put(bytes, 0, 4);
    put(bytes, 0x89ABCDEFu, 4);
    put(bytes, 0xDEADBEEFu, 4);
    std::istringstream stream(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeightsHeader header, read_darknet_header(stream));

    EXPECT_EQ(header.seen, 0x89ABCDEFu);
    EXPECT_EQ(static_cast<std::streamoff>(stream.tellg()), 16);
}

/** @brief Tests that a major of 1 gives a 64-bit seen count whatever the minor. */
TEST(Small_DarknetWeights, ReadsAMajorOneHeaderWithA64BitSeen)
{
    std::string bytes;
    put(bytes, 1, 4);
    put(bytes, 0, 4);
    put(bytes, 0, 4);
    put(bytes, 0x1122334455667788ull, 8);
    std::istringstream stream(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeightsHeader header, read_darknet_header(stream));

    EXPECT_EQ(header.major, 1);
    EXPECT_EQ(header.seen, 0x1122334455667788ull);
    EXPECT_EQ(static_cast<std::streamoff>(stream.tellg()), 20);
}

/** @brief Holds one header version and the status code reading it gives. */
struct VersionLimit
{
    std::uint32_t major;
    std::uint32_t minor;
    StatusCode expected;
};

/** @brief Runs the header reader over each version around the limit of one thousand. */
class Small_DarknetWeightsVersionLimit : public ::testing::TestWithParam<VersionLimit>
{
};

/** @brief Tests that a major or a minor above 1000 is Unsupported and 1000 is read. */
TEST_P(Small_DarknetWeightsVersionLimit, GivesTheExpectedStatusCode)
{
    std::istringstream stream(header_bytes(GetParam().major, GetParam().minor));

    EXPECT_EQ(read_darknet_header(stream).status().code(), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    Versions, Small_DarknetWeightsVersionLimit,
    ::testing::Values(VersionLimit{1000, 0, StatusCode::Ok},
                      VersionLimit{1001, 0, StatusCode::Unsupported},
                      VersionLimit{0, 1001, StatusCode::Unsupported}));

/** @brief Tests that a stream that ends inside the header is InvalidFile. */
TEST(Small_DarknetWeights, RefusesAHeaderThatEndsEarly)
{
    std::string bytes;
    put(bytes, 0, 4);
    put(bytes, 2, 4);
    put(bytes, 0, 2);
    std::istringstream stream(bytes);

    SA_EXPECT_FAILURE(read_darknet_header(stream), StatusCode::InvalidFile, "revision");
}

/** @brief Tests that a stream that ends inside the seen count is InvalidFile. */
TEST(Small_DarknetWeights, RefusesAHeaderThatEndsInsideTheSeenCount)
{
    std::string bytes;
    put(bytes, 0, 4);
    put(bytes, 2, 4);
    put(bytes, 0, 4);
    put(bytes, 0, 4);
    std::istringstream stream(bytes);

    SA_EXPECT_FAILURE(read_darknet_header(stream), StatusCode::InvalidFile, "seen count");
}

/** @brief Tests that index-valued floats land in each vector at their own indices. */
TEST(Small_DarknetWeights, ReadsEveryVectorInFileOrder)
{
    // The floats are 0 to 21 in file order, so each vector starts at its own index.
    const std::string bytes = sample_bytes(22);
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeights weights, read_bytes(bytes, shapes));

    ASSERT_EQ(weights.convolutions.size(), 2u);
    const DarknetConvolution& plain = weights.convolutions[0];
    const DarknetConvolution& normalised = weights.convolutions[1];
    EXPECT_TRUE(holds_indices(plain.biases, 0));
    EXPECT_TRUE(holds_indices(plain.weights, 2));
    EXPECT_TRUE(holds_indices(normalised.biases, 4));
    EXPECT_TRUE(holds_indices(normalised.scales, 7));
    EXPECT_TRUE(holds_indices(normalised.rolling_mean, 10));
    EXPECT_TRUE(holds_indices(normalised.rolling_variance, 13));
    EXPECT_TRUE(holds_indices(normalised.weights, 16));
    EXPECT_EQ(weights.header.major, 0);
    EXPECT_EQ(weights.header.minor, 2);
}

/** @brief Tests that only a normalised layer fills scales, means and variances. */
TEST(Small_DarknetWeights, KeepsTheBatchNormVectorsOnlyForNormalisedLayers)
{
    const std::string bytes = sample_bytes(22);
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeights weights, read_bytes(bytes, shapes));

    const DarknetConvolution& plain = weights.convolutions[0];
    const DarknetConvolution& normalised = weights.convolutions[1];
    EXPECT_TRUE(plain.scales.empty());
    EXPECT_TRUE(plain.rolling_mean.empty());
    EXPECT_TRUE(plain.rolling_variance.empty());
    EXPECT_EQ(plain.weights.size(), 2u);
    EXPECT_EQ(normalised.scales.size(), 3u);
    EXPECT_EQ(normalised.rolling_mean.size(), 3u);
    EXPECT_EQ(normalised.rolling_variance.size(), 3u);
    EXPECT_EQ(normalised.weights.size(), 6u);
}

/** @brief Tests that negative zero, a denormal and a NaN payload keep their bits. */
TEST(Small_DarknetWeights, DecodesEveryBitPatternAsItIs)
{
    // Biases are -0.0 and the smallest subnormal; weights are a NaN with payload 1 and 1.0.
    std::string bytes = header_bytes(0, 2);
    put(bytes, 0x80000000u, 4);
    put(bytes, 0x00000001u, 4);
    put(bytes, 0x7FC00001u, 4);
    put(bytes, 0x3F800000u, 4);
    const std::vector<DarknetConvolutionShape> shapes{DarknetConvolutionShape{0, 2, 1, 1, false}};

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeights weights, read_bytes(bytes, shapes));

    const DarknetConvolution& layer = weights.convolutions[0];
    ASSERT_EQ(layer.biases.size(), 2u);
    ASSERT_EQ(layer.weights.size(), 2u);
    EXPECT_EQ(bits_of(layer.biases[0]), 0x80000000u);
    EXPECT_EQ(bits_of(layer.biases[1]), 0x00000001u);
    EXPECT_EQ(bits_of(layer.weights[0]), 0x7FC00001u);
    EXPECT_EQ(bits_of(layer.weights[1]), 0x3F800000u);
}

/** @brief Holds a float count that disagrees with the sample shapes and what the refusal names. */
struct ByteCountRefusal
{
    std::size_t floats;
    const char* found;
    const char* expected;
    const char* detail;
};

/** @brief Runs the reader over a sample file that is one float short or one float over. */
class Small_DarknetWeightsByteCount : public ::testing::TestWithParam<ByteCountRefusal>
{
};

/** @brief Tests that a file of the wrong size is InvalidFile with the layer and both counts. */
TEST_P(Small_DarknetWeightsByteCount, IsRefusedWithTheLayerAndBothByteCounts)
{
    // The header is 20 bytes. Layer 0 needs 16 bytes, so layer 2 ends at 20 + 16 + 72 = 108.
    const std::string bytes = sample_bytes(GetParam().floats);
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_EXPECT_FAILURE(read_bytes(bytes, shapes), StatusCode::InvalidFile, GetParam().found,
                      GetParam().expected, GetParam().detail);
}

INSTANTIATE_TEST_SUITE_P(
    Floats, Small_DarknetWeightsByteCount,
    ::testing::Values(ByteCountRefusal{21, "104 bytes", "up to 108",
                                       "convolution layer 2 needs bytes up to 108"},
                      ByteCountRefusal{23, "112 bytes", "108 bytes",
                                       "4 bytes are left over after convolution layer 2"}));

/** @brief Tests that bytes after the header with no layer listed are InvalidFile. */
TEST(Small_DarknetWeights, RefusesBytesAfterAHeaderWithNoConvolution)
{
    const std::string bytes = sample_bytes(1);
    const std::vector<DarknetConvolutionShape> shapes;

    SA_EXPECT_FAILURE(read_bytes(bytes, shapes), StatusCode::InvalidFile,
                      "no convolution is listed", "4 bytes are left over");
}

/** @brief Holds a shape the reader cannot size and what its refusal names. */
struct ShapeRefusal
{
    DarknetConvolutionShape shape;
    const char* detail;
};

/** @brief Runs the reader over a header-only file and one shape it cannot size. */
class Small_DarknetWeightsShape : public ::testing::TestWithParam<ShapeRefusal>
{
};

/** @brief Tests that a shape with no positive size or too large to count is InvalidDescription. */
TEST_P(Small_DarknetWeightsShape, IsRefusedNamingTheLayer)
{
    const std::string bytes = sample_bytes(0);
    const std::vector<DarknetConvolutionShape> shapes{GetParam().shape};

    SA_EXPECT_FAILURE(read_bytes(bytes, shapes), StatusCode::InvalidDescription,
                      "convolution layer 0", GetParam().detail);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, Small_DarknetWeightsShape,
    ::testing::Values(
        ShapeRefusal{DarknetConvolutionShape{0, 0, 1, 1, false}, "filters 0"},
        ShapeRefusal{DarknetConvolutionShape{0, std::int64_t{1} << 40, std::int64_t{1} << 40, 1,
                                             false},
                     "too large to read"}));

/** @brief Tests that a path that cannot be opened gives FileNotFound. */
TEST(Medium_DarknetWeights, ReportsAMissingFileAsFileNotFound)
{
    const std::filesystem::path missing = std::filesystem::temp_directory_path() /
                                          "sushiai_weights_absent.weights";
    std::error_code ignored;
    std::filesystem::remove(missing, ignored);
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_EXPECT_FAILURE(read_darknet_weights(missing, shapes), StatusCode::FileNotFound,
                      "could not open");
}
