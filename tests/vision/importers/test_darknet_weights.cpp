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
 * @file test_darknet_weights.cpp
 * @brief Tests the .weights header forms, the version refusals, the vector order and the byte-count
 *        refusals of the Darknet weights reader.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
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

    /**
     * @brief Returns a version 0.2.0 file: a 64-bit seen count of zero, then the floats 0 to
     *        @p float_count - 1, each equal to its own index.
     */
    std::string sample_bytes(std::size_t float_count)
    {
        std::string bytes;
        put(bytes, 0, 4);
        put(bytes, 2, 4);
        put(bytes, 0, 4);
        put(bytes, 0, 8);
        for (std::size_t i = 0; i < float_count; ++i)
            put_float(bytes, static_cast<float>(i));
        return bytes;
    }

    /**
     * @brief Owns a .weights file the test writes under the temporary directory, and removes
     *        it on destruction.
     */
    class TemporaryWeightsFile
    {
        public:
            /** @brief Writes @p bytes to the file named @p name, replacing any earlier one. */
            TemporaryWeightsFile(const char* name, const std::string& bytes)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::ofstream out(path_, std::ios::binary);
                out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            }

            /** @brief Removes the file. */
            ~TemporaryWeightsFile()
            {
                std::error_code ignored;
                std::filesystem::remove(path_, ignored);
            }

            TemporaryWeightsFile(const TemporaryWeightsFile&) = delete;
            TemporaryWeightsFile& operator=(const TemporaryWeightsFile&) = delete;

            /** @brief Returns the path of the file. */
            const std::filesystem::path& path() const noexcept
            {
                return path_;
            }

        private:
            /** @brief Holds the path the file was written to. */
            std::filesystem::path path_;
    };

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
TEST(Medium_DarknetWeights, ReadsAVersionZeroTwoHeaderWithA64BitSeen)
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
TEST(Medium_DarknetWeights, ReadsAVersionZeroOneHeaderWithA32BitSeen)
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
TEST(Medium_DarknetWeights, ReadsAMajorOneHeaderWithA64BitSeen)
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

/** @brief Tests that a major of exactly 1000 is read. */
TEST(Medium_DarknetWeights, AcceptsAMajorAtTheLimitOfOneThousand)
{
    std::string bytes;
    put(bytes, 1000, 4);
    put(bytes, 0, 4);
    put(bytes, 0, 4);
    put(bytes, 0, 8);
    std::istringstream stream(bytes);

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeightsHeader header, read_darknet_header(stream));

    EXPECT_EQ(header.major, 1000);
}

/** @brief Tests that a major of 1001 is Unsupported. */
TEST(Medium_DarknetWeights, RefusesAMajorAboveOneThousandAsUnsupported)
{
    std::string bytes;
    put(bytes, 1001, 4);
    put(bytes, 0, 4);
    put(bytes, 0, 4);
    std::istringstream stream(bytes);

    SA_EXPECT_FAILURE(read_darknet_header(stream), StatusCode::Unsupported, "1001");
}

/** @brief Tests that a minor of 1001 is Unsupported. */
TEST(Medium_DarknetWeights, RefusesAMinorAboveOneThousandAsUnsupported)
{
    std::string bytes;
    put(bytes, 0, 4);
    put(bytes, 1001, 4);
    put(bytes, 0, 4);
    std::istringstream stream(bytes);

    SA_EXPECT_FAILURE(read_darknet_header(stream), StatusCode::Unsupported, "1001");
}

/** @brief Tests that a stream that ends inside the header is InvalidFile. */
TEST(Medium_DarknetWeights, RefusesAHeaderThatEndsEarly)
{
    std::string bytes;
    put(bytes, 0, 4);
    put(bytes, 2, 4);
    put(bytes, 0, 2);
    std::istringstream stream(bytes);

    SA_EXPECT_FAILURE(read_darknet_header(stream), StatusCode::InvalidFile, "revision");
}

/** @brief Tests that index-valued floats land in each vector at their own indices. */
TEST(Medium_DarknetWeights, ReadsEveryVectorInFileOrder)
{
    // The floats are 0 to 21 in file order, so each vector starts at its own index.
    TemporaryWeightsFile file("sushiai_weights_order.weights", sample_bytes(22));
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeights weights,
                            read_darknet_weights(file.path(), shapes));

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
TEST(Medium_DarknetWeights, KeepsTheBatchNormVectorsOnlyForNormalisedLayers)
{
    TemporaryWeightsFile file("sushiai_weights_vectors.weights", sample_bytes(22));
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeights weights,
                            read_darknet_weights(file.path(), shapes));

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
TEST(Medium_DarknetWeights, DecodesEveryBitPatternAsItIs)
{
    // Biases are -0.0 and the smallest subnormal; weights are a NaN with payload 1 and 1.0.
    std::string bytes;
    put(bytes, 0, 4);
    put(bytes, 2, 4);
    put(bytes, 0, 4);
    put(bytes, 0, 8);
    put(bytes, 0x80000000u, 4);
    put(bytes, 0x00000001u, 4);
    put(bytes, 0x7FC00001u, 4);
    put(bytes, 0x3F800000u, 4);
    TemporaryWeightsFile file("sushiai_weights_bits.weights", bytes);
    const std::vector<DarknetConvolutionShape> shapes{DarknetConvolutionShape{0, 2, 1, 1, false}};

    SA_ASSERT_OK_AND_ASSIGN(const DarknetWeights weights,
                            read_darknet_weights(file.path(), shapes));

    const DarknetConvolution& layer = weights.convolutions[0];
    ASSERT_EQ(layer.biases.size(), 2u);
    ASSERT_EQ(layer.weights.size(), 2u);
    EXPECT_EQ(bits_of(layer.biases[0]), 0x80000000u);
    EXPECT_EQ(bits_of(layer.biases[1]), 0x00000001u);
    EXPECT_EQ(bits_of(layer.weights[0]), 0x7FC00001u);
    EXPECT_EQ(bits_of(layer.weights[1]), 0x3F800000u);
}

/** @brief Tests that a file one float short is InvalidFile with the layer and both counts. */
TEST(Medium_DarknetWeights, RefusesOneFloatShortWithTheLayerAndBothByteCounts)
{
    // The header is 20 bytes. Layer 0 needs 16 bytes, so layer 2 ends at 20 + 16 + 72 = 108.
    TemporaryWeightsFile file("sushiai_weights_short.weights", sample_bytes(21));
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_EXPECT_FAILURE(read_darknet_weights(file.path(), shapes), StatusCode::InvalidFile,
                      "104 bytes", "convolution layer 2 needs bytes up to 108");
}

/** @brief Tests that a file one float over is InvalidFile with the layer and both counts. */
TEST(Medium_DarknetWeights, RefusesOneFloatOverWithTheLayerAndBothByteCounts)
{
    TemporaryWeightsFile file("sushiai_weights_over.weights", sample_bytes(23));
    const std::vector<DarknetConvolutionShape> shapes = sample_shapes();

    SA_EXPECT_FAILURE(read_darknet_weights(file.path(), shapes), StatusCode::InvalidFile,
                      "112 bytes", "108 bytes", "4 bytes are left over after convolution layer 2");
}

/** @brief Tests that bytes after the header with no layer listed are InvalidFile. */
TEST(Medium_DarknetWeights, RefusesBytesAfterAHeaderWithNoConvolution)
{
    TemporaryWeightsFile file("sushiai_weights_no_layer.weights", sample_bytes(1));
    const std::vector<DarknetConvolutionShape> shapes;

    SA_EXPECT_FAILURE(read_darknet_weights(file.path(), shapes), StatusCode::InvalidFile,
                      "no convolution is listed", "4 bytes are left over");
}

/** @brief Tests that a shape with zero filters is InvalidDescription naming the layer. */
TEST(Medium_DarknetWeights, RefusesAShapeThatIsNotPositive)
{
    TemporaryWeightsFile file("sushiai_weights_zero.weights", sample_bytes(0));
    const std::vector<DarknetConvolutionShape> shapes{DarknetConvolutionShape{0, 0, 1, 1, false}};

    SA_EXPECT_FAILURE(read_darknet_weights(file.path(), shapes), StatusCode::InvalidDescription,
                      "convolution layer 0", "filters 0");
}

/** @brief Tests that a shape too large to count is InvalidDescription. */
TEST(Medium_DarknetWeights, RefusesAShapeWhoseCountOverflows)
{
    TemporaryWeightsFile file("sushiai_weights_huge.weights", sample_bytes(0));
    const std::int64_t big = std::int64_t{1} << 40;
    const std::vector<DarknetConvolutionShape> shapes{
        DarknetConvolutionShape{0, big, big, 1, false},
    };

    SA_EXPECT_FAILURE(read_darknet_weights(file.path(), shapes), StatusCode::InvalidDescription,
                      "convolution layer 0", "too large to read");
}

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
