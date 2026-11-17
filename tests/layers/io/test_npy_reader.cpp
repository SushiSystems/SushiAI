/**************************************************************************/
/* test_npy_reader.cpp                                                    */
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
 * @file test_npy_reader.cpp
 * @brief Tests the .npy reader: header versions, element types, array order and each refusal.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/npy_reader.hpp>
#include <SushiAI/shape/shape.hpp>
#include "archive_bytes.hpp"

using namespace SushiAI;

namespace
{
    /** @brief Holds the alignment numpy.lib.format pads the preamble and the header to. */
    constexpr std::size_t K_HEADER_ALIGNMENT = 64;

    /**
     * @brief Holds where the data of every file below starts when its header is the 59 bytes
     *        numpy writes for a float32 array of shape (2, 3): 10 + 59 + 1 and 12 + 59 + 1
     *        bytes round up to 128.
     */
    constexpr std::size_t K_DATA_OFFSET = 128;

    /** @brief Holds the origin the tests give the reader. */
    constexpr const char* K_ORIGIN = "fixture.npy";

    /** @brief Returns the header text numpy writes for @p descr, @p fortran and @p shape. */
    std::string header_of(const std::string& descr, bool fortran, const std::string& shape)
    {
        return "{'descr': '" + descr + "', 'fortran_order': " + (fortran ? "True" : "False") +
               ", 'shape': " + shape + ", }";
    }

    /**
     * @brief Returns a .npy file of format version @p major.@p minor.
     *
     * The header text is padded with spaces and ends in a line end, so that the preamble and
     * the header take a multiple of 64 bytes, as numpy.lib.format writes it.
     */
    std::vector<std::byte> npy_file(unsigned major, unsigned minor, const std::string& header_text,
                                    const std::string& data)
    {
        const std::size_t width = major == 1 ? 2 : 4;
        const std::size_t unpadded = 8 + width + header_text.size() + 1;
        const std::size_t padded =
            (unpadded + K_HEADER_ALIGNMENT - 1) / K_HEADER_ALIGNMENT * K_HEADER_ALIGNMENT;
        const std::string header = header_text + std::string(padded - unpadded, ' ') + "\n";

        std::string out = "\x93NUMPY";
        out.push_back(static_cast<char>(major));
        out.push_back(static_cast<char>(minor));
        SushiAI::Test::put_le(out, header.size(), width);
        out += header;
        out += data;
        return SushiAI::Test::bytes_of(out);
    }

    /** @brief Returns the bits of @p value as the raw word the file stores. */
    std::uint64_t bits_of(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    /** @brief Returns the elements @p raw, each as @p width little-endian bytes. */
    std::string data_of(const std::vector<std::uint64_t>& raw, std::size_t width)
    {
        std::string out;
        for (const std::uint64_t value : raw)
            SushiAI::Test::put_le(out, value, width);
        return out;
    }

    /** @brief Returns the float32 elements @p values as the words a '<f4' file stores. */
    std::string float_data_of(const std::vector<float>& values)
    {
        std::vector<std::uint64_t> raw;
        for (const float value : values)
            raw.push_back(bits_of(value));
        return data_of(raw, 4);
    }

    /** @brief Returns the extents of @p shape. */
    std::vector<std::int64_t> extents_of(const Shape& shape)
    {
        std::vector<std::int64_t> out;
        for (std::int32_t axis = 0; axis < shape.rank(); ++axis)
            out.push_back(shape[axis]);
        return out;
    }

    /** @brief Returns the number of elements of an array of extents @p extents. */
    std::int64_t element_count(const std::vector<std::int64_t>& extents)
    {
        std::int64_t out = 1;
        for (const std::int64_t extent : extents)
            out *= extent;
        return out;
    }

    /** @brief Returns the value of element (i, j, k) of the test cube: 100 i + 10 j + k. */
    float cube_value(int i, int j, int k)
    {
        return static_cast<float>(100 * i + 10 * j + k);
    }

    /** @brief Returns the elements of the cube of extents (2, 3, 4) in row-major order. */
    std::vector<float> cube_row_major()
    {
        std::vector<float> out;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 4; ++k)
                    out.push_back(cube_value(i, j, k));
            }
        }
        return out;
    }

    /** @brief Returns the elements of the cube in column-major order: i varies fastest. */
    std::vector<float> cube_column_major()
    {
        std::vector<float> out;
        for (int k = 0; k < 4; ++k)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int i = 0; i < 2; ++i)
                    out.push_back(cube_value(i, j, k));
            }
        }
        return out;
    }

    /** @brief Reads @p bytes, which the caller keeps alive. */
    Result<IO::NpyReader> read_npy(const std::vector<std::byte>& bytes)
    {
        return IO::NpyReader::read(SushiAI::Test::view_of(bytes), K_ORIGIN);
    }

    /** @brief Names a format version. */
    struct VersionRow
    {
        std::string name;  /**< The row's name. */
        unsigned major;    /**< The major version the file states. */
    };

    /** @brief Holds an element type, its elements as raw words and the float32 they widen to. */
    struct TypeRow
    {
        std::string name;                 /**< The row's name. */
        std::string descr;                /**< The descriptor the header states. */
        std::size_t width;                /**< The bytes of one element. */
        std::vector<std::uint64_t> raw;   /**< Two elements, as the words the file stores. */
        std::vector<float> expected;      /**< The elements read as float32. */
    };

    /** @brief Holds a shape as the header spells it and the extents it names. */
    struct ShapeRow
    {
        std::string name;                    /**< The row's name. */
        std::string text;                    /**< The header's tuple. */
        std::vector<std::int64_t> extents;   /**< The extents it names. */
    };

    /** @brief Holds a file the reader refuses and what the refusal names. */
    struct RefusedRow
    {
        std::string name;      /**< The row's name. */
        unsigned major;        /**< The major version the file states. */
        unsigned minor;        /**< The minor version the file states. */
        std::string header;    /**< The header text, before its padding. */
        std::string data;      /**< The data bytes. */
        StatusCode code;       /**< The code of the refusal. */
        std::string first;     /**< A part of the message. */
        std::string second;    /**< Another part of the message. */
    };

    /** @brief Runs the reader over each format version. */
    class Small_NpyReaderVersion : public ::testing::TestWithParam<VersionRow>
    {
    };

    /** @brief Runs the reader over each element type. */
    class Small_NpyReaderType : public ::testing::TestWithParam<TypeRow>
    {
    };

    /** @brief Runs the reader over each shape spelling. */
    class Small_NpyReaderShape : public ::testing::TestWithParam<ShapeRow>
    {
    };

    /** @brief Runs the reader over each file it refuses. */
    class Small_NpyReaderRefusal : public ::testing::TestWithParam<RefusedRow>
    {
    };
} // namespace

/** @brief Tests that each format version gives the array's type, shape, offset and values. */
TEST_P(Small_NpyReaderVersion, ReadReturnsTheArrayOfEachFormatVersion)
{
    const std::vector<std::byte> bytes =
        npy_file(GetParam().major, 0, header_of("<f4", false, "(2, 3)"),
                 float_data_of({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_EQ(reader.array().type, IO::NpyType::Float32);
    EXPECT_EQ(extents_of(reader.array().shape), (std::vector<std::int64_t>{2, 3}));
    EXPECT_FALSE(reader.array().fortran_order);
    EXPECT_EQ(reader.array().offset, K_DATA_OFFSET);
    EXPECT_EQ(reader.array().length, 24u);
    EXPECT_EQ(reader.contents().size(), 24u);
    EXPECT_EQ(reader.read_float32(), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_NpyReaderVersion,
    ::testing::Values(VersionRow{"Version1", 1}, VersionRow{"Version2", 2},
                      VersionRow{"Version3", 3}),
    [](const ::testing::TestParamInfo<VersionRow>& info) { return info.param.name; });

/** @brief Tests that each element type widens to the float32 its bits spell. */
TEST_P(Small_NpyReaderType, ReadFloat32WidensEachElementType)
{
    const std::vector<std::byte> bytes =
        npy_file(1, 0, header_of(GetParam().descr, false, "(2,)"),
                 data_of(GetParam().raw, GetParam().width));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_EQ(IO::npy_type_name(reader.array().type), GetParam().descr);
    EXPECT_EQ(IO::npy_type_size(reader.array().type), GetParam().width);
    EXPECT_EQ(reader.read_float32(), GetParam().expected);
}

/**
 * @brief Runs ReadFloat32WidensEachElementType over each type. The raw words are IEEE 754 and
 *        two's-complement encodings: binary16 0x3C00 is 1 and 0xC000 is -2, and 0xFE, 0xFFFE,
 *        0xFFFFFFFE and 0xFFFFFFFFFFFFFFFE are -2 in 8, 16, 32 and 64 bits.
 */
INSTANTIATE_TEST_SUITE_P(
    Rows, Small_NpyReaderType,
    ::testing::Values(
        TypeRow{"Float16", "<f2", 2, {0x3C00, 0xC000}, {1.0f, -2.0f}},
        TypeRow{"Float32", "<f4", 4, {0x3F800000, 0xC0000000}, {1.0f, -2.0f}},
        TypeRow{"Float64", "<f8", 8, {0x3FF0000000000000, 0xC000000000000000}, {1.0f, -2.0f}},
        TypeRow{"Int8", "|i1", 1, {0x01, 0xFE}, {1.0f, -2.0f}},
        TypeRow{"UInt8", "|u1", 1, {0x01, 200}, {1.0f, 200.0f}},
        TypeRow{"Int16", "<i2", 2, {0x0001, 0xFFFE}, {1.0f, -2.0f}},
        TypeRow{"UInt16", "<u2", 2, {0x0001, 60000}, {1.0f, 60000.0f}},
        TypeRow{"Int32", "<i4", 4, {0x00000001, 0xFFFFFFFE}, {1.0f, -2.0f}},
        TypeRow{"UInt32", "<u4", 4, {0x00000001, 65536}, {1.0f, 65536.0f}},
        TypeRow{"Int64", "<i8", 8, {0x1, 0xFFFFFFFFFFFFFFFE}, {1.0f, -2.0f}},
        TypeRow{"UInt64", "<u8", 8, {0x1, 0x10000000000}, {1.0f, 1099511627776.0f}}),
    [](const ::testing::TestParamInfo<TypeRow>& info) { return info.param.name; });

/** @brief Tests that a column-major matrix reads as the row-major matrix it stores. */
TEST(Small_NpyReader, ReadFloat32TransposesAColumnMajorMatrix)
{
    // The matrix [[0, 1, 2], [3, 4, 5]] stored column by column.
    const std::vector<std::byte> bytes =
        npy_file(1, 0, header_of("<f4", true, "(2, 3)"),
                 float_data_of({0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f}));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_TRUE(reader.array().fortran_order);
    EXPECT_EQ(reader.read_float32(), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
}

/** @brief Tests that a column-major array of rank 3 reads as the row-major array it stores. */
TEST(Small_NpyReader, ReadFloat32TransposesAColumnMajorCube)
{
    const std::vector<std::byte> bytes = npy_file(
        1, 0, header_of("<f4", true, "(2, 3, 4)"), float_data_of(cube_column_major()));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_EQ(reader.read_float32(), cube_row_major());
}

/** @brief Tests that a scalar reads as one element of rank 0. */
TEST(Small_NpyReader, ReadReturnsAScalarAsOneElement)
{
    const std::vector<std::byte> bytes =
        npy_file(1, 0, header_of("<f4", false, "()"), float_data_of({7.0f}));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_EQ(reader.array().shape.rank(), 0);
    EXPECT_EQ(reader.read_float32(), (std::vector<float>{7.0f}));
}

/** @brief Tests that each spelling of a shape gives its extents and its data length. */
TEST_P(Small_NpyReaderShape, ReadNamesTheExtentsOfEachShapeSpelling)
{
    const std::int64_t elements = element_count(GetParam().extents);
    const std::vector<std::byte> bytes =
        npy_file(1, 0, header_of("<f4", false, GetParam().text),
                 float_data_of(std::vector<float>(static_cast<std::size_t>(elements), 0.0f)));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_EQ(extents_of(reader.array().shape), GetParam().extents);
    EXPECT_EQ(reader.array().length, static_cast<std::size_t>(elements) * 4u);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_NpyReaderShape,
    ::testing::Values(ShapeRow{"Scalar", "()", {}}, ShapeRow{"Vector", "(3,)", {3}},
                      ShapeRow{"Matrix", "(2, 3)", {2, 3}},
                      ShapeRow{"Cube", "(2, 3, 4)", {2, 3, 4}}),
    [](const ::testing::TestParamInfo<ShapeRow>& info) { return info.param.name; });

/** @brief Tests that a header with double quotes and no spaces reads as numpy's own spelling. */
TEST(Small_NpyReader, ReadAcceptsDoubleQuotesAndNoSpaces)
{
    const std::vector<std::byte> bytes =
        npy_file(1, 0, "{\"descr\":\"<f4\",\"fortran_order\":False,\"shape\":(2,)}",
                 float_data_of({1.0f, 2.0f}));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_EQ(reader.read_float32(), (std::vector<float>{1.0f, 2.0f}));
}

/** @brief Tests that bytes after the data the shape needs are left alone. */
TEST(Small_NpyReader, ReadAllowsBytesAfterTheData)
{
    const std::vector<std::byte> bytes = npy_file(1, 0, header_of("<f4", false, "(1,)"),
                                                  float_data_of({3.0f}) + std::string(8, 'x'));

    SA_ASSERT_OK_AND_ASSIGN(const IO::NpyReader reader, read_npy(bytes));

    EXPECT_EQ(reader.array().length, 4u);
    EXPECT_EQ(reader.read_float32(), (std::vector<float>{3.0f}));
}

/** @brief Tests that each file the reader lacks the means to read is refused by what it names. */
TEST_P(Small_NpyReaderRefusal, ReadRefusesAFileAndNamesWhy)
{
    const std::vector<std::byte> bytes =
        npy_file(GetParam().major, GetParam().minor, GetParam().header, GetParam().data);

    SA_EXPECT_FAILURE(read_npy(bytes), GetParam().code, "'fixture.npy'", GetParam().first,
                      GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_NpyReaderRefusal,
    ::testing::Values(
        RefusedRow{"ObjectArray", 1, 0, header_of("|O", false, "(2,)"), std::string(16, '\0'),
                   StatusCode::Unsupported, "object array", "'|O'"},
        RefusedRow{"BigEndianFloat", 1, 0, header_of(">f4", false, "(2,)"), std::string(8, '\0'),
                   StatusCode::Unsupported, "big-endian", "'>f4'"},
        RefusedRow{"UnreadType", 1, 0, header_of("<c8", false, "(2,)"), std::string(16, '\0'),
                   StatusCode::Unsupported, "'<c8'", "lacks"},
        RefusedRow{"StructuredType", 1, 0,
                   "{'descr': [('a', '<f4')], 'fortran_order': False, 'shape': (2,), }",
                   std::string(8, '\0'), StatusCode::Unsupported, "structured", "not read"},
        RefusedRow{"VersionFour", 4, 0, header_of("<f4", false, "(2,)"), std::string(8, '\0'),
                   StatusCode::Unsupported, "version 4.0", "not read"},
        RefusedRow{"VersionOnePointOne", 1, 1, header_of("<f4", false, "(2,)"),
                   std::string(8, '\0'), StatusCode::Unsupported, "version 1.1", "not read"},
        RefusedRow{"MissingShape", 1, 0, "{'descr': '<f4', 'fortran_order': False, }",
                   std::string(8, '\0'), StatusCode::InvalidFile, "lacks", "'shape'"},
        RefusedRow{"UnknownKey", 1, 0, "{'descr': '<f4', 'order': False, 'shape': (2,), }",
                   std::string(8, '\0'), StatusCode::InvalidFile, "'order'", "allowed"},
        RefusedRow{"RepeatedKey", 1, 0,
                   "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (2,), }",
                   std::string(8, '\0'), StatusCode::InvalidFile, "'descr'", "twice"},
        RefusedRow{"TextAfterTheDictionary", 1, 0, header_of("<f4", false, "(2,)") + " x",
                   std::string(8, '\0'), StatusCode::InvalidFile, "after its dictionary", "byte"},
        RefusedRow{"UnclosedString", 1, 0, "{'descr': '<f4", std::string(8, '\0'),
                   StatusCode::InvalidFile, "never closes", "byte"},
        RefusedRow{"ShapeWithNoNumber", 1, 0, header_of("<f4", false, "(a,)"),
                   std::string(8, '\0'), StatusCode::InvalidFile, "no whole number", "byte"},
        RefusedRow{"ExtentOfZero", 1, 0, header_of("<f4", false, "(0,)"), "",
                   StatusCode::Unsupported, "extent of 0", "empty array"},
        RefusedRow{"DataShorterThanTheShape", 1, 0, header_of("<f4", false, "(2, 3)"),
                   std::string(20, '\0'), StatusCode::InvalidFile, "needs 24 bytes", "holds 20"}),
    [](const ::testing::TestParamInfo<RefusedRow>& info) { return info.param.name; });

/** @brief Tests that a file without the magic bytes is refused. */
TEST(Small_NpyReader, ReadRefusesAFileWithoutTheMagic)
{
    const std::vector<std::byte> bytes = SushiAI::Test::bytes_of(std::string(32, 'x'));

    SA_EXPECT_FAILURE(read_npy(bytes), StatusCode::InvalidFile, "'fixture.npy'", "magic");
}

/** @brief Tests that a header longer than the file is refused. */
TEST(Small_NpyReader, ReadRefusesAHeaderLongerThanTheFile)
{
    std::vector<std::byte> bytes =
        npy_file(1, 0, header_of("<f4", false, "(2,)"), float_data_of({1.0f, 2.0f}));
    bytes.resize(40);

    SA_EXPECT_FAILURE(read_npy(bytes), StatusCode::InvalidFile, "'fixture.npy'", "header states");
}
