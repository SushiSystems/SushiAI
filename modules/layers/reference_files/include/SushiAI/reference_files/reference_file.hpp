/**************************************************************************/
/* reference_file.hpp                                                     */
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
 * @file reference_file.hpp
 * @brief Declares the reader of the YOLO26 reference files and the comparisons against them.
 * @author Mustafa Garip
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Returns the directory SA_TEST_FIXTURE_DIR names, or an empty string. */
        inline std::string fixture_directory() noexcept
        {
#ifdef SA_TEST_FIXTURE_DIR
            return SA_TEST_FIXTURE_DIR;
#else
            return {};
#endif
        }

        /** @brief Holds one tensor read from a reference file, or why it could not be read. */
        struct ReferenceTensor
        {
            bool found = false;                 /**< Whether the tensor was read. */
            std::string reason;                 /**< Names what is missing when not found. */
            DType dtype = DType::Float32;       /**< Element type as stored. */
            Shape shape{};                      /**< Extents as stored. */
            std::vector<unsigned char> payload; /**< Element bytes, row-major, little-endian. */

            /** @brief Returns the elements of a float32 tensor. */
            [[nodiscard]] std::vector<float> floats() const noexcept
            {
                return elements<float>(DType::Float32);
            }

            /** @brief Returns the elements of an int32 tensor. */
            [[nodiscard]] std::vector<std::int32_t> int32s() const noexcept
            {
                return elements<std::int32_t>(DType::Int32);
            }

            /** @brief Returns the elements of an int64 tensor. */
            [[nodiscard]] std::vector<std::int64_t> int64s() const noexcept
            {
                return elements<std::int64_t>(DType::Int64);
            }

            private:
                /**
                 * @brief Returns the payload as @p Element values.
                 * @pre The tensor was found and is stored as @p expected.
                 */
                template <typename Element>
                [[nodiscard]] std::vector<Element> elements(DType expected) const
                {
                    SA_FATAL_IF(!found, "a reference tensor that was not found was read: {}",
                                reason);
                    SA_FATAL_IF(dtype != expected, "a {} reference tensor was read as {}",
                                dtype_name(dtype), dtype_name(expected));

                    std::vector<Element> values(payload.size() / sizeof(Element));
                    if (!values.empty())
                        std::memcpy(values.data(), payload.data(), values.size() * sizeof(Element));
                    return values;
                }
        };

        /** @brief Reads named tensors from one SACP file of the YOLO26 reference set. */
        class ReferenceFile final
        {
            public:
                /** @brief Opens @p file_name, a path below the fixture directory, and indexes its records. */
                explicit ReferenceFile(std::string_view file_name) noexcept
                {
                    const std::string directory = fixture_directory();
                    if (directory.empty())
                    {
                        reason_ = "SA_TEST_FIXTURE_DIR is not defined for this binary, so " +
                                  std::string(file_name) + " cannot be located";
                        return;
                    }

                    path_ = directory + "/";
                    path_ += file_name;
                    stream_.open(path_, std::ios::binary);
                    if (!stream_.is_open())
                    {
                        reason_ = "the reference file is missing: " + path_;
                        return;
                    }

                    const Status indexed = index_records();
                    if (!indexed.is_ok())
                    {
                        reason_ = path_ + " is not a readable reference file: ";
                        reason_ += indexed.message();
                        ADD_FAILURE() << reason_;
                        return;
                    }
                    present_ = true;
                }

                /** @brief Reports whether the file was found and indexed. */
                [[nodiscard]] bool is_present() const noexcept
                {
                    return present_;
                }

                /** @brief Returns what is missing when the file is not present. */
                [[nodiscard]] const std::string& reason() const noexcept
                {
                    return reason_;
                }

                /** @brief Reads the tensor named @p tensor_name, or reports that it is absent. */
                [[nodiscard]] ReferenceTensor read(std::string_view tensor_name) noexcept
                {
                    ReferenceTensor tensor;
                    if (!present_)
                    {
                        tensor.reason = reason_;
                        return tensor;
                    }

                    const auto entry = entries_.find(tensor_name);
                    if (entry == entries_.end())
                    {
                        tensor.reason = "the reference tensor '" + std::string(tensor_name) +
                                        "' is not in " + path_;
                        return tensor;
                    }

                    const IO::TensorRecord& record = entry->second.record;
                    tensor.dtype = record.dtype;
                    tensor.shape = record.shape;
                    tensor.payload.resize(record.payload_bytes());
                    stream_.clear();
                    stream_.seekg(entry->second.offset);
                    stream_.read(reinterpret_cast<char*>(tensor.payload.data()),
                                 static_cast<std::streamsize>(tensor.payload.size()));
                    if (!stream_)
                    {
                        tensor.payload.clear();
                        tensor.reason = path_ + " ended inside the payload of '" + record.name + "'";
                        ADD_FAILURE() << tensor.reason;
                        return tensor;
                    }

                    tensor.found = true;
                    return tensor;
                }

            private:
                /** @brief Holds one record's header and where its payload starts. */
                struct Entry
                {
                    IO::TensorRecord record; /**< Name, dtype and extents. */
                    std::streampos offset;   /**< Position of the first payload byte. */
                };

                /**
                 * @brief Reads every record header of the open file into the index.
                 * @return Success, or the failure of the header or of the first bad record.
                 * @pre The stream is open at the first byte of the file.
                 */
                [[nodiscard]] Status index_records()
                {
                    SA_TRY_ASSIGN(IO::CheckpointReader reader, IO::CheckpointReader::open(stream_));
                    while (reader.read_count() < reader.count())
                    {
                        SA_TRY_ASSIGN(IO::TensorRecord record, reader.next());
                        const std::streampos offset = stream_.tellg();
                        stream_.seekg(static_cast<std::streamoff>(record.payload_bytes()),
                                      std::ios::cur);
                        std::string name = record.name;
                        entries_.emplace(std::move(name), Entry{std::move(record), offset});
                    }
                    return Status();
                }

                /// @brief Holds the path of the file, empty when the fixture directory is unknown.
                std::string path_;

                /// @brief Holds what is missing when the file is not present.
                std::string reason_;

                /// @brief Holds the open file.
                std::ifstream stream_;

                /// @brief Holds the records of the file by tensor name.
                std::map<std::string, Entry, std::less<>> entries_;

                /// @brief Holds whether the file was found and indexed.
                bool present_ = false;
        };

        /**
         * @brief Checks that every element of @p actual is within @p tolerance of @p reference.
         * @param tolerance Largest absolute difference allowed; 0 demands equal values.
         * @return Success, or a failure naming the first and the worst element.
         */
        inline ::testing::AssertionResult near_reference(SushiRuntime::span<const float> actual,
                                                         SushiRuntime::span<const float> reference,
                                                         float tolerance) noexcept
        {
            if (actual.size() != reference.size())
                return ::testing::AssertionFailure() << actual.size() << " values against "
                                                     << reference.size() << " reference values";

            std::size_t mismatches = 0;
            std::size_t first = 0;
            std::size_t worst = 0;
            float worst_gap = 0.0f;
            for (std::size_t i = 0; i < actual.size(); ++i)
            {
                const float gap = std::fabs(actual[i] - reference[i]);
                if (gap <= tolerance)
                    continue;

                if (mismatches == 0)
                    first = i;
                if (mismatches == 0 || !(gap <= worst_gap))
                {
                    worst = i;
                    worst_gap = gap;
                }
                ++mismatches;
            }
            if (mismatches == 0)
                return ::testing::AssertionSuccess();

            return ::testing::AssertionFailure()
                   << mismatches << " of " << actual.size() << " values differ by more than "
                   << tolerance << "; first at " << first << " (" << actual[first] << " against "
                   << reference[first] << "), worst at " << worst << " (" << actual[worst]
                   << " against " << reference[worst] << ")";
        }

        /**
         * @brief Checks that two integer arrays hold the same values in the same order.
         * @return Success, or a failure naming the first differing element and the count.
         */
        template <typename Actual, typename Reference>
        ::testing::AssertionResult equal_reference(const std::vector<Actual>& actual,
                                                   const std::vector<Reference>& reference) noexcept
        {
            if (actual.size() != reference.size())
                return ::testing::AssertionFailure() << actual.size() << " values against "
                                                     << reference.size() << " reference values";

            std::size_t mismatches = 0;
            std::size_t first = 0;
            for (std::size_t i = 0; i < actual.size(); ++i)
            {
                if (static_cast<std::int64_t>(actual[i]) == static_cast<std::int64_t>(reference[i]))
                    continue;

                if (mismatches == 0)
                    first = i;
                ++mismatches;
            }
            if (mismatches == 0)
                return ::testing::AssertionSuccess();

            return ::testing::AssertionFailure()
                   << mismatches << " of " << actual.size() << " values differ; first at " << first
                   << " (" << static_cast<std::int64_t>(actual[first]) << " against "
                   << static_cast<std::int64_t>(reference[first]) << ")";
        }

    } // namespace Test
} // namespace SushiAI

/**
 * @brief Declares @p file as an open reference file, or skips the test naming the missing file.
 * @pre Used as a statement in a function returning void.
 */
#define SA_OPEN_REFERENCE_OR_SKIP(file, name)                                  \
    ::SushiAI::Test::ReferenceFile file(name);                                 \
    if (!file.is_present())                                                    \
        GTEST_SKIP() << file.reason()

/**
 * @brief Declares @p tensor as a tensor of @p file, or fails the test naming the missing tensor.
 * @pre Used as a statement in a function returning void, after SA_OPEN_REFERENCE_OR_SKIP.
 * @pre MANIFEST.json lists @p name for the file.
 */
#define SA_READ_REFERENCE_OR_FAIL(tensor, file, name)                          \
    const ::SushiAI::Test::ReferenceTensor tensor = file.read(name);           \
    ASSERT_TRUE(tensor.found) << tensor.reason
