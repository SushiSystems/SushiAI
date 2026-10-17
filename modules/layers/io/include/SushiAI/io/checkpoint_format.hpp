/**************************************************************************/
/* checkpoint_format.hpp                                                  */
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
 * @file checkpoint_format.hpp
 * @brief Defines the binary serialization format and streaming reader and writer.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @brief Holds the first four bytes of every checkpoint, the ASCII "SACP", as the
         *        little-endian word a reader decodes them to.
         */
        inline constexpr std::uint32_t K_CHECKPOINT_MAGIC = 0x50434153u;

        /** @brief Holds the oldest format version this build reads. */
        inline constexpr std::uint32_t K_CHECKPOINT_FIRST_VERSION = 1u;

        /**
         * @brief Holds the format version this build writes, and the newest it reads.
         *
         * Version 2 adds the run-state section; README.md has the layout of each version.
         */
        inline constexpr std::uint32_t K_CHECKPOINT_VERSION = 2u;

        /** @brief Holds the largest count of model-state records a file may declare. */
        inline constexpr std::uint32_t K_MAX_TENSOR_COUNT = 4096u;

        /** @brief Holds the largest count of run-state records a file may declare. */
        inline constexpr std::uint32_t K_MAX_RUN_STATE_COUNT = 16384u;

        /** @brief Holds the longest record name or owner name a file may declare, in bytes. */
        inline constexpr std::uint32_t K_MAX_NAME_LENGTH = 256u;

        /**
         * @struct TensorRecord
         * @brief Holds one tensor's identity in a checkpoint: its name, type and extents.
         */
        struct TensorRecord
        {
            /** @brief Holds the dotted parameter name, such as "0.weight". */
            std::string name;

            /** @brief Holds the element type. */
            DType dtype = DType::Float32;

            /** @brief Holds the extents. */
            Shape shape{};

            /**
             * @brief Returns the byte size of this record's payload.
             * @return The element count times the element size.
             */
            [[nodiscard]] std::size_t payload_bytes() const noexcept
            {
                return static_cast<std::size_t>(shape.num_elements()) * dtype_size(dtype);
            }
        };

        /**
         * @class CheckpointWriter
         * @brief Serializes the header, the model-state records and the run-state section.
         */
        class CheckpointWriter final
        {
            public:
                /**
                 * @brief Writes the file header and returns a writer for the records.
                 * @param out Where the bytes go; must outlive the writer.
                 * @param count How many model-state records will follow.
                 * @param run_state_count How many run-state records follow those; 0 for none.
                 * @return The writer; InvalidDescription for a count above its bound;
                 *         Unsupported on a host that is not little-endian; InvalidFile.
                 */
                [[nodiscard]] static Result<CheckpointWriter> create(
                    std::ostream& out, std::uint32_t count, std::uint32_t run_state_count = 0);

                /**
                 * @brief Opens the run-state section; every later write() is a run-state record.
                 * @param owner Names what the state belongs to, such as "sgd".
                 * @pre Every declared model-state record is written, and the header declared
                 *      at least one run-state record; either broken stops the process.
                 * @return InvalidDescription for an owner that is empty or longer than
                 *         @ref K_MAX_NAME_LENGTH; InvalidFile when the write fails.
                 */
                [[nodiscard]] Status begin_run_state(std::string_view owner);

                /**
                 * @brief Writes one record's header and its payload to the open section.
                 * @param record What the tensor is called, and its type and extents.
                 * @param payload @c record.payload_bytes() readable little-endian bytes.
                 * @return Success, InvalidDescription for a name longer than
                 *         @ref K_MAX_NAME_LENGTH, or InvalidFile when the write fails.
                 */
                [[nodiscard]] Status write(const TensorRecord& record, const void* payload);

                /**
                 * @brief Checks that both sections hold exactly the records declared.
                 * @pre write() has succeeded once for each declared record.
                 */
                void finish() const;

                /** @brief Returns how many model-state records this file declared. */
                [[nodiscard]] std::uint32_t declared_count() const noexcept
                {
                    return declared_;
                }

                /** @brief Returns how many model-state records have been written so far. */
                [[nodiscard]] std::uint32_t written_count() const noexcept
                {
                    return written_;
                }

                /** @brief Returns how many run-state records this file declared. */
                [[nodiscard]] std::uint32_t declared_run_state_count() const noexcept
                {
                    return run_state_declared_;
                }

                /** @brief Returns how many run-state records have been written so far. */
                [[nodiscard]] std::uint32_t written_run_state_count() const noexcept
                {
                    return run_state_written_;
                }

            private:
                /**
                 * @brief Stores the stream and the declared counts; writes nothing.
                 */
                CheckpointWriter(std::ostream& out, std::uint32_t count,
                                 std::uint32_t run_state_count) noexcept;

                /** @brief Holds the stream the header and records go to. */
                std::ostream* out_ = nullptr;

                /** @brief Holds the model-state record count the header declared. */
                std::uint32_t declared_ = 0;

                /** @brief Holds the number of model-state records written so far. */
                std::uint32_t written_ = 0;

                /** @brief Holds the run-state record count the header declared. */
                std::uint32_t run_state_declared_ = 0;

                /** @brief Holds the number of run-state records written so far. */
                std::uint32_t run_state_written_ = 0;

                /** @brief Says whether begin_run_state() has succeeded. */
                bool in_run_state_ = false;
        };

        /**
         * @class CheckpointReader
         * @brief Deserializes the header and the records of either section from a stream.
         */
        class CheckpointReader final
        {
            public:
                /**
                 * @brief Reads and validates the file header of any version this build reads.
                 * @param in Where the bytes come from; must outlive the reader.
                 * @return The reader; InvalidFile when the magic, the version or a count is
                 *         wrong, or the header is truncated; Unsupported on a host that is
                 *         not little-endian.
                 */
                [[nodiscard]] static Result<CheckpointReader> open(std::istream& in);

                /** @brief Returns the format version the file states. */
                [[nodiscard]] std::uint32_t version() const noexcept
                {
                    return version_;
                }

                /** @brief Returns how many model-state records the file declares. */
                [[nodiscard]] std::uint32_t count() const noexcept
                {
                    return count_;
                }

                /** @brief Returns how many model-state records have been read so far. */
                [[nodiscard]] std::uint32_t read_count() const noexcept
                {
                    return read_;
                }

                /** @brief Returns how many run-state records the file declares; 0 for none. */
                [[nodiscard]] std::uint32_t run_state_count() const noexcept
                {
                    return run_state_count_;
                }

                /** @brief Returns how many run-state records have been read so far. */
                [[nodiscard]] std::uint32_t run_state_read_count() const noexcept
                {
                    return run_state_read_;
                }

                /**
                 * @brief Opens the run-state section; every later next() is a run-state record.
                 * @pre Every model-state record and its payload are read, and run_state_count()
                 *      is not 0; either broken stops the process.
                 * @return The owner the section names; InvalidFile for an owner that is empty,
                 *         longer than @ref K_MAX_NAME_LENGTH or truncated.
                 */
                [[nodiscard]] Result<std::string> begin_run_state();

                /**
                 * @brief Reads the next record's header of the open section, not its payload.
                 * @return The record, or InvalidFile when a bound is exceeded, a declared
                 *         extent is not positive, the dtype is unknown, or the header is
                 *         truncated.
                 * @pre The open section has a record left; otherwise the process stops.
                 */
                [[nodiscard]] Result<TensorRecord> next();

                /**
                 * @brief Reads the payload of the record @ref next just returned.
                 * @param record The record @ref next returned.
                 * @param destination @c record.payload_bytes() writable bytes.
                 * @return Success, or InvalidFile when the stream ends early.
                 * @pre @p destination is not null.
                 */
                [[nodiscard]] Status read_payload(const TensorRecord& record, void* destination);

            private:
                /**
                 * @brief Stores the stream and what the header declared; reads nothing.
                 */
                CheckpointReader(std::istream& in, std::uint32_t version, std::uint32_t count,
                                 std::uint32_t run_state_count) noexcept;

                /** @brief Holds the stream the header and records come from. */
                std::istream* in_ = nullptr;

                /** @brief Holds the version the header stated. */
                std::uint32_t version_ = 0;

                /** @brief Holds the model-state record count the header declared. */
                std::uint32_t count_ = 0;

                /** @brief Holds the number of model-state records read so far. */
                std::uint32_t read_ = 0;

                /** @brief Holds the run-state record count, 0 for a version 1 file. */
                std::uint32_t run_state_count_ = 0;

                /** @brief Holds the number of run-state records read so far. */
                std::uint32_t run_state_read_ = 0;

                /** @brief Says whether begin_run_state() has succeeded. */
                bool in_run_state_ = false;
        };

    } // namespace IO
} // namespace SushiAI
