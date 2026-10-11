/**************************************************************************/
/* checkpoint_format.cpp                                                  */
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
 * @file checkpoint_format.cpp
 * @brief Implements binary checkpoint serialization and deserialization helpers.
 * @author Mustafa Garip
 */

#include <SushiAI/io/checkpoint_format.hpp>

#include <array>
#include <cstring>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Lists every element type a checkpoint may encode. */
            constexpr std::array<DType, 7> K_KNOWN_DTYPES = {
                DType::Half, DType::Float32, DType::Float64,
                DType::Complex32, DType::Complex64,
                DType::Int32, DType::Int64,
            };

            /** @brief Holds what the header of a version 1 file declares after its version. */
            struct HeaderV1
            {
                std::uint32_t count = 0; /**< Holds the count of model-state records. */
            };

            /** @brief Holds what the header of a version 2 file declares after its version. */
            struct HeaderV2
            {
                std::uint32_t count = 0;           /**< Holds the model-state record count. */
                std::uint32_t run_state_count = 0; /**< Holds the run-state record count. */
            };

            /**
             * @brief Returns whether @p encoded is the value of an entry of @ref K_KNOWN_DTYPES.
             */
            [[nodiscard]] bool is_known_dtype(std::uint32_t encoded) noexcept
            {
                for (DType dtype : K_KNOWN_DTYPES)
                {
                    if (static_cast<std::uint32_t>(dtype) == encoded)
                        return true;
                }

                return false;
            }

            /** @brief Reports whether this host stores the low byte of a word first. */
            [[nodiscard]] bool is_little_endian_host() noexcept
            {
                const std::uint32_t word = 1u;
                unsigned char first = 0;
                std::memcpy(&first, &word, 1);
                return first == 1;
            }

            /** @brief Returns Unsupported on a host whose tensor bytes are not little-endian. */
            [[nodiscard]] Status check_host_byte_order()
            {
                SA_FAIL_IF(!is_little_endian_host(), StatusCode::Unsupported,
                           "a checkpoint holds little-endian payloads, and this host is not "
                           "little-endian");
                return Status();
            }

            /** @brief Writes @p value as four little-endian bytes. */
            void put_u32(std::ostream& out, std::uint32_t value)
            {
                char bytes[4];
                for (int i = 0; i < 4; ++i)
                    bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFFu);
                out.write(bytes, 4);
            }

            /** @brief Writes @p value as eight little-endian bytes, two's complement. */
            void put_i64(std::ostream& out, std::int64_t value)
            {
                const std::uint64_t bits = static_cast<std::uint64_t>(value);
                char bytes[8];
                for (int i = 0; i < 8; ++i)
                    bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFFu);
                out.write(bytes, 8);
            }

            /**
             * @brief Reads four little-endian bytes.
             * @param what Names what is being read, in the failure's message.
             * @return The decoded value, or InvalidFile when the stream ends first.
             */
            [[nodiscard]] Result<std::uint32_t> get_u32(std::istream& in, std::string_view what)
            {
                char bytes[4];
                in.read(bytes, 4);
                SA_FAIL_IF(!in, StatusCode::InvalidFile,
                           "the checkpoint ended while reading {}", what);

                std::uint32_t value = 0;
                for (int i = 0; i < 4; ++i)
                    value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i]))
                             << (8 * i);
                return value;
            }

            /**
             * @brief Reads eight little-endian bytes as a signed value.
             * @param what Names what is being read, in the failure's message.
             * @return The decoded value, or InvalidFile when the stream ends first.
             */
            [[nodiscard]] Result<std::int64_t> get_i64(std::istream& in, std::string_view what)
            {
                char bytes[8];
                in.read(bytes, 8);
                SA_FAIL_IF(!in, StatusCode::InvalidFile,
                           "the checkpoint ended while reading {}", what);

                std::uint64_t bits = 0;
                for (int i = 0; i < 8; ++i)
                    bits |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                            << (8 * i);
                return static_cast<std::int64_t>(bits);
            }

            /**
             * @brief Reads a length and that many bytes of a name.
             * @param subject Names what declares the name, in the message of the bound.
             * @param what Names the name, in the message of a stream that ends first.
             * @return The name; InvalidFile for a length above @ref K_MAX_NAME_LENGTH and
             *         for a stream that ends first.
             */
            [[nodiscard]] Result<std::string> get_name(std::istream& in,
                                                       std::string_view subject,
                                                       std::string_view what)
            {
                SA_TRY_ASSIGN(const std::uint32_t length,
                              get_u32(in, std::string(what) + " length"));
                SA_FAIL_IF(length > K_MAX_NAME_LENGTH, StatusCode::InvalidFile,
                           "{} declares a {}-byte name; at most {} are allowed",
                           subject, length, K_MAX_NAME_LENGTH);

                std::string name;
                SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
                {
                    name.resize(length);
                    return Status();
                }));
                if (length != 0)
                {
                    in.read(name.data(), static_cast<std::streamsize>(length));
                    SA_FAIL_IF(!in, StatusCode::InvalidFile,
                               "the checkpoint ended while reading {}", what);
                }

                return std::move(name);
            }

            /** @brief Reads what a version 1 header holds after its version. */
            [[nodiscard]] Result<HeaderV1> read_header_v1(std::istream& in)
            {
                HeaderV1 header;
                SA_TRY_ASSIGN(header.count, get_u32(in, "the tensor count"));
                return header;
            }

            /** @brief Reads what a version 2 header holds after its version. */
            [[nodiscard]] Result<HeaderV2> read_header_v2(std::istream& in)
            {
                HeaderV2 header;
                SA_TRY_ASSIGN(header.count, get_u32(in, "the tensor count"));
                SA_TRY_ASSIGN(header.run_state_count, get_u32(in, "the run-state count"));
                return header;
            }

            /** @brief Returns a version 1 header as version 2 reads it: with no run state. */
            [[nodiscard]] HeaderV2 upgrade_header(const HeaderV1& header) noexcept
            {
                HeaderV2 upgraded;
                upgraded.count = header.count;
                upgraded.run_state_count = 0;
                return upgraded;
            }

            /** @brief Reads the header of @p version and upgrades it to the current one. */
            [[nodiscard]] Result<HeaderV2> read_header(std::istream& in, std::uint32_t version)
            {
                if (version == K_CHECKPOINT_FIRST_VERSION)
                {
                    SA_TRY_ASSIGN(const HeaderV1 first, read_header_v1(in));
                    return upgrade_header(first);
                }

                return read_header_v2(in);
            }

        } // namespace

        Result<CheckpointWriter> CheckpointWriter::create(std::ostream& out, std::uint32_t count,
                                                          std::uint32_t run_state_count)
        {
            SA_FAIL_IF(count > K_MAX_TENSOR_COUNT, StatusCode::InvalidDescription,
                       "a checkpoint may hold at most {} tensors; {} were declared",
                       K_MAX_TENSOR_COUNT, count);
            SA_FAIL_IF(run_state_count > K_MAX_RUN_STATE_COUNT, StatusCode::InvalidDescription,
                       "a checkpoint may hold at most {} run-state records; {} were declared",
                       K_MAX_RUN_STATE_COUNT, run_state_count);
            SA_TRY(check_host_byte_order());

            put_u32(out, K_CHECKPOINT_MAGIC);
            put_u32(out, K_CHECKPOINT_VERSION);
            put_u32(out, count);
            put_u32(out, run_state_count);
            SA_FAIL_IF(!out, StatusCode::InvalidFile, "failed writing the checkpoint header");

            return CheckpointWriter(out, count, run_state_count);
        }

        CheckpointWriter::CheckpointWriter(std::ostream& out, std::uint32_t count,
                                           std::uint32_t run_state_count) noexcept
            : out_(&out), declared_(count), run_state_declared_(run_state_count)
        {
        }

        Status CheckpointWriter::begin_run_state(std::string_view owner)
        {
            SA_FATAL_IF(in_run_state_, "the run-state section of this checkpoint is already open");
            SA_FATAL_IF(written_ != declared_,
                        "this checkpoint declared {} records but {} were written before its "
                        "run-state section was opened", declared_, written_);
            SA_FATAL_IF(run_state_declared_ == 0,
                        "this checkpoint declared no run-state record, and its run-state "
                        "section was opened for '{}'", owner);
            SA_FAIL_IF(owner.empty() || owner.size() > K_MAX_NAME_LENGTH,
                       StatusCode::InvalidDescription,
                       "the owner of a run-state section is {} bytes long; it may be 1 to {}",
                       owner.size(), K_MAX_NAME_LENGTH);

            put_u32(*out_, static_cast<std::uint32_t>(owner.size()));
            out_->write(owner.data(), static_cast<std::streamsize>(owner.size()));
            SA_FAIL_IF(!*out_, StatusCode::InvalidFile,
                       "failed writing the run-state owner '{}' to the checkpoint", owner);

            in_run_state_ = true;
            return Status();
        }

        Status CheckpointWriter::write(const TensorRecord& record, const void* payload)
        {
            SA_FATAL_IF(!in_run_state_ && written_ >= declared_,
                        "this checkpoint declared {} records and a further one "
                        "('{}') was written", declared_, record.name);
            SA_FATAL_IF(in_run_state_ && run_state_written_ >= run_state_declared_,
                        "this checkpoint declared {} run-state records and a further one "
                        "('{}') was written", run_state_declared_, record.name);
            SA_FATAL_IF(payload == nullptr,
                        "'{}' was written with no payload", record.name);
            SA_FAIL_IF(record.name.size() > K_MAX_NAME_LENGTH, StatusCode::InvalidDescription,
                       "'{}...' is {} bytes long; a checkpoint name may be at most {}",
                       record.name.substr(0, 32), record.name.size(), K_MAX_NAME_LENGTH);
            SA_FATAL_IF(record.shape.rank() > K_MAX_RANK,
                        "'{}' is rank {}; a checkpoint tensor may be at most rank {}",
                        record.name, record.shape.rank(), K_MAX_RANK);

            put_u32(*out_, static_cast<std::uint32_t>(record.name.size()));
            out_->write(record.name.data(), static_cast<std::streamsize>(record.name.size()));
            put_u32(*out_, static_cast<std::uint32_t>(record.dtype));
            put_u32(*out_, static_cast<std::uint32_t>(record.shape.rank()));
            for (int32_t i = 0; i < record.shape.rank(); ++i)
                put_i64(*out_, record.shape[i]);

            out_->write(static_cast<const char*>(payload),
                        static_cast<std::streamsize>(record.payload_bytes()));
            SA_FAIL_IF(!*out_, StatusCode::InvalidFile,
                       "failed writing '{}' to the checkpoint", record.name);

            if (in_run_state_)
                ++run_state_written_;
            else
                ++written_;

            return Status();
        }

        void CheckpointWriter::finish() const
        {
            SA_FATAL_IF(written_ != declared_,
                        "this checkpoint declared {} records but {} were written",
                        declared_, written_);
            SA_FATAL_IF(run_state_written_ != run_state_declared_,
                        "this checkpoint declared {} run-state records but {} were written",
                        run_state_declared_, run_state_written_);
        }

        Result<CheckpointReader> CheckpointReader::open(std::istream& in)
        {
            SA_TRY(check_host_byte_order());

            SA_TRY_ASSIGN(const std::uint32_t magic, get_u32(in, "the magic number"));
            SA_FAIL_IF(magic != K_CHECKPOINT_MAGIC, StatusCode::InvalidFile,
                       "this is not a SushiAI checkpoint: expected the magic bytes "
                       "'SACP' (0x{:08X}), got 0x{:08X}", K_CHECKPOINT_MAGIC, magic);

            SA_TRY_ASSIGN(const std::uint32_t version, get_u32(in, "the version"));
            SA_FAIL_IF(version < K_CHECKPOINT_FIRST_VERSION || version > K_CHECKPOINT_VERSION,
                       StatusCode::InvalidFile,
                       "this checkpoint is version {}; this build reads versions {} to {}",
                       version, K_CHECKPOINT_FIRST_VERSION, K_CHECKPOINT_VERSION);

            SA_TRY_ASSIGN(const HeaderV2 header, read_header(in, version));
            SA_FAIL_IF(header.count > K_MAX_TENSOR_COUNT, StatusCode::InvalidFile,
                       "this checkpoint declares {} tensors; at most {} are allowed",
                       header.count, K_MAX_TENSOR_COUNT);
            SA_FAIL_IF(header.run_state_count > K_MAX_RUN_STATE_COUNT, StatusCode::InvalidFile,
                       "this checkpoint declares {} run-state records; at most {} are allowed",
                       header.run_state_count, K_MAX_RUN_STATE_COUNT);

            return CheckpointReader(in, version, header.count, header.run_state_count);
        }

        CheckpointReader::CheckpointReader(std::istream& in, std::uint32_t version,
                                           std::uint32_t count,
                                           std::uint32_t run_state_count) noexcept
            : in_(&in), version_(version), count_(count), run_state_count_(run_state_count)
        {
        }

        Result<std::string> CheckpointReader::begin_run_state()
        {
            SA_FATAL_IF(in_run_state_, "the run-state section of this checkpoint is already open");
            SA_FATAL_IF(read_ != count_,
                        "this checkpoint declared {} records and {} were read before its "
                        "run-state section was opened", count_, read_);
            SA_FATAL_IF(run_state_count_ == 0,
                        "this checkpoint declares no run-state record, and its run-state "
                        "section was opened");

            SA_TRY_ASSIGN(std::string owner,
                          get_name(*in_, "a checkpoint's run-state section",
                                   "the run-state owner's name"));
            SA_FAIL_IF(owner.empty(), StatusCode::InvalidFile,
                       "the run-state section of this checkpoint names no owner");

            in_run_state_ = true;
            return std::move(owner);
        }

        Result<TensorRecord> CheckpointReader::next()
        {
            SA_FATAL_IF(!in_run_state_ && read_ >= count_,
                        "this checkpoint declared {} records and a further one was requested",
                        count_);
            SA_FATAL_IF(in_run_state_ && run_state_read_ >= run_state_count_,
                        "this checkpoint declared {} run-state records and a further one was "
                        "requested", run_state_count_);

            TensorRecord record;
            SA_TRY_ASSIGN(record.name, get_name(*in_, "a checkpoint record", "a record's name"));

            SA_TRY_ASSIGN(const std::uint32_t encoded_dtype,
                          get_u32(*in_, "'" + record.name + "'s dtype"));
            SA_FAIL_IF(!is_known_dtype(encoded_dtype), StatusCode::InvalidFile,
                       "'{}' declares dtype {}, which this build does not know",
                       record.name, encoded_dtype);
            record.dtype = static_cast<DType>(encoded_dtype);

            SA_TRY_ASSIGN(const std::uint32_t rank, get_u32(*in_, "'" + record.name + "'s rank"));
            SA_FAIL_IF(rank > static_cast<std::uint32_t>(K_MAX_RANK), StatusCode::InvalidFile,
                       "'{}' declares rank {}; at most {} is allowed",
                       record.name, rank, K_MAX_RANK);

            // rank is at most K_MAX_RANK and every extent is positive when the Shape is built.
            std::array<std::int64_t, static_cast<std::size_t>(K_MAX_RANK)> dims{};
            for (std::uint32_t i = 0; i < rank; ++i)
            {
                SA_TRY_ASSIGN(const std::int64_t extent,
                              get_i64(*in_, "'" + record.name + "'s extents"));
                SA_FAIL_IF(extent <= 0, StatusCode::InvalidFile,
                           "'{}' declares extent {} for dimension {}; "
                           "every extent must be positive",
                           record.name, extent, i);
                dims[i] = extent;
            }
            record.shape = Shape(SushiRuntime::span<const int64_t>(dims.data(), rank));

            if (in_run_state_)
                ++run_state_read_;
            else
                ++read_;

            return record;
        }

        Status CheckpointReader::read_payload(const TensorRecord& record, void* destination)
        {
            SA_FATAL_IF(destination == nullptr,
                        "'{}' was read with no destination", record.name);

            in_->read(static_cast<char*>(destination),
                      static_cast<std::streamsize>(record.payload_bytes()));
            SA_FAIL_IF(!*in_, StatusCode::InvalidFile,
                       "the checkpoint ended while reading '{}'s {} payload bytes",
                       record.name, record.payload_bytes());

            return Status();
        }

    } // namespace IO
} // namespace SushiAI
