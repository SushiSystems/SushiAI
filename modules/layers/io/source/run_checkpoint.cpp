/**************************************************************************/
/* run_checkpoint.cpp                                                     */
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
 * @file run_checkpoint.cpp
 * @brief Writes and loads checkpoints that hold an optimizer's run state.
 * @author Mustafa Garip
 */

#include <SushiAI/io/run_checkpoint.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/optim/run_state.hpp>
#include "model_records.hpp"

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Counts the bytes of the payload of a scalar record. */
            constexpr std::size_t K_SCALAR_BYTES = 8;

            /** @brief Holds the payload of a scalar record. */
            using ScalarBytes = std::array<char, K_SCALAR_BYTES>;

            /** @brief Returns the element type a scalar is written as: Int64 or Float64. */
            [[nodiscard]] DType dtype_of(const Optim::StateScalar& scalar) noexcept
            {
                return std::holds_alternative<double>(scalar.value) ? DType::Float64
                                                                    : DType::Int64;
            }

            /** @brief Returns a scalar as 64 bits: a count as it is, a real as IEEE 754. */
            [[nodiscard]] std::uint64_t bits_of(const Optim::StateScalar& scalar) noexcept
            {
                if (const std::uint64_t* const count = std::get_if<std::uint64_t>(&scalar.value))
                    return *count;

                std::uint64_t bits = 0;
                std::memcpy(&bits, std::get_if<double>(&scalar.value), sizeof(bits));
                return bits;
            }

            /** @brief Returns @p bits as eight bytes, the low byte first. */
            [[nodiscard]] ScalarBytes encode(std::uint64_t bits) noexcept
            {
                ScalarBytes bytes{};
                for (std::size_t i = 0; i < K_SCALAR_BYTES; ++i)
                    bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFFu);

                return bytes;
            }

            /** @brief Returns the 64 bits whose low byte @p bytes holds first. */
            [[nodiscard]] std::uint64_t decode(const ScalarBytes& bytes) noexcept
            {
                std::uint64_t bits = 0;
                for (std::size_t i = 0; i < K_SCALAR_BYTES; ++i)
                    bits |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
                            << (8 * i);

                return bits;
            }

            /** @brief Sets the value of @p scalar from 64 bits, keeping which kind it holds. */
            void set_bits(Optim::StateScalar& scalar, std::uint64_t bits) noexcept
            {
                if (std::holds_alternative<double>(scalar.value))
                {
                    double real = 0.0;
                    std::memcpy(&real, &bits, sizeof(real));
                    scalar.value = real;
                    return;
                }

                scalar.value = bits;
            }

            /** @brief Describes @p scalar as a record of one element. */
            [[nodiscard]] TensorRecord record_of(const Optim::StateScalar& scalar)
            {
                TensorRecord record;
                record.name = scalar.name;
                record.dtype = dtype_of(scalar);
                record.shape = Shape{1};
                return record;
            }

            /** @brief Writes the scalars and then the tensors of @p run as records. */
            [[nodiscard]] Status write_run_records(CheckpointWriter& writer, Optim::RunState& run)
            {
                for (const Optim::StateScalar& scalar : run.scalars)
                {
                    const ScalarBytes bytes = encode(bits_of(scalar));
                    SA_TRY(writer.write(record_of(scalar), bytes.data()));
                }

                for (Optim::StateTensor& held : run.tensors)
                {
                    SA_TRY(writer.write(Detail::record_for(held.name, held.tensor),
                                        held.tensor.data()));
                }

                return Status();
            }

            /** @brief Pairs the name of a progress record with the field it holds. */
            struct ProgressField
            {
                std::string_view name;               /**< Names the record. */
                std::uint64_t RunProgress::* member; /**< Points at the field. */
            };

            /** @brief Lists the progress records in the order a file holds them. */
            constexpr ProgressField K_PROGRESS_FIELDS[] = {
                {K_RUN_SEED_STATE, &RunProgress::seed},
                {K_RUN_STEP_RUNS_STATE, &RunProgress::step_runs},
                {K_RUN_EPOCH_STATE, &RunProgress::epoch},
                {K_RUN_BATCH_STATE, &RunProgress::batch},
            };

            /** @brief Counts the progress records every run-state section holds. */
            constexpr std::size_t K_PROGRESS_RECORDS =
                sizeof(K_PROGRESS_FIELDS) / sizeof(K_PROGRESS_FIELDS[0]);

            /** @brief Holds the largest epoch index a dataset can be given. */
            constexpr std::uint64_t K_MAX_EPOCH =
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

            /** @brief Writes each field of @p progress as an Int64 record of one element. */
            [[nodiscard]] Status write_progress_records(CheckpointWriter& writer,
                                                        const RunProgress& progress)
            {
                for (const ProgressField& field : K_PROGRESS_FIELDS)
                {
                    TensorRecord record;
                    record.name = std::string(field.name);
                    record.dtype = DType::Int64;
                    record.shape = Shape{1};

                    const ScalarBytes bytes = encode(progress.*field.member);
                    SA_TRY(writer.write(record, bytes.data()));
                }

                return Status();
            }

            /** @brief Names a file and the optimizer whose run state it is read into. */
            struct Subject
            {
                const std::string& file;    /**< Names the file. */
                std::string_view optimizer; /**< Names the optimizer type. */
            };

            /**
             * @brief Reads the payload of @p record into the tensor it names.
             * @return Incompatible for a record of another type or shape than the tensor.
             */
            [[nodiscard]] Status read_tensor(CheckpointReader& reader, const TensorRecord& record,
                                             SushiBLAS::Tensor& tensor, const Subject& subject)
            {
                SA_FAIL_IF(record.dtype != tensor.dtype, StatusCode::Incompatible,
                           "'{}' holds the run-state record '{}' as {}, but this '{}' "
                           "optimizer's is {}",
                           subject.file, record.name, dtype_name(record.dtype),
                           subject.optimizer, dtype_name(tensor.dtype));

                const Shape expected = shape_of(tensor);
                SA_FAIL_IF(record.shape != expected, StatusCode::Incompatible,
                           "'{}' holds the run-state record '{}' with shape {}, but this '{}' "
                           "optimizer's is {}",
                           subject.file, record.name, record.shape.to_string(),
                           subject.optimizer, expected.to_string());

                return reader.read_payload(record, tensor.data());
            }

            /**
             * @brief Reads the payload of @p record into the scalar it names.
             * @return Incompatible for a record that is not one element of the scalar's type.
             */
            [[nodiscard]] Status read_scalar(CheckpointReader& reader, const TensorRecord& record,
                                             Optim::StateScalar& scalar, const Subject& subject)
            {
                SA_FAIL_IF(record.dtype != dtype_of(scalar) || record.shape != Shape{1},
                           StatusCode::Incompatible,
                           "'{}' holds the run-state record '{}' as {} of shape {}, but this "
                           "'{}' optimizer's is {} of shape [1]",
                           subject.file, record.name, dtype_name(record.dtype),
                           record.shape.to_string(), subject.optimizer,
                           dtype_name(dtype_of(scalar)));

                ScalarBytes bytes{};
                SA_TRY(reader.read_payload(record, bytes.data()));
                set_bits(scalar, decode(bytes));
                return Status();
            }

            /**
             * @brief Reads the payload of @p record into a field of a run's progress.
             * @return Incompatible for a record that is not one Int64 element.
             */
            [[nodiscard]] Status read_progress(CheckpointReader& reader,
                                               const TensorRecord& record, std::uint64_t& field,
                                               const Subject& subject)
            {
                SA_FAIL_IF(record.dtype != DType::Int64 || record.shape != Shape{1},
                           StatusCode::Incompatible,
                           "'{}' holds the run-state record '{}' as {} of shape {}, but the "
                           "progress of a run is {} of shape [1]",
                           subject.file, record.name, dtype_name(record.dtype),
                           record.shape.to_string(), dtype_name(DType::Int64));

                ScalarBytes bytes{};
                SA_TRY(reader.read_payload(record, bytes.data()));
                field = decode(bytes);
                return Status();
            }

            /**
             * @brief Reads every run-state record into the tensor, scalar or field it names.
             * @param run The optimizer's own listing; its scalars take the values read.
             * @param progress Takes the four progress records.
             * @return Incompatible for a record nothing keeps or the file repeats, and for a
             *         tensor, a scalar or a progress record the file does not supply.
             */
            [[nodiscard]] Status read_run_records(CheckpointReader& reader, Optim::RunState& run,
                                                  RunProgress& progress, const Subject& subject)
            {
                std::unordered_set<std::string> supplied;
                while (reader.run_state_read_count() < reader.run_state_count())
                {
                    SA_TRY_ASSIGN(const TensorRecord record, reader.next());
                    SA_FAIL_IF(!supplied.insert(record.name).second, StatusCode::Incompatible,
                               "'{}' supplies the run-state record '{}' twice", subject.file,
                               record.name);

                    bool is_read = false;
                    for (Optim::StateTensor& held : run.tensors)
                    {
                        if (held.name != record.name)
                            continue;

                        SA_TRY(read_tensor(reader, record, held.tensor, subject));
                        is_read = true;
                    }

                    for (Optim::StateScalar& scalar : run.scalars)
                    {
                        if (scalar.name != record.name)
                            continue;

                        SA_TRY(read_scalar(reader, record, scalar, subject));
                        is_read = true;
                    }

                    for (const ProgressField& field : K_PROGRESS_FIELDS)
                    {
                        if (field.name != record.name)
                            continue;

                        SA_TRY(read_progress(reader, record, progress.*field.member, subject));
                        is_read = true;
                    }

                    SA_FAIL_IF(!is_read, StatusCode::Incompatible,
                               "'{}' holds the run-state record '{}', which this '{}' "
                               "optimizer does not keep",
                               subject.file, record.name, subject.optimizer);
                }

                for (const Optim::StateTensor& held : run.tensors)
                {
                    SA_FAIL_IF(supplied.count(held.name) == 0, StatusCode::Incompatible,
                               "'{}' does not supply the run-state record '{}' of this '{}' "
                               "optimizer",
                               subject.file, held.name, subject.optimizer);
                }

                for (const Optim::StateScalar& scalar : run.scalars)
                {
                    SA_FAIL_IF(supplied.count(scalar.name) == 0, StatusCode::Incompatible,
                               "'{}' does not supply the run-state record '{}' of this '{}' "
                               "optimizer",
                               subject.file, scalar.name, subject.optimizer);
                }

                for (const ProgressField& field : K_PROGRESS_FIELDS)
                {
                    SA_FAIL_IF(supplied.count(std::string(field.name)) == 0,
                               StatusCode::Incompatible,
                               "'{}' does not supply the run-state record '{}'; a checkpoint "
                               "without the seed and the position of its run cannot be "
                               "resumed, and it can be the 'weights' a run starts from",
                               subject.file, field.name);
                }

                SA_FAIL_IF(progress.epoch > K_MAX_EPOCH, StatusCode::InvalidFile,
                           "'{}' holds the run-state record '{}' of {}, above the largest "
                           "epoch index {}",
                           subject.file, K_RUN_EPOCH_STATE, progress.epoch, K_MAX_EPOCH);

                return Status();
            }

        } // namespace

        Status save_run_checkpoint(const std::filesystem::path& path, NN::ModelState& state,
                                   Optim::Optimizer& optimizer, SushiBLAS::Engine& engine,
                                   const RunProgress& progress)
        {
            SA_TRY(SushiAI::Detail::run_recorded(engine));

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                SA_TRY_ASSIGN(Optim::RunState run, optimizer.run_state());

                std::ofstream out(path, std::ios::binary);
                SA_FAIL_IF(!out, StatusCode::InvalidFile, "could not open '{}' for writing",
                           path.string());

                SA_TRY_ASSIGN(CheckpointWriter writer,
                              CheckpointWriter::create(
                                  out, static_cast<std::uint32_t>(state.entries().size()),
                                  static_cast<std::uint32_t>(run.scalars.size() +
                                                             run.tensors.size() +
                                                             K_PROGRESS_RECORDS)));
                SA_TRY(Detail::write_model_records(writer, state));
                SA_TRY(writer.begin_run_state(optimizer.type_name()));
                SA_TRY(write_run_records(writer, run));
                SA_TRY(write_progress_records(writer, progress));
                writer.finish();

                SA_FAIL_IF(!out, StatusCode::InvalidFile,
                           "failed writing the checkpoint to '{}'", path.string());
                return Status();
            });
        }

        Status load_run_checkpoint(const std::filesystem::path& path, NN::ModelState& state,
                                   Optim::Optimizer& optimizer, RunProgress* progress)
        {
            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                const std::string file = path.string();
                std::ifstream in(path, std::ios::binary);
                SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                           file);

                SA_TRY_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
                SA_TRY(Detail::read_model_records(reader, path, state, {}));
                SA_FAIL_IF(reader.run_state_count() == 0, StatusCode::Incompatible,
                           "'{}' is a version {} checkpoint with model state and no run state, "
                           "so a run cannot resume from it; it can be the 'weights' a run "
                           "starts from",
                           file, reader.version());

                SA_TRY_ASSIGN(const std::string owner, reader.begin_run_state());
                SA_TRY(SA_ANNOTATE(optimizer.check_state_owner(owner), "'{}'", file));

                SA_TRY_ASSIGN(Optim::RunState run, optimizer.run_state());
                RunProgress read;
                SA_TRY(read_run_records(reader, run, read,
                                        Subject{file, optimizer.type_name()}));

                SA_TRY(SA_ANNOTATE(
                    optimizer.restore_scalars(SushiRuntime::span<const Optim::StateScalar>(
                        run.scalars.data(), run.scalars.size())),
                    "'{}'", file));

                if (progress != nullptr)
                    *progress = read;

                return Status();
            });
        }

    } // namespace IO
} // namespace SushiAI
