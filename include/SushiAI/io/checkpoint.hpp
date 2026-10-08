/**************************************************************************/
/* checkpoint.hpp                                                         */
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
 * @file checkpoint.hpp
 * @brief Binds checkpoint records to module parameters for save and load.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace Detail
        {
            /**
             * @struct BoundParameter
             * @brief One parameter's name and the tensor behind it.
             */
            struct BoundParameter
            {
                /** @brief The dotted name the walk produced. */
                std::string name;

                /** @brief The tensor the pool holds for it. */
                SushiBLAS::Tensor* tensor = nullptr;
            };

            /**
             * @brief Collects module parameters and bound tensors in walk order.
             * @param module Traced model whose parameters are traversed.
             * @param pool Tensor pool storing parameter storage.
             * @return Vector of bound parameter records.
             */
            template<typename M>
            std::vector<BoundParameter> bind(M& module, Graph::TensorPool& pool)
            {
                std::vector<BoundParameter> bound;
                NN::for_each_parameter(module, [&](const std::string& name,
                                                   NN::Parameter& parameter)
                {
                    SushiBLAS::Tensor& tensor = pool.tensor(parameter.id());
                    SA_FATAL_IF(!tensor.is_contiguous(),
                                "'{}' is a strided view, and a checkpoint records its "
                                "elements in layout order", name);
                    bound.push_back(BoundParameter{name, &tensor});
                });
                return bound;
            }

            /**
             * @brief Describes @p tensor as a checkpoint record called @p name.
             * @param name The dotted parameter name.
             * @param tensor The tensor to describe.
             * @return The record.
             */
            inline TensorRecord record_for(const std::string& name,
                                           const SushiBLAS::Tensor& tensor)
            {
                TensorRecord record;
                record.name = name;
                record.dtype = tensor.dtype;
                record.shape = Shape(SushiRuntime::span<const int64_t>(
                    tensor.shape.data(), static_cast<std::size_t>(tensor.rank)));
                return record;
            }
        }

        /**
         * @brief Writes every declared parameter of the module to disk.
         * @param path Target filesystem path for the checkpoint.
         * @param module Traced model providing declared parameters.
         * @param pool Storage pool containing parameter tensors.
         * @param engine Engine whose recorded tasks run before the tensor memory is read.
         * @return Success, or InvalidFile when the file cannot be written.
         */
        template<typename M>
        [[nodiscard]] Status save_checkpoint(const std::filesystem::path& path, M& module,
                                             Graph::TensorPool& pool,
                                             SushiBLAS::Engine& engine)
        {
            SA_TRY(SushiAI::Detail::run_recorded(engine));

            SA_TRY_ASSIGN(const std::vector<Detail::BoundParameter> bound,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return Detail::bind(module, pool);
            }));

            std::ofstream out(path, std::ios::binary);
            SA_FAIL_IF(!out, StatusCode::InvalidFile, "could not open '{}' for writing",
                       path.string());

            SA_TRY_ASSIGN(CheckpointWriter writer,
                          CheckpointWriter::create(out, static_cast<std::uint32_t>(bound.size())));
            for (const Detail::BoundParameter& entry : bound)
            {
                SA_TRY(writer.write(Detail::record_for(entry.name, *entry.tensor),
                                    entry.tensor->data()));
            }
            writer.finish();

            SA_FAIL_IF(!out, StatusCode::InvalidFile, "failed writing the checkpoint to '{}'",
                       path.string());
            return Status();
        }

        /**
         * @brief Loads parameter tensors from a checkpoint file into a module.
         * @param path Checkpoint path to read from disk.
         * @param module Traced model matching the checkpoint structure.
         * @param pool Storage pool where parameters are materialized.
         * @return Success, FileNotFound when the file cannot be opened, or Incompatible when
         *         a record does not match a parameter of the model.
         */
        template<typename M>
        [[nodiscard]] Status load_checkpoint(const std::filesystem::path& path, M& module,
                                             Graph::TensorPool& pool)
        {
            SA_TRY_ASSIGN(const std::vector<Detail::BoundParameter> bound,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return Detail::bind(module, pool);
            }));

            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());

            // Tracking filled slots prevents duplicate or missing parameter bindings.
            SA_TRY_ASSIGN(std::vector<bool> filled,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::vector<bool>(bound.size(), false);
            }));

            SA_TRY_ASSIGN(CheckpointReader reader, CheckpointReader::open(in));
            while (reader.read_count() < reader.count())
            {
                SA_TRY_ASSIGN(const TensorRecord record, reader.next());

                std::size_t index = bound.size();
                for (std::size_t i = 0; i < bound.size(); ++i)
                {
                    if (bound[i].name == record.name)
                    {
                        index = i;
                        break;
                    }
                }

                SA_FAIL_IF(index == bound.size(), StatusCode::Incompatible,
                           "'{}' holds a parameter '{}', which this model does not have",
                           path.string(), record.name);
                SA_FAIL_IF(filled[index], StatusCode::Incompatible,
                           "'{}' supplies '{}' twice", path.string(), record.name);

                const TensorRecord expected =
                    Detail::record_for(record.name, *bound[index].tensor);

                SA_FAIL_IF(record.dtype != expected.dtype, StatusCode::Incompatible,
                           "'{}' holds '{}' as {}, but this model's is {}",
                           path.string(), record.name,
                           dtype_name(record.dtype), dtype_name(expected.dtype));
                SA_FAIL_IF(record.shape != expected.shape, StatusCode::Incompatible,
                           "'{}' holds '{}' with shape {}, but this model's is {}",
                           path.string(), record.name,
                           record.shape.to_string(), expected.shape.to_string());

                // Payloads are streamed directly into allocated destination memory.
                SA_TRY(reader.read_payload(record, bound[index].tensor->data()));
                filled[index] = true;
            }

            for (std::size_t i = 0; i < bound.size(); ++i)
            {
                SA_FAIL_IF(!filled[i], StatusCode::Incompatible,
                           "'{}' does not supply this model's '{}'",
                           path.string(), bound[i].name);
            }
            return Status();
        }

    } // namespace IO
} // namespace SushiAI
