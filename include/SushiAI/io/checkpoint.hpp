/****************************************************************/
/* checkpoint.hpp                                               */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/lowering.hpp>
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
                    SA_THROW_IF(!tensor.is_contiguous(),
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
         * @param engine Engine flushed prior to reading tensor memory.
         */
        template<typename M>
        void save_checkpoint(const std::filesystem::path& path, M& module,
                             Graph::TensorPool& pool, SushiBLAS::Engine& engine)
        {
            engine.execute();
            engine.get_context().wait_all();

            const std::vector<Detail::BoundParameter> bound = Detail::bind(module, pool);

            std::ofstream out(path, std::ios::binary);
            SA_THROW_IF(!out, "could not open '{}' for writing", path.string());

            CheckpointWriter writer(out, static_cast<std::uint32_t>(bound.size()));
            for (const Detail::BoundParameter& entry : bound)
                writer.write(Detail::record_for(entry.name, *entry.tensor),
                             entry.tensor->data());
            writer.finish();

            SA_THROW_IF(!out, "failed writing the checkpoint to '{}'", path.string());
        }

        /**
         * @brief Loads parameter tensors from a checkpoint file into a module.
         * @param path Checkpoint path to read from disk.
         * @param module Traced model matching the checkpoint structure.
         * @param pool Storage pool where parameters are materialized.
         */
        template<typename M>
        void load_checkpoint(const std::filesystem::path& path, M& module,
                             Graph::TensorPool& pool)
        {
            std::vector<Detail::BoundParameter> bound = Detail::bind(module, pool);

            std::ifstream in(path, std::ios::binary);
            SA_THROW_IF(!in, "could not open '{}' for reading", path.string());

            // Tracking filled slots prevents duplicate or missing parameter bindings.
            std::vector<bool> filled(bound.size(), false);

            CheckpointReader reader(in);
            while (reader.read_count() < reader.count())
            {
                const TensorRecord record = reader.next();

                std::size_t index = bound.size();
                for (std::size_t i = 0; i < bound.size(); ++i)
                    if (bound[i].name == record.name) { index = i; break; }

                SA_THROW_IF(index == bound.size(),
                            "'{}' holds a parameter '{}', which this model does not have",
                            path.string(), record.name);
                SA_THROW_IF(filled[index],
                            "'{}' supplies '{}' twice", path.string(), record.name);

                const TensorRecord expected =
                    Detail::record_for(record.name, *bound[index].tensor);

                SA_THROW_IF(record.dtype != expected.dtype,
                            "'{}' holds '{}' as {}, but this model's is {}",
                            path.string(), record.name,
                            dtype_name(record.dtype), dtype_name(expected.dtype));
                SA_THROW_IF(record.shape != expected.shape,
                            "'{}' holds '{}' with shape {}, but this model's is {}",
                            path.string(), record.name,
                            record.shape.to_string(), expected.shape.to_string());

                // Payloads are streamed directly into allocated destination memory.
                reader.read_payload(record, bound[index].tensor->data());
                filled[index] = true;
            }

            for (std::size_t i = 0; i < bound.size(); ++i)
                SA_THROW_IF(!filled[i],
                            "'{}' does not supply this model's '{}'",
                            path.string(), bound[i].name);
        }

    } // namespace IO
} // namespace SushiAI
