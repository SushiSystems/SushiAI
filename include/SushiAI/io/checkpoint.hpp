/**************************************************************************/
/* checkpoint.hpp                                                         */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file checkpoint.hpp
 * @brief Binding a checkpoint's records to a module's parameters.
 *
 * Device lane: this half knows about @c TensorPool and @c SushiBLAS::Tensor.
 * @c io/checkpoint_format.hpp is the other half - the byte stream alone, with
 * no tensors in it, which is what lets the whole malformed-file surface be a
 * device-free unit test.
 *
 * @section walk One walk, so nothing can disagree
 *
 * Both directions go through @c NN::for_each_parameter, which is the same walk
 * the optimizer and the initializer use. So a checkpoint cannot contain a
 * different set of tensors from the one being trained, and - because the walk
 * has a runtime arm - a @c LayerStack built from a model.json is saved and
 * loaded by this code with no change at all.
 *
 * @section defects What a naive loader gets wrong
 *
 * Each of these is closed by making the wrong state unrepresentable rather
 * than by adding a check beside it:
 *
 * - **Counting records instead of filled parameters.** A file with one
 *   duplicated name and one missing name has the right count. The parameter
 *   nothing filled then reads uninitialised device memory, because
 *   @c allocate_tensor does not zero - and a forward pass over it produces
 *   numbers, not a crash. A @c std::vector<bool> of filled slots is what makes
 *   the count meaningless.
 * - **Comparing element counts instead of shapes.** [256, 6] loads happily
 *   into [6, 256]; the model still runs and is wrong. The format records a
 *   @c Shape and both are compared.
 * - **Assuming FLOAT32.** @c data_as<float>() over half data reads two
 *   elements as one. The dtype is recorded and checked.
 * - **Sizing a buffer from the file.** There is no intermediate buffer: the
 *   payload is read straight into the destination tensor once its record has
 *   been checked against it. See checkpoint_format.hpp.
 */

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
             * @brief Collects @p module's parameters and their tensors, in walk order.
             * @param module The model; must already have been traced.
             * @param pool Where the parameters' tensors live.
             * @return One entry per declared parameter.
             * @throws Error If a parameter has no tensor bound, or its tensor is
             *         not contiguous - a checkpoint records elements in layout
             *         order and a strided view would be written with gaps.
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
         * @brief Writes every declared parameter of @p module, by name.
         *
         * Waits for the device first, so what reaches the file is what the last
         * step produced rather than whatever had been flushed by chance.
         *
         * @param path Where to write.
         * @param module The trained model; must already have been traced.
         * @param pool Where the parameters' tensors live.
         * @param engine The engine to flush before reading the parameters.
         * @throws Error If the file cannot be opened, a parameter has no tensor,
         *         or the write fails.
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
         * @brief Loads @p path into @p module's already-materialised parameters.
         *
         * Every parameter the walk yields must be supplied exactly once. A name
         * the model does not have, a name supplied twice, a parameter the file
         * does not supply, and a dtype or Shape that disagrees are each an
         * error naming what is wrong - see this file's header for why each of
         * them would otherwise be silent.
         *
         * @param path The checkpoint.
         * @param module A traced model of the same architecture.
         * @param pool Where the parameters' tensors were materialised.
         * @throws Error On any of the conditions above, or an I/O failure.
         */
        template<typename M>
        void load_checkpoint(const std::filesystem::path& path, M& module,
                             Graph::TensorPool& pool)
        {
            std::vector<Detail::BoundParameter> bound = Detail::bind(module, pool);

            std::ifstream in(path, std::ios::binary);
            SA_THROW_IF(!in, "could not open '{}' for reading", path.string());

            // Filled slots rather than a record count: a file with one
            // duplicated name and one missing one has the right count, and the
            // parameter nothing filled would then read uninitialised device
            // memory.
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

                // Straight into the destination, which is why nothing here is
                // sized from the file.
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
