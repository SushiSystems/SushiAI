/**************************************************************************/
/* darknet_tensors.cpp                                                    */
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
 * @file darknet_tensors.cpp
 * @brief Implements the listing of Darknet convolution vectors as named host tensors.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_tensors.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

#include "darknet_weight_layout.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Names one of a convolution's vectors and the target that names its entry. */
            struct Field
            {
                /** @brief Holds the entry name; empty means the vector is not listed. */
                const std::string& name;

                /** @brief Holds the convolution's vector. */
                const std::vector<float>& values;
            };

            /** @brief Returns the four per-filter vectors of @p convolution with their targets. */
            [[nodiscard]] std::array<Field, 4> list_filter_fields(
                const DarknetConvolution& convolution, const DarknetWeightTarget& target) noexcept
            {
                return {{
                    {target.bias, convolution.biases},
                    {target.scale, convolution.scales},
                    {target.mean, convolution.rolling_mean},
                    {target.variance, convolution.rolling_variance},
                }};
            }

            /**
             * @brief Checks one vector against the count its layer gives it and claims its name.
             * @return Incompatible naming the layer for another length or a name stated twice.
             */
            [[nodiscard]] Status claim_field(const Field& field, std::size_t expected,
                                             std::int64_t layer,
                                             std::unordered_set<std::string_view>& claimed)
            {
                if (field.name.empty())
                    return Status();

                SA_FAIL_IF(field.values.size() != expected, StatusCode::Incompatible,
                           "convolution layer {} has {} values for '{}', which holds {}", layer,
                           field.values.size(), field.name, expected);
                SA_FAIL_IF(!claimed.insert(field.name).second, StatusCode::Incompatible,
                           "convolution layer {} fills '{}', which an earlier target already "
                           "fills",
                           layer, field.name);
                return Status();
            }

            /** @brief Returns a float32 tensor named @p name that holds a copy of @p values. */
            [[nodiscard]] IO::ImportedTensor make_tensor(const std::string& name,
                                                         const Shape& shape,
                                                         const std::vector<float>& values)
            {
                IO::ImportedTensor tensor;
                tensor.record.name = name;
                tensor.record.dtype = DType::Float32;
                tensor.record.shape = shape;
                tensor.payload.resize(values.size() * sizeof(float));
                if (!values.empty())
                    std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());

                return tensor;
            }

            /**
             * @brief Checks one convolution against its shape and claims its target names.
             * @param claimed Receives every name that is not empty; a repeat is refused.
             * @return InvalidDescription for a shape that is not positive; Incompatible naming
             *         the layer for a vector of another length or a name stated twice.
             */
            [[nodiscard]] Status validate_convolution(const DarknetConvolution& convolution,
                                                      const DarknetConvolutionShape& shape,
                                                      const DarknetWeightTarget& target,
                                                      std::unordered_set<std::string_view>& claimed)
            {
                SA_FAIL_IF(shape.filters <= 0 || shape.channels <= 0 || shape.size <= 0,
                           StatusCode::InvalidDescription,
                           "convolution layer {} has filters {}, channels {} and size {}; each "
                           "must be positive",
                           shape.layer, shape.filters, shape.channels, shape.size);

                const auto filters = static_cast<std::size_t>(shape.filters);
                const std::size_t kernel = static_cast<std::size_t>(shape.channels)
                                         * static_cast<std::size_t>(shape.size)
                                         * static_cast<std::size_t>(shape.size);
                SA_TRY(claim_field(Field{target.weight, convolution.weights}, filters * kernel,
                                   shape.layer, claimed));
                for (const Field& field : list_filter_fields(convolution, target))
                {
                    SA_TRY(claim_field(field, filters, shape.layer, claimed));
                }

                return Status();
            }

            /**
             * @brief Appends the tensors of one convolution to @p tensors.
             * @pre validate_convolution succeeds on the same three arguments.
             */
            void append_convolution(const DarknetConvolution& convolution,
                                    const DarknetConvolutionShape& shape,
                                    const DarknetWeightTarget& target,
                                    std::vector<IO::ImportedTensor>& tensors)
            {
                if (!target.weight.empty())
                {
                    std::vector<float> moved(convolution.weights.size());
                    Detail::copy_kernel_last(convolution.weights, shape, moved.data());
                    tensors.push_back(make_tensor(
                        target.weight,
                        Shape{shape.filters, shape.size, shape.size, shape.channels}, moved));
                }

                for (const Field& field : list_filter_fields(convolution, target))
                {
                    if (!field.name.empty())
                        tensors.push_back(
                            make_tensor(field.name, Shape{shape.filters}, field.values));
                }
            }

        } // namespace

        Result<std::vector<IO::ImportedTensor>> list_darknet_tensors(
            const DarknetWeights& weights,
            SushiRuntime::span<const DarknetConvolutionShape> shapes,
            SushiRuntime::span<const DarknetWeightTarget> targets)
        {
            SA_FAIL_IF(weights.convolutions.size() != shapes.size()
                           || shapes.size() != targets.size(),
                       StatusCode::Incompatible,
                       "the weights hold {} convolutions, the shapes {} and the targets {}; the "
                       "three must match",
                       weights.convolutions.size(), shapes.size(), targets.size());

            std::size_t count = 0;
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                std::unordered_set<std::string_view> claimed;
                for (std::size_t i = 0; i < shapes.size(); ++i)
                {
                    SA_TRY(validate_convolution(weights.convolutions[i], shapes[i], targets[i],
                                                claimed));
                }

                count = claimed.size();
                return Status();
            }));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<IO::ImportedTensor> tensors;
                tensors.reserve(count);
                for (std::size_t i = 0; i < shapes.size(); ++i)
                    append_convolution(weights.convolutions[i], shapes[i], targets[i], tensors);

                return tensors;
            });
        }

    } // namespace Vision::Importers
} // namespace SushiAI
