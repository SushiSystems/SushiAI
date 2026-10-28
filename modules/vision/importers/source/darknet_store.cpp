/**************************************************************************/
/* darknet_store.cpp                                                      */
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
 * @file darknet_store.cpp
 * @brief Defines the checks and the copy that fill a model state from Darknet convolutions.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_store.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string_view>
#include <vector>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

#include "darknet_weight_layout.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Names how a vector is written into its tensor. */
            enum class CopyKind : std::uint8_t
            {
                Vector,  /**< One value per filter, copied in order. */
                Weights, /**< Weights moved from [out, in, kh, kw] to [out, kh, kw, in]. */
            };

            /** @brief Holds one validated write: the tensor, the vector and its layout. */
            struct Transfer
            {
                /** @brief Holds the state tensor the write fills. */
                SushiBLAS::Tensor* tensor = nullptr;

                /** @brief Holds the vector the write reads; the caller keeps it alive. */
                const std::vector<float>* source = nullptr;

                /** @brief Holds how the vector is laid out in the tensor. */
                CopyKind kind = CopyKind::Vector;

                /** @brief Holds the layer, read for the weights' permutation only. */
                DarknetConvolutionShape shape;
            };

            /**
             * @brief Holds the writes a store has validated, and applies them only when all pass.
             */
            class StorePlan
            {
                public:
                    /**
                     * @brief Validates one vector against its state entry and records the write.
                     * @param shape The layer, named in a failure; @p source must outlive apply().
                     * @return Incompatible naming the layer and the entry when it does not fit.
                     */
                    [[nodiscard]] Status add(NN::ModelState& state,
                                             const DarknetConvolutionShape& shape,
                                             std::string_view name,
                                             const std::vector<float>& source,
                                             const Shape& expected, CopyKind kind)
                    {
                        const Result<SushiBLAS::Tensor*> found = state.tensor(name);
                        SA_FAIL_IF(!found.has_value(), StatusCode::Incompatible,
                                   "convolution layer {} names '{}', and this model's state has no "
                                   "such entry",
                                   shape.layer, name);
                        SushiBLAS::Tensor* tensor = found.value();

                        SA_FAIL_IF(!claimed_.insert(tensor).second, StatusCode::Incompatible,
                                   "convolution layer {} fills '{}', which an earlier target "
                                   "already fills",
                                   shape.layer, name);
                        SA_FAIL_IF(tensor->dtype != SushiBLAS::Core::DataType::Float32,
                                   StatusCode::Incompatible, "'{}' is not float32 in this model",
                                   name);

                        const Shape actual = shape_of(*tensor);
                        SA_FAIL_IF(actual != expected, StatusCode::Incompatible,
                                   "convolution layer {} fills '{}' with shape {}, but this "
                                   "model's is {}",
                                   shape.layer, name, expected.to_string(), actual.to_string());
                        SA_FAIL_IF(static_cast<std::int64_t>(source.size()) != tensor->num_elements,
                                   StatusCode::Incompatible,
                                   "convolution layer {} has {} values for '{}', which holds {}",
                                   shape.layer, source.size(), name, tensor->num_elements);
                        SA_FAIL_IF(tensor->data_or_null() == nullptr, StatusCode::Incompatible,
                                   "'{}' has no storage to receive a convolution", name);

                        transfers_.push_back(Transfer{tensor, &source, kind, shape});
                        return Status();
                    }

                    /** @brief Copies every recorded vector into its tensor; cannot fail. */
                    void apply() const
                    {
                        for (const Transfer& transfer : transfers_)
                        {
                            float* destination = transfer.tensor->data_as<float>();
                            if (transfer.kind == CopyKind::Weights)
                                Detail::copy_kernel_last(*transfer.source, transfer.shape,
                                                         destination);
                            else
                                std::copy(transfer.source->begin(), transfer.source->end(),
                                          destination);
                        }
                    }

                private:
                    /** @brief Holds the validated writes, in the order they were added. */
                    std::vector<Transfer> transfers_;

                    /** @brief Holds the tensors already claimed, so no entry is filled twice. */
                    std::set<const SushiBLAS::Tensor*> claimed_;
            };

            /** @brief Names one of a convolution's vectors and the target that names its entry. */
            struct Field
            {
                /** @brief Holds the entry name; empty means the vector is not stored. */
                const std::string& name;

                /** @brief Holds the convolution's vector. */
                const std::vector<float>& values;
            };

            /**
             * @brief Plans the writes of one convolution, one per target name that is not empty.
             * @param state The state the entries are looked up in.
             * @param plan Receives the validated writes.
             * @return InvalidDescription for a shape that is not positive; otherwise the first
             *         failure of StorePlan::add().
             */
            Status plan_convolution(const DarknetConvolution& convolution,
                                    const DarknetConvolutionShape& shape,
                                    const DarknetWeightTarget& target, NN::ModelState& state,
                                    StorePlan& plan)
            {
                SA_FAIL_IF(shape.filters <= 0 || shape.channels <= 0 || shape.size <= 0,
                           StatusCode::InvalidDescription,
                           "convolution layer {} has filters {}, channels {} and size {}; each "
                           "must be positive",
                           shape.layer, shape.filters, shape.channels, shape.size);

                const Shape weight_extent{shape.filters, shape.size, shape.size, shape.channels};
                const Shape vector_extent{shape.filters};
                if (!target.weight.empty())
                {
                    SA_TRY(plan.add(state, shape, target.weight, convolution.weights, weight_extent,
                                    CopyKind::Weights));
                }

                const Field fields[] = {
                    {target.bias, convolution.biases},
                    {target.scale, convolution.scales},
                    {target.mean, convolution.rolling_mean},
                    {target.variance, convolution.rolling_variance},
                };
                for (const Field& field : fields)
                {
                    if (field.name.empty())
                        continue;

                    SA_TRY(plan.add(state, shape, field.name, field.values, vector_extent,
                                    CopyKind::Vector));
                }
                return Status();
            }

        } // namespace

        Status store_darknet_weights(const DarknetWeights& weights,
                                     SushiRuntime::span<const DarknetConvolutionShape> shapes,
                                     SushiRuntime::span<const DarknetWeightTarget> targets,
                                     NN::ModelState& state) noexcept
        {
            SA_FAIL_IF(weights.convolutions.size() != shapes.size()
                           || shapes.size() != targets.size(),
                       StatusCode::Incompatible,
                       "the weights hold {} convolutions, the shapes {} and the targets {}; the "
                       "three must match",
                       weights.convolutions.size(), shapes.size(), targets.size());

            StorePlan plan;
            for (std::size_t i = 0; i < shapes.size(); ++i)
            {
                SA_TRY(plan_convolution(weights.convolutions[i], shapes[i], targets[i], state,
                                        plan));
            }
            plan.apply();
            return Status();
        }

    } // namespace Vision::Importers
} // namespace SushiAI
