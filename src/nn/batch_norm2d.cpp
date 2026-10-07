/**************************************************************************/
/* batch_norm2d.cpp                                                       */
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
 * @file batch_norm2d.cpp
 * @brief Implements BatchNorm2d: its two trace forms, its node factory and the update.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/batch_norm2d.hpp>

#include <cstdint>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include "blocks/block_attributes.hpp"

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Holds the two parameter leaves both trace forms declare. */
            struct DeclaredState
            {
                Graph::ValueId gamma; /**< The leaf of the scale. */
                Graph::ValueId beta;  /**< The leaf of the shift. */
            };

            /** @brief Lists every key a "batchnorm2d" row accepts. */
            constexpr std::string_view K_BATCH_NORM2D_KEYS[] = {Detail::K_NORM_EPSILON_KEY,
                                                                Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Returns InvalidDescription unless the channels and both constants fit. */
            [[nodiscard]] Status check_description(const BatchNorm2d& layer)
            {
                SA_FAIL_IF(layer.channels <= 0, StatusCode::InvalidDescription,
                           "BatchNorm2d needs a positive channel count; got {}", layer.channels);
                SA_FAIL_IF(!(layer.options.epsilon > 0.0), StatusCode::InvalidDescription,
                           "BatchNorm2d({}) needs a positive epsilon; got {}", layer.channels,
                           layer.options.epsilon);
                SA_FAIL_IF(!(layer.options.momentum >= 0.0 && layer.options.momentum <= 1.0),
                           StatusCode::InvalidDescription,
                           "BatchNorm2d({}) needs a momentum in [0, 1]; got {}", layer.channels,
                           layer.options.momentum);

                return Status();
            }

            /**
             * @brief Returns the shape of @p x once it is known to fit @p layer.
             * @param x A value of builder.graph(); anything else is fatal.
             * @return A copy, which stays valid while later declarations grow the value table.
             */
            [[nodiscard]] Result<Shape> checked_image(const BatchNorm2d& layer,
                                                      const Graph::GraphBuilder& builder,
                                                      Graph::ValueId x)
            {
                SA_FATAL_IF(x >= builder.graph().value_count(),
                            "BatchNorm2d({}) was traced over value {}; the graph holds {} values",
                            layer.channels, x, builder.graph().value_count());
                SA_TRY(check_description(layer));

                const Shape image = builder.graph().value(x).shape;
                SA_FAIL_IF(image.rank() != 4 || image[3] != layer.channels,
                           StatusCode::InvalidDescription,
                           "BatchNorm2d({}) needs a rank-4 [N, H, W, {}] input; got {}",
                           layer.channels, layer.channels, image.to_string());
                SA_FAIL_IF(builder.graph().value(x).layout != Graph::TensorLayout::NHWC,
                           StatusCode::InvalidDescription,
                           "BatchNorm2d({}) needs an NHWC input; got {} tagged NCHW. Cross with "
                           "layout_convert first (CV-3).",
                           layer.channels, image.to_string());
                SA_FAIL_IF(!is_real_float(builder.graph().value(x).dtype),
                           StatusCode::InvalidDescription,
                           "BatchNorm2d({}) needs a half, float32 or float64 input; got {}",
                           layer.channels, dtype_name(builder.graph().value(x).dtype));

                return image;
            }

            /**
             * @brief Declares gamma and beta in the builder's graph and defines both buffers.
             * @return InvalidDescription when an earlier trace described the state otherwise.
             */
            [[nodiscard]] Result<DeclaredState> declare_state(BatchNorm2d& layer,
                                                              const Graph::GraphBuilder& builder,
                                                              Graph::ValueId x)
            {
                // The type is held by value; declaring a parameter grows the value table.
                const DType dtype = builder.graph().value(x).dtype;
                const Shape per_channel{layer.channels};

                SA_TRY_ASSIGN(const Graph::ValueId gamma, Detail::bridge_throwing([&]()
                {
                    return layer.weight.declare(builder, per_channel, ParameterRole::SCALE,
                                                layer.channels, layer.channels, "weight", dtype);
                }));
                SA_TRY_ASSIGN(const Graph::ValueId beta, Detail::bridge_throwing([&]()
                {
                    return layer.bias.declare(builder, per_channel, ParameterRole::BIAS,
                                              layer.channels, layer.channels, "bias", dtype);
                }));

                SA_TRY(SA_ANNOTATE(layer.running_mean.define(per_channel, 0.0, dtype),
                                   "BatchNorm2d({}) 'running_mean'", layer.channels));
                SA_TRY(SA_ANNOTATE(layer.running_var.define(per_channel, 1.0, dtype),
                                   "BatchNorm2d({}) 'running_var'", layer.channels));

                return DeclaredState{gamma, beta};
            }

            /**
             * @brief Returns the factor the biased batch variance carries into the running one.
             * @param count The values one channel's statistics are taken over; at least two.
             * @return count / (count - 1), which tools/reference/README.md records for torch.
             */
            [[nodiscard]] double running_variance_scale(int64_t count) noexcept
            {
                return static_cast<double>(count) / static_cast<double>(count - 1);
            }

            /** @brief Hands one update of @p target to the sink of a Training trace. */
            void request_update(const Trace& trace, Buffer& target, std::string_view debug_name,
                                Graph::ValueId source, double momentum, double source_scale)
            {
                const Status requested = trace.update(target, source, momentum, source_scale);
                SA_FATAL_IF(!requested.is_ok(),
                            "BatchNorm2d could not request the update of '{}': {}", debug_name,
                            requested.message());
            }

            /** @brief Traces the training form, or returns why @p x or @p layer does not fit. */
            [[nodiscard]] Result<Graph::ValueId> trace_batch_statistics(BatchNorm2d& layer,
                                                                        const Trace& trace,
                                                                        Graph::ValueId x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY_ASSIGN(const Shape image, checked_image(layer, builder, x));

                const int64_t count = image[0] * image[1] * image[2];
                SA_FAIL_IF(count < 2, StatusCode::InvalidDescription,
                           "BatchNorm2d({}) needs more than one value per channel to train; got "
                           "{} from an input of {}",
                           layer.channels, count, image.to_string());

                SA_TRY_ASSIGN(const DeclaredState state, declare_state(layer, builder, x));
                SA_TRY_ASSIGN(const Graph::BatchNormResult result, Detail::bridge_throwing([&]()
                {
                    return builder.batchnorm2d(x, state.gamma, state.beta, layer.options.epsilon);
                }));

                request_update(trace, layer.running_mean, "running_mean", result.saved_mean,
                               layer.options.momentum, 1.0);
                request_update(trace, layer.running_var, "running_var", result.batch_variance,
                               layer.options.momentum, running_variance_scale(count));

                return result.output;
            }

            /** @brief Traces the evaluation form, or returns why @p x or @p layer does not fit. */
            [[nodiscard]] Result<Graph::ValueId> trace_running_statistics(BatchNorm2d& layer,
                                                                          const Trace& trace,
                                                                          Graph::ValueId x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_TRY(checked_image(layer, builder, x));

                SA_TRY_ASSIGN(const DeclaredState state, declare_state(layer, builder, x));
                SA_TRY_ASSIGN(const Graph::ValueId mean, Detail::bridge_throwing([&]()
                {
                    return layer.running_mean.read(builder, "running_mean");
                }));
                SA_TRY_ASSIGN(const Graph::ValueId variance, Detail::bridge_throwing([&]()
                {
                    return layer.running_var.read(builder, "running_var");
                }));

                return Detail::bridge_throwing([&]()
                {
                    const Graph::BatchNormFold folded = builder.batchnorm_fold(
                        state.gamma, state.beta, mean, variance, layer.options.epsilon);

                    return builder.channel_affine(x, folded.scale, folded.shift);
                });
            }

        } // namespace

        BatchNorm2d::BatchNorm2d(int64_t channels, NormOptions options) noexcept
            : channels(channels), options(options)
        {

        }

        Graph::ValueId BatchNorm2d::trace_training(const Trace& trace, Graph::ValueId x)
        {
            SA_FATAL_IF(trace.mode() != TraceMode::Training,
                        "BatchNorm2d::trace_training was given an evaluation trace; trace_layer "
                        "picks the form of the trace's mode");

            return trace.value_or_fail(trace_batch_statistics(*this, trace, x), x);
        }

        Graph::ValueId BatchNorm2d::trace_evaluation(const Trace& trace, Graph::ValueId x)
        {
            return trace.value_or_fail(trace_running_statistics(*this, trace, x), x);
        }

        Shape BatchNorm2d::output_shape(const Shape& input) const
        {
            return input;
        }

        Result<std::unique_ptr<Node>> make_batch_norm2d_node(ShapeSpan inputs,
                                                             const Config::Value& attributes,
                                                             std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_BATCH_NORM2D_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const NormOptions norm, Detail::read_norm_options(attributes, context));

            return Detail::wrap_block(BatchNorm2d(input[2], norm), input);
        }

        double updated_statistic(double running, double batch, double momentum,
                                 double source_scale) noexcept
        {
            return (1.0 - momentum) * running + momentum * source_scale * batch;
        }

    } // namespace NN
} // namespace SushiAI
