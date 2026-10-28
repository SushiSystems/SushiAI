/**************************************************************************/
/* pixel_objective.cpp                                                    */
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
 * @file pixel_objective.cpp
 * @brief Implements what the objectives over [N, H, W, C] logits and a mask share.
 * @author Mustafa Garip
 */

#include "pixel_objective.hpp"

#include <limits>
#include <string>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/vision/segment/ignore_index.hpp>

namespace SushiAI::Vision::Training::Detail
{
    namespace
    {
        /** @brief Names the rank of the [N, H, W, C] logits a per-pixel objective reads. */
        constexpr std::int32_t K_LOGITS_RANK = 4;

        /** @brief Names the rank of the [H, W, channels] sample of an image model. */
        constexpr std::int32_t K_IMAGE_SAMPLE_RANK = 3;

        /**
         * @brief Reports whether N * H * W * C is at most @p largest.
         * @pre Every extent is at least 1.
         */
        [[nodiscard]] bool fits(const PixelExtents& extents, std::int64_t largest) noexcept
        {
            // Dividing first keeps each product below what int64 holds.
            std::int64_t room = largest / extents.batch;
            room /= extents.height;
            room /= extents.width;

            return extents.classes <= room;
        }
    } // namespace

    Status validate_pixel_objective(const PixelExtents& extents, std::int32_t ignore_index,
                                    DType dtype, std::int64_t largest,
                                    std::string_view objective)
    {
        SA_FAIL_IF(extents.batch < 1 || extents.height < 1 || extents.width < 1
                   || extents.classes < 1,
                   StatusCode::InvalidDescription,
                   "A {} objective needs a batch, a height, a width and a class count of at "
                   "least 1; got {}, {}, {} and {}",
                   objective, extents.batch, extents.height, extents.width, extents.classes);
        SA_FAIL_IF(!fits(extents, largest), StatusCode::InvalidDescription,
                   "A {} objective holds batch * height * width * classes scores; "
                   "{} * {} * {} * {} is above {}",
                   objective, extents.batch, extents.height, extents.width, extents.classes,
                   largest);
        SA_FAIL_IF(ignore_index >= 0 && ignore_index < extents.classes,
                   StatusCode::InvalidDescription,
                   "A {} objective was given the ignore index {}, which is one of its {} "
                   "classes",
                   objective, ignore_index, extents.classes);
        SA_FAIL_IF(dtype != DType::Float32 && dtype != DType::Float64,
                   StatusCode::InvalidDescription,
                   "A {} objective writes its leaves as {} or {}; got {}", objective,
                   dtype_name(DType::Float32), dtype_name(DType::Float64), dtype_name(dtype));

        return Status();
    }

    Data::TargetDescription describe_mask_role(const PixelExtents& extents)
    {
        // No class count is stated: a mask of fewer classes serves the same step.
        return Data::TargetDescription{std::string(Data::K_ROLE_MASK),
                                       Shape{extents.height, extents.width}, DType::Int32, 0};
    }

    Graph::Traced emit_pixel_rows(const Graph::GraphBuilder& builder,
                                  const Graph::Graph& graph, const NN::ModelOutputs& outputs,
                                  const PixelExtents& extents, std::string_view objective)
    {
        const Result<Graph::ValueID> single = outputs.only();
        SA_FATAL_IF(!single.has_value(), "The {} objective reads one output: {}", objective,
                    single.status().message());

        // By value: the reshape below can reallocate the value vector.
        const Shape produced = graph.value(single.value()).shape;
        const Shape expected{extents.batch, extents.height, extents.width, extents.classes};
        SA_FATAL_IF(produced != expected,
                    "The model produced {} but the {} objective was made for {}",
                    produced.to_string(), objective, expected.to_string());
        SA_FATAL_IF(graph.value(single.value()).layout != Graph::TensorLayout::NHWC,
                    "The {} objective reads its logits with the classes innermost, but the "
                    "model's output is tagged NCHW",
                    objective);

        return builder.reshape(single.value(), Shape{extents.pixels(), extents.classes});
    }

    Result<PixelExtents> read_pixel_extents(const Train::ObjectiveContext& given,
                                            std::string_view context)
    {
        SA_FAIL_IF(given.outputs.size() != 1, StatusCode::InvalidDescription,
                   "{} reads one [N, H, W, classes] output, but the model publishes {}",
                   context, given.outputs.size());

        const Train::OutputShape& output = given.outputs.front();
        SA_FAIL_IF(output.shape.rank() != K_LOGITS_RANK || output.shape[0] != given.batch,
                   StatusCode::InvalidDescription,
                   "{} reads [{}, H, W, classes] logits, but the model's output '{}' is {}",
                   context, given.batch, output.name, output.shape.to_string());

        const Shape& input = given.input_sample;
        SA_FAIL_IF(input.rank() == K_IMAGE_SAMPLE_RANK
                   && (input[0] != output.shape[1] || input[1] != output.shape[2]),
                   StatusCode::InvalidDescription,
                   "{} compares a score with a mask pixel by pixel, but the model's output "
                   "'{}' is {} over an input of {} per sample; the head must end at the input "
                   "size",
                   context, output.name, output.shape.to_string(), input.to_string());
        SA_FAIL_IF(given.classes > output.shape[3], StatusCode::InvalidDescription,
                   "{}: the data holds {} classes but the model scores {}", context,
                   given.classes, output.shape[3]);

        return PixelExtents{output.shape[0], output.shape[1], output.shape[2], output.shape[3]};
    }

    Result<std::int32_t> read_ignore_index(const Config::Value& attributes,
                                           std::string_view context)
    {
        for (const std::string& key : attributes.keys())
        {
            SA_FAIL_IF(key != K_IGNORE_INDEX_KEY, StatusCode::InvalidDescription,
                       "{} states the key '{}', but the only key it reads is '{}'", context,
                       key, K_IGNORE_INDEX_KEY);
        }

        if (!attributes.has(K_IGNORE_INDEX_KEY))
            return Segment::K_DEFAULT_IGNORE_INDEX;

        SA_TRY_ASSIGN(const Config::Value& stated, attributes.at(K_IGNORE_INDEX_KEY, context));
        SA_FAIL_IF(stated.kind() != Config::Kind::Int, StatusCode::InvalidDescription,
                   "{}: '{}' is a whole number, but the file holds a {}", context,
                   K_IGNORE_INDEX_KEY, Config::kind_name(stated.kind()));

        SA_TRY_ASSIGN(const std::int64_t index, stated.as_int(context));
        SA_FAIL_IF(index < std::numeric_limits<std::int32_t>::min()
                   || index > std::numeric_limits<std::int32_t>::max(),
                   StatusCode::InvalidDescription,
                   "{}: '{}' is {}, which the int32 of a mask cannot hold", context,
                   K_IGNORE_INDEX_KEY, index);

        return static_cast<std::int32_t>(index);
    }

} // namespace SushiAI::Vision::Training::Detail
