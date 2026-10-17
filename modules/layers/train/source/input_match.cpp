/**************************************************************************/
/* input_match.cpp                                                        */
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
 * @file input_match.cpp
 * @brief Implements the check of the input a step reads against the one a dataset publishes.
 * @author Mustafa Garip
 */

#include <SushiAI/train/input_match.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <SushiLog/text/text_format.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Renders @p input as "int32 [16] per sample". */
            std::string describe(const Data::InputDescription& input)
            {
                return SushiLog::format_text("{} {} per sample", dtype_name(input.dtype),
                                               input.sample_shape.to_string());
            }
        } // namespace

        Result<Data::InputDescription> describe_input_leaf(const Graph::Value& leaf)
        {
            SA_FAIL_IF(leaf.shape.rank() == 0, StatusCode::InvalidDescription,
                       "The input leaf has no batch extent; a step reads [rows, sample...]");

            return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Allocate,
                                                  [&leaf]()
            {
                const SushiRuntime::span<const int64_t> sample(
                    leaf.shape.data() + 1, static_cast<std::size_t>(leaf.shape.rank()) - 1);

                return Data::InputDescription{Shape(sample), leaf.dtype};
            });
        }

        Status match_input(const Data::InputDescription& required,
                           const Data::Dataset& dataset)
        {
            return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                            [&required, &dataset]() -> Status
            {
                const Data::InputDescription published = Data::describe_input(dataset);

                SA_FAIL_IF(published.dtype != required.dtype
                           || published.sample_shape != required.sample_shape,
                           StatusCode::Incompatible,
                           "The dataset '{}' publishes its input as {}, but the step was "
                           "compiled for {}",
                           dataset.name(), describe(published), describe(required));

                return Status();
            });
        }

    } // namespace Train
} // namespace SushiAI
