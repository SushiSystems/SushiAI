/**************************************************************************/
/* host_state_support.hpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file host_state_support.hpp
 * @brief Declares the state writes the vision blocks tests share when a traced block runs on
 *        Graph::HostEvaluator.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/host_evaluator.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/tolerance.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Names the elements to write for the state @p name of extents @p shape. */
        using StateRule = std::function<std::vector<double>(const std::string& name,
                                                            const Shape& shape)>;

        /**
         * @brief Holds the norm options under which a norm of gamma 2, running variance 1 and
         *        running mean 0 is the identity: 2 / sqrt(1 + 3) is exactly 1.
         */
        inline constexpr NN::NormOptions K_IDENTITY_NORM{3.0, 0.03};

        /** @brief Returns @p value once for each element of @p shape. */
        inline std::vector<double> filled(const Shape& shape, double value)
        {
            return std::vector<double>(static_cast<std::size_t>(shape.num_elements()), value);
        }

        /** @brief Returns 1, 2, ..., @p count. */
        inline std::vector<double> ramp(std::size_t count)
        {
            std::vector<double> values;
            values.reserve(count);
            for (std::size_t at = 1; at <= count; ++at)
                values.push_back(static_cast<double>(at));

            return values;
        }

        /** @brief Holds the rounded operations on the longest path of a block of a few layers. */
        inline constexpr std::size_t K_BLOCK_OPERATIONS = 8;

        /**
         * @brief Expects each element of @p result to sit within the rounding bound of its
         *        element of @p expected; an expected zero must be produced exactly.
         */
        inline void expect_values_near(const std::vector<double>& result,
                                       const std::vector<double>& expected)
        {
            ASSERT_EQ(result.size(), expected.size());
            for (std::size_t at = 0; at < expected.size(); ++at)
            {
                if (expected[at] == 0.0)
                    EXPECT_EQ(result[at], 0.0) << "element " << at;
                else
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<double>(
                        result[at], expected[at], K_BLOCK_OPERATIONS)))
                        << "element " << at;
            }
        }

        /** @brief Tells whether @p name ends with @p tail. */
        inline bool ends_with(const std::string& name, std::string_view tail)
        {
            return name.size() >= tail.size()
                && std::string_view(name).substr(name.size() - tail.size()) == tail;
        }

        /**
         * @brief Writes @p rule's elements into every parameter and buffer of @p module.
         * @param prefix The prefix the names handed to @p rule hang under.
         * @return A success, or the first failure HostEvaluator::write() returns.
         * @pre @p module was traced into @p graph.
         */
        template<typename Module>
        Status write_state_by(Graph::HostEvaluator& evaluator, const Graph::Graph& graph,
                              Module& module, std::string_view prefix, const StateRule& rule)
        {
            Status outcome;
            const auto write = [&](Graph::ValueID id, const std::string& name, const Shape& shape)
            {
                const Status written = evaluator.write(id, rule(name, shape));
                if (outcome.is_ok())
                    outcome = written;
            };
            const auto on_parameter = [&](const std::string& name, NN::Parameter& parameter)
            {
                write(parameter.id_in(graph), name, parameter.shape());
            };
            const auto on_buffer = [&](const std::string& name, NN::Buffer& buffer)
            {
                write(buffer.id_in(graph), name, buffer.shape());
            };
            const NN::StateVisitor visitor(on_parameter, on_buffer);
            NN::walk_state(module, prefix, visitor);
            return outcome;
        }
    } // namespace Test
} // namespace SushiAI
