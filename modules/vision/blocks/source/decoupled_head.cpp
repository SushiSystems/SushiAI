/**************************************************************************/
/* decoupled_head.cpp                                                     */
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
 * @file decoupled_head.cpp
 * @brief Implements DecoupledHead and the factory that builds it from a "decoupled_head" row.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/decoupled_head.hpp>

#include <string>
#include <utility>
#include <SushiAI/config/value.hpp>
#include <SushiAI/graph/builder.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            using Detect::Yolox::K_YOLOX_BOXES;
            using Detect::Yolox::K_YOLOX_CLASSES;
            using Detect::Yolox::K_YOLOX_OBJECTNESS;
            using Detect::Yolox::yolox_port_name;
            using Detect::Yolox::yolox_prior_bias;

            /** @brief Lists every key a "decoupled_head" row accepts. */
            constexpr std::string_view K_DECOUPLED_KEYS[] = {
                "classes", "hidden", Detail::K_NORM_EPSILON_KEY, Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Names the rank of a per-sample [H, W, C] feature map. */
            constexpr int32_t K_FEATURE_MAP_RANK = 3;

            /** @brief Holds the side of the window of a tower's convolutions. */
            constexpr int64_t K_TOWER_KERNEL = 3;

            /** @brief Builds the biased, unpadded 1x1 convolution that ends a branch. */
            [[nodiscard]] NN::Conv2d biased_pointwise(int64_t in_channels, int64_t out_channels,
                                                      double bias_init)
            {
                NN::Conv2d layer;
                layer.in_channels = in_channels;
                layer.out_channels = out_channels;
                layer.bias_init = bias_init;

                return layer;
            }

            /** @brief Builds the two 3x3 Conv of a tower of width @p hidden. */
            [[nodiscard]] DecoupledTower tower(int64_t hidden, NN::NormOptions norm)
            {
                return DecoupledTower(Conv(hidden, hidden, K_TOWER_KERNEL, 1, norm),
                                      Conv(hidden, hidden, K_TOWER_KERNEL, 1, norm));
            }

            /** @brief Appends the layers of one level, of input width @p in_channels. */
            void add_level(DecoupledBranches& branches, int64_t in_channels, int64_t classes,
                           int64_t hidden, NN::NormOptions norm)
            {
                branches.stems.push_back(
                    Conv(in_channels, hidden, Detail::K_POINTWISE, Detail::K_POINTWISE, norm));
                branches.cls_convs.push_back(tower(hidden, norm));
                branches.reg_convs.push_back(tower(hidden, norm));
                branches.cls_preds.push_back(biased_pointwise(hidden, classes, yolox_prior_bias()));
                branches.reg_preds.push_back(
                    biased_pointwise(hidden, static_cast<int64_t>(Detect::Yolox::K_YOLOX_BOX_CHANNELS),
                                     0.0));
                branches.obj_preds.push_back(biased_pointwise(
                    hidden, static_cast<int64_t>(Detect::Yolox::K_YOLOX_OBJECTNESS_CHANNELS),
                    yolox_prior_bias()));
            }

            /**
             * @brief Appends the three ports of level @p level, from the shape its stem yields.
             * @return The first failure of a prediction's output_shape().
             */
            [[nodiscard]] Status append_ports(DecoupledBranches& branches, std::size_t level,
                                              const Shape& narrowed, std::vector<NN::Port>& ports)
            {
                SA_TRY_ASSIGN(Shape classes, branches.cls_preds[level].output_shape(narrowed));
                SA_TRY_ASSIGN(Shape boxes, branches.reg_preds[level].output_shape(narrowed));
                SA_TRY_ASSIGN(Shape objectness, branches.obj_preds[level].output_shape(narrowed));
                ports.push_back(NN::Port{yolox_port_name(level, K_YOLOX_CLASSES), std::move(classes)});
                ports.push_back(NN::Port{yolox_port_name(level, K_YOLOX_BOXES), std::move(boxes)});
                ports.push_back(
                    NN::Port{yolox_port_name(level, K_YOLOX_OBJECTNESS), std::move(objectness)});

                return Status();
            }

        } // namespace

        Result<std::unique_ptr<DecoupledHead>> DecoupledHead::create(NN::ShapeSpan levels,
                                                                     int64_t classes,
                                                                     int64_t hidden,
                                                                     NN::NormOptions norm) noexcept
        {
            SA_FAIL_IF(levels.empty(), StatusCode::InvalidDescription,
                       "decoupled_head needs at least one feature level and was given none");
            SA_FAIL_IF(classes < 1, StatusCode::InvalidDescription,
                       "decoupled_head needs at least one class; the class count it was given is "
                       "{}",
                       classes);
            SA_FAIL_IF(hidden < 1, StatusCode::InvalidDescription,
                       "decoupled_head needs a hidden width of at least one; the width it was "
                       "given is {}",
                       hidden);
            for (std::size_t level = 0; level < levels.size(); ++level)
            {
                const Shape& input = levels[level];
                SA_FAIL_IF(input.rank() != K_FEATURE_MAP_RANK || input[2] < 1,
                           StatusCode::InvalidDescription,
                           "decoupled_head level {} has the per-sample shape {}; a level is a "
                           "rank-3 [H, W, C] feature map of at least one channel",
                           level, input.to_string());
            }

            DecoupledBranches branches;
            for (const Shape& input : levels)
                add_level(branches, input[2], classes, hidden, norm);

            std::vector<NN::Port> ports;
            ports.reserve(3 * levels.size());
            for (std::size_t level = 0; level < levels.size(); ++level)
            {
                SA_TRY_ASSIGN(const Shape narrowed, branches.stems[level].output_shape(levels[level]));
                SA_TRY(append_ports(branches, level, narrowed, ports));
            }

            return std::unique_ptr<DecoupledHead>(
                new DecoupledHead(std::move(branches), std::move(ports), levels.size()));
        }

        DecoupledHead::DecoupledHead(DecoupledBranches branches, std::vector<NN::Port> ports,
                                     std::size_t level_count)
            : branches_(std::move(branches)), ports_(std::move(ports)), level_count_(level_count)
        {

        }

        Status DecoupledHead::forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                      std::vector<Graph::ValueID>& outputs)
        {
            SA_TRY(NN::Detail::require_input_count(inputs, level_count_));

            std::vector<Graph::ValueID> results;
            results.reserve(ports_.size());
            for (std::size_t level = 0; level < level_count_; ++level)
            {
                const Graph::ValueID narrowed =
                    NN::trace_layer(branches_.stems[level], trace, inputs[level]);
                const Graph::ValueID class_feature =
                    NN::trace_layer(branches_.cls_convs[level], trace, narrowed);
                const Graph::ValueID box_feature =
                    NN::trace_layer(branches_.reg_convs[level], trace, narrowed);

                results.push_back(NN::trace_layer(branches_.cls_preds[level], trace, class_feature));
                results.push_back(NN::trace_layer(branches_.reg_preds[level], trace, box_feature));
                results.push_back(NN::trace_layer(branches_.obj_preds[level], trace, box_feature));
            }

            SA_TRY(trace.status());

            outputs.insert(outputs.end(), results.begin(), results.end());
            return Status();
        }

        void DecoupledHead::walk_state(std::string_view prefix, const NN::StateVisitor& visitor)
        {
            NN::walk_state(branches_, prefix, visitor);
        }

        std::size_t DecoupledHead::input_count() const noexcept
        {
            return level_count_;
        }

        SushiRuntime::span<const NN::Port> DecoupledHead::ports() const noexcept
        {
            return SushiRuntime::span<const NN::Port>(ports_.data(), ports_.size());
        }

        Result<std::unique_ptr<NN::Node>> make_decoupled_head_node(NN::ShapeSpan inputs,
                                                                   const Config::Value& attributes,
                                                                   std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_DECOUPLED_KEYS),
                                               context));
            SA_TRY_ASSIGN(const int64_t classes,
                          Detail::read_positive_int(attributes, "classes", context));
            SA_TRY_ASSIGN(const int64_t hidden,
                          Detail::read_positive_int(attributes, "hidden", context));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            Result<std::unique_ptr<DecoupledHead>> made =
                DecoupledHead::create(inputs, classes, hidden, norm);
            SA_TRY(SA_ANNOTATE(made.status(), "{}", context));

            return std::unique_ptr<NN::Node>(std::move(made).value());
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
