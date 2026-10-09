/**************************************************************************/
/* detect_head.cpp                                                        */
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
 * @file detect_head.cpp
 * @brief Implements DetectHead and the factory that builds it from a "detect" row.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/detect_head.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <SushiAI/config/value.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "detect" row accepts. */
            constexpr std::string_view K_DETECT_KEYS[] = {
                "classes", Detail::K_NORM_EPSILON_KEY, Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Names the rank of a per-sample [H, W, C] feature map. */
            constexpr int32_t K_FEATURE_MAP_RANK = 3;

            /** @brief Holds the side of the window of a branch's 3x3 convolutions. */
            constexpr int64_t K_BRANCH_KERNEL = 3;

            /** @brief Holds the channels of a box output: four distances per position. */
            constexpr int64_t K_BOX_CHANNELS = 4;

            /** @brief Holds the narrowest hidden width of a box branch. */
            constexpr int64_t K_SMALLEST_BOX_WIDTH = 16;

            /** @brief Holds what the first level's width is divided by for a box branch. */
            constexpr int64_t K_BOX_WIDTH_DIVISOR = 4;

            /** @brief Holds the class count above which a class branch stops widening. */
            constexpr int64_t K_CLASS_WIDTH_CAP = 100;

            /** @brief Names the copy whose ports come first. */
            constexpr std::string_view K_ONE_TO_MANY = "one2many";

            /** @brief Names the copy whose ports come second. */
            constexpr std::string_view K_ONE_TO_ONE = "one2one";

            /** @brief Names the role of a box port. */
            constexpr std::string_view K_BOXES = "boxes";

            /** @brief Names the role of a class-logit port. */
            constexpr std::string_view K_SCORES = "scores";

            /** @brief Builds the biased, unpadded 1x1 convolution that ends a branch. */
            [[nodiscard]] NN::Conv2d biased_pointwise(int64_t in_channels, int64_t out_channels)
            {
                NN::Conv2d layer;
                layer.in_channels = in_channels;
                layer.out_channels = out_channels;

                return layer;
            }

            /** @brief Builds the box branch that reads @p in_channels through @p width. */
            [[nodiscard]] BoxBranch box_branch(int64_t in_channels, int64_t width,
                                               NN::NormOptions norm)
            {
                return BoxBranch(Conv(in_channels, width, K_BRANCH_KERNEL, 1, norm),
                                 Conv(width, width, K_BRANCH_KERNEL, 1, norm),
                                 biased_pointwise(width, K_BOX_CHANNELS));
            }

            /** @brief Builds the stage that filters @p in_channels and widens to @p width. */
            [[nodiscard]] ClassStage class_stage(int64_t in_channels, int64_t width,
                                                 NN::NormOptions norm)
            {
                return ClassStage(DWConv(in_channels, in_channels, K_BRANCH_KERNEL, 1, norm),
                                  Conv(in_channels, width, Detail::K_POINTWISE,
                                       Detail::K_POINTWISE, norm));
            }

            /** @brief Builds the class branch that reads @p in_channels through @p width. */
            [[nodiscard]] ClassBranch class_branch(int64_t in_channels, int64_t width,
                                                   int64_t classes, NN::NormOptions norm)
            {
                return ClassBranch(class_stage(in_channels, width, norm),
                                   class_stage(width, width, norm),
                                   biased_pointwise(width, classes));
            }

            /** @brief Appends one port per shape, named "<copy>.level<l>.<role>". */
            void append_ports(std::vector<NN::Port>& ports, std::string_view copy,
                              std::string_view role, const std::vector<Shape>& shapes)
            {
                for (std::size_t level = 0; level < shapes.size(); ++level)
                {
                    std::string name(copy);
                    name.append(".level").append(std::to_string(level)).append(".");
                    name.append(role);
                    ports.push_back(NN::Port{std::move(name), shapes[level]});
                }
            }

            /** @brief Appends the ports of one copy: every box port, then every score port. */
            void append_copy_ports(std::vector<NN::Port>& ports, std::string_view copy,
                                   const std::vector<Shape>& boxes,
                                   const std::vector<Shape>& scores)
            {
                append_ports(ports, copy, K_BOXES, boxes);
                append_ports(ports, copy, K_SCORES, scores);
            }

            /**
             * @brief Traces one copy and appends the boxes of every level, then the scores.
             * @param features One value per level; @p boxes and @p classes hold as many.
             */
            void trace_copy(NN::ModuleList<BoxBranch>& boxes, NN::ModuleList<ClassBranch>& classes,
                            const NN::Trace& trace, NN::ValueSpan features,
                            std::vector<Graph::ValueId>& results)
            {
                for (std::size_t level = 0; level < features.size(); ++level)
                    results.push_back(NN::trace_layer(boxes[level], trace, features[level]));

                for (std::size_t level = 0; level < features.size(); ++level)
                    results.push_back(NN::trace_layer(classes[level], trace, features[level]));
            }

            /**
             * @brief Lists what the one-to-one copy reads: each feature behind a STOP_GRADIENT
             *        in a training trace, the features themselves otherwise.
             */
            [[nodiscard]] std::vector<Graph::ValueId> one_to_one_features(const NN::Trace& trace,
                                                                          NN::ValueSpan features)
            {
                std::vector<Graph::ValueId> read(features.begin(), features.end());
                if (trace.mode() != NN::TraceMode::Training)
                    return read;

                for (Graph::ValueId& feature : read)
                    feature = trace.builder().stop_gradient(feature);

                return read;
            }

        } // namespace

        Result<std::unique_ptr<DetectHead>> DetectHead::create(NN::ShapeSpan levels,
                                                               int64_t classes,
                                                               NN::NormOptions norm)
        {
            SA_FAIL_IF(levels.empty(), StatusCode::InvalidDescription,
                       "detect needs at least one feature level and was given none");
            SA_FAIL_IF(classes < 1, StatusCode::InvalidDescription,
                       "detect needs at least one class; the class count it was given is {}",
                       classes);

            for (std::size_t level = 0; level < levels.size(); ++level)
            {
                const Shape& input = levels[level];
                SA_FAIL_IF(input.rank() != K_FEATURE_MAP_RANK || input[2] < 1,
                           StatusCode::InvalidDescription,
                           "detect level {} has the per-sample shape {}; a level is a rank-3 "
                           "[H, W, C] feature map of at least one channel",
                           level, input.to_string());
            }

            const int64_t first_width = levels[0][2];
            const int64_t box_width = std::max(
                {K_SMALLEST_BOX_WIDTH, first_width / K_BOX_WIDTH_DIVISOR, K_BOX_CHANNELS});
            const int64_t class_width = std::max(first_width, std::min(classes, K_CLASS_WIDTH_CAP));

            return NN::Detail::bridge_throwing([&]()
            {
                Branches branches;
                std::vector<Shape> boxes;
                std::vector<Shape> scores;
                boxes.reserve(levels.size());
                scores.reserve(levels.size());

                for (const Shape& input : levels)
                {
                    const int64_t width = input[2];
                    branches.cv2.push_back(box_branch(width, box_width, norm));
                    branches.cv3.push_back(class_branch(width, class_width, classes, norm));
                    branches.one2one_cv2.push_back(box_branch(width, box_width, norm));
                    branches.one2one_cv3.push_back(
                        class_branch(width, class_width, classes, norm));

                    // Both copies are built alike, so one copy's shapes serve the other.
                    boxes.push_back(branches.cv2[boxes.size()].output_shape(input));
                    scores.push_back(branches.cv3[scores.size()].output_shape(input));
                }

                std::vector<NN::Port> ports;
                ports.reserve(2 * (boxes.size() + scores.size()));
                append_copy_ports(ports, K_ONE_TO_MANY, boxes, scores);
                append_copy_ports(ports, K_ONE_TO_ONE, boxes, scores);

                return std::unique_ptr<DetectHead>(
                    new DetectHead(std::move(branches), std::move(ports), levels.size()));
            });
        }

        DetectHead::DetectHead(Branches branches, std::vector<NN::Port> ports,
                               std::size_t level_count)
            : branches_(std::move(branches)), ports_(std::move(ports)), level_count_(level_count)
        {

        }

        Status DetectHead::forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                   std::vector<Graph::ValueId>& outputs)
        {
            SA_TRY(NN::Detail::require_input_count(inputs, level_count_));
            const Result<std::vector<Graph::ValueId>> traced =
                NN::Detail::bridge_throwing([this, &trace, &inputs]()
            {
                std::vector<Graph::ValueId> results;
                results.reserve(ports_.size());
                trace_copy(branches_.cv2, branches_.cv3, trace, inputs, results);

                const std::vector<Graph::ValueId> read = one_to_one_features(trace, inputs);
                trace_copy(branches_.one2one_cv2, branches_.one2one_cv3, trace,
                           NN::ValueSpan(read.data(), read.size()), results);

                return results;
            });

            // A failure a layer recorded is the cause of whatever it threw afterwards.
            SA_TRY(trace.status());
            SA_TRY(traced);

            outputs.insert(outputs.end(), traced.value().begin(), traced.value().end());
            return Status();
        }

        void DetectHead::walk_state(std::string_view prefix, const NN::StateVisitor& visitor)
        {
            NN::walk_state(branches_, prefix, visitor);
        }

        std::size_t DetectHead::input_count() const noexcept
        {
            return level_count_;
        }

        SushiRuntime::span<const NN::Port> DetectHead::ports() const noexcept
        {
            return SushiRuntime::span<const NN::Port>(ports_.data(), ports_.size());
        }

        Result<std::unique_ptr<NN::Node>> make_detect_node(NN::ShapeSpan inputs,
                                                           const Config::Value& attributes,
                                                           std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_DETECT_KEYS),
                                               context));
            SA_TRY_ASSIGN(const int64_t classes,
                          Detail::read_positive_int(attributes, "classes", context));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            Result<std::unique_ptr<DetectHead>> made = DetectHead::create(inputs, classes, norm);
            SA_TRY(SA_ANNOTATE(made.status(), "{}", context));

            return std::unique_ptr<NN::Node>(std::move(made).value());
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
