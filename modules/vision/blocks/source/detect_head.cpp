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
#include <cmath>
#include <string>
#include <utility>
#include <variant>
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
                "classes", "reg_max", "end2end", "class_branch", Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Names the key that holds the box side's distribution bins. */
            constexpr std::string_view K_REG_MAX_KEY = "reg_max";

            /** @brief Names the key that tells whether the one-to-one copy exists. */
            constexpr std::string_view K_END2END_KEY = "end2end";

            /** @brief Names the key that holds the form of the class branches. */
            constexpr std::string_view K_CLASS_BRANCH_KEY = "class_branch";

            /** @brief Names the value of K_CLASS_BRANCH_KEY for the depthwise form. */
            constexpr std::string_view K_DEPTHWISE = "depthwise";

            /** @brief Names the value of K_CLASS_BRANCH_KEY for the plain form. */
            constexpr std::string_view K_PLAIN = "plain";

            /** @brief Holds the value the last box convolution's bias starts from. */
            constexpr double K_BOX_BIAS_INIT = 1.0;

            /** @brief Holds the expected object count per image the class bias assumes. */
            constexpr double K_OBJECTS_PER_IMAGE = 5.0;

            /** @brief Holds the image side, in pixels, the class bias assumes. */
            constexpr double K_REFERENCE_IMAGE_SIDE = 640.0;

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
            [[nodiscard]] NN::Conv2d biased_pointwise(int64_t in_channels, int64_t out_channels,
                                                      double bias_init)
            {
                NN::Conv2d layer;
                layer.in_channels = in_channels;
                layer.out_channels = out_channels;
                layer.bias_init = bias_init;

                return layer;
            }

            /** @brief Holds what one level's branches are built from. */
            struct LevelSpec
            {
                /** @brief Holds the channels of the level's feature map. */
                int64_t in_channels;

                /** @brief Holds the hidden width of the box branch. */
                int64_t box_width;

                /** @brief Holds the output channels of the box branch. */
                int64_t box_channels;

                /** @brief Holds the hidden width of the class branch. */
                int64_t class_width;

                /** @brief Holds the output channels of the class branch. */
                int64_t classes;

                /** @brief Holds the bias the last box convolution starts from. */
                double box_bias;

                /** @brief Holds the bias the last class convolution starts from. */
                double class_bias;

                /** @brief Holds the epsilon and the momentum of every norm. */
                NN::NormOptions norm;
            };

            /** @brief Returns the class bias of a level of @p stride over @p classes. */
            [[nodiscard]] double class_bias_for(int64_t classes, double stride)
            {
                const double cells = K_REFERENCE_IMAGE_SIDE / stride;

                return std::log(K_OBJECTS_PER_IMAGE / static_cast<double>(classes) /
                                (cells * cells));
            }

            /** @brief Builds the box branch of @p spec. */
            [[nodiscard]] BoxBranch box_branch(const LevelSpec& spec)
            {
                return BoxBranch(
                    Conv(spec.in_channels, spec.box_width, K_BRANCH_KERNEL, 1, spec.norm),
                    Conv(spec.box_width, spec.box_width, K_BRANCH_KERNEL, 1, spec.norm),
                    biased_pointwise(spec.box_width, spec.box_channels, spec.box_bias));
            }

            /** @brief Builds the stage that filters @p in_channels and widens to @p width. */
            [[nodiscard]] ClassStage class_stage(int64_t in_channels, int64_t width,
                                                 NN::NormOptions norm)
            {
                return ClassStage(DWConv(in_channels, in_channels, K_BRANCH_KERNEL, 1, norm),
                                  Conv(in_channels, width, Detail::K_POINTWISE,
                                       Detail::K_POINTWISE, norm));
            }

            /** @brief Builds the depthwise-form class branch of @p spec. */
            [[nodiscard]] ClassBranch depthwise_class_branch(const LevelSpec& spec)
            {
                return ClassBranch(class_stage(spec.in_channels, spec.class_width, spec.norm),
                                   class_stage(spec.class_width, spec.class_width, spec.norm),
                                   biased_pointwise(spec.class_width, spec.classes,
                                                    spec.class_bias));
            }

            /** @brief Builds the plain-form class branch of @p spec. */
            [[nodiscard]] PlainClassBranch plain_class_branch(const LevelSpec& spec)
            {
                return PlainClassBranch(
                    Conv(spec.in_channels, spec.class_width, K_BRANCH_KERNEL, 1, spec.norm),
                    Conv(spec.class_width, spec.class_width, K_BRANCH_KERNEL, 1, spec.norm),
                    biased_pointwise(spec.class_width, spec.classes, spec.class_bias));
            }

            /** @brief Appends the depthwise-form class branch of @p spec to @p list. */
            void add_class_branch(NN::ModuleList<ClassBranch>& list, const LevelSpec& spec)
            {
                list.push_back(depthwise_class_branch(spec));
            }

            /** @brief Appends the plain-form class branch of @p spec to @p list. */
            void add_class_branch(NN::ModuleList<PlainClassBranch>& list, const LevelSpec& spec)
            {
                list.push_back(plain_class_branch(spec));
            }

            /** @brief Appends one level's box and class branches to a copy's lists. */
            template<typename ClassList>
            void add_level(NN::ModuleList<BoxBranch>& boxes, ClassList& classes,
                           const LevelSpec& spec)
            {
                boxes.push_back(box_branch(spec));
                add_class_branch(classes, spec);
            }

            /** @brief Reads the class_branch key, or the depthwise form when it is absent. */
            [[nodiscard]] Result<ClassBranchStyle> read_class_branch(
                const Config::Value& attributes, std::string_view context)
            {
                if (!attributes.has(K_CLASS_BRANCH_KEY))
                    return ClassBranchStyle::Depthwise;

                const Config::Value& value = attributes.at(K_CLASS_BRANCH_KEY, context);
                SA_FAIL_IF(value.kind() != Config::Kind::String, StatusCode::InvalidDescription,
                           "{} '{}' must be \"{}\" or \"{}\", but it is {}", context,
                           K_CLASS_BRANCH_KEY, K_DEPTHWISE, K_PLAIN,
                           Config::kind_name(value.kind()));

                const std::string& form = value.as_string(K_CLASS_BRANCH_KEY);
                if (form == K_DEPTHWISE)
                    return ClassBranchStyle::Depthwise;

                SA_FAIL_IF(form != K_PLAIN, StatusCode::InvalidDescription,
                           "{} '{}' must be \"{}\" or \"{}\", but it is \"{}\"", context,
                           K_CLASS_BRANCH_KEY, K_DEPTHWISE, K_PLAIN, form);

                return ClassBranchStyle::Plain;
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
            template<typename ClassType>
            void trace_copy(NN::ModuleList<BoxBranch>& boxes, NN::ModuleList<ClassType>& classes,
                            const NN::Trace& trace, NN::ValueSpan features,
                            std::vector<Graph::ValueID>& results)
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
            [[nodiscard]] std::vector<Graph::ValueID> one_to_one_features(const NN::Trace& trace,
                                                                          NN::ValueSpan features)
            {
                std::vector<Graph::ValueID> read(features.begin(), features.end());
                if (trace.mode() != NN::TraceMode::Training)
                    return read;

                for (Graph::ValueID& feature : read)
                    feature = trace.builder().stop_gradient(feature);

                return read;
            }

            /** @brief Holds what building the branches of one head needs. */
            struct HeadPlan
            {
                /** @brief Holds the per-sample shape of each level. */
                NN::ShapeSpan levels;

                /** @brief Holds the class count. */
                int64_t classes;

                /** @brief Holds the epsilon and the momentum of every norm. */
                NN::NormOptions norm;

                /** @brief Holds the options the head was asked for. */
                const DetectOptions& options;

                /** @brief Holds the hidden width of the box branches. */
                int64_t box_width;

                /** @brief Holds the output channels of the box branches. */
                int64_t box_channels;

                /** @brief Holds the hidden width of the class branches. */
                int64_t class_width;
            };

            /** @brief Builds the branches of every level, for the copies @p plan asks for. */
            template<typename ClassType>
            [[nodiscard]] DetectBranches<ClassType> build_branches(const HeadPlan& plan)
            {
                const bool has_strides = !plan.options.strides.empty();
                DetectBranches<ClassType> branches;

                for (std::size_t level = 0; level < plan.levels.size(); ++level)
                {
                    const LevelSpec spec{
                        plan.levels[level][2], plan.box_width, plan.box_channels,
                        plan.class_width, plan.classes,
                        has_strides ? K_BOX_BIAS_INIT : 0.0,
                        has_strides ? class_bias_for(plan.classes, plan.options.strides[level])
                                    : 0.0,
                        plan.norm};

                    add_level(branches.cv2, branches.cv3, spec);
                    if (plan.options.end2end)
                        add_level(branches.one2one_cv2, branches.one2one_cv3, spec);
                }

                return branches;
            }

            /** @brief Lists the box and score shapes of the first copy of @p branches. */
            template<typename ClassType>
            void collect_shapes(DetectBranches<ClassType>& branches, NN::ShapeSpan levels,
                                std::vector<Shape>& boxes, std::vector<Shape>& scores)
            {
                for (std::size_t level = 0; level < levels.size(); ++level)
                {
                    boxes.push_back(branches.cv2[level].output_shape(levels[level]));
                    scores.push_back(branches.cv3[level].output_shape(levels[level]));
                }
            }

            /** @brief Traces the one-to-many copy, then the one-to-one copy when @p end2end. */
            template<typename ClassType>
            void trace_branches(DetectBranches<ClassType>& branches, bool end2end,
                                const NN::Trace& trace, NN::ValueSpan inputs,
                                std::vector<Graph::ValueID>& results)
            {
                trace_copy(branches.cv2, branches.cv3, trace, inputs, results);
                if (!end2end)
                    return;

                const std::vector<Graph::ValueID> read = one_to_one_features(trace, inputs);
                trace_copy(branches.one2one_cv2, branches.one2one_cv3, trace,
                           NN::ValueSpan(read.data(), read.size()), results);
            }

        } // namespace

        Result<std::unique_ptr<DetectHead>> DetectHead::create(NN::ShapeSpan levels,
                                                               int64_t classes,
                                                               NN::NormOptions norm,
                                                               const DetectOptions& options)
        {
            SA_FAIL_IF(levels.empty(), StatusCode::InvalidDescription,
                       "detect needs at least one feature level and was given none");
            SA_FAIL_IF(classes < 1, StatusCode::InvalidDescription,
                       "detect needs at least one class; the class count it was given is {}",
                       classes);

            SA_FAIL_IF(options.reg_max < 1, StatusCode::InvalidDescription,
                       "detect needs a reg_max of at least one; the reg_max it was given is {}",
                       options.reg_max);

            SA_FAIL_IF(!options.strides.empty() && options.strides.size() != levels.size(),
                       StatusCode::InvalidDescription,
                       "detect needs one stride per level or none; it was given {} strides for "
                       "{} levels",
                       options.strides.size(), levels.size());
            for (const double stride : options.strides)
            {
                SA_FAIL_IF(!(stride > 0.0) || !std::isfinite(stride),
                           StatusCode::InvalidDescription,
                           "detect needs positive finite strides; one of them is {}", stride);
            }

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
            const int64_t box_channels = K_BOX_CHANNELS * options.reg_max;
            const int64_t box_width = std::max(
                {K_SMALLEST_BOX_WIDTH, first_width / K_BOX_WIDTH_DIVISOR, box_channels});
            const int64_t class_width = std::max(first_width, std::min(classes, K_CLASS_WIDTH_CAP));

            const HeadPlan plan{levels, classes, norm, options, box_width, box_channels,
                                class_width};

            return NN::Detail::bridge_throwing([&]()
            {
                Branches branches =
                    options.class_branch == ClassBranchStyle::Plain
                        ? Branches(build_branches<PlainClassBranch>(plan))
                        : Branches(build_branches<ClassBranch>(plan));
                std::vector<Shape> boxes;
                std::vector<Shape> scores;
                boxes.reserve(levels.size());
                scores.reserve(levels.size());
                std::visit([&](auto& built) { collect_shapes(built, levels, boxes, scores); },
                           branches);

                std::vector<NN::Port> ports;
                ports.reserve(2 * (boxes.size() + scores.size()));
                append_copy_ports(ports, K_ONE_TO_MANY, boxes, scores);
                if (options.end2end)
                    append_copy_ports(ports, K_ONE_TO_ONE, boxes, scores);

                return std::unique_ptr<DetectHead>(new DetectHead(
                    std::move(branches), std::move(ports), levels.size(), options.end2end));
            });
        }

        DetectHead::DetectHead(Branches branches, std::vector<NN::Port> ports,
                               std::size_t level_count, bool end2end)
            : branches_(std::move(branches)), ports_(std::move(ports)), level_count_(level_count),
              end2end_(end2end)
        {

        }

        Status DetectHead::forward(const NN::Trace& trace, NN::ValueSpan inputs,
                                   std::vector<Graph::ValueID>& outputs)
        {
            SA_TRY(NN::Detail::require_input_count(inputs, level_count_));
            const Result<std::vector<Graph::ValueID>> traced =
                NN::Detail::bridge_throwing([this, &trace, &inputs]()
            {
                std::vector<Graph::ValueID> results;
                results.reserve(ports_.size());
                std::visit([&](auto& built)
                {
                    trace_branches(built, end2end_, trace, inputs, results);
                }, branches_);

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
            std::visit([&](auto& built) { NN::walk_state(built, prefix, visitor); },
                       branches_);
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

            DetectOptions options;
            SA_TRY_ASSIGN(options.reg_max,
                          Detail::read_positive_int_or(attributes, K_REG_MAX_KEY, context,
                                                       options.reg_max));
            SA_TRY_ASSIGN(options.end2end,
                          Detail::read_bool_or(attributes, K_END2END_KEY, context,
                                               options.end2end));
            SA_TRY_ASSIGN(options.class_branch, read_class_branch(attributes, context));

            Result<std::unique_ptr<DetectHead>> made =
                DetectHead::create(inputs, classes, norm, options);
            SA_TRY(SA_ANNOTATE(made.status(), "{}", context));

            return std::unique_ptr<NN::Node>(std::move(made).value());
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
