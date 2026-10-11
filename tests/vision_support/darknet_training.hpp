/**************************************************************************/
/* darknet_training.hpp                                                   */
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
 * @file darknet_training.hpp
 * @brief Declares the loaders of what a Darknet run dumped of a head's training pass, and the
 *        training suite the six model tests run against those dumps.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <vision_support/darknet_reference.hpp>
#include <common/test_utils.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds the name of the reference of the truth rows in every run folder. */
        inline constexpr std::string_view K_DARKNET_TRUTH = "truth";

        /** @brief Holds the name of the reference of a step's network cost and solver. */
        inline constexpr std::string_view K_DARKNET_NETWORK_COST = "network_cost";

        /**
         * @brief Reads feature maps, or the updates of a kernel, and moves the channels last.
         * @return `[N, H, W, C]` from `NCHW`, or from `CHW` with N of 1; `[out, kh, kw, in]`
         *         from `OIHW`; FileNotFound naming a missing file; InvalidFile for a description
         *         or a `.bin` the loader cannot read; Unsupported for another `order` or another
         *         count of axes.
         */
        [[nodiscard]] Result<DarknetTensor> load_darknet_maps(std::string_view run,
                                                              std::string_view name);

        /** @brief Holds what one run dumped of the training pass of one head. */
        struct DarknetHeadDump
        {
            Vision::Detect::Anchor::HeadShape shape;  /**< Images and grid of the head. */
            std::vector<float> input;                 /**< Raw head, `[B, H, W, A * (5 + C)]`. */
            std::vector<float> delta;                 /**< Delta, in the layout of the input. */
            bool has_assignments = false;             /**< Whether the run dumped the two below. */
            std::vector<std::int64_t> assigned_row;   /**< `[B, H, W, A]` truth row, else -1. */
            std::vector<std::int64_t> assigned_class; /**< `[B, H, W, A]` class, else -1. */
            double cost = 0.0;                        /**< The cost the layer stored. */
            Config::Value settings;                   /**< The description of the delta. */
        };

        /**
         * @brief Loads the head input, the delta, the assignments and the cost of one head.
         * @param yolo_layer The index of the [yolo] section among the layers of the file.
         * @return The dump. The cost is the `cost` of the delta's description, or the tensor
         *         `layer_<k>_yolo_cost` when the description has none. Incompatible when the
         *         delta or an assignment does not have the extents of the input.
         */
        [[nodiscard]] Result<DarknetHeadDump> load_darknet_head_dump(std::string_view run,
                                                                     std::int64_t yolo_layer);

        /**
         * @brief Loads the `truth` rows of a run as the padded batch a head delta reads.
         * @param batch Images of the run; a row without an `image` column belongs to image 0.
         * @return The batch in pixel corners of @p network; InvalidFile for a description
         *         whose `columns` lack a box or a class column; Incompatible for a row that
         *         names an image outside the batch or a class that is not a whole number.
         */
        [[nodiscard]] Result<Vision::Detect::GroundTruthBatch> load_darknet_truth(
            std::string_view run, std::int64_t batch, Vision::Detect::Anchor::NetworkSize network);

        /** @brief Holds the bounds a delta is measured against, by the kind of its entry. */
        struct DarknetDeltaBounds
        {
            double box_absolute = 0.0;   /**< Absolute bound on a box entry. */
            double box_relative = 0.0;   /**< Bound on a box entry relative to the reference. */
            double other_absolute = 0.0; /**< Absolute bound on an objectness or class entry. */
        };

        /**
         * @brief Checks a computed delta against a dumped one and prints the two largest
         *        differences measured, of the box entries and of the others.
         * @param slot_entries The entries of an anchor slot, `5 + C`; positive.
         * @return Success, or a failure naming the count outside each bound and its worst
         *         element; a failure as well when the sizes differ or hold no whole slot.
         */
        [[nodiscard]] ::testing::AssertionResult delta_within(
            std::string_view what, SushiRuntime::span<const float> actual,
            SushiRuntime::span<const float> reference, std::int64_t slot_entries,
            const DarknetDeltaBounds& bounds);

        /** @brief Holds the bounds a training suite measures one model against. */
        struct DarknetTrainingBounds
        {
            DarknetDeltaBounds host_delta;   /**< A delta computed from a dumped head. */
            double host_cost_relative = 0.0; /**< The stored cost of such a delta. */
            double head_absolute = 0.0;      /**< A head under batch statistics, absolute. */
            double head_relative = 0.0;      /**< A head under batch statistics, relative. */
            DarknetDeltaBounds step_delta;   /**< A delta computed from the library's head. */
            double step_cost_relative = 0.0; /**< The summed cost of the deltas of the step. */
            double head_update_share = 0.0;  /**< A head update, as a share of the dump's peak. */
            double first_update_share = 0.0; /**< A first update, as a share of the dump's peak. */
            double update_relative = 0.0;    /**< Every update, relative to the dumped entry. */
            double rolling_absolute = 0.0;   /**< A rolling statistic after the step. */
        };

        /** @brief States what one model's cfg and training runs hold, as its test wrote it. */
        struct DarknetTrainingCase
        {
            std::string_view stem;     /**< Names `<stem>.cfg` and `<stem>.weights`. */
            std::string_view host_run; /**< The run of one image with deltas and assignments. */
            std::string_view step_run; /**< The run of one step of two images. */
            Vision::Detect::Anchor::BoxLoss box_loss =
                Vision::Detect::Anchor::BoxLoss::SquaredError; /**< Box loss of every head. */
            Vision::Detect::Anchor::ObjectnessTarget objectness_target =
                Vision::Detect::Anchor::ObjectnessTarget::One; /**< Target of every head. */
            DarknetTrainingBounds bounds;                      /**< The tolerances of the suite. */
            std::int64_t overfit_steps = 0;  /**< Steps the overfit check runs on one batch. */
            double overfit_cost_share = 0.0; /**< Its last cost, as a share of the first. */
        };

        /**
         * @brief Looks for the cfg, the weights and every reference the suite of @p stated reads.
         * @return Why the first of them is missing, or an empty string when all are found.
         */
        [[nodiscard]] std::string find_missing_training_file(const DarknetTrainingCase& stated);

        /** @brief Runs on an engine, and skips every case when a file of its model is missing. */
        class DarknetTrainingTest : public EngineTest
        {
            protected:
                /** @brief Keeps the case the suite of the derived fixture measures. */
                explicit DarknetTrainingTest(DarknetTrainingCase stated);

                /** @brief Builds the engine, then skips naming the first missing file. */
                void SetUp() override;

                /** @brief Holds what the test states of its model. */
                DarknetTrainingCase training;
        };

        /**
         * @brief Checks the training options read from the cfg against what the test states and
         *        against the settings the host run's delta descriptions hold for each head.
         */
        void check_training_settings(const DarknetTrainingCase& stated);

        /**
         * @brief Checks the library's delta of every head the host run dumped: the assignments
         *        exactly, the delta within the host bounds and the stored cost.
         */
        void check_host_deltas(const DarknetTrainingCase& stated);

        /**
         * @brief Checks the library's delta of every head the step run dumped, two images each,
         *        within the host bounds, and the stored cost.
         */
        void check_two_image_deltas(const DarknetTrainingCase& stated);

        /**
         * @brief Checks each head of one step under batch statistics, and the [yolo] output
         *        computed from it, against the step run, within the head bounds.
         *
         * A NaN of a dumped [yolo] output is read as 0, the value its logistic has there.
         */
        void check_step_heads(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated);

        /**
         * @brief Checks the deltas the step fed, the summed cost against the layers' stored
         *        costs, and its mean over the heads against the network cost.
         */
        void check_step_deltas(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated);

        /**
         * @brief Checks the weight and bias gradients of the convolution before each head, times
         *        minus the batch, against the updates the step run dumped.
         */
        void check_head_updates(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated);

        /**
         * @brief Checks the weight and scale gradients of the first convolution, times minus the
         *        batch, against the dumped updates, and that its shift has a gradient.
         *
         * The dumped bias update of that layer is not a reference and is not read.
         */
        void check_first_updates(SushiBLAS::Engine& engine, const DarknetTrainingCase& stated);

        /**
         * @brief Checks the rolling mean and variance of the first batch norm: equal to the
         *        dump before the step, within the rolling bound after it.
         */
        void check_rolling_statistics(SushiBLAS::Engine& engine,
                                      const DarknetTrainingCase& stated);

        /**
         * @brief Checks that `overfit_steps` steps on the batch and truth of the step run bring
         *        the summed `HeadCost::total` below `overfit_cost_share` of the first step's,
         *        and prints both costs.
         *
         * The solver is the one every step check builds. A cost that is not finite fails at
         * its step.
         */
        void check_one_batch_overfit(SushiBLAS::Engine& engine,
                                     const DarknetTrainingCase& stated);

    } // namespace Test
} // namespace SushiAI
