/**************************************************************************/
/* darknet_forward.hpp                                                    */
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
 * @file darknet_forward.hpp
 * @brief Declares the table of the Darknet parity models, the forward run of one of them over
 *        its reference input, and the four checks every model's parity suite makes of it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <vision_support/darknet_reference.hpp>
#include <common/test_utils.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds the name of the reference of the network input in every run folder. */
        inline constexpr std::string_view K_DARKNET_INPUT = "input";

        /** @brief Holds the name of the reference of the detection rows in every run folder. */
        inline constexpr std::string_view K_DARKNET_DETECTIONS = "detections";

        /** @brief Names one reference run of a model and the score threshold it was taken at. */
        struct DarknetParityRun
        {
            std::string_view directory; /**< The folder of the run under `references`. */
            float thresh;               /**< The score threshold of the run's `detections`. */
        };

        /** @brief Names one model of the parity tests and where its files are. */
        struct DarknetParityModel
        {
            std::string_view stem;                 /**< Names `<stem>.cfg` and `<stem>.weights`. */
            std::int64_t side;                     /**< The side of the network input, in pixels. */
            std::vector<DarknetParityRun> runs;    /**< The run folders, default threshold first. */
            std::vector<std::int64_t> yolo_layers; /**< The [yolo] layer indices, in file order. */
        };

        /** @brief Lists the six models REPORT sections 6, 12 and 14 hold reference runs of. */
        [[nodiscard]] std::vector<DarknetParityModel> list_darknet_parity_models();

        /**
         * @brief Returns the row of list_darknet_parity_models whose stem is @p stem.
         * @pre The table has such a row; a stem it lacks stops the process.
         */
        [[nodiscard]] DarknetParityModel find_darknet_parity_model(std::string_view stem);

        /**
         * @brief Returns the reference name of layer @p layer, such as "layer_016_yolo".
         * @param kind The part after the index: the section type, or "yolo_input".
         */
        [[nodiscard]] std::string name_darknet_layer(std::int64_t layer, std::string_view kind);

        /** @brief Returns the name a forward run publishes the observed layer @p layer under. */
        [[nodiscard]] std::string name_observed_layer(std::int64_t layer);

        /**
         * @brief Looks for the cfg and the weights of @p stem and for @p names in @p run.
         * @return Why the first of them is missing, or an empty string when all are found.
         */
        [[nodiscard]] std::string find_missing_darknet_file(
            std::string_view stem, std::string_view run,
            SushiRuntime::span<const std::string> names);

        /** @brief Loads the reference tensor @p name of the run folder @p run. */
        [[nodiscard]] Result<DarknetTensor> load_darknet_reference(std::string_view run,
                                                                   std::string_view name);

        /**
         * @brief Translates the cfg text @p text, parsed under the origin "test.cfg".
         * @return The model, or the failure of the parse or of the translation.
         */
        [[nodiscard]] Result<Vision::Importers::DarknetModel> translate_darknet_text(
            std::string_view text, Vision::Importers::DarknetNormalisation normalisation =
                                       Vision::Importers::DarknetNormalisation::Separate);

        /**
         * @brief Loads the sections of `<stem>.cfg` of the reference data directory.
         * @return The sections; FileNotFound naming the file; or the failure of the parse.
         */
        [[nodiscard]] Result<Vision::Importers::DarknetConfig> load_darknet_config_file(
            std::string_view stem);

        /**
         * @brief Translates `<stem>.cfg` of the reference data directory.
         * @return The model; FileNotFound naming the file; or the failure of the parse or of the
         *         translation.
         */
        [[nodiscard]] Result<Vision::Importers::DarknetModel> translate_darknet_file(
            std::string_view stem, Vision::Importers::DarknetNormalisation normalisation);

        /**
         * @brief Returns the rows that row @p row of @p model reads, in the order it reads them.
         * @pre @p row is a row of the model; another index stops the process.
         */
        [[nodiscard]] std::vector<std::int64_t> sources_of_row(
            const Vision::Importers::DarknetModel& model, std::int64_t row);

        /** @brief Names one forward pass: the model, its form, its input and what is observed. */
        struct DarknetForwardRequest
        {
            /** @brief Names `<stem>.cfg` and `<stem>.weights` in the reference data directory. */
            std::string_view stem;

            /** @brief Names the run folder whose `input` the pass is fed. */
            std::string_view run;

            /** @brief Holds the form the weights are imported in. */
            Vision::Importers::DarknetNormalisation normalisation =
                Vision::Importers::DarknetNormalisation::Folded;

            /** @brief Lists the layers of the file whose outputs are published beside the heads. */
            std::vector<std::int64_t> observed_layers;
        };

        /** @brief Holds a translated model and what one forward pass of it published. */
        struct DarknetForwardRun
        {
            /** @brief Holds the translation of the cfg, without the observed outputs. */
            Vision::Importers::DarknetModel model;

            /** @brief Holds every published tensor by name, each row-major over `[1, H, W, C]`. */
            std::map<std::string, std::vector<float>, std::less<>> outputs;
        };

        /**
         * @brief Runs one evaluation step of the imported model over the run's `input`.
         *
         * A head is published under the name the translation gives it and an observed layer
         * under name_observed_layer.
         * @return The run; or the first failure of the files, the import, the build or the step;
         *         Incompatible for an observed layer the file lacks or an input of another shape.
         */
        [[nodiscard]] Result<DarknetForwardRun> run_darknet_forward(
            SushiBLAS::Engine& engine, const DarknetForwardRequest& request);

        /** @brief Returns the output @p name of @p run, or null when the run published none. */
        [[nodiscard]] const std::vector<float>* find_output(const DarknetForwardRun& run,
                                                            std::string_view name);

        /** @brief Holds the largest difference of a comparison and how many values fail it. */
        struct ToleranceGap
        {
            double largest = 0.0;    /**< The largest absolute difference. */
            std::size_t at = 0;      /**< The element that carries it. */
            std::size_t outside = 0; /**< Elements beyond the absolute plus relative bound. */
        };

        /**
         * @brief Measures @p actual against @p reference under `absolute + relative * |reference|`.
         * @pre Both spans hold the same number of values.
         */
        [[nodiscard]] ToleranceGap measure_gap(SushiRuntime::span<const float> actual,
                                               SushiRuntime::span<const float> reference,
                                               double absolute, double relative);

        /**
         * @brief Checks @p actual against @p reference and prints the largest difference measured.
         * @return Success, or a failure naming the count outside the bound and the worst element.
         */
        [[nodiscard]] ::testing::AssertionResult within_tolerance(
            std::string_view what, SushiRuntime::span<const float> actual,
            SushiRuntime::span<const float> reference, double absolute, double relative);

        /**
         * @brief Returns the [yolo] output SPEC 4.4 states for the head @p head under @p options.
         *
         * A Logit head takes the logistic of every entry but its two sizes; an Activated head
         * takes none. Both then stretch x and y to `v * s - (s - 1) / 2`, in double.
         * @param head The raw head, whose last axis holds slots of `5 + num_classes` entries.
         * @pre The class count is positive and the slot length divides the head's size.
         */
        [[nodiscard]] std::vector<float> compute_yolo_output(
            SushiRuntime::span<const float> head,
            const Vision::Detect::Anchor::HeadOptions& options);

        /** @brief Holds the bounds a parity suite measures a model against (SPEC 6.3). */
        struct DarknetParityBounds
        {
            double head_absolute = 0.0;    /**< Absolute bound on a head value. */
            double head_relative = 0.0;    /**< Relative bound on a head value. */
            double row_absolute = 0.0;     /**< Bound on a box entry and a score of a row. */
            double threshold_margin = 0.0; /**< Distance from a threshold that names a decision. */
        };

        /** @brief States what one model's cfg and reference runs hold, as its test wrote it. */
        struct DarknetParityCase
        {
            std::string_view stem;                 /**< The model's row of the parity table. */
            std::vector<std::int64_t> grids;       /**< Rows and columns of each head's grid. */
            std::vector<float> scales;             /**< The scale_x_y of each head. */
            std::vector<std::int64_t> run_rows;    /**< Detection rows of each run of the table. */
            std::int64_t classes = 0;              /**< Class entries of an anchor slot. */
            std::int64_t slots = 0;                /**< Anchor slots of each head. */
            float suppression = 0.0f;              /**< The suppression threshold of every run. */
            float distance_exponent = 0.0f;        /**< Read for DistanceIou alone. */
            DarknetParityBounds bounds;            /**< The tolerances of the suite. */
            Vision::Detect::Anchor::HeadCoding coding =
                Vision::Detect::Anchor::HeadCoding::Logit; /**< Coding of every head. */
            Vision::Detect::SuppressionCriterion criterion =
                Vision::Detect::SuppressionCriterion::Iou; /**< Criterion of the cfg. */
        };

        /**
         * @brief Looks for every file the four parity checks of @p parity read.
         * @return Why the first of them is missing, or an empty string when all are found.
         */
        [[nodiscard]] std::string find_missing_parity_file(const DarknetParityCase& parity);

        /** @brief Runs on an engine, and skips every case when a file of its model is missing. */
        class DarknetParityTest : public EngineTest
        {
            protected:
                /** @brief Keeps the case the suite of the derived fixture measures. */
                explicit DarknetParityTest(DarknetParityCase stated);

                /** @brief Builds the engine, then skips naming the first missing file. */
                void SetUp() override;

                /** @brief Holds what the test states of its model. */
                DarknetParityCase parity;
        };

        /**
         * @brief Checks every published head of the folded model against the dumped raw head of
         *        each run, within the head bounds, and prints the largest difference.
         */
        void check_folded_heads(SushiBLAS::Engine& engine, const DarknetParityCase& parity);

        /**
         * @brief Checks compute_yolo_output over each head of the folded model against the dumped
         *        [yolo] layer of each run, within the head bounds.
         */
        void check_yolo_outputs(SushiBLAS::Engine& engine, const DarknetParityCase& parity);

        /**
         * @brief Checks decode, threshold and suppression over the folded heads against the
         *        `detections` of each run: the row count, then each row sorted by class and score.
         *
         * A row matches on an equal class and on box entries and a score within the row bound.
         * Every decision within the threshold margin is printed. The reference's objectness
         * column has no counterpart in a library row and is not compared.
         */
        void check_detections(SushiBLAS::Engine& engine, const DarknetParityCase& parity);

        /**
         * @brief Checks the heads of the Separate model against those of the Folded model, within
         *        the head bounds relative to the folded value.
         */
        void check_separate_form(SushiBLAS::Engine& engine, const DarknetParityCase& parity);

    } // namespace Test
} // namespace SushiAI
