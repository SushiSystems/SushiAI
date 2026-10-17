/**************************************************************************/
/* inference_run.hpp                                                      */
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
 * @file inference_run.hpp
 * @brief Declares the table of the reports `sushiai inference` picks by the task of a model.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The report table".
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/evaluation_step.hpp>

namespace SushiAI
{
    namespace NN
    {
        class ModelGraph;
        class ModelState;
    } // namespace NN

    namespace Apps
    {
        /**
         * @struct PreparedInference
         * @brief Holds what `sushiai inference` built before a report starts; a report owns none.
         */
        struct PreparedInference
        {
            /** @brief The model file, as the command line names it. */
            const std::filesystem::path& model_path;

            /** @brief The weight file, as the command line names it. */
            const std::filesystem::path& weights_path;

            /** @brief The parsed model file: its input shape and its task section. */
            const Config::ModelSpec& model;

            /** @brief How many rows the built model has. */
            std::size_t layers;

            /** @brief The batches the report runs over. */
            const Data::Dataset& dataset;

            /** @brief The compiled forward pass, with the weights loaded. */
            Train::EvaluationStep& step;

            /** @brief What the model publishes, by name, in the order it publishes them. */
            const std::vector<NN::ModelOutputs::Entry>& outputs;

            /** @brief The engine the step runs on, for a report that allocates a feed. */
            SushiBLAS::Engine& engine;

            /** @brief The built model, for a report that traces another form; may be null. */
            NN::ModelGraph* built_model = nullptr;

            /** @brief The loaded state every form of the model reads; may be null. */
            const NN::ModelState* state = nullptr;
        };

        /**
         * @brief Runs the model over every batch and prints what its outputs mean.
         * @return Success, or the first failure of a read, a step or the task section.
         */
        using InferenceFunction = Status (*)(PreparedInference& run);

        /**
         * @struct InferenceEntry
         * @brief One row of an InferenceRegistry.
         */
        struct InferenceEntry
        {
            /** @brief The "type" of the task section whose models the row reports. */
            std::string_view task;

            /** @brief What runs the model and prints the report. */
            InferenceFunction report;
        };

        /** @brief Holds the reports `sushiai inference` may pick, by task type. */
        class InferenceRegistry final
        {
            public:
                /**
                 * @brief Appends a report row.
                 * @param row Its type text must outlive the registry; an empty type or a null
                 *            function is fatal.
                 * @return InvalidDescription when a row holds the type.
                 */
                [[nodiscard]] Status add(const InferenceEntry& row);

                /** @brief Returns the row of @p task, or null when none holds it. */
                [[nodiscard]] const InferenceEntry* find(std::string_view task) const noexcept;

                /**
                 * @brief Returns the rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const InferenceEntry> rows() const noexcept;

            private:
                /** @brief Holds the rows in insertion order. */
                std::vector<InferenceEntry> rows_;
        };

        /**
         * @brief Prints the header every report opens with: the model, the weights, the data.
         * @param outputs What the model publishes, in the words of the report.
         */
        void print_inference_header(const PreparedInference& run, std::string_view outputs);

        /**
         * @brief Runs the report of the row @p task names.
         * @return What the row returns; InvalidDescription, naming every task type the
         *         registry reports, when @p task is no row.
         */
        [[nodiscard]] Status report_inference(const InferenceRegistry& registry,
                                              std::string_view task, PreparedInference& run);

    } // namespace Apps
} // namespace SushiAI
