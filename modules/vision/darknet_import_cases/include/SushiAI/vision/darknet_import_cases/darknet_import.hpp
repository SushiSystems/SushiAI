/**************************************************************************/
/* darknet_import.hpp                                                     */
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
 * @file darknet_import.hpp
 * @brief Declares the import of a Darknet model and the checks against its dump.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds one parity model as its .cfg states it and as the importer returns it. */
        struct DarknetImportRun
        {
            /** @brief Holds the sections of `<stem>.cfg`. */
            Vision::Importers::DarknetConfig config;

            /** @brief Holds the translation, every batch norm in its own row. */
            Vision::Importers::DarknetModel model;

            /** @brief Holds the convolutions of `<stem>.weights`, unfolded. */
            Vision::Importers::DarknetWeights weights;

            /** @brief Holds what import_darknet_model returns for the three above. */
            IO::ImportedModel imported;
        };

        /**
         * @brief Imports `<stem>.cfg` and `<stem>.weights` of the reference data in memory.
         * @return The run; FileNotFound naming a missing file; or the failure of the parse, the
         *         translation, the read or the import. Nothing is written to disk.
         */
        [[nodiscard]] Result<DarknetImportRun> import_darknet_stem(std::string_view stem) noexcept;

        /**
         * @brief Checks the imported tensors against the entries of a state, strictly in both ways.
         * @param tensors The imported tensors.
         * @param entries The entries a built model's state holds; no storage is read.
         * @return Incompatible naming a tensor without an entry, of another dtype, shape or payload
         *         size, or stated twice, and an entry no tensor supplies.
         */
        [[nodiscard]] Status check_tensors_against_entries(
            SushiRuntime::span<const IO::ImportedTensor> tensors,
            SushiRuntime::span<const NN::StateEntry> entries) noexcept;

        /**
         * @brief Copies each tensor into the state entry of its name, strictly in both ways.
         * @return The refusal of check_tensors_against_entries, naming the tensor or the entry;
         *         nothing is copied when it refuses.
         */
        [[nodiscard]] Status fill_imported_state(
            SushiRuntime::span<const IO::ImportedTensor> tensors, NN::ModelState& state) noexcept;

        /**
         * @brief Checks that the imported rows and outputs equal those of the translation: the
         *        model file text without its task.
         */
        void check_imported_rows(const DarknetImportRun& run) noexcept;

        /** @brief Checks that the task reads back as the anchors, heads and criterion. */
        void check_imported_task(const DarknetImportRun& run) noexcept;

        /**
         * @brief Checks that a state built and filled from the imported model holds the bytes
         *        of the state store_darknet_weights fills from the translation, entry by entry.
         */
        void check_imported_weights(SushiBLAS::Engine& engine, const DarknetImportRun& run) noexcept;

        /**
         * @brief Checks that the imported tensors supply every entry of the imported model's
         *        state exactly once, and no other.
         */
        void check_imported_entries(SushiBLAS::Engine& engine, const DarknetImportRun& run) noexcept;

        /**
         * @brief Writes the import of @p run into @p directory and checks that the written
         *        model and training files read back as the import.
         * @param directory Absent or empty; the three files are written into it.
         */
        void check_written_files(const DarknetImportRun& run,
                                 const std::filesystem::path& directory) noexcept;

        /**
         * @brief Writes the import of @p run into @p directory and checks that load_checkpoint
         *        fills a state with the bytes of the state store_darknet_weights fills.
         * @param directory Absent or empty; the three files are written into it.
         */
        void check_written_weights(SushiBLAS::Engine& engine, const DarknetImportRun& run,
                                   const std::filesystem::path& directory) noexcept;

        /** @brief Imports the model its parameter names; skips when a file of it is missing. */
        class DarknetImportTest : public EngineTest,
                                  public ::testing::WithParamInterface<std::string_view>
        {
            protected:
                /** @brief Builds the engine, then imports the stem or skips naming a file. */
                void SetUp() override;

                /** @brief Holds the import of the stem the case was instantiated with. */
                DarknetImportRun run;
        };

    } // namespace Test
} // namespace SushiAI
