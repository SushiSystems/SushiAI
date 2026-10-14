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
 * @file darknet_import.hpp
 * @brief Declares the import of a Darknet model and the checks against its dump.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <SushiAI/SushiAI.h>
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
        [[nodiscard]] Result<DarknetImportRun> import_darknet_stem(std::string_view stem);

        /**
         * @brief Copies each tensor into the state entry of its name, strictly in both ways.
         * @return Incompatible naming a tensor without an entry, of another dtype or shape or
         *         stated twice, and an entry no tensor supplies.
         */
        [[nodiscard]] Status fill_imported_state(
            SushiRuntime::span<const IO::ImportedTensor> tensors, NN::ModelState& state);

        /**
         * @brief Checks that the imported rows, outputs and task equal those of the translation:
         *        the model file text without its task, then the task read back field by field.
         */
        void check_imported_rows(const DarknetImportRun& run);

        /**
         * @brief Checks that a state built and filled from the imported model holds the bytes
         *        of the state store_darknet_weights fills from the translation, entry by entry.
         */
        void check_imported_weights(SushiBLAS::Engine& engine, const DarknetImportRun& run);

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
