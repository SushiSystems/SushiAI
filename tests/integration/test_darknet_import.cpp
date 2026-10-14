/**************************************************************************/
/* test_darknet_import.cpp                                                */
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
 * @file test_darknet_import.cpp
 * @brief Tests that each Darknet parity model, imported in memory, has the rows, the task and
 *        the weights the parity suites build from its .cfg.
 * @author Mustafa Garip
 */

#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/vision/darknet_forward_cases/darknet_forward.hpp>
#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>

namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Runs once per parity model; a case skips when the model's files are missing. */
    class Integration_DarknetImport : public SaTest::DarknetImportTest {};

    /** @brief Lists the stems of the parity table, one case per forward suite. */
    std::vector<std::string_view> list_stems()
    {
        std::vector<std::string_view> stems;
        for (const SaTest::DarknetParityModel& model : SaTest::list_darknet_parity_models())
            stems.push_back(model.stem);

        return stems;
    }
} // namespace

/**
 * @brief Tests that the imported model file, without its task, is the text of the translated
 *        model, and that the task reads back as the anchors, the heads and the criterion.
 */
TEST_P(Integration_DarknetImport, TheImportedRowsAndTaskAreThoseOfTheTranslation)
{
    SaTest::check_imported_rows(run);
}

/**
 * @brief Tests that a state filled from the imported tensors equals, byte for byte, the state
 *        store_darknet_weights fills, and that the tensors supply every entry exactly once.
 */
TEST_P(Integration_DarknetImport, TheImportedWeightsAreThoseOfTheDirectStore)
{
    SaTest::check_imported_weights(*engine, run);
}

INSTANTIATE_TEST_SUITE_P(ParityModels, Integration_DarknetImport,
                         ::testing::ValuesIn(list_stems()));
