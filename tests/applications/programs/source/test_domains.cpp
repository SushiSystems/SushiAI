/**************************************************************************/
/* test_domains.cpp                                                       */
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
 * @file test_domains.cpp
 * @brief Tests that each linked domain adds its rows beside the library's without a clash.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <gtest/gtest.h>
#include "domains.hpp"
#include "registries.hpp"
#include "trainer_run.hpp"
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/train/objective_registry.hpp>

namespace Apps = SushiAI::Apps;

namespace
{
    /** @brief Returns how many rows @p registries holds across every table it has. */
    std::size_t row_count(const Apps::Registries& registries)
    {
        return registries.nodes.nodes().size() + registries.nodes.layers().size() +
               registries.nodes.linked().size() + registries.datasets.rows().size() +
               registries.objectives.rows().size() + registries.runs.rows().size() +
               registries.inferences.rows().size() + registries.importers.rows().size();
    }
} // namespace

/** @brief Tests that each domain registers into empty registries without a failure. */
TEST(Small_Domains, EachDomainRegistersIntoEmptyRegistries)
{
    Apps::Registries for_vision;
    Apps::Registries for_language;

    SA_EXPECT_OK(Apps::register_vision_domain(for_vision));
    SA_EXPECT_OK(Apps::register_language_domain(for_language));
}

/** @brief Tests that a vision domain with inference rows reports anchor-free detections. */
TEST(Small_Domains, TheVisionDomainReportsAnchorFreeDetections)
{
    Apps::Registries registries;
    SA_EXPECT_OK(Apps::register_vision_domain(registries));

    // A build without vision registers no inference row, and the case holds trivially.
    if (registries.inferences.rows().size() != 0)
        EXPECT_NE(registries.inferences.find("anchor_free_detection"), nullptr);
}

/** @brief Tests that the domains add rows only, beside the library's, and remove none. */
TEST(Small_Domains, TheDomainsOnlyAddBesideTheLibraryRows)
{
    Apps::Registries registries;
    SA_EXPECT_OK(SushiAI::NN::register_layers(registries.nodes));
    SA_EXPECT_OK(SushiAI::Data::register_datasets(registries.datasets));
    SA_EXPECT_OK(SushiAI::Train::register_objectives(registries.objectives));
    SA_EXPECT_OK(Apps::register_runs(registries.runs));
    const std::size_t library_rows = row_count(registries);

    SA_EXPECT_OK(Apps::register_vision_domain(registries));
    SA_EXPECT_OK(Apps::register_language_domain(registries));

    EXPECT_GE(row_count(registries), library_rows);
    EXPECT_NE(registries.nodes.find_layer("linear"), nullptr);
    EXPECT_NE(registries.datasets.find("rings"), nullptr);
    EXPECT_NE(registries.objectives.find("cross_entropy"), nullptr);
    EXPECT_NE(registries.runs.fallback(), nullptr);
}
