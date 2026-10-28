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
 * @brief Tests the strict copy of imported tensors into a model state and a missing import.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/darknet_import_cases/darknet_import.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides an engine for a state built from the two entries every case fills. */
    class Medium_DarknetImportCases : public SaTest::EngineTest
    {

    };

    /** @brief Describes a float32 entry of a given name, identity and extents. */
    NN::StateEntry make_entry(std::string name, std::uint64_t state, Shape shape)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.state = Graph::StateID{state};
        entry.shape = shape;
        return entry;
    }

    /** @brief Returns the entries "0.weight" of [2, 3] and "0.bias" of [3]. */
    std::vector<NN::StateEntry> two_entries()
    {
        return {make_entry("0.weight", 1, Shape{2, 3}), make_entry("0.bias", 2, Shape{3})};
    }

    /** @brief Returns a float32 tensor of @p shape whose element i holds base + i. */
    IO::ImportedTensor make_tensor(std::string name, Shape shape, float base)
    {
        IO::ImportedTensor tensor;
        tensor.record.name = std::move(name);
        tensor.record.shape = shape;
        std::vector<float> values(static_cast<std::size_t>(shape.num_elements()));
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = base + static_cast<float>(i);
        tensor.payload.resize(values.size() * sizeof(float));
        std::memcpy(tensor.payload.data(), values.data(), tensor.payload.size());
        return tensor;
    }

    /** @brief Returns the values of the entry @p name of @p state. */
    std::vector<float> values_of(NN::ModelState& state, const std::string& name)
    {
        SushiBLAS::Tensor& tensor = *state.tensor(name).value();
        const float* data = tensor.data_as<float>();
        return std::vector<float>(data, data + tensor.num_elements);
    }
} // namespace

/** @brief Tests that the import of a stem no reference holds fails naming its cfg. */
TEST(Small_DarknetImport, AStemWithoutFilesIsNotFound)
{
    SA_EXPECT_FAILURE(SaTest::import_darknet_stem("sushiai_no_such_model"),
                      StatusCode::FileNotFound, "sushiai_no_such_model.cfg");
}

/** @brief Tests that every tensor lands in the entry of its name, byte for byte. */
TEST_F(Medium_DarknetImportCases, EveryTensorLandsInTheEntryOfItsName)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, two_entries()));
    const std::vector<IO::ImportedTensor> tensors = {make_tensor("0.bias", Shape{3}, 10.0f),
                                                     make_tensor("0.weight", Shape{2, 3}, 0.0f)};

    SA_EXPECT_OK(SaTest::fill_imported_state(tensors, state));

    // The payloads are copied and never computed, so they compare exactly.
    EXPECT_EQ(values_of(state, "0.weight"),
              (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
    EXPECT_EQ(values_of(state, "0.bias"), (std::vector<float>{10.0f, 11.0f, 12.0f}));
}

/** @brief Tests that a tensor the state has no entry for is refused by name. */
TEST_F(Medium_DarknetImportCases, ATensorWithoutAnEntryIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, two_entries()));
    const std::vector<IO::ImportedTensor> tensors = {make_tensor("1.weight", Shape{3}, 0.0f)};

    SA_EXPECT_FAILURE(SaTest::fill_imported_state(tensors, state), StatusCode::Incompatible,
                      "'1.weight'", "no such entry");
}

/** @brief Tests that a tensor stated twice, or of another shape, is refused by name. */
TEST_F(Medium_DarknetImportCases, ARepeatedOrMisshapenTensorIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, two_entries()));
    const std::vector<IO::ImportedTensor> twice = {make_tensor("0.bias", Shape{3}, 0.0f),
                                                   make_tensor("0.bias", Shape{3}, 0.0f)};
    const std::vector<IO::ImportedTensor> misshapen = {make_tensor("0.weight", Shape{3, 2}, 0.0f)};

    SA_EXPECT_FAILURE(SaTest::fill_imported_state(twice, state), StatusCode::Incompatible,
                      "'0.bias' twice");
    SA_EXPECT_FAILURE(SaTest::fill_imported_state(misshapen, state), StatusCode::Incompatible,
                      "'0.weight'", "shape");
}

/** @brief Tests that an entry no tensor supplies is refused by name. */
TEST_F(Medium_DarknetImportCases, AnEntryNoTensorSuppliesIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state,
                            NN::ModelState::from_entries(*engine, two_entries()));
    const std::vector<IO::ImportedTensor> tensors = {make_tensor("0.weight", Shape{2, 3}, 0.0f)};

    SA_EXPECT_FAILURE(SaTest::fill_imported_state(tensors, state), StatusCode::Incompatible,
                      "does not supply", "'0.bias'");
}
