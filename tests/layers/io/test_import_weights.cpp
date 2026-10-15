/**************************************************************************/
/* test_import_weights.cpp                                                */
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
 * @file test_import_weights.cpp
 * @brief Tests the prefix rules of the strict loader and its refusals against a ModelState.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_ImportWeights : public SaTest::EngineTest {};

    using SaTest::ScratchFile;

    /** @brief One record to write, with the tensor whose bytes its payload carries. */
    struct Source
    {
        /** @brief The header the record is written with. */
        IO::TensorRecord record;

        /** @brief The tensor whose bytes form the payload. */
        const SushiBLAS::Tensor* tensor;
    };

    /** @brief Returns a record called @p name with the extents @p shape, in float32. */
    IO::TensorRecord make_record(std::string name, Shape shape)
    {
        IO::TensorRecord record;
        record.name = std::move(name);
        record.shape = shape;
        return record;
    }

    /** @brief Describes an entry of a given name, identity and extents, for from_entries. */
    NN::StateEntry make_entry(std::string name, std::uint64_t state, Shape shape)
    {
        NN::StateEntry entry;
        entry.name = std::move(name);
        entry.state = Graph::StateID{state};
        entry.shape = shape;
        return entry;
    }

    /** @brief The two entries every integration case builds a state from. */
    std::vector<NN::StateEntry> two_entries()
    {
        return {make_entry("0.weight", 1, Shape{2, 3}), make_entry("0.bias", 2, Shape{3})};
    }

    /** @brief Returns the tensor of the entry @p name in @p state. */
    SushiBLAS::Tensor& tensor_of(NN::ModelState& state, std::string_view name)
    {
        return *state.tensor(name).value();
    }

    /** @brief Writes @p base + 0.5 * index into every element of the entry @p name. */
    void fill_tensor(NN::ModelState& state, std::string_view name, float base)
    {
        SushiBLAS::Tensor& tensor = tensor_of(state, name);
        float* values = tensor.data_as<float>();
        for (std::int64_t i = 0; i < tensor.num_elements; ++i)
            values[i] = base + static_cast<float>(i) * 0.5f;
    }

    /** @brief Writes a checkpoint of @p sources to @p path, in the order given. */
    void write_checkpoint(const std::filesystem::path& path, const std::vector<Source>& sources)
    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        SA_ASSERT_OK_AND_ASSIGN(IO::CheckpointWriter writer,
                                IO::CheckpointWriter::create(
                                    out, static_cast<std::uint32_t>(sources.size())));
        for (const Source& source : sources)
            SA_EXPECT_OK(writer.write(source.record, source.tensor->data()));
        writer.finish();
    }
} // namespace

/** @brief Tests that a rule replaces a matching prefix and keeps the rest of the name. */
TEST(Unit_RenameRule, RewritesAMatchingPrefix)
{
    const IO::RenameRule rules[] = {{"model.", ""}};
    EXPECT_EQ(IO::renamed("model.0.weight", rules), "0.weight");
}

/** @brief Tests that a name no rule's prefix starts is returned unchanged. */
TEST(Unit_RenameRule, LeavesANameNoRuleMatches)
{
    const IO::RenameRule rules[] = {{"model.", ""}};
    EXPECT_EQ(IO::renamed("other.0.weight", rules), "other.0.weight");
}

/** @brief Tests that the first rule whose prefix matches decides the name, not the longest. */
TEST(Unit_RenameRule, TheFirstMatchingRuleWins)
{
    const IO::RenameRule rules[] = {{"model.", "first."}, {"model.0", "second"}};
    EXPECT_EQ(IO::renamed("model.0.weight", rules), "first.0.weight");
}

/** @brief Tests that an empty to-prefix removes the matched prefix from the name. */
TEST(Unit_RenameRule, AnEmptyToPrefixStripsThePrefix)
{
    const IO::RenameRule rules[] = {{"model.", ""}};
    EXPECT_EQ(IO::renamed("model.1.bias", rules), "1.bias");
}

/**
 * @brief Tests that a file written under foreign names loads through a rule, with every
 *        record landing in its entry bit for bit.
 */
TEST_F(Integration_ImportWeights, ARuleStripsAForeignPrefixAndEveryRecordLands)
{
    const ScratchFile file("sushiai_import_rule.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState source,
                            NN::ModelState::from_entries(*engine, two_entries()));
    fill_tensor(source, "0.weight", 1.25f);
    fill_tensor(source, "0.bias", -3.0f);
    write_checkpoint(file.path(),
                     {{make_record("model.0.weight", Shape{2, 3}),
                       &tensor_of(source, "0.weight")},
                      {make_record("model.0.bias", Shape{3}), &tensor_of(source, "0.bias")}});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    fill_tensor(target, "0.weight", 9.0f);
    fill_tensor(target, "0.bias", 9.0f);
    ASSERT_NE(SaTest::read_tensor_bytes(tensor_of(target, "0.weight")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.weight")));

    const IO::RenameRule rules[] = {{"model.", ""}};
    SA_EXPECT_OK(IO::import_weights(file.path(), target, rules));
    EXPECT_EQ(SaTest::read_tensor_bytes(tensor_of(target, "0.weight")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.weight")));
    EXPECT_EQ(SaTest::read_tensor_bytes(tensor_of(target, "0.bias")),
              SaTest::read_tensor_bytes(tensor_of(source, "0.bias")));
}

/** @brief Tests that without a rule the first foreign record is refused, and named. */
TEST_F(Integration_ImportWeights, WithoutTheRuleTheFirstRecordIsRefused)
{
    const ScratchFile file("sushiai_import_no_rule.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState source,
                            NN::ModelState::from_entries(*engine, two_entries()));
    write_checkpoint(file.path(),
                     {{make_record("model.0.weight", Shape{2, 3}),
                       &tensor_of(source, "0.weight")}});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    SA_EXPECT_FAILURE(IO::import_weights(file.path(), target,
                                         SushiRuntime::span<const IO::RenameRule>()),
                      StatusCode::Incompatible, "'model.0.weight'");
}

/**
 * @brief Tests that a record whose name matches no entry is refused, with the file's name and
 *        the renamed one both in the text.
 */
TEST_F(Integration_ImportWeights, ARecordTheStateLacksIsRefusedUnderBothNames)
{
    const ScratchFile file("sushiai_import_unknown.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState source,
                            NN::ModelState::from_entries(*engine, two_entries()));
    write_checkpoint(file.path(),
                     {{make_record("model.9.weight", Shape{2, 3}),
                       &tensor_of(source, "0.weight")}});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    const IO::RenameRule rules[] = {{"model.", ""}};
    SA_EXPECT_FAILURE(IO::import_weights(file.path(), target, rules),
                      StatusCode::Incompatible, "holds 'model.9.weight'",
                      "renames to '9.weight'");
}

/** @brief Tests that an entry no record supplies is refused, and the refusal names it. */
TEST_F(Integration_ImportWeights, AnEntryTheFileLacksIsRefusedAndNamed)
{
    const ScratchFile file("sushiai_import_missing.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState source,
                            NN::ModelState::from_entries(*engine, two_entries()));
    write_checkpoint(file.path(),
                     {{make_record("model.0.weight", Shape{2, 3}),
                       &tensor_of(source, "0.weight")}});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    const IO::RenameRule rules[] = {{"model.", ""}};
    SA_EXPECT_FAILURE(IO::import_weights(file.path(), target, rules),
                      StatusCode::Incompatible, "does not supply", "'0.bias'");
}

/** @brief Tests that a record whose shape differs from its entry is refused with both shapes. */
TEST_F(Integration_ImportWeights, AShapeMismatchIsRefusedWithBothShapes)
{
    const ScratchFile file("sushiai_import_shape.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState source,
                            NN::ModelState::from_entries(*engine, two_entries()));
    // The same six elements, under a shape the entry does not have.
    write_checkpoint(file.path(),
                     {{make_record("model.0.weight", Shape{3, 2}),
                       &tensor_of(source, "0.weight")}});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    const IO::RenameRule rules[] = {{"model.", ""}};
    const std::string file_shape = Shape{3, 2}.to_string();
    const std::string model_shape = Shape{2, 3}.to_string();
    SA_EXPECT_FAILURE(IO::import_weights(file.path(), target, rules),
                      StatusCode::Incompatible, file_shape, model_shape);
}

/** @brief Tests that a path that cannot be opened is refused as FileNotFound. */
TEST_F(Integration_ImportWeights, AMissingFileIsFileNotFound)
{
    const ScratchFile file("sushiai_import_absent.sushi");

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState target,
                            NN::ModelState::from_entries(*engine, two_entries()));
    SA_EXPECT_FAILURE(IO::import_weights(file.path(), target,
                                         SushiRuntime::span<const IO::RenameRule>()),
                      StatusCode::FileNotFound, "could not open");
}
