/**************************************************************************/
/* test_train_config.cpp                                                  */
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
 * @file test_train_config.cpp
 * @brief Tests how train and eval documents parse into their typed blocks and which ones they
 *        refuse.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/model_text/model_text.hpp>

using namespace SushiAI;
using Config::Kind;
using Config::TrainSpec;
using Config::Value;
using SushiAI::Config::Kind;
using SushiAI::Config::Value;
using SushiAI::Config::parse_json;

namespace
{
    /** @brief Holds the members every training file states, without the closing brace. */
    constexpr std::string_view K_REQUIRED_MEMBERS = R"({
        "sushiai_train": 1,
        "model": "model.json",
        "objective": {"type": "anchor"},
        "data": {"type": "rings"},
        "optimizer": {"type": "sgd", "momentum": 0.9})";

    /** @brief Holds a schedule member with a key the section reader does not have. */
    constexpr std::string_view K_SCHEDULE_MEMBER =
        R"("schedule": {"type": "step", "base_rate": 0.001, "unread_here": [1, 2]})";

    /** @brief Parses @p text as a train document, resolving its paths against @p base. */
    Result<TrainSpec> parse_text(const std::string& text, const std::filesystem::path& base)
    {
        SA_TRY_ASSIGN(const Value root, Config::parse_json(text, "<test>"));
        return Config::parse_train(root, "<test>", base);
    }

    /** @brief Parses the required members followed by @p members, each led by a comma. */
    Result<TrainSpec> parse_with(std::string_view members)
    {
        return parse_text(std::string(K_REQUIRED_MEMBERS) + std::string(members) + "}",
                          std::filesystem::path("."));
    }
} // namespace

/** @brief Tests that a file without the section parses as before, with a null schedule. */
TEST(Medium_TrainConfig, AFileWithoutAScheduleHoldsTheNullValue)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec, parse_with(""));

    EXPECT_TRUE(spec.schedule.is_null());
    EXPECT_EQ(spec.optimizer_type, "sgd");
    EXPECT_EQ(spec.epochs, 1);
}

/** @brief Tests that the section is kept whole, its "type" and an unread key included. */
TEST(Medium_TrainConfig, TheScheduleIsCarriedAsTheFileStatesIt)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec,
                            parse_with(", " + std::string(K_SCHEDULE_MEMBER)));

    ASSERT_EQ(spec.schedule.kind(), Kind::Object);
    ASSERT_EQ(spec.schedule.keys().size(), 3u);
    SA_ASSERT_OK_AND_ASSIGN(const Value& type, spec.schedule.at("type", "schedule"));
    SA_ASSERT_OK_AND_ASSIGN(const std::string& type_name, type.as_string("type"));
    EXPECT_EQ(type_name, "step");
    SA_ASSERT_OK_AND_ASSIGN(const Value& base_rate, spec.schedule.at("base_rate", "schedule"));
    EXPECT_EQ(base_rate.kind(), Kind::Real);
    EXPECT_TRUE(spec.schedule.has("unread_here"));
}

/** @brief Tests that the parser accepts a schedule of any kind and leaves the check to optim. */
TEST(Medium_TrainConfig, AScheduleOfAnotherKindIsCarriedUnchecked)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec, parse_with(R"(, "schedule": "later")"));

    EXPECT_EQ(spec.schedule.kind(), Kind::String);
}

/** @brief Tests that a key beside the known ones is still refused by name. */
TEST(Medium_TrainConfig, AKeyBesideTheScheduleIsStillRefused)
{
    SA_EXPECT_FAILURE(parse_with(R"(, "schedules": {})"), StatusCode::InvalidDescription,
                      "'schedules'");
}

/** @brief Tests that a file without "weights" leaves the path empty. */
TEST(Medium_TrainConfig, AFileWithoutWeightsHoldsAnEmptyPath)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec, parse_with(""));
    EXPECT_TRUE(spec.weights.empty());
}

/** @brief Tests that "weights" resolves against the directory of the file, as "model" does. */
TEST(Medium_TrainConfig, TheWeightsPathResolvesAgainstTheFile)
{
    const std::string text =
        std::string(K_REQUIRED_MEMBERS) + R"(, "weights": "start.sushi"})";
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec,
                            parse_text(text, std::filesystem::path("runs")));

    EXPECT_EQ(spec.weights, std::filesystem::path("runs") / "start.sushi");
    EXPECT_EQ(spec.model_path, std::filesystem::path("runs") / "model.json");
}

/** @brief Tests that "weights" of another kind than a string is refused by name. */
TEST(Medium_TrainConfig, WeightsThatAreNoStringAreRefused)
{
    SA_EXPECT_FAILURE(parse_with(R"(, "weights": 3)"), StatusCode::InvalidDescription, "'weights'");
}

/** @brief Tests that "resume" is empty when left out and resolves against the file. */
TEST(Medium_TrainConfig, TheResumePathResolvesAgainstTheFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec bare, parse_with(""));
    EXPECT_TRUE(bare.resume.empty());

    const std::string text =
        std::string(K_REQUIRED_MEMBERS) + R"(, "resume": "last.sushi"})";
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec,
                            parse_text(text, std::filesystem::path("runs")));

    EXPECT_EQ(spec.resume, std::filesystem::path("runs") / "last.sushi");
    EXPECT_TRUE(spec.weights.empty());
}

/** @brief Tests that "resume" of another kind than a string is refused by name. */
TEST(Medium_TrainConfig, AResumeThatIsNoStringIsRefused)
{
    SA_EXPECT_FAILURE(parse_with(R"(, "resume": true)"), StatusCode::InvalidDescription,
                      "'resume'");
}

/** @brief Tests that a file states "weights" or "resume", and is refused with both. */
TEST(Medium_TrainConfig, WeightsAndResumeTogetherAreRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec bare, parse_with(""));
    SA_EXPECT_OK(Config::check_start(bare, "train.json"));

    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec weights, parse_with(R"(, "weights": "a.sushi")"));
    SA_EXPECT_OK(Config::check_start(weights, "train.json"));

    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec resume, parse_with(R"(, "resume": "a.sushi")"));
    SA_EXPECT_OK(Config::check_start(resume, "train.json"));

    SA_ASSERT_OK_AND_ASSIGN(
        const TrainSpec both, parse_with(R"(, "weights": "a.sushi", "resume": "b.sushi")"));
    SA_EXPECT_FAILURE(Config::check_start(both, "train.json"), StatusCode::InvalidDescription,
                      "'train.json'", "'weights'", "'resume'");
}

/** @brief Tests that "seed" is 0 when left out and holds the integer the file states. */
TEST(Medium_TrainConfig, TheSeedIsZeroUnlessTheFileStatesOne)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec bare, parse_with(""));
    EXPECT_EQ(bare.seed, 0);

    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec, parse_with(R"(, "seed": 20261011)"));
    EXPECT_EQ(spec.seed, 20261011);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t seed, Config::read_run_seed(spec, "train.json"));
    EXPECT_EQ(seed, 20261011u);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t absent,
                            Config::read_run_seed(bare, "train.json"));
    EXPECT_EQ(absent, 0u);
}

/** @brief Tests that a "seed" that is no integer is refused by name. */
TEST(Medium_TrainConfig, ASeedThatIsNoIntegerIsRefused)
{
    SA_EXPECT_FAILURE(parse_with(R"(, "seed": "seven")"), StatusCode::InvalidDescription, "'seed'");
    SA_EXPECT_FAILURE(parse_with(R"(, "seed": 0.5)"), StatusCode::InvalidDescription, "'seed'");
}

/** @brief Tests that a negative "seed" parses and is refused with the file and the value. */
TEST(Medium_TrainConfig, ANegativeSeedIsRefused)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec negative, parse_with(R"(, "seed": -3)"));
    SA_EXPECT_FAILURE(Config::read_run_seed(negative, "train.json"),
                      StatusCode::InvalidDescription, "'train.json'", "'seed'", "-3");
}

namespace
{
    /** @brief Parses a training file whose data block is @p data. */
    Result<TrainSpec> parse_with_data(std::string_view data)
    {
        const std::string text = R"({"sushiai_train": 1, "model": "model.json",
            "objective": {"type": "anchor"}, "optimizer": {"type": "sgd"}, "data": )"
            + std::string(data) + "}";
        return parse_text(text, std::filesystem::path("."));
    }
} // namespace

/** @brief Tests that a data block with a reader and no type takes the reader type whole. */
TEST(Medium_TrainConfig, ADataBlockWithAReaderTakesTheReaderType)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const TrainSpec spec,
        parse_with_data(R"({"reader": "yolo_txt", "task": "detection", "root": "data"})"));

    EXPECT_EQ(spec.data.type, Config::K_READER_DATA_TYPE);
    ASSERT_EQ(spec.data.attributes.keys().size(), 3u);
    SA_ASSERT_OK_AND_ASSIGN(const Value& reader, spec.data.attributes.at("reader", "data"));
    SA_ASSERT_OK_AND_ASSIGN(const std::string& reader_name, reader.as_string("reader"));
    EXPECT_EQ(reader_name, "yolo_txt");
}

/** @brief Tests that a typed data block is read as before, its type stripped. */
TEST(Medium_TrainConfig, ATypedDataBlockKeepsItsType)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec,
                            parse_with_data(R"({"type": "mnist", "directory": "data"})"));

    EXPECT_EQ(spec.data.type, "mnist");
    EXPECT_FALSE(spec.data.attributes.has("type"));
    EXPECT_TRUE(spec.data.attributes.has("directory"));
}

/** @brief Tests that a block with both keys is a typed block that still carries "reader". */
TEST(Medium_TrainConfig, ATypeBesideAReaderStaysATypedBlock)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec,
                            parse_with_data(R"({"type": "mnist", "reader": "idx"})"));

    EXPECT_EQ(spec.data.type, "mnist");
    EXPECT_TRUE(spec.data.attributes.has("reader"));
}

/** @brief Tests that a data block with neither key is refused for its missing type. */
TEST(Medium_TrainConfig, ADataBlockWithNeitherKeyIsRefused)
{
    SA_EXPECT_FAILURE(parse_with_data(R"({"root": "data"})"), StatusCode::InvalidDescription,
                      "'type'");
}

namespace
{
    /** @brief Parses @p text as a train document, resolving paths against ".". */
    Result<SushiAI::Config::TrainSpec> train_from(std::string_view text)
    {
        SA_TRY_ASSIGN(const Value root, parse_json(text, "train.json"));
        return SushiAI::Config::parse_train(root, "train.json", ".");
    }

    constexpr std::string_view K_GOOD_TRAIN = R"({
        "sushiai_train": 1,
        "model": "model.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings", "batch_size": 8, "batches": 2, "classes": 4},
        "optimizer": {"type": "adamw", "learning_rate": 0.003},
        "epochs": 3,
        "checkpoint": "out/model.sushi"
    })";

    /**
     * @class TemporaryDirectory
     * @brief A directory that removes itself, so a failing test leaves nothing behind.
     */
    class TemporaryDirectory
    {
        public:
            explicit TemporaryDirectory(const char* name)
                : path_(std::filesystem::temp_directory_path() / name)
            {
                std::filesystem::remove_all(path_);
                std::filesystem::create_directories(path_);
            }

            ~TemporaryDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all(path_, ignored);
            }

            TemporaryDirectory(const TemporaryDirectory&) = delete;
            TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

            const std::filesystem::path& path() const noexcept { return path_; }

            /** @brief Writes @p text into this directory as @p name. */
            std::filesystem::path write(const char* name, std::string_view text) const
            {
                const std::filesystem::path file = path_ / name;
                std::ofstream out(file, std::ios::binary);
                out.write(text.data(), static_cast<std::streamsize>(text.size()));
                return file;
            }

        private:
            std::filesystem::path path_;
    };

} // namespace

TEST(Medium_TrainConfig, ParsesATrainDocumentIntoItsFourTypedBlocks)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::TrainSpec spec, train_from(K_GOOD_TRAIN));

    EXPECT_EQ(spec.objective_type, "cross_entropy");
    EXPECT_EQ(spec.data.type, "rings");
    EXPECT_EQ(spec.optimizer_type, "adamw");
    EXPECT_EQ(spec.epochs, 3);

    // Each block's "type" is separated out, so no factory has to allow it.
    EXPECT_FALSE(spec.optimizer_attributes.has("type"));
    SA_ASSERT_OK_AND_ASSIGN(const Value& rate,
                            spec.optimizer_attributes.at("learning_rate", "optimizer"));
    SA_ASSERT_OK_AND_ASSIGN(const double learning_rate, rate.as_real("learning_rate"));
    EXPECT_DOUBLE_EQ(learning_rate, 0.003);
    SA_ASSERT_OK_AND_ASSIGN(const Value& batch, spec.data.attributes.at("batch_size", "data"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t batch_size, batch.as_int("batch_size"));
    EXPECT_EQ(batch_size, 8);
}

TEST(Medium_TrainConfig, ResolvesRelativePathsAgainstTheConfigsOwnDirectory)
{
    const TemporaryDirectory directory("sushiai_config_paths");
    const std::filesystem::path config = directory.write("train.json", K_GOOD_TRAIN);

    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::TrainSpec spec,
                            SushiAI::Config::load_train(config));

    EXPECT_EQ(spec.model_path, directory.path() / "model.json");
    EXPECT_EQ(spec.checkpoint, directory.path() / "out" / "model.sushi");
}

TEST(Medium_TrainConfig, LeavesAnAbsolutePathAlone)
{
    const std::filesystem::path absolute =
        std::filesystem::temp_directory_path() / "elsewhere" / "model.json";

    // generic_string() writes '/', which JSON does not escape; see tests/core/config/README.md.
    const std::string text =
        R"({"sushiai_train": 1, "model": ")" + absolute.generic_string() +
        R"(", "objective": {"type": "cross_entropy"},
            "data": {"type": "rings"}, "optimizer": {"type": "adamw"}})";

    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(text, "train.json"));
    SA_ASSERT_OK_AND_ASSIGN(
        const SushiAI::Config::TrainSpec spec,
        SushiAI::Config::parse_train(root, "train.json", "/some/other/place"));
    EXPECT_EQ(spec.model_path, absolute);
}

TEST(Medium_TrainConfig, EpochsDefaultsToOneAndMustBePositive)
{
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::TrainSpec defaulted, train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"));
    EXPECT_EQ(defaulted.epochs, 1);
    EXPECT_TRUE(defaulted.checkpoint.empty());

    SA_EXPECT_FAILURE(train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "epochs": 0
    })"), StatusCode::InvalidDescription, "'epochs'", "positive");
}

TEST(Medium_TrainConfig, AnUnknownTopLevelTrainKeyIsAnError)
{
    SA_EXPECT_FAILURE(train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "epochz": 3
    })"), StatusCode::InvalidDescription, "'epochz'", "'epochs'");
}

TEST(Medium_TrainConfig, AMissingTrainBlockNamesIt)
{
    SA_EXPECT_FAILURE(train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "optimizer": {"type": "adamw"}
    })"), StatusCode::InvalidDescription, "'data'");
}

TEST(Medium_TrainConfig, ATypedBlockThatIsNotAnObjectIsAnError)
{
    SA_EXPECT_FAILURE(train_from(R"({
        "sushiai_train": 1, "model": "m.json",
        "objective": "cross_entropy",
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"), StatusCode::InvalidDescription, "'objective'", "a string");
}

TEST(Medium_TrainConfig, AFutureTrainVersionIsAnError)
{
    SA_EXPECT_FAILURE(train_from(R"({
        "sushiai_train": 4, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"), StatusCode::InvalidFile, "version 4", "version 3");
}

/** @brief Tests that a version 1 file without a cluster section still reads. */
TEST(Medium_TrainConfig, AVersionOneFileStillReads)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec, parse_with(""));

    EXPECT_EQ(spec.optimizer_type, "sgd");
    EXPECT_TRUE(spec.cluster.is_null());
}

/** @brief Tests that a version 1 file that states "cluster" is refused, naming the key. */
TEST(Medium_TrainConfig, AClusterInAVersionOneFileIsRefused)
{
    SA_EXPECT_FAILURE(parse_with(R"(, "cluster": {})"), StatusCode::InvalidDescription,
                      "'cluster'");
}

/** @brief Tests that a version 2 file carries its cluster section as the file states it. */
TEST(Medium_TrainConfig, AVersionTwoFileCarriesTheClusterUnread)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec, train_from(R"({
        "sushiai_train": 2, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "cluster": {"listen": "0.0.0.0:7700", "unread_here": 1}
    })"));

    ASSERT_EQ(spec.cluster.kind(), Kind::Object);
    EXPECT_TRUE(spec.cluster.has("unread_here"));
}

/** @brief Tests that a file that does not state "deterministic" asks for a deterministic run. */
TEST(Medium_TrainConfig, ARunIsDeterministicUnlessTheFileSaysOtherwise)
{
    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec spec, train_from(R"({
        "sushiai_train": 3, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"}
    })"));

    EXPECT_TRUE(spec.deterministic);
}

/** @brief Tests that "deterministic" reads back as the file states it, true or false. */
TEST(Medium_TrainConfig, TheDeterministicKeyReadsBackAsStated)
{
    const std::string head = R"({
        "sushiai_train": 3, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "deterministic": )";

    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec off, train_from(head + "false}"));
    EXPECT_FALSE(off.deterministic);

    SA_ASSERT_OK_AND_ASSIGN(const TrainSpec on, train_from(head + "true}"));
    EXPECT_TRUE(on.deterministic);
}

/** @brief Tests that "deterministic" must be a boolean and names the key when it is not. */
TEST(Medium_TrainConfig, ADeterministicKeyThatIsNoBooleanNamesTheKey)
{
    SA_EXPECT_FAILURE(train_from(R"({
        "sushiai_train": 3, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "deterministic": 1
    })"), StatusCode::InvalidDescription, "'deterministic'");
}

/** @brief Tests that a version 2 file that states "deterministic" is refused, naming the key. */
TEST(Medium_TrainConfig, ADeterministicKeyInAVersionTwoFileIsRefused)
{
    SA_EXPECT_FAILURE(train_from(R"({
        "sushiai_train": 2, "model": "m.json",
        "objective": {"type": "cross_entropy"},
        "data": {"type": "rings"},
        "optimizer": {"type": "adamw"},
        "deterministic": false
    })"), StatusCode::InvalidDescription, "'deterministic'");
}

TEST(Medium_TrainConfig, ParsesAnEvalDocument)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value root, parse_json(R"({
            "sushiai_data": 1,
            "data": {"type": "rings", "batch_size": 512, "batches": 1, "seed": 3735928575}
        })", "eval.json"));
    SA_ASSERT_OK_AND_ASSIGN(const SushiAI::Config::DataSpec data,
                            SushiAI::Config::parse_data(root, "eval.json"));

    EXPECT_EQ(data.type, "rings");
    SA_ASSERT_OK_AND_ASSIGN(const Value& batch, data.attributes.at("batch_size", "data"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t batch_size, batch.as_int("batch_size"));
    EXPECT_EQ(batch_size, 512);
    SA_ASSERT_OK_AND_ASSIGN(const Value& seed, data.attributes.at("seed", "data"));
    SA_ASSERT_OK_AND_ASSIGN(const std::int64_t seed_value, seed.as_int("seed"));
    EXPECT_EQ(seed_value, 3735928575);
}

TEST(Medium_TrainConfig, AnEvalDocumentCarriesNoModelBlock)
{
    // An eval document naming a model is an error; see tests/core/config/README.md.
    SA_ASSERT_OK_AND_ASSIGN(
        const Value root,
        parse_json(R"({"sushiai_data": 1, "data": {"type": "rings"}, "model": "m.json"})",
                   "eval.json"));
    SA_EXPECT_FAILURE(SushiAI::Config::parse_data(root, "eval.json"),
                      StatusCode::InvalidDescription, "'model'", "'data'");
}

TEST(Medium_TrainConfig, LoadingAnAbsentFileNamesThePath)
{
    SA_EXPECT_FAILURE(SushiAI::Config::load_train("no/such/train.json"),
                      StatusCode::FileNotFound, "no/such/train.json");
}
