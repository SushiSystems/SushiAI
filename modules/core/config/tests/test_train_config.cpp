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
 * @brief Tests that a training file carries its schedule section as an opaque value.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

using namespace SushiAI;
using Config::Kind;
using Config::TrainSpec;
using Config::Value;

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

    /** @brief Parses the required members followed by @p members, each led by a comma. */
    TrainSpec parse_with(std::string_view members)
    {
        const std::string text = std::string(K_REQUIRED_MEMBERS) + std::string(members) + "}";
        return Config::parse_train(Config::parse_json(text, "<test>"), "<test>",
                                   std::filesystem::path("."));
    }
} // namespace

/** @brief Tests that a file without the section parses as before, with a null schedule. */
TEST(Unit_TrainConfig, AFileWithoutAScheduleHoldsTheNullValue)
{
    const TrainSpec spec = parse_with("");

    EXPECT_TRUE(spec.schedule.is_null());
    EXPECT_EQ(spec.optimizer_type, "sgd");
    EXPECT_EQ(spec.epochs, 1);
}

/** @brief Tests that the section is kept whole, its "type" and an unread key included. */
TEST(Unit_TrainConfig, TheScheduleIsCarriedAsTheFileStatesIt)
{
    const TrainSpec spec = parse_with(", " + std::string(K_SCHEDULE_MEMBER));

    ASSERT_EQ(spec.schedule.kind(), Kind::Object);
    ASSERT_EQ(spec.schedule.keys().size(), 3u);
    EXPECT_EQ(spec.schedule.at("type", "schedule").as_string("type"), "step");
    EXPECT_EQ(spec.schedule.at("base_rate", "schedule").kind(), Kind::Real);
    EXPECT_TRUE(spec.schedule.has("unread_here"));
}

/** @brief Tests that the parser accepts a schedule of any kind and leaves the check to optim. */
TEST(Unit_TrainConfig, AScheduleOfAnotherKindIsCarriedUnchecked)
{
    const TrainSpec spec = parse_with(R"(, "schedule": "later")");

    EXPECT_EQ(spec.schedule.kind(), Kind::String);
}

/** @brief Tests that a key beside the known ones is still refused by name. */
TEST(Unit_TrainConfig, AKeyBesideTheScheduleIsStillRefused)
{
    SA_EXPECT_ERROR((void)parse_with(R"(, "schedules": {})"), "'schedules'");
}

/** @brief Tests that a file without "weights" leaves the path empty. */
TEST(Unit_TrainConfig, AFileWithoutWeightsHoldsAnEmptyPath)
{
    EXPECT_TRUE(parse_with("").weights.empty());
}

/** @brief Tests that "weights" resolves against the directory of the file, as "model" does. */
TEST(Unit_TrainConfig, TheWeightsPathResolvesAgainstTheFile)
{
    const std::string text =
        std::string(K_REQUIRED_MEMBERS) + R"(, "weights": "start.sushi"})";
    const TrainSpec spec = Config::parse_train(Config::parse_json(text, "<test>"), "<test>",
                                               std::filesystem::path("runs"));

    EXPECT_EQ(spec.weights, std::filesystem::path("runs") / "start.sushi");
    EXPECT_EQ(spec.model_path, std::filesystem::path("runs") / "model.json");
}

/** @brief Tests that "weights" of another kind than a string is refused by name. */
TEST(Unit_TrainConfig, WeightsThatAreNoStringAreRefused)
{
    SA_EXPECT_ERROR((void)parse_with(R"(, "weights": 3)"), "'weights'");
}

/** @brief Tests that "resume" is empty when left out and resolves against the file. */
TEST(Unit_TrainConfig, TheResumePathResolvesAgainstTheFile)
{
    EXPECT_TRUE(parse_with("").resume.empty());

    const std::string text =
        std::string(K_REQUIRED_MEMBERS) + R"(, "resume": "last.sushi"})";
    const TrainSpec spec = Config::parse_train(Config::parse_json(text, "<test>"), "<test>",
                                               std::filesystem::path("runs"));

    EXPECT_EQ(spec.resume, std::filesystem::path("runs") / "last.sushi");
    EXPECT_TRUE(spec.weights.empty());
}

/** @brief Tests that "resume" of another kind than a string is refused by name. */
TEST(Unit_TrainConfig, AResumeThatIsNoStringIsRefused)
{
    SA_EXPECT_ERROR((void)parse_with(R"(, "resume": true)"), "'resume'");
}

/** @brief Tests that a file states "weights" or "resume", and is refused with both. */
TEST(Unit_TrainConfig, WeightsAndResumeTogetherAreRefused)
{
    SA_EXPECT_OK(Config::check_start(parse_with(""), "train.json"));
    SA_EXPECT_OK(Config::check_start(parse_with(R"(, "weights": "a.sushi")"), "train.json"));
    SA_EXPECT_OK(Config::check_start(parse_with(R"(, "resume": "a.sushi")"), "train.json"));

    SA_EXPECT_FAILURE(
        Config::check_start(parse_with(R"(, "weights": "a.sushi", "resume": "b.sushi")"),
                            "train.json"),
        StatusCode::InvalidDescription, "'train.json'", "'weights'", "'resume'");
}

/** @brief Tests that "seed" is 0 when left out and holds the integer the file states. */
TEST(Unit_TrainConfig, TheSeedIsZeroUnlessTheFileStatesOne)
{
    EXPECT_EQ(parse_with("").seed, 0);

    const TrainSpec spec = parse_with(R"(, "seed": 20261011)");
    EXPECT_EQ(spec.seed, 20261011);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t seed, Config::read_run_seed(spec, "train.json"));
    EXPECT_EQ(seed, 20261011u);

    SA_ASSERT_OK_AND_ASSIGN(const std::uint64_t absent,
                            Config::read_run_seed(parse_with(""), "train.json"));
    EXPECT_EQ(absent, 0u);
}

/** @brief Tests that a "seed" that is no integer is refused by name. */
TEST(Unit_TrainConfig, ASeedThatIsNoIntegerIsRefused)
{
    SA_EXPECT_ERROR((void)parse_with(R"(, "seed": "seven")"), "'seed'");
    SA_EXPECT_ERROR((void)parse_with(R"(, "seed": 0.5)"), "'seed'");
}

/** @brief Tests that a negative "seed" parses and is refused with the file and the value. */
TEST(Unit_TrainConfig, ANegativeSeedIsRefused)
{
    SA_EXPECT_FAILURE(Config::read_run_seed(parse_with(R"(, "seed": -3)"), "train.json"),
                      StatusCode::InvalidDescription, "'train.json'", "'seed'", "-3");
}

namespace
{
    /** @brief Parses a training file whose data block is @p data. */
    TrainSpec parse_with_data(std::string_view data)
    {
        const std::string text = R"({"sushiai_train": 1, "model": "model.json",
            "objective": {"type": "anchor"}, "optimizer": {"type": "sgd"}, "data": )"
            + std::string(data) + "}";
        return Config::parse_train(Config::parse_json(text, "<test>"), "<test>",
                                   std::filesystem::path("."));
    }
} // namespace

/** @brief Tests that a data block with a reader and no type takes the reader type whole. */
TEST(Unit_TrainConfig, ADataBlockWithAReaderTakesTheReaderType)
{
    const TrainSpec spec =
        parse_with_data(R"({"reader": "yolo_txt", "task": "detection", "root": "data"})");

    EXPECT_EQ(spec.data.type, Config::K_READER_DATA_TYPE);
    ASSERT_EQ(spec.data.attributes.keys().size(), 3u);
    EXPECT_EQ(spec.data.attributes.at("reader", "data").as_string("reader"), "yolo_txt");
}

/** @brief Tests that a typed data block is read as before, its type stripped. */
TEST(Unit_TrainConfig, ATypedDataBlockKeepsItsType)
{
    const TrainSpec spec = parse_with_data(R"({"type": "mnist", "directory": "data"})");

    EXPECT_EQ(spec.data.type, "mnist");
    EXPECT_FALSE(spec.data.attributes.has("type"));
    EXPECT_TRUE(spec.data.attributes.has("directory"));
}

/** @brief Tests that a block with both keys is a typed block that still carries "reader". */
TEST(Unit_TrainConfig, ATypeBesideAReaderStaysATypedBlock)
{
    const TrainSpec spec = parse_with_data(R"({"type": "mnist", "reader": "idx"})");

    EXPECT_EQ(spec.data.type, "mnist");
    EXPECT_TRUE(spec.data.attributes.has("reader"));
}

/** @brief Tests that a data block with neither key is refused for its missing type. */
TEST(Unit_TrainConfig, ADataBlockWithNeitherKeyIsRefused)
{
    SA_EXPECT_ERROR((void)parse_with_data(R"({"root": "data"})"), "'type'");
}
