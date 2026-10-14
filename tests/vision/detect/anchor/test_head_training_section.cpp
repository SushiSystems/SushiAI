/**************************************************************************/
/* test_head_training_section.cpp                                         */
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
 * @file test_head_training_section.cpp
 * @brief Tests that training options and the anchor objective section convert both ways.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Vision::Detect::BoxGradient;
    using SushiAI::Vision::Detect::Anchor::BoxLoss;
    using SushiAI::Vision::Detect::Anchor::HeadTrainingOptions;
    using SushiAI::Vision::Detect::Anchor::K_ANCHOR_OBJECTIVE_TYPE;
    using SushiAI::Vision::Detect::Anchor::ObjectnessTarget;
    using SushiAI::Vision::Detect::Anchor::read_head_training_section;
    using SushiAI::Vision::Detect::Anchor::write_head_training_section;

    /**
     * @brief Holds a section that states every key, as the writer orders them.
     *
     * Every real is a sum of a few powers of two, so the float it narrows to widens back to
     * the same double and the section can be compared with what the writer returns.
     */
    constexpr std::string_view K_FULL_SECTION = R"({
        "type": "anchor",
        "match_threshold": 0.25,
        "ignore_threshold": 0.5,
        "box_loss": "complete_iou",
        "box_gradient": "darknet_compatible",
        "objectness_target": "best_iou",
        "box_weight": 0.0625,
        "object_weight": 1.5,
        "max_delta": 5.0,
        "class_gate": 0.125
    })";

    /** @brief Parses @p text as an objective section. */
    Value section_from(std::string_view text)
    {
        return SushiAI::Config::parse_json(text, "<test>");
    }

    /** @brief Parses the section that states the one member @p member, as JSON text. */
    Value section_of(std::string_view member)
    {
        return section_from("{" + std::string(member) + "}");
    }

    /** @brief Returns options whose floats no short decimal states, with no default left. */
    HeadTrainingOptions make_imported_options()
    {
        HeadTrainingOptions options;
        options.thresholds.match_threshold = 0.213f;
        options.thresholds.ignore_threshold = 0.7f;
        options.box_loss = BoxLoss::CompleteIoU;
        options.box_gradient = BoxGradient::DarknetCompatible;
        options.objectness_target = ObjectnessTarget::BestIoU;
        options.box_weight = 0.07f;
        options.object_weight = 0.4f;
        options.max_delta = 5.0f;
        options.class_gate = 0.3f;
        return options;
    }

    /** @brief Reports whether @p left and @p right are the same float, bit for bit. */
    bool same_bits(float left, float right)
    {
        std::uint32_t left_bits = 0;
        std::uint32_t right_bits = 0;
        std::memcpy(&left_bits, &left, sizeof(left_bits));
        std::memcpy(&right_bits, &right, sizeof(right_bits));
        return left_bits == right_bits;
    }

    /**
     * @brief Expects every field of @p found to equal that of @p wanted.
     *
     * A float is compared bit for bit: the tolerance is zero, absolute and relative.
     */
    void expect_same_options(const HeadTrainingOptions& found, const HeadTrainingOptions& wanted)
    {
        EXPECT_TRUE(
            same_bits(found.thresholds.match_threshold, wanted.thresholds.match_threshold))
            << found.thresholds.match_threshold;
        EXPECT_TRUE(
            same_bits(found.thresholds.ignore_threshold, wanted.thresholds.ignore_threshold))
            << found.thresholds.ignore_threshold;
        EXPECT_EQ(found.box_loss, wanted.box_loss);
        EXPECT_EQ(found.box_gradient, wanted.box_gradient);
        EXPECT_EQ(found.objectness_target, wanted.objectness_target);
        EXPECT_TRUE(same_bits(found.box_weight, wanted.box_weight)) << found.box_weight;
        EXPECT_TRUE(same_bits(found.object_weight, wanted.object_weight)) << found.object_weight;
        EXPECT_TRUE(same_bits(found.max_delta, wanted.max_delta)) << found.max_delta;
        EXPECT_TRUE(same_bits(found.class_gate, wanted.class_gate)) << found.class_gate;
    }

    /** @brief Expects @p options to survive the writer, JSON text, the parser and the reader. */
    void expect_options_round_trip(const HeadTrainingOptions& options)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Value section, write_head_training_section(options));
        SA_ASSERT_OK_AND_ASSIGN(const std::string text, SushiAI::Config::write_json(section));
        SA_ASSERT_OK_AND_ASSIGN(const HeadTrainingOptions read,
                                read_head_training_section(section_from(text)));

        expect_same_options(read, options);
    }
} // namespace

/** @brief Tests that the defaults read back from their section unchanged. */
TEST(Unit_DetectAnchorTrainingSection, TheDefaultsRoundTripThroughTheirSection)
{
    expect_options_round_trip(HeadTrainingOptions{});
}

/** @brief Tests that options with no default left read back from their section unchanged. */
TEST(Unit_DetectAnchorTrainingSection, OptionsWithNoDefaultRoundTripThroughTheirSection)
{
    expect_options_round_trip(make_imported_options());
}

/**
 * @brief Tests that a section stating every key is written back as it was read.
 *
 * write_json spells a real in the shortest form that reads back to the same double, so equal
 * text means equal doubles: the tolerance is zero.
 */
TEST(Unit_DetectAnchorTrainingSection, AFullSectionRoundTripsThroughItsOptions)
{
    const Value section = section_from(K_FULL_SECTION);
    SA_ASSERT_OK_AND_ASSIGN(const HeadTrainingOptions options,
                            read_head_training_section(section));
    SA_ASSERT_OK_AND_ASSIGN(const Value written, write_head_training_section(options));

    SA_ASSERT_OK_AND_ASSIGN(const std::string found, SushiAI::Config::write_json(written));
    SA_ASSERT_OK_AND_ASSIGN(const std::string wanted, SushiAI::Config::write_json(section));
    EXPECT_EQ(found, wanted);
}

/** @brief Tests that the reader fills every field of the options from the section. */
TEST(Unit_DetectAnchorTrainingSection, ReadsEveryFieldOfASection)
{
    SA_ASSERT_OK_AND_ASSIGN(const HeadTrainingOptions options,
                            read_head_training_section(section_from(K_FULL_SECTION)));

    HeadTrainingOptions wanted;
    wanted.thresholds.match_threshold = 0.25f;
    wanted.thresholds.ignore_threshold = 0.5f;
    wanted.box_loss = BoxLoss::CompleteIoU;
    wanted.box_gradient = BoxGradient::DarknetCompatible;
    wanted.objectness_target = ObjectnessTarget::BestIoU;
    wanted.box_weight = 0.0625f;
    wanted.object_weight = 1.5f;
    wanted.max_delta = 5.0f;
    wanted.class_gate = 0.125f;

    expect_same_options(options, wanted);
}

/** @brief Tests that a section with no key, or with its type alone, gives the defaults. */
TEST(Unit_DetectAnchorTrainingSection, AKeyLeftOutKeepsTheDefaultOfItsField)
{
    SA_ASSERT_OK_AND_ASSIGN(const HeadTrainingOptions empty,
                            read_head_training_section(section_from("{}")));
    expect_same_options(empty, HeadTrainingOptions{});

    SA_ASSERT_OK_AND_ASSIGN(const HeadTrainingOptions typed,
                            read_head_training_section(section_of(R"("type": "anchor")")));
    expect_same_options(typed, HeadTrainingOptions{});

    SA_ASSERT_OK_AND_ASSIGN(const HeadTrainingOptions bounded,
                            read_head_training_section(section_of(R"("max_delta": 2.5)")));
    HeadTrainingOptions wanted;
    wanted.max_delta = 2.5f;
    expect_same_options(bounded, wanted);
}

/** @brief Tests that the writer states the type, every key in order and a float as a double. */
TEST(Unit_DetectAnchorTrainingSection, WritesTheTypeEveryKeyInOrderAndAFloatAsItsDouble)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value section,
                            write_head_training_section(make_imported_options()));

    EXPECT_EQ(section.keys(),
              (std::vector<std::string>{"type", "match_threshold", "ignore_threshold",
                                        "box_loss", "box_gradient", "objectness_target",
                                        "box_weight", "object_weight", "max_delta",
                                        "class_gate"}));
    EXPECT_EQ(section.at("type", "objective").as_string("type"), K_ANCHOR_OBJECTIVE_TYPE);
    EXPECT_EQ(section.at("box_loss", "objective").as_string("box_loss"), "complete_iou");

    // The tolerance is zero: a float widens to exactly one double.
    EXPECT_EQ(section.at("match_threshold", "objective").as_real("match_threshold"),
              static_cast<double>(0.213f));
    EXPECT_EQ(section.at("box_weight", "objective").as_real("box_weight"),
              static_cast<double>(0.07f));
}

/** @brief Tests that every string of the three enumerations is read and written. */
TEST(Unit_DetectAnchorTrainingSection, ReadsAndWritesEachEnumeratorString)
{
    HeadTrainingOptions options;

    options.box_loss = BoxLoss::SquaredError;
    SA_ASSERT_OK_AND_ASSIGN(const Value squared, write_head_training_section(options));
    EXPECT_EQ(squared.at("box_loss", "objective").as_string("box_loss"), "squared_error");
    options.box_loss = BoxLoss::CompleteIoU;
    SA_ASSERT_OK_AND_ASSIGN(const Value complete, write_head_training_section(options));
    EXPECT_EQ(complete.at("box_loss", "objective").as_string("box_loss"), "complete_iou");

    options.box_gradient = BoxGradient::Analytic;
    SA_ASSERT_OK_AND_ASSIGN(const Value analytic, write_head_training_section(options));
    EXPECT_EQ(analytic.at("box_gradient", "objective").as_string("box_gradient"), "analytic");
    options.box_gradient = BoxGradient::DarknetCompatible;
    SA_ASSERT_OK_AND_ASSIGN(const Value compatible, write_head_training_section(options));
    EXPECT_EQ(compatible.at("box_gradient", "objective").as_string("box_gradient"),
              "darknet_compatible");

    options.objectness_target = ObjectnessTarget::One;
    SA_ASSERT_OK_AND_ASSIGN(const Value one, write_head_training_section(options));
    EXPECT_EQ(one.at("objectness_target", "objective").as_string("objectness_target"), "one");
    options.objectness_target = ObjectnessTarget::BestIoU;
    SA_ASSERT_OK_AND_ASSIGN(const Value best, write_head_training_section(options));
    EXPECT_EQ(best.at("objectness_target", "objective").as_string("objectness_target"),
              "best_iou");

    SA_ASSERT_OK_AND_ASSIGN(
        const HeadTrainingOptions read,
        read_head_training_section(section_from(
            R"({"box_loss": "squared_error", "box_gradient": "analytic",
                "objectness_target": "one"})")));
    EXPECT_EQ(read.box_loss, BoxLoss::SquaredError);
    EXPECT_EQ(read.box_gradient, BoxGradient::Analytic);
    EXPECT_EQ(read.objectness_target, ObjectnessTarget::One);
}

/** @brief Tests that a whole number is read where the section holds a number. */
TEST(Unit_DetectAnchorTrainingSection, ReadsAWholeNumberAsAWeight)
{
    SA_ASSERT_OK_AND_ASSIGN(const HeadTrainingOptions options,
                            read_head_training_section(section_of(R"("object_weight": 2)")));
    EXPECT_TRUE(same_bits(options.object_weight, 2.0f));
}

/** @brief Tests that a section that is no object is refused with the kind it is. */
TEST(Unit_DetectAnchorTrainingSection, RefusesASectionThatIsNotAnObject)
{
    SA_EXPECT_FAILURE(read_head_training_section(section_from("[]")),
                      StatusCode::InvalidDescription, "objective", "an object");
    SA_EXPECT_FAILURE(read_head_training_section(Value()), StatusCode::InvalidDescription,
                      "objective", "an object");
}

/** @brief Tests that a type of another objective or of another kind is refused. */
TEST(Unit_DetectAnchorTrainingSection, RefusesAnotherType)
{
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("type": "cross_entropy")")),
                      StatusCode::InvalidDescription, "objective 'type'", "'cross_entropy'",
                      "'anchor'");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("type": 3)")),
                      StatusCode::InvalidDescription, "objective 'type'", "a string");
}

/** @brief Tests that a key outside the section is refused with the keys it accepts. */
TEST(Unit_DetectAnchorTrainingSection, RefusesAnUnknownKey)
{
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("iou_normalizer": 0.07)")),
                      StatusCode::InvalidDescription, "objective", "'iou_normalizer'",
                      "'type', 'match_threshold'", "'class_gate'");
    SA_EXPECT_FAILURE(
        read_head_training_section(section_of(R"("thresholds": {"match_threshold": 1.0})")),
        StatusCode::InvalidDescription, "objective", "'thresholds'");
}

/** @brief Tests that a value of another kind is refused with its key and its kind. */
TEST(Unit_DetectAnchorTrainingSection, RefusesAValueOfAnotherKind)
{
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("box_weight": "heavy")")),
                      StatusCode::InvalidDescription, "objective 'box_weight'", "a number");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("ignore_threshold": [0.5])")),
                      StatusCode::InvalidDescription, "objective 'ignore_threshold'",
                      "a number");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("box_loss": 1)")),
                      StatusCode::InvalidDescription, "objective 'box_loss'", "a string");
}

/** @brief Tests that a string no enumerator has is refused with the strings of its key. */
TEST(Unit_DetectAnchorTrainingSection, RefusesAStringNoEnumeratorHas)
{
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("box_loss": "ciou")")),
                      StatusCode::InvalidDescription, "objective 'box_loss'", "'ciou'",
                      "'squared_error', 'complete_iou'");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("box_gradient": "exact")")),
                      StatusCode::InvalidDescription, "objective 'box_gradient'", "'exact'",
                      "'analytic', 'darknet_compatible'");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("objectness_target": "iou")")),
                      StatusCode::InvalidDescription, "objective 'objectness_target'", "'iou'",
                      "'one', 'best_iou'");
}

/** @brief Tests that a number float32 cannot hold as a finite value is refused. */
TEST(Unit_DetectAnchorTrainingSection, RefusesANumberFloat32CannotHold)
{
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("max_delta": 1e39)")),
                      StatusCode::InvalidDescription, "objective 'max_delta'", "float32");
}

/** @brief Tests that the reader refuses the values validate_head_training refuses. */
TEST(Unit_DetectAnchorTrainingSection, RefusesTheValuesTheOptionsCheckRefuses)
{
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("match_threshold": 0.0)")),
                      StatusCode::InvalidDescription, "objective", "match threshold", "(0, 1]");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("ignore_threshold": 1.5)")),
                      StatusCode::InvalidDescription, "objective", "ignore threshold", "1.5");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("class_gate": -0.25)")),
                      StatusCode::InvalidDescription, "objective", "class gate", "-0.25");
    SA_EXPECT_FAILURE(read_head_training_section(section_of(R"("box_weight": -1.0)")),
                      StatusCode::InvalidDescription, "objective", "box weight", "-1");
}

/** @brief Tests that the writer refuses the values validate_head_training refuses. */
TEST(Unit_DetectAnchorTrainingSection, WriterRefusesTheValuesTheOptionsCheckRefuses)
{
    HeadTrainingOptions weightless;
    weightless.object_weight = std::numeric_limits<float>::quiet_NaN();
    SA_EXPECT_FAILURE(write_head_training_section(weightless), StatusCode::InvalidDescription,
                      "objective", "object weight");

    HeadTrainingOptions unbounded;
    unbounded.max_delta = std::numeric_limits<float>::infinity();
    SA_EXPECT_FAILURE(write_head_training_section(unbounded), StatusCode::InvalidDescription,
                      "objective", "max_delta");

    HeadTrainingOptions lossless;
    lossless.box_loss = static_cast<BoxLoss>(7);
    SA_EXPECT_FAILURE(write_head_training_section(lossless), StatusCode::InvalidDescription,
                      "objective", "BoxLoss", "7");
}
