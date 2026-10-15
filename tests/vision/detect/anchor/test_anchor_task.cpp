/**************************************************************************/
/* test_anchor_task.cpp                                                   */
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
 * @file test_anchor_task.cpp
 * @brief Tests that an anchor task and its task section convert both ways without loss.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

namespace
{
    using SushiAI::StatusCode;
    using SushiAI::Config::Value;
    using SushiAI::Vision::Detect::K_DEFAULT_DISTANCE_EXPONENT;
    using SushiAI::Vision::Detect::SuppressionCriterion;
    using SushiAI::Vision::Detect::Anchor::AnchorTask;
    using SushiAI::Vision::Detect::Anchor::AnchorTaskHead;
    using SushiAI::Vision::Detect::Anchor::HeadCoding;
    using SushiAI::Vision::Detect::Anchor::HeadOptions;
    using SushiAI::Vision::Detect::Anchor::K_ANCHOR_TASK_TYPE;
    using SushiAI::Vision::Detect::Anchor::read_anchor_task;
    using SushiAI::Vision::Detect::Anchor::write_anchor_task;

    /**
     * @brief Holds a section of three heads over nine anchors, as the writer orders its keys.
     *
     * Every real is a sum of a few powers of two, so the float it narrows to widens back to
     * the same double and the section can be compared with what the writer returns.
     */
    constexpr std::string_view K_THREE_HEAD_SECTION = R"({
        "type": "anchor_detection",
        "classes": 80,
        "anchors": [[0.015625, 0.03125], [0.03125, 0.0625], [0.0625, 0.046875],
                    [0.0625, 0.125], [0.125, 0.09375], [0.125, 0.25],
                    [0.25, 0.1875], [0.375, 0.5], [0.75, 0.625]],
        "heads": [
            {"output": "head_0", "anchors": [6, 7, 8], "coding": "logit", "scale_xy": 1.25},
            {"output": "head_1", "anchors": [3, 4, 5], "coding": "logit", "scale_xy": 1.125},
            {"output": "head_2", "anchors": [0, 1, 2], "coding": "activated", "scale_xy": 2.0}
        ],
        "suppression": {"criterion": "distance_iou", "distance_exponent": 0.5}
    })";

    /** @brief Holds a section of two heads over six anchors, as the writer orders its keys. */
    constexpr std::string_view K_TWO_HEAD_SECTION = R"({
        "type": "anchor_detection",
        "classes": 3,
        "anchors": [[0.03125, 0.0625], [0.0625, 0.125], [0.125, 0.09375],
                    [0.25, 0.1875], [0.375, 0.5], [0.75, 0.625]],
        "heads": [
            {"output": "head_0", "anchors": [3, 4, 5], "coding": "logit", "scale_xy": 1.0},
            {"output": "head_1", "anchors": [0, 1, 2], "coding": "logit", "scale_xy": 1.5}
        ],
        "suppression": {"criterion": "iou"}
    })";

    /** @brief Holds the members of a section as JSON text, valid until a test replaces one. */
    struct SectionText
    {
        /** @brief Holds the value of "type". */
        std::string type = R"("anchor_detection")";

        /** @brief Holds the value of "classes". */
        std::string classes = "3";

        /** @brief Holds the value of "anchors": four anchors. */
        std::string anchors = "[[0.125, 0.25], [0.25, 0.5], [0.5, 0.25], [0.75, 0.5]]";

        /** @brief Holds the value of "heads": two heads of two anchors each. */
        std::string heads =
            R"([{"output": "a", "anchors": [2, 3], "coding": "logit", "scale_xy": 1.0},
                {"output": "b", "anchors": [0, 1], "coding": "logit", "scale_xy": 1.0}])";

        /** @brief Holds the value of "suppression". */
        std::string suppression = R"({"criterion": "iou"})";
    };

    /** @brief Parses @p text as a task section. */
    Value section_from(std::string_view text)
    {
        return SushiAI::Config::parse_json(text, "<test>");
    }

    /** @brief Parses the section whose five members are those of @p text. */
    Value section_from(const SectionText& text)
    {
        return section_from(R"({"type": )" + text.type + R"(, "classes": )" + text.classes
                            + R"(, "anchors": )" + text.anchors + R"(, "heads": )" + text.heads
                            + R"(, "suppression": )" + text.suppression + "}");
    }

    /** @brief Returns one head entry over anchors 0 and 1 with the given coding and scale. */
    std::string head_text(std::string_view coding, std::string_view scale_xy)
    {
        return R"({"output": "a", "anchors": [0, 1], "coding": )" + std::string(coding)
             + R"(, "scale_xy": )" + std::string(scale_xy) + "}";
    }

    /** @brief Returns a head that reads @p output with @p classes classes. */
    AnchorTaskHead make_head(std::string output, std::int64_t classes, HeadCoding coding,
                             float scale_xy)
    {
        AnchorTaskHead head;
        head.output = std::move(output);
        head.options = HeadOptions{classes, coding, scale_xy};
        return head;
    }

    /** @brief Returns three heads over nine anchors whose floats no short decimal states. */
    AnchorTask make_three_head_task()
    {
        AnchorTask task;
        task.num_classes = 80;
        task.layout.anchors = {{0.024f, 0.031f}, {0.055f, 0.072f}, {0.089f, 0.061f},
                               {0.091f, 0.178f}, {0.186f, 0.135f}, {0.142f, 0.286f},
                               {0.277f, 0.216f}, {0.375f, 0.476f}, {0.897f, 0.784f}};
        task.layout.head_anchors = {{6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
        task.heads = {make_head("head_0", 80, HeadCoding::Logit, 1.05f),
                      make_head("head_1", 80, HeadCoding::Logit, 1.1f),
                      make_head("head_2", 80, HeadCoding::Activated, 1.2f)};
        task.criterion = SuppressionCriterion::DistanceIoU;
        task.distance_exponent = 0.6f;
        return task;
    }

    /** @brief Returns two heads over six anchors under the plain IoU criterion. */
    AnchorTask make_two_head_task()
    {
        AnchorTask task;
        task.num_classes = 3;
        task.layout.anchors = {{0.02f, 0.03f}, {0.05f, 0.06f}, {0.09f, 0.14f},
                               {0.2f, 0.2f},   {0.3f, 0.4f},   {0.8f, 0.7f}};
        task.layout.head_anchors = {{3, 4, 5}, {0, 1, 2}};
        task.heads = {make_head("head_0", 3, HeadCoding::Logit, 1.0f),
                      make_head("head_1", 3, HeadCoding::Logit, 1.05f)};
        return task;
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
    void expect_same_task(const AnchorTask& found, const AnchorTask& wanted)
    {
        EXPECT_EQ(found.num_classes, wanted.num_classes);
        EXPECT_EQ(found.criterion, wanted.criterion);
        EXPECT_TRUE(same_bits(found.distance_exponent, wanted.distance_exponent))
            << found.distance_exponent << " against " << wanted.distance_exponent;

        ASSERT_EQ(found.layout.anchors.size(), wanted.layout.anchors.size());
        for (std::size_t i = 0; i < wanted.layout.anchors.size(); ++i)
        {
            EXPECT_TRUE(same_bits(found.layout.anchors[i].width, wanted.layout.anchors[i].width))
                << "width of anchor " << i;
            EXPECT_TRUE(
                same_bits(found.layout.anchors[i].height, wanted.layout.anchors[i].height))
                << "height of anchor " << i;
        }

        EXPECT_EQ(found.layout.head_anchors, wanted.layout.head_anchors);

        ASSERT_EQ(found.heads.size(), wanted.heads.size());
        for (std::size_t k = 0; k < wanted.heads.size(); ++k)
        {
            EXPECT_EQ(found.heads[k].output, wanted.heads[k].output) << "head " << k;
            EXPECT_EQ(found.heads[k].options.num_classes, wanted.heads[k].options.num_classes)
                << "head " << k;
            EXPECT_EQ(found.heads[k].options.coding, wanted.heads[k].options.coding)
                << "head " << k;
            EXPECT_TRUE(
                same_bits(found.heads[k].options.scale_xy, wanted.heads[k].options.scale_xy))
                << "scale_xy of head " << k;
        }
    }

    /**
     * @brief Expects @p found and @p wanted to be the same tree, by their JSON text.
     *
     * write_json spells a real in the shortest form that reads back to the same double, so
     * equal text means equal doubles: the tolerance is zero, absolute and relative.
     */
    void expect_same_section(const Value& found, const Value& wanted)
    {
        SA_ASSERT_OK_AND_ASSIGN(const std::string found_text, SushiAI::Config::write_json(found));
        SA_ASSERT_OK_AND_ASSIGN(const std::string wanted_text,
                                SushiAI::Config::write_json(wanted));
        EXPECT_EQ(found_text, wanted_text);
    }

    /** @brief Expects @p task to survive the writer, JSON text, the parser and the reader. */
    void expect_task_round_trip(const AnchorTask& task)
    {
        SA_ASSERT_OK_AND_ASSIGN(const Value section, write_anchor_task(task));
        SA_ASSERT_OK_AND_ASSIGN(const std::string text, SushiAI::Config::write_json(section));
        SA_ASSERT_OK_AND_ASSIGN(const AnchorTask read, read_anchor_task(section_from(text)));

        expect_same_task(read, task);
    }

    /** @brief Expects the section in @p text to survive the reader and the writer. */
    void expect_section_round_trip(std::string_view text)
    {
        const Value section = section_from(text);
        SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task, read_anchor_task(section));
        SA_ASSERT_OK_AND_ASSIGN(const Value written, write_anchor_task(task));

        expect_same_section(written, section);
    }
} // namespace

/** @brief Tests that three heads over nine anchors read back from their section unchanged. */
TEST(Small_DetectAnchorTask, AThreeHeadTaskRoundTripsThroughItsSection)
{
    expect_task_round_trip(make_three_head_task());
}

/** @brief Tests that two heads over six anchors read back from their section unchanged. */
TEST(Small_DetectAnchorTask, ATwoHeadTaskRoundTripsThroughItsSection)
{
    expect_task_round_trip(make_two_head_task());
}

/** @brief Tests that a section of three heads is written back as it was read. */
TEST(Small_DetectAnchorTask, AThreeHeadSectionRoundTripsThroughItsTask)
{
    expect_section_round_trip(K_THREE_HEAD_SECTION);
}

/** @brief Tests that a section of two heads is written back as it was read. */
TEST(Small_DetectAnchorTask, ATwoHeadSectionRoundTripsThroughItsTask)
{
    expect_section_round_trip(K_TWO_HEAD_SECTION);
}

/** @brief Tests that the reader fills every field of the task from the section. */
TEST(Small_DetectAnchorTask, ReadsEveryFieldOfASection)
{
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task,
                            read_anchor_task(section_from(K_THREE_HEAD_SECTION)));

    AnchorTask wanted;
    wanted.num_classes = 80;
    wanted.layout.anchors = {{0.015625f, 0.03125f}, {0.03125f, 0.0625f}, {0.0625f, 0.046875f},
                             {0.0625f, 0.125f},     {0.125f, 0.09375f},  {0.125f, 0.25f},
                             {0.25f, 0.1875f},      {0.375f, 0.5f},      {0.75f, 0.625f}};
    wanted.layout.head_anchors = {{6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
    wanted.heads = {make_head("head_0", 80, HeadCoding::Logit, 1.25f),
                    make_head("head_1", 80, HeadCoding::Logit, 1.125f),
                    make_head("head_2", 80, HeadCoding::Activated, 2.0f)};
    wanted.criterion = SuppressionCriterion::DistanceIoU;
    wanted.distance_exponent = 0.5f;

    expect_same_task(task, wanted);
}

/** @brief Tests that the writer states the type, the keys in order and each float as a double. */
TEST(Small_DetectAnchorTask, WritesTheTypeTheKeysInOrderAndEachFloatAsItsDouble)
{
    SA_ASSERT_OK_AND_ASSIGN(const Value section, write_anchor_task(make_two_head_task()));

    EXPECT_EQ(section.keys(),
              (std::vector<std::string>{"type", "classes", "anchors", "heads", "suppression"}));
    EXPECT_EQ(section.at("type", "task").as_string("type"), K_ANCHOR_TASK_TYPE);
    EXPECT_EQ(section.at("classes", "task").as_int("classes"), 3);

    const Value& head = section.at("heads", "task").as_array("heads")[1];
    EXPECT_EQ(head.keys(),
              (std::vector<std::string>{"output", "anchors", "coding", "scale_xy"}));

    // The tolerance is zero: a float widens to exactly one double.
    EXPECT_EQ(head.at("scale_xy", "head").as_real("scale_xy"), static_cast<double>(1.05f));
    const Value& anchor = section.at("anchors", "task").as_array("anchors")[2];
    EXPECT_EQ(anchor.as_array("anchor")[0].as_real("width"), static_cast<double>(0.09f));
    EXPECT_EQ(anchor.as_array("anchor")[1].as_real("height"), static_cast<double>(0.14f));
}

/** @brief Tests that both strings of "coding" are read and written as their enumerators. */
TEST(Small_DetectAnchorTask, ReadsAndWritesEachCodingString)
{
    /** @brief Pairs a string of "coding" with its enumerator. */
    struct Case
    {
        std::string_view name; /**< The string a section states. */
        HeadCoding coding;     /**< The enumerator it reads as. */
    };

    for (const Case& entry : {Case{"logit", HeadCoding::Logit},
                              Case{"activated", HeadCoding::Activated}})
    {
        SectionText text;
        text.heads = "[" + head_text("\"" + std::string(entry.name) + "\"", "1.0") + "]";
        SA_ASSERT_OK_AND_ASSIGN(const AnchorTask read, read_anchor_task(section_from(text)));
        EXPECT_EQ(read.heads[0].options.coding, entry.coding) << entry.name;

        SA_ASSERT_OK_AND_ASSIGN(const Value written, write_anchor_task(read));
        const Value& head = written.at("heads", "task").as_array("heads")[0];
        EXPECT_EQ(head.at("coding", "head").as_string("coding"), entry.name);
    }
}

/** @brief Tests that both strings of "criterion" are read and written as their enumerators. */
TEST(Small_DetectAnchorTask, ReadsAndWritesEachCriterionString)
{
    /** @brief Pairs a string of "criterion" with its enumerator and a section stating it. */
    struct Case
    {
        std::string_view name;          /**< The string a section states. */
        std::string_view suppression;   /**< A "suppression" object that states it. */
        SuppressionCriterion criterion; /**< The enumerator it reads as. */
    };

    for (const Case& entry :
         {Case{"iou", R"({"criterion": "iou"})", SuppressionCriterion::IoU},
          Case{"distance_iou", R"({"criterion": "distance_iou", "distance_exponent": 0.75})",
               SuppressionCriterion::DistanceIoU}})
    {
        SectionText text;
        text.suppression = std::string(entry.suppression);
        SA_ASSERT_OK_AND_ASSIGN(const AnchorTask read, read_anchor_task(section_from(text)));
        EXPECT_EQ(read.criterion, entry.criterion) << entry.name;

        SA_ASSERT_OK_AND_ASSIGN(const Value written, write_anchor_task(read));
        const Value& suppression = written.at("suppression", "task");
        EXPECT_EQ(suppression.at("criterion", "suppression").as_string("criterion"),
                  entry.name);
    }
}

/** @brief Tests that the exponent is written and read under DistanceIoU and only there. */
TEST(Small_DetectAnchorTask, TheExponentBelongsToTheCriterionThatReadsIt)
{
    AnchorTask plain = make_two_head_task();
    plain.distance_exponent = 0.25f;
    SA_ASSERT_OK_AND_ASSIGN(const Value plain_section, write_anchor_task(plain));
    EXPECT_EQ(plain_section.at("suppression", "task").keys(),
              (std::vector<std::string>{"criterion"}));

    // The tolerance is zero: an exponent the section leaves out reads as the default.
    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask plain_read, read_anchor_task(plain_section));
    EXPECT_TRUE(same_bits(plain_read.distance_exponent, K_DEFAULT_DISTANCE_EXPONENT));

    AnchorTask distance = make_two_head_task();
    distance.criterion = SuppressionCriterion::DistanceIoU;
    distance.distance_exponent = 0.25f;
    SA_ASSERT_OK_AND_ASSIGN(const Value distance_section, write_anchor_task(distance));
    const Value& suppression = distance_section.at("suppression", "task");
    EXPECT_EQ(suppression.keys(),
              (std::vector<std::string>{"criterion", "distance_exponent"}));
    EXPECT_EQ(suppression.at("distance_exponent", "suppression").as_real("exponent"), 0.25);
}

/** @brief Tests that a whole number is read where a section states a real. */
TEST(Small_DetectAnchorTask, ReadsAWholeNumberAsAScale)
{
    SectionText text;
    text.heads = "[" + head_text(R"("logit")", "2") + "]";

    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task, read_anchor_task(section_from(text)));
    EXPECT_TRUE(same_bits(task.heads[0].options.scale_xy, 2.0f));
}

/** @brief Tests that two heads may share an anchor, as validate_anchor_layout allows. */
TEST(Small_DetectAnchorTask, AcceptsAnAnchorTwoHeadsShare)
{
    SectionText text;
    text.heads =
        R"([{"output": "a", "anchors": [1, 2], "coding": "logit", "scale_xy": 1.0},
            {"output": "b", "anchors": [0, 1], "coding": "logit", "scale_xy": 1.0}])";

    SA_ASSERT_OK_AND_ASSIGN(const AnchorTask task, read_anchor_task(section_from(text)));
    EXPECT_EQ(task.layout.head_anchors,
              (std::vector<std::vector<std::int64_t>>{{1, 2}, {0, 1}}));
}

/** @brief Tests that a section that is not an object is refused. */
TEST(Small_DetectAnchorTask, RefusesASectionThatIsNotAnObject)
{
    SA_EXPECT_FAILURE(read_anchor_task(section_from("[1]")), StatusCode::InvalidDescription,
                      "task", "an array");
    SA_EXPECT_FAILURE(read_anchor_task(Value()), StatusCode::InvalidDescription, "task",
                      "null");
}

/** @brief Tests that a type other than the anchor task's, or none, is refused by its key. */
TEST(Small_DetectAnchorTask, RefusesAnotherTypeOrNone)
{
    SectionText other;
    other.type = R"("anchor_free_detection")";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(other)), StatusCode::InvalidDescription,
                      "'type'", "'anchor_free_detection'", "'anchor_detection'");

    SectionText numbered;
    numbered.type = "3";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(numbered)),
                      StatusCode::InvalidDescription, "'type'", "a whole number");

    SA_EXPECT_FAILURE(read_anchor_task(section_from(R"({"classes": 3})")),
                      StatusCode::InvalidDescription, "task has no 'type'");
}

/** @brief Tests that a key the section does not have is refused at each level. */
TEST(Small_DetectAnchorTask, RefusesAnUnknownKeyAtEachLevel)
{
    SA_EXPECT_FAILURE(read_anchor_task(section_from(R"({"type": "anchor_detection",
                                                        "clases": 3})")),
                      StatusCode::InvalidDescription, "'clases'", "'classes'");

    SectionText head;
    head.heads = R"([{"output": "a", "anchors": [0], "coding": "logit", "scale_xy": 1.0,
                      "scale": 2.0}])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(head)), StatusCode::InvalidDescription,
                      "'heads' entry 0", "'scale'", "'scale_xy'");

    SectionText suppression;
    suppression.suppression = R"({"criterion": "iou", "threshold": 0.5})";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(suppression)),
                      StatusCode::InvalidDescription, "'suppression'", "'threshold'");
}

/** @brief Tests that each of the five members is required. */
TEST(Small_DetectAnchorTask, RefusesASectionWithoutOneOfItsMembers)
{
    const std::string type = R"("type": "anchor_detection")";
    const std::string classes = R"("classes": 3)";
    const std::string anchors = R"("anchors": [[0.5, 0.5]])";
    const std::string heads =
        R"("heads": [{"output": "a", "anchors": [0], "coding": "logit", "scale_xy": 1.0}])";
    const std::string suppression = R"("suppression": {"criterion": "iou"})";

    const auto joined = [](std::initializer_list<std::string> members)
    {
        std::string text;
        for (const std::string& member : members)
            text += (text.empty() ? "{" : ", ") + member;
        return section_from(text + "}");
    };

    SA_EXPECT_OK(read_anchor_task(joined({type, classes, anchors, heads, suppression})));
    SA_EXPECT_FAILURE(read_anchor_task(joined({type, anchors, heads, suppression})),
                      StatusCode::InvalidDescription, "task has no 'classes'");
    SA_EXPECT_FAILURE(read_anchor_task(joined({type, classes, heads, suppression})),
                      StatusCode::InvalidDescription, "task has no 'anchors'");
    SA_EXPECT_FAILURE(read_anchor_task(joined({type, classes, anchors, suppression})),
                      StatusCode::InvalidDescription, "task has no 'heads'");
    SA_EXPECT_FAILURE(read_anchor_task(joined({type, classes, anchors, heads})),
                      StatusCode::InvalidDescription, "task has no 'suppression'");
}

/** @brief Tests that a class count that is not a whole number of at least 1 is refused. */
TEST(Small_DetectAnchorTask, RefusesAClassCountBelowOneOrOfAnotherKind)
{
    SectionText none;
    none.classes = "0";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(none)), StatusCode::InvalidDescription,
                      "'classes'", "at least 1", "0");

    SectionText real;
    real.classes = "3.5";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(real)), StatusCode::InvalidDescription,
                      "'classes'", "a whole number");
}

/** @brief Tests that an anchor that is not a pair of numbers is refused with its index. */
TEST(Small_DetectAnchorTask, RefusesAnAnchorThatIsNotAPairOfNumbers)
{
    SectionText listed;
    listed.anchors = "0.5";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(listed)), StatusCode::InvalidDescription,
                      "'anchors'", "an array");

    SectionText triple;
    triple.anchors = "[[0.125, 0.25], [0.25, 0.5, 0.75]]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(triple)), StatusCode::InvalidDescription,
                      "'anchors' entry 1", "[width, height]", "3 values");

    SectionText scalar;
    scalar.anchors = "[[0.125, 0.25], [0.25, 0.5], 0.5]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(scalar)), StatusCode::InvalidDescription,
                      "'anchors' entry 2", "[width, height]");

    SectionText text;
    text.anchors = R"([[0.125, "wide"]])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(text)), StatusCode::InvalidDescription,
                      "'anchors' entry 0 height", "a string");
}

/** @brief Tests that an anchor side that is not a positive finite float is refused. */
TEST(Small_DetectAnchorTask, RefusesAnAnchorSideThatIsNotPositiveAndFinite)
{
    SectionText zero;
    zero.anchors = "[[0.125, 0.25], [0.25, 0.5], [0.0, 0.25], [0.75, 0.5]]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(zero)), StatusCode::InvalidDescription,
                      "'anchors' entry 2", "above zero");

    SectionText negative;
    negative.anchors = "[[0.125, 0.25], [0.25, -0.5], [0.5, 0.25], [0.75, 0.5]]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(negative)),
                      StatusCode::InvalidDescription, "'anchors' entry 1", "above zero");

    // 1e-60 is positive as a double and narrows to a float of zero.
    SectionText vanishing;
    vanishing.anchors = "[[1e-60, 0.25], [0.25, 0.5], [0.5, 0.25], [0.75, 0.5]]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(vanishing)),
                      StatusCode::InvalidDescription, "'anchors' entry 0", "above zero");

    SectionText vast;
    vast.anchors = "[[0.125, 0.25], [0.25, 0.5], [0.5, 0.25], [0.75, 1e60]]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(vast)), StatusCode::InvalidDescription,
                      "'anchors' entry 3 height", "float32");

    SectionText none;
    none.anchors = "[]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(none)), StatusCode::InvalidDescription,
                      "'anchors'", "no anchor");
}

/** @brief Tests that a section without a head, or with a head that is no object, is refused. */
TEST(Small_DetectAnchorTask, RefusesNoHeadAndAHeadThatIsNotAnObject)
{
    SectionText none;
    none.heads = "[]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(none)), StatusCode::InvalidDescription,
                      "'heads'", "no head");

    SectionText listed;
    listed.heads = R"({"output": "a"})";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(listed)), StatusCode::InvalidDescription,
                      "'heads'", "an object");

    SectionText scalar;
    scalar.heads = "[" + head_text(R"("logit")", "1.0") + ", 7]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(scalar)), StatusCode::InvalidDescription,
                      "'heads' entry 1", "a whole number");
}

/** @brief Tests that a mask index that names no anchor is refused with head and slot. */
TEST(Small_DetectAnchorTask, RefusesAMaskIndexThatNamesNoAnchor)
{
    SectionText past;
    past.heads =
        R"([{"output": "a", "anchors": [2, 3], "coding": "logit", "scale_xy": 1.0},
            {"output": "b", "anchors": [0, 1, 4], "coding": "logit", "scale_xy": 1.0}])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(past)), StatusCode::InvalidDescription,
                      "'heads' entry 1 'anchors' entry 2", "anchor 4", "4 anchors");

    SectionText negative;
    negative.heads =
        R"([{"output": "a", "anchors": [-1, 3], "coding": "logit", "scale_xy": 1.0}])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(negative)),
                      StatusCode::InvalidDescription, "'heads' entry 0 'anchors' entry 0",
                      "anchor -1");

    SectionText real;
    real.heads = R"([{"output": "a", "anchors": [0, 1.5], "coding": "logit", "scale_xy": 1.0}])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(real)), StatusCode::InvalidDescription,
                      "'heads' entry 0 'anchors' entry 1", "a whole number");

    SectionText empty;
    empty.heads = R"([{"output": "a", "anchors": [], "coding": "logit", "scale_xy": 1.0}])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(empty)), StatusCode::InvalidDescription,
                      "'heads' entry 0 'anchors'", "no anchor");
}

/** @brief Tests that a head's output, coding and scale are each checked by key. */
TEST(Small_DetectAnchorTask, RefusesAHeadWithABadOutputCodingOrScale)
{
    SectionText unnamed;
    unnamed.heads = R"([{"output": "", "anchors": [0], "coding": "logit", "scale_xy": 1.0}])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(unnamed)), StatusCode::InvalidDescription,
                      "'heads' entry 0 'output'", "empty string");

    SectionText silent;
    silent.heads = R"([{"anchors": [0], "coding": "logit", "scale_xy": 1.0}])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(silent)), StatusCode::InvalidDescription,
                      "'heads' entry 0", "no 'output'");

    SectionText coded;
    coded.heads = "[" + head_text(R"("logit")", "1.0") + ", " + head_text(R"("linear")", "1.0")
                + "]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(coded)), StatusCode::InvalidDescription,
                      "'heads' entry 1 'coding'", "'linear'", "'logit', 'activated'");

    SectionText flat;
    flat.heads = "[" + head_text(R"("logit")", "0.0") + "]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(flat)), StatusCode::InvalidDescription,
                      "'heads' entry 0 'scale_xy'", "above zero");

    SectionText worded;
    worded.heads = "[" + head_text(R"("logit")", R"("wide")") + "]";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(worded)), StatusCode::InvalidDescription,
                      "'heads' entry 0 'scale_xy'", "a string");
}

/** @brief Tests that a criterion and its exponent are checked by key. */
TEST(Small_DetectAnchorTask, RefusesABadCriterionOrExponent)
{
    SectionText unknown;
    unknown.suppression = R"({"criterion": "giou"})";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(unknown)), StatusCode::InvalidDescription,
                      "'suppression' 'criterion'", "'giou'", "'iou', 'distance_iou'");

    SectionText unread;
    unread.suppression = R"({"criterion": "iou", "distance_exponent": 0.5})";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(unread)), StatusCode::InvalidDescription,
                      "'suppression' 'distance_exponent'", "'iou'", "does not read");

    SectionText absent;
    absent.suppression = R"({"criterion": "distance_iou"})";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(absent)), StatusCode::InvalidDescription,
                      "'suppression'", "no 'distance_exponent'");

    SectionText listed;
    listed.suppression = R"(["iou"])";
    SA_EXPECT_FAILURE(read_anchor_task(section_from(listed)), StatusCode::InvalidDescription,
                      "'suppression'", "an array");
}

/** @brief Tests that the writer refuses a head whose class count is not the task's. */
TEST(Small_DetectAnchorTask, WriterRefusesAHeadOfAnotherClassCount)
{
    AnchorTask task = make_two_head_task();
    task.heads[1].options.num_classes = 4;

    SA_EXPECT_FAILURE(write_anchor_task(task), StatusCode::InvalidDescription,
                      "'heads' entry 1", "4 classes", "'classes'", "3");
}

/** @brief Tests that the writer refuses what the reader refuses in a task's values. */
TEST(Small_DetectAnchorTask, WriterRefusesTheValuesTheReaderRefuses)
{
    AnchorTask classless = make_two_head_task();
    classless.num_classes = 0;
    SA_EXPECT_FAILURE(write_anchor_task(classless), StatusCode::InvalidDescription,
                      "'classes'", "at least 1");

    AnchorTask flat = make_two_head_task();
    flat.layout.anchors[4].height = std::numeric_limits<float>::quiet_NaN();
    SA_EXPECT_FAILURE(write_anchor_task(flat), StatusCode::InvalidDescription,
                      "'anchors' entry 4", "above zero");

    AnchorTask past = make_two_head_task();
    past.layout.head_anchors[0][1] = 6;
    SA_EXPECT_FAILURE(write_anchor_task(past), StatusCode::InvalidDescription,
                      "'heads' entry 0 'anchors' entry 1", "anchor 6", "6 anchors");

    AnchorTask headless = make_two_head_task();
    headless.heads.clear();
    SA_EXPECT_FAILURE(write_anchor_task(headless), StatusCode::InvalidDescription,
                      "'heads'", "no head");

    AnchorTask uneven = make_two_head_task();
    uneven.heads.pop_back();
    SA_EXPECT_FAILURE(write_anchor_task(uneven), StatusCode::InvalidDescription,
                      "'heads'", "1 heads", "2 masks");

    AnchorTask unnamed = make_two_head_task();
    unnamed.heads[0].output.clear();
    SA_EXPECT_FAILURE(write_anchor_task(unnamed), StatusCode::InvalidDescription,
                      "'heads' entry 0 'output'", "empty string");

    AnchorTask stretched = make_two_head_task();
    stretched.heads[1].options.scale_xy = std::numeric_limits<float>::infinity();
    SA_EXPECT_FAILURE(write_anchor_task(stretched), StatusCode::InvalidDescription,
                      "'heads' entry 1 'scale_xy'", "finite");
}

/** @brief Tests that the writer refuses an enumerator its tables do not list. */
TEST(Small_DetectAnchorTask, WriterRefusesAnEnumeratorWithoutAString)
{
    AnchorTask coded = make_two_head_task();
    coded.heads[1].options.coding = static_cast<HeadCoding>(7);
    SA_EXPECT_FAILURE(write_anchor_task(coded), StatusCode::InvalidDescription,
                      "'heads' entry 1 'coding'", "7", "'logit', 'activated'");

    AnchorTask measured = make_two_head_task();
    measured.criterion = static_cast<SuppressionCriterion>(9);
    SA_EXPECT_FAILURE(write_anchor_task(measured), StatusCode::InvalidDescription,
                      "'suppression' 'criterion'", "9", "'iou', 'distance_iou'");
}

/** @brief Tests that the writer refuses an exponent that is not finite where it is read. */
TEST(Small_DetectAnchorTask, WriterRefusesAnExponentThatIsNotFiniteUnderDistanceIoU)
{
    AnchorTask task = make_two_head_task();
    task.distance_exponent = std::numeric_limits<float>::infinity();
    SA_EXPECT_OK(write_anchor_task(task));

    task.criterion = SuppressionCriterion::DistanceIoU;
    SA_EXPECT_FAILURE(write_anchor_task(task), StatusCode::InvalidDescription,
                      "'suppression' 'distance_exponent'", "finite");
}
