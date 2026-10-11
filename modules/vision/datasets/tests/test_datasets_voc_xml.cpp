/**************************************************************************/
/* test_datasets_voc_xml.cpp                                              */
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
 * @file test_datasets_voc_xml.cpp
 * @brief Tests the reader of a dataset kept as images and one XML annotation per image.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/voc_xml.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <common/test_status.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::NormalizedBox;

namespace
{
    /** @brief Holds the absolute difference a box fraction may have from its quotient. */
    constexpr float K_FRACTION_TOLERANCE = 1e-6f;

    /** @brief Returns options that read the split "train" of a directory. */
    VOCXMLOptions options_of(const Support::ScratchDirectory& scratch)
    {
        VOCXMLOptions options;
        options.root = scratch.path();
        return options;
    }

    /** @brief Returns an object element: a name, a difficult flag and four box edges. */
    std::string object_of(const char* name, const char* difficult, const char* edges)
    {
        return std::string("  <object>\n    <name>") + name + "</name>\n    <difficult>" +
               difficult + "</difficult>\n    <bndbox>" + edges + "</bndbox>\n  </object>\n";
    }

    /** @brief Returns an annotation of a 200 by 100 image that holds some objects. */
    std::string annotation_of(const std::string& objects)
    {
        return "<annotation>\n  <filename>first.jpg</filename>\n"
               "  <size><width>200</width><height>100</height><depth>3</depth></size>\n" +
               objects + "</annotation>\n";
    }

    /** @brief Holds the edges of a box that spans [10, 60] by [20, 40] once converted. */
    constexpr const char* K_EDGES = "<xmin>11</xmin><ymin>21</ymin><xmax>60</xmax><ymax>40</ymax>";

    /** @brief Returns the path of a file of a list below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t sample,
                              const std::filesystem::path& base)
    {
        return list.files[list.samples[sample].file].lexically_relative(base).generic_string();
    }
} // namespace

/** @brief Tests that the listed annotations are read and their classes sorted by name. */
TEST(Unit_DatasetsVocXml, TheListedAnnotationsAreReadWithSortedClasses)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_list");
    scratch.write_text("ImageSets/Main/train.txt", "second\r\n\nfirst  1\n");
    scratch.write_text("Annotations/first.xml",
                       annotation_of(object_of("dog", "0", K_EDGES) +
                                     object_of("cat &amp; co", "0", K_EDGES)));
    scratch.write_text("Annotations/second.xml",
                       "<annotation><size><width>10</width><height>10</height></size>"
                       "</annotation>");
    scratch.write_text("Annotations/unlisted.xml", "not read");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options_of(scratch)));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"cat & co", "dog"}));
    EXPECT_EQ(list.class_count, 2);
    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "JPEGImages/second.jpg");
    EXPECT_EQ(relative_file(list, 1, scratch.path()), "JPEGImages/first.jpg");
    EXPECT_TRUE(list.samples[0].objects.is_empty());

    const Sample& sample = list.samples[1];
    EXPECT_FALSE(sample.class_index.has_value());
    ASSERT_EQ(sample.objects.size(), 2u);
    EXPECT_EQ(sample.objects.classes()[0], 1);
    EXPECT_EQ(sample.objects.classes()[1], 0);

    // The box spans [10, 60] by [20, 40] of a 200 by 100 image.
    const NormalizedBox& box = sample.objects.boxes()[0];
    EXPECT_NEAR(box.x_center, 35.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.y_center, 30.0f / 100.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.width, 50.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.height, 20.0f / 100.0f, K_FRACTION_TOLERANCE);
}

/** @brief Tests that an empty split reads every annotation of the folder, sorted. */
TEST(Unit_DatasetsVocXml, AnEmptySplitReadsEveryAnnotation)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_folder");
    scratch.write_text("Annotations/b.xml", annotation_of(object_of("dog", "0", K_EDGES)));
    scratch.write_text("Annotations/a.xml",
                       "<annotation><size><width>4</width><height>4</height></size>"
                       "</annotation>");
    scratch.write_text("Annotations/readme.txt", "not read");

    VOCXMLOptions options = options_of(scratch);
    options.split.clear();
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "JPEGImages/a.jpg");
    EXPECT_EQ(relative_file(list, 1, scratch.path()), "JPEGImages/first.jpg");
}

/** @brief Tests that a difficult object is dropped unless the options keep it. */
TEST(Unit_DatasetsVocXml, ADifficultObjectIsDroppedUnlessKept)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_difficult");
    scratch.write_text("ImageSets/Main/train.txt", "first\n");
    scratch.write_text("Annotations/first.xml",
                       annotation_of(object_of("dog", "1", K_EDGES) +
                                     object_of("cat", "0", K_EDGES)));

    VOCXMLOptions options = options_of(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList dropped, read_voc_xml(options));
    EXPECT_EQ(dropped.class_names, (std::vector<std::string>{"cat"}));
    EXPECT_EQ(dropped.samples[0].objects.size(), 1u);

    options.difficult = VOCDifficult::Kept;
    SA_ASSERT_OK_AND_ASSIGN(const SampleList kept, read_voc_xml(options));
    EXPECT_EQ(kept.class_names, (std::vector<std::string>{"cat", "dog"}));
    EXPECT_EQ(kept.samples[0].objects.size(), 2u);
}

/** @brief Tests that a stated class list gives the indices and refuses a name it lacks. */
TEST(Unit_DatasetsVocXml, AStatedClassListGivesTheIndices)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_classes");
    scratch.write_text("ImageSets/Main/train.txt", "first\n");
    scratch.write_text("Annotations/first.xml",
                       annotation_of(object_of("dog", "0", K_EDGES)));

    VOCXMLOptions options = options_of(scratch);
    options.class_names = {"person", "dog", "cat"};
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options));
    EXPECT_EQ(list.class_count, 3);
    EXPECT_EQ(list.samples[0].objects.classes()[0], 1);

    // The object opens on line 4 of the annotation.
    options.class_names = {"person", "cat"};
    SA_EXPECT_FAILURE(read_voc_xml(options), StatusCode::InvalidFile, "Annotations/first.xml",
                      "line 4", "read_voc_xml", "the class 'dog'");

    options.class_names = {"dog", "cat", "dog"};
    SA_EXPECT_FAILURE(read_voc_xml(options), StatusCode::InvalidDescription, "read_voc_xml",
                      "names 'dog' twice", "at 0 and at 2");
}

/** @brief Tests that what parse_xml refuses is returned with the annotation and its line. */
TEST(Unit_DatasetsVocXml, AMalformedAnnotationNamesItsFileAndLine)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_malformed");
    scratch.write_text("ImageSets/Main/train.txt", "first\n");
    scratch.write_text("Annotations/first.xml", "<annotation>\n<size>\n</annotation>\n");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(scratch)), StatusCode::InvalidFile,
                      "Annotations/first.xml", "line 3", "parse_xml", "closes '<size>'");
}

/** @brief Tests that an element or a number the layout requires is refused with its line. */
TEST(Unit_DatasetsVocXml, AMissingElementOrNumberIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_missing");
    scratch.write_text("ImageSets/Main/train.txt", "first\n");
    const VOCXMLOptions options = options_of(scratch);

    scratch.write_text("Annotations/first.xml", "<annotation>\n</annotation>\n");
    SA_EXPECT_FAILURE(read_voc_xml(options), StatusCode::InvalidFile, "first.xml", "line 1",
                      "read_voc_xml", "'annotation' has no 'size'");

    scratch.write_text("Annotations/first.xml", "<image/>");
    SA_EXPECT_FAILURE(read_voc_xml(options), StatusCode::InvalidFile, "first.xml", "line 1",
                      "the root element is 'image'");

    scratch.write_text("Annotations/first.xml",
                       "<annotation>\n<size>\n<width>wide</width><height>1</height></size>\n"
                       "</annotation>");
    SA_EXPECT_FAILURE(read_voc_xml(options), StatusCode::InvalidFile, "first.xml", "line 3",
                      "'width' holds 'wide'");

    scratch.write_text("Annotations/first.xml",
                       annotation_of("  <object>\n    <name>dog</name>\n  </object>\n"));
    SA_EXPECT_FAILURE(read_voc_xml(options), StatusCode::InvalidFile, "first.xml", "line 4",
                      "'object' has no 'bndbox'");

    scratch.write_text(
        "Annotations/first.xml",
        annotation_of(object_of(
            "dog", "0", "<xmin>9</xmin><ymin>1</ymin><xmax>5</xmax><ymax>2</ymax>")));
    SA_EXPECT_FAILURE(read_voc_xml(options), StatusCode::InvalidFile, "first.xml", "line 7",
                      "the box ends at (5, 2), before its start (9, 1)");
}

/** @brief Tests that an absent list, and a stem without an annotation, are refused. */
TEST(Unit_DatasetsVocXml, TheListAndItsStemsMustExist)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_absent");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(scratch)), StatusCode::FileNotFound,
                      "read_voc_xml", "the split 'train'", "ImageSets/Main/train.txt");

    scratch.write_text("ImageSets/Main/train.txt", "\n\nmissing\n");
    SA_EXPECT_FAILURE(read_voc_xml(options_of(scratch)), StatusCode::FileNotFound,
                      "ImageSets/Main/train.txt", "line 3", "read_voc_xml",
                      "Annotations/missing.xml");

    scratch.write_text("ImageSets/Main/train.txt", "\n");
    SA_EXPECT_FAILURE(read_voc_xml(options_of(scratch)), StatusCode::InvalidFile,
                      "holds no annotation");
}
