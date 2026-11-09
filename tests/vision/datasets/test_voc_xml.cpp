/**************************************************************************/
/* test_voc_xml.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_voc_xml.cpp
 * @brief Tests the reader of a dataset kept as images and one XML annotation per image.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/voc_xml.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::NormalizedBox;

namespace
{
    /**
     * @brief Holds the absolute difference a box fraction may have from its quotient: 32 times the
     *        float unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_FRACTION_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Returns options that read the split "train" of the folder data. */
    VOCXMLOptions options_of()
    {
        VOCXMLOptions options;
        options.root = "data";
        return options;
    }

    /** @brief Returns options that read the split "train" of a directory on disk. */
    VOCXMLOptions disk_options_of(const Support::ScratchDirectory& scratch)
    {
        VOCXMLOptions options;
        options.root = scratch.path();
        return options;
    }

    /** @brief Adds a text file below the folder data of a source. */
    void put(MemorySource& source, const std::string& relative, const std::string& text)
    {
        source.add_file("data/" + relative, text);
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

    /** @brief Returns the text of an annotation of a 4 by 4 image that holds no object. */
    std::string empty_annotation()
    {
        return "<annotation><size><width>4</width><height>4</height></size></annotation>";
    }

    /** @brief Returns a source of two listed annotations, one of them with two objects. */
    MemorySource listed_source()
    {
        MemorySource source;
        put(source, "ImageSets/Main/train.txt", "second\r\n\nfirst  1\n");
        put(source, "Annotations/first.xml",
            annotation_of(object_of("dog", "0", K_EDGES) +
                          object_of("cat &amp; co", "0", K_EDGES)));
        put(source, "Annotations/second.xml", empty_annotation());
        put(source, "Annotations/unlisted.xml", "not read");
        return source;
    }

    /** @brief Returns a source of one listed annotation with the text given. */
    MemorySource one_annotation(const std::string& text)
    {
        MemorySource source;
        put(source, "ImageSets/Main/train.txt", "first\n");
        put(source, "Annotations/first.xml", text);
        return source;
    }

    /** @brief Returns the path of a file of a list below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t sample,
                              const std::filesystem::path& base)
    {
        return list.files[list.samples[sample].file].lexically_relative(base).generic_string();
    }
} // namespace

/** @brief Tests that the annotations of the split are read in the order the list gives. */
TEST(Small_DatasetsVocXml, TheListedAnnotationsAreReadInTheOrderOfTheList)
{
    const MemorySource source = listed_source();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options_of(), source));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, "data"), "JPEGImages/second.jpg");
    EXPECT_EQ(relative_file(list, 1, "data"), "JPEGImages/first.jpg");
    EXPECT_TRUE(list.samples[0].objects.is_empty());
}

/** @brief Tests that the classes are the object names, sorted, and an object takes its index. */
TEST(Small_DatasetsVocXml, TheClassesAreSortedByName)
{
    const MemorySource source = listed_source();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options_of(), source));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"cat & co", "dog"}));
    EXPECT_EQ(list.class_count, 2);
    const Sample& sample = list.samples[1];
    EXPECT_FALSE(sample.class_index.has_value());
    ASSERT_EQ(sample.objects.size(), 2u);
    EXPECT_EQ(sample.objects.classes()[0], 1);
    EXPECT_EQ(sample.objects.classes()[1], 0);
}

/** @brief Tests that a box of pixel edges becomes a centre and a size in fractions. */
TEST(Small_DatasetsVocXml, ABoxBecomesACentreAndASizeInFractions)
{
    const MemorySource source = listed_source();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options_of(), source));

    // The box spans [10, 60] by [20, 40] of a 200 by 100 image.
    ASSERT_EQ(list.samples[1].objects.size(), 2u);
    const NormalizedBox& box = list.samples[1].objects.boxes()[0];
    EXPECT_NEAR(box.x_center, 35.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.y_center, 30.0f / 100.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.width, 50.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.height, 20.0f / 100.0f, K_FRACTION_TOLERANCE);
}

/** @brief Tests that an empty split reads every annotation of the folder, sorted. */
TEST(Small_DatasetsVocXml, AnEmptySplitReadsEveryAnnotation)
{
    MemorySource source;
    put(source, "Annotations/b.xml", annotation_of(object_of("dog", "0", K_EDGES)));
    put(source, "Annotations/a.xml", empty_annotation());
    put(source, "Annotations/readme.txt", "not read");
    VOCXMLOptions options = options_of();
    options.split.clear();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options, source));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, "data"), "JPEGImages/a.jpg");
    EXPECT_EQ(relative_file(list, 1, "data"), "JPEGImages/first.jpg");
}

/** @brief Tests that a difficult object is dropped unless the options keep it. */
TEST(Small_DatasetsVocXml, ADifficultObjectIsDroppedByDefault)
{
    const MemorySource source = one_annotation(
        annotation_of(object_of("dog", "1", K_EDGES) + object_of("cat", "0", K_EDGES)));

    SA_ASSERT_OK_AND_ASSIGN(const SampleList dropped, read_voc_xml(options_of(), source));

    EXPECT_EQ(dropped.class_names, (std::vector<std::string>{"cat"}));
    EXPECT_EQ(dropped.samples[0].objects.size(), 1u);
}

/** @brief Tests that a difficult object is kept when the options say so. */
TEST(Small_DatasetsVocXml, ADifficultObjectIsKeptWhenAsked)
{
    const MemorySource source = one_annotation(
        annotation_of(object_of("dog", "1", K_EDGES) + object_of("cat", "0", K_EDGES)));
    VOCXMLOptions options = options_of();
    options.difficult = VOCDifficult::Kept;

    SA_ASSERT_OK_AND_ASSIGN(const SampleList kept, read_voc_xml(options, source));

    EXPECT_EQ(kept.class_names, (std::vector<std::string>{"cat", "dog"}));
    EXPECT_EQ(kept.samples[0].objects.size(), 2u);
}

/** @brief Tests that a stated class list gives the indices. */
TEST(Small_DatasetsVocXml, AStatedClassListGivesTheIndices)
{
    const MemorySource source = one_annotation(annotation_of(object_of("dog", "0", K_EDGES)));
    VOCXMLOptions options = options_of();
    options.class_names = {"person", "dog", "cat"};

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(options, source));

    EXPECT_EQ(list.class_count, 3);
    EXPECT_EQ(list.samples[0].objects.classes()[0], 1);
}

/** @brief Tests that a stated class list that lacks an object name is refused with its line. */
TEST(Small_DatasetsVocXml, AStatedClassListThatLacksAnObjectNameIsRefused)
{
    const MemorySource source = one_annotation(annotation_of(object_of("dog", "0", K_EDGES)));
    VOCXMLOptions options = options_of();
    options.class_names = {"person", "cat"};

    // The object opens on line 4 of the annotation.
    SA_EXPECT_FAILURE(read_voc_xml(options, source), StatusCode::InvalidFile,
                      "Annotations/first.xml", "line 4", "read_voc_xml", "the class 'dog'");
}

/** @brief Tests that a stated class list that names a class twice is refused. */
TEST(Small_DatasetsVocXml, AStatedClassListThatNamesAClassTwiceIsRefused)
{
    const MemorySource source = one_annotation(annotation_of(object_of("dog", "0", K_EDGES)));
    VOCXMLOptions options = options_of();
    options.class_names = {"dog", "cat", "dog"};

    SA_EXPECT_FAILURE(read_voc_xml(options, source), StatusCode::InvalidDescription,
                      "read_voc_xml", "names 'dog' twice", "at 0 and at 2");
}

/** @brief Tests that what parse_xml refuses is returned with the annotation and its line. */
TEST(Small_DatasetsVocXml, AMalformedAnnotationNamesItsFileAndLine)
{
    const MemorySource source = one_annotation("<annotation>\n<size>\n</annotation>\n");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::InvalidFile,
                      "Annotations/first.xml", "line 3", "parse_xml", "closes '<size>'");
}

/** @brief Tests that an annotation without a size is refused with its line. */
TEST(Small_DatasetsVocXml, AnAnnotationWithoutASizeIsRefused)
{
    const MemorySource source = one_annotation("<annotation>\n</annotation>\n");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::InvalidFile, "first.xml",
                      "line 1", "read_voc_xml", "'annotation' has no 'size'");
}

/** @brief Tests that a root element that is no annotation is refused with its line. */
TEST(Small_DatasetsVocXml, ARootThatIsNoAnnotationIsRefused)
{
    const MemorySource source = one_annotation("<image/>");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::InvalidFile, "first.xml",
                      "line 1", "the root element is 'image'");
}

/** @brief Tests that a size that is no number is refused with its line. */
TEST(Small_DatasetsVocXml, ASizeThatIsNoNumberIsRefused)
{
    const MemorySource source = one_annotation(
        "<annotation>\n<size>\n<width>wide</width><height>1</height></size>\n</annotation>");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::InvalidFile, "first.xml",
                      "line 3", "'width' holds 'wide'");
}

/** @brief Tests that an object without a box is refused with its line. */
TEST(Small_DatasetsVocXml, AnObjectWithoutABoxIsRefused)
{
    const MemorySource source = one_annotation(
        annotation_of("  <object>\n    <name>dog</name>\n  </object>\n"));

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::InvalidFile, "first.xml",
                      "line 4", "'object' has no 'bndbox'");
}

/** @brief Tests that a box that ends before it starts is refused with its line. */
TEST(Small_DatasetsVocXml, ABoxThatEndsBeforeItStartsIsRefused)
{
    const MemorySource source = one_annotation(annotation_of(object_of(
        "dog", "0", "<xmin>9</xmin><ymin>1</ymin><xmax>5</xmax><ymax>2</ymax>")));

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::InvalidFile, "first.xml",
                      "line 7", "the box ends at (5, 2), before its start (9, 1)");
}

/** @brief Tests that a split without a list is refused with the path of the list. */
TEST(Small_DatasetsVocXml, ASplitWithoutAListIsRefused)
{
    const MemorySource source;

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::FileNotFound,
                      "read_voc_xml", "the split 'train'", "ImageSets/Main/train.txt");
}

/** @brief Tests that a stem without an annotation is refused with its line of the list. */
TEST(Small_DatasetsVocXml, AStemWithoutAnAnnotationIsRefused)
{
    MemorySource source;
    put(source, "ImageSets/Main/train.txt", "\n\nmissing\n");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::FileNotFound,
                      "ImageSets/Main/train.txt", "line 3", "read_voc_xml",
                      "Annotations/missing.xml");
}

/** @brief Tests that a list with no stem is refused. */
TEST(Small_DatasetsVocXml, AListWithoutAStemIsRefused)
{
    MemorySource source;
    put(source, "ImageSets/Main/train.txt", "\n");

    SA_EXPECT_FAILURE(read_voc_xml(options_of(), source), StatusCode::InvalidFile,
                      "holds no annotation");
}

/** @brief Tests that the file-system form reads the list and the annotations written to disk. */
TEST(Medium_DatasetsVocXml, AnnotationsOnDiskAreReadInTheOrderOfTheList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_disk");
    scratch.write_text("ImageSets/Main/train.txt", "second\r\n\nfirst  1\n");
    scratch.write_text("Annotations/first.xml",
                       annotation_of(object_of("dog", "0", K_EDGES)));
    scratch.write_text("Annotations/second.xml", empty_annotation());

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_voc_xml(disk_options_of(scratch)));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "JPEGImages/second.jpg");
    EXPECT_EQ(relative_file(list, 1, scratch.path()), "JPEGImages/first.jpg");
    EXPECT_EQ(list.class_names, (std::vector<std::string>{"dog"}));
}

/** @brief Tests that a split whose list is absent from the disk is reported as not found. */
TEST(Medium_DatasetsVocXml, ASplitWithoutAListOnDiskIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_voc_absent");

    SA_EXPECT_FAILURE(read_voc_xml(disk_options_of(scratch)), StatusCode::FileNotFound,
                      "read_voc_xml", "the split 'train'", "ImageSets/Main/train.txt");
}
