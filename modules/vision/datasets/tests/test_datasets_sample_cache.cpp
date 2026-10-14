/**************************************************************************/
/* test_datasets_sample_cache.cpp                                         */
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
 * @file test_datasets_sample_cache.cpp
 * @brief Tests the raw cache of a sample list: its names, its writer, its use and what is stale.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_cache.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::Image;
using Vision::Imaging::ImageSize;

namespace
{
    /** @brief Returns an RGB image whose bytes rise from a first value. */
    Image rising_image(ImageSize size, std::uint8_t first)
    {
        Image image(size, 3);
        for (std::size_t i = 0; i < image.byte_count(); ++i)
            image.data()[i] = static_cast<std::uint8_t>(first + i);

        return image;
    }

    /** @brief Returns a list of one sample per file, each file being one image. */
    SampleList list_of(const std::vector<std::filesystem::path>& files)
    {
        SampleList list;
        for (const std::filesystem::path& file : files)
        {
            Sample sample;
            sample.file = static_cast<std::uint32_t>(list.files.size());
            list.files.push_back(file);
            list.samples.push_back(sample);
        }

        return list;
    }

    /** @brief Returns the options of a cache below a scratch directory. */
    SampleCacheOptions options_of(const Support::ScratchDirectory& scratch)
    {
        SampleCacheOptions options;
        options.directory = scratch.path() / "cache";
        return options;
    }

    /** @brief Returns the byte count of the raw image file of an RGB image of a size. */
    std::uint64_t raw_bytes(ImageSize size)
    {
        return Vision::Imaging::K_RAW_IMAGE_HEADER_BYTES +
               static_cast<std::uint64_t>(size.height) * size.width * 3;
    }
} // namespace

/** @brief Tests that the first pass writes every image and the second writes none. */
TEST(Unit_DatasetsSampleCache, TheFirstPassWritesAndTheSecondReuses)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_passes");
    const SampleList list = list_of({
        Support::write_raw_image(scratch, "images/a.jpg", rising_image(ImageSize{2, 3}, 0)),
        Support::write_raw_image(scratch, "images/b.jpg", rising_image(ImageSize{4, 2}, 9)),
    });
    const SampleCacheOptions options = options_of(scratch);

    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport first, write_sample_cache(list, options));
    EXPECT_EQ(first.written, 2u);
    EXPECT_EQ(first.reused, 0u);
    EXPECT_EQ(first.bytes_written, raw_bytes(ImageSize{2, 3}) + raw_bytes(ImageSize{4, 2}));

    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport second, write_sample_cache(list, options));
    EXPECT_EQ(second.written, 0u);
    EXPECT_EQ(second.reused, 2u);
    EXPECT_EQ(second.bytes_written, 0u);
}

/** @brief Tests that a list pointed at the cache loads the pixels of its images from it. */
TEST(Unit_DatasetsSampleCache, AListPointedAtTheCacheLoadsTheSamePixels)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_use");
    const Image original = rising_image(ImageSize{2, 3}, 40);
    const SampleList list =
        list_of({Support::write_raw_image(scratch, "images/a.jpg", original)});
    const SampleCacheOptions options = options_of(scratch);

    SA_EXPECT_OK(write_sample_cache(list, options));
    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));

    ASSERT_EQ(cached.files.size(), 1u);
    EXPECT_EQ(cached.files[0].parent_path(), options.directory);
    EXPECT_EQ(cached.files[0].extension(), ".saim");
    EXPECT_EQ(cached.files[0].filename().string().rfind("a-", 0), 0u);

    Image loaded;
    SA_EXPECT_OK(load_sample_image(cached, 0, Vision::Codecs::DecodeOptions{}, loaded));
    ASSERT_EQ(loaded.size(), original.size());
    ASSERT_EQ(loaded.channels(), 3);
    for (std::size_t i = 0; i < original.byte_count(); ++i)
        EXPECT_EQ(loaded.data()[i], original.data()[i]) << "byte " << i;
}

/** @brief Tests that a longest side reduces a larger image and keeps a smaller one. */
TEST(Unit_DatasetsSampleCache, ALongestSideReducesOnlyWhatExceedsIt)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_side");
    const SampleList list = list_of({
        Support::write_raw_image(scratch, "images/wide.jpg",
                                 Support::uniform_image(ImageSize{4, 8}, 3, 200)),
        Support::write_raw_image(scratch, "images/small.jpg",
                                 Support::uniform_image(ImageSize{2, 2}, 3, 100)),
    });

    SampleCacheOptions options = options_of(scratch);
    options.longest_side = 4;

    SA_EXPECT_OK(write_sample_cache(list, options));
    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));

    Image loaded;
    SA_EXPECT_OK(load_sample_image(cached, 0, Vision::Codecs::DecodeOptions{}, loaded));
    EXPECT_EQ(loaded.height(), 2);
    EXPECT_EQ(loaded.width(), 4);
    EXPECT_EQ(loaded.data()[0], 200);

    SA_EXPECT_OK(load_sample_image(cached, 1, Vision::Codecs::DecodeOptions{}, loaded));
    EXPECT_EQ(loaded.height(), 2);
    EXPECT_EQ(loaded.width(), 2);
}

/** @brief Tests that an image that changed has no cache file until it is cached again. */
TEST(Unit_DatasetsSampleCache, AnImageThatChangedIsStale)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_stale");
    const SampleList list = list_of({Support::write_raw_image(
        scratch, "images/a.jpg", Support::uniform_image(ImageSize{2, 2}, 3, 1))});
    const SampleCacheOptions options = options_of(scratch);

    SA_EXPECT_OK(write_sample_cache(list, options));

    // Another byte count is another cache file, whatever the two modification times are.
    Support::write_raw_image(scratch, "images/a.jpg",
                             Support::uniform_image(ImageSize{2, 3}, 3, 2));
    SA_EXPECT_FAILURE(use_sample_cache(list, options), StatusCode::FileNotFound,
                      "images/a.jpg", "sample cache", "changed since");

    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport again, write_sample_cache(list, options));
    EXPECT_EQ(again.written, 1u);
    EXPECT_EQ(again.reused, 0u);

    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));
    Image loaded;
    SA_EXPECT_OK(load_sample_image(cached, 0, Vision::Codecs::DecodeOptions{}, loaded));
    EXPECT_EQ(loaded.width(), 3);
    EXPECT_EQ(loaded.data()[0], 2);
}

/** @brief Tests that a file of stored pixels and a mask keep their paths and are not cached. */
TEST(Unit_DatasetsSampleCache, StoredPixelsAndMasksAreLeftAlone)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_alone");

    SampleList list;
    list.files = {
        scratch.write("pixels.bin", Support::rising_bytes(4, 0)),
        Support::write_raw_image(scratch, "images/a.jpg",
                                 Support::uniform_image(ImageSize{2, 2}, 3, 7)),
        Support::write_raw_image(scratch, "masks/a.png",
                                 Support::uniform_image(ImageSize{2, 2}, 1, 0)),
    };

    Sample stored;
    stored.file = 0;
    stored.stored = StoredPixels{0, ImageSize{2, 2}, 1, PlaneLayout::Interleaved};
    Sample masked;
    masked.file = 1;
    masked.mask = 2;
    list.samples = {stored, masked};

    const SampleCacheOptions options = options_of(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const SampleCacheReport report, write_sample_cache(list, options));
    EXPECT_EQ(report.written, 1u);

    SA_ASSERT_OK_AND_ASSIGN(const SampleList cached, use_sample_cache(list, options));
    EXPECT_EQ(cached.files[0], list.files[0]);
    EXPECT_NE(cached.files[1], list.files[1]);
    EXPECT_EQ(cached.files[2], list.files[2]);
}

/** @brief Tests that the name of a cache file follows the source and the options alone. */
TEST(Unit_DatasetsSampleCache, ANameFollowsTheSourceAndTheOptions)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_names");
    const std::filesystem::path source = Support::write_raw_image(
        scratch, "images/a.jpg", Support::uniform_image(ImageSize{2, 2}, 3, 1));

    SampleCacheOptions options = options_of(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path first,
                            sample_cache_path(source, options));
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path again,
                            sample_cache_path(source, options));
    EXPECT_EQ(first, again);

    // "a-", sixteen digits of the hash and ".saim".
    EXPECT_EQ(first.filename().string().size(), 2u + 16u + 5u);

    options.longest_side = 320;
    SA_ASSERT_OK_AND_ASSIGN(const std::filesystem::path reduced,
                            sample_cache_path(source, options));
    EXPECT_NE(first, reduced);

    SA_EXPECT_FAILURE(sample_cache_path(scratch.path() / "images/gone.jpg", options),
                      StatusCode::FileNotFound, "gone.jpg");
}

/** @brief Tests each refusal: no directory, a side out of range, and a side over masks. */
TEST(Unit_DatasetsSampleCache, RefusesOptionsItCannotServe)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_cache_refusals");
    SampleList list = list_of({Support::write_raw_image(
        scratch, "images/a.jpg", Support::uniform_image(ImageSize{2, 2}, 3, 1))});

    SampleCacheOptions options;
    SA_EXPECT_FAILURE(write_sample_cache(list, options), StatusCode::InvalidDescription,
                      "no directory");

    options = options_of(scratch);
    options.longest_side = -1;
    SA_EXPECT_FAILURE(write_sample_cache(list, options), StatusCode::InvalidDescription,
                      "a longest side of -1");

    list.files.push_back(Support::write_raw_image(
        scratch, "masks/a.png", Support::uniform_image(ImageSize{2, 2}, 1, 0)));
    list.samples[0].mask = 1;
    options.longest_side = 1;
    SA_EXPECT_FAILURE(write_sample_cache(list, options), StatusCode::InvalidDescription,
                      "a longest side of 1", "masks");
}
