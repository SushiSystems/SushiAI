/**************************************************************************/
/* jpeg_decoder.cpp                                                       */
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
 * @file jpeg_decoder.cpp
 * @brief Implements the JPEG decoder over libjpeg, with its error exit returned to the caller.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/codecs/jpeg_decoder.hpp>

#include <array>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <jpeglib.h>
#include <jerror.h>

#include <SushiRuntime/core/span.hpp>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>

static_assert(BITS_IN_JSAMPLE == 8, "the JPEG decoder writes libjpeg's samples as 8-bit pixels");

namespace SushiAI::Vision::Codecs
{
    namespace
    {
        /** @brief Holds the sample count of one CMYK pixel as libjpeg hands it over. */
        constexpr std::int32_t K_CMYK_COMPONENTS = 4;

        /** @brief Holds the largest value of a sample. */
        constexpr unsigned K_MAX_SAMPLE = 255u;

        /**
         * @brief Holds everything a libjpeg call and its error handlers read and write.
         *
         * Every member is trivial, so a longjmp back to run_step skips no destructor.
         */
        struct Session
        {
            /** @brief Holds the library's decompressor. */
            jpeg_decompress_struct info;

            /** @brief Holds the library's error manager, with the three handlers below. */
            jpeg_error_mgr errors;

            /** @brief Holds the point in run_step a fatal error returns to. */
            std::jmp_buf resume;

            /** @brief Holds the library's text for the fatal error or the first warning. */
            char message[JMSG_LENGTH_MAX];

            /** @brief Points at the file's bytes. */
            const unsigned char* source;

            /** @brief Counts the file's bytes. */
            unsigned long source_bytes;

            /** @brief Points at the first destination row. */
            JSAMPLE* rows;

            /** @brief Counts the bytes of one destination row. */
            std::size_t row_bytes;

            /** @brief Points at one row of CMYK samples, or is null when rows are read in place. */
            JSAMPLE* cmyk_row;

            /** @brief Holds 0 when a CMYK sample of 255 means no ink, and 255 otherwise. */
            unsigned ink_flip;

            /** @brief Reports whether the library stopped before the last row. */
            bool stopped_early;
        };

        /** @brief Names one library call sequence that run_step guards. */
        using Step = void (*)(Session&);

        extern "C"
        {
            /** @brief Records the library's text for a fatal error and returns to run_step. */
            [[noreturn]] void abandon_decode(j_common_ptr common)
            {
                Session* session = static_cast<Session*>(common->client_data);
                (*common->err->format_message)(common, session->message);
                std::longjmp(session->resume, 1);
            }

            /** @brief Counts a corrupt-data warning and records the text of the first one. */
            void record_warning(j_common_ptr common, int level)
            {
                if (level >= 0)
                    return;

                Session* session = static_cast<Session*>(common->client_data);
                if (common->err->num_warnings == 0)
                    (*common->err->format_message)(common, session->message);

                ++common->err->num_warnings;
            }

            /** @brief Writes nothing where the library would print to the standard error. */
            void discard_message(j_common_ptr)
            {

            }
        }

        /** @brief Creates the decompressor over the session's bytes and reads the header. */
        void open_stream(Session& session)
        {
            jpeg_create_decompress(&session.info);
            jpeg_mem_src(&session.info, session.source, session.source_bytes);
            (void)jpeg_read_header(&session.info, TRUE);
        }

        /** @brief Starts the output in the colour space the session's decompressor names. */
        void start_output(Session& session)
        {
            (void)jpeg_start_decompress(&session.info);
        }

        /** @brief Converts one row of CMYK samples to RGB: each channel is its ink times black. */
        void convert_cmyk_row(const JSAMPLE* cmyk, JSAMPLE* rgb, JDIMENSION width,
                              unsigned ink_flip)
        {
            for (JDIMENSION x = 0; x < width; ++x)
            {
                const unsigned key = cmyk[3] ^ ink_flip;
                for (int channel = 0; channel < Imaging::K_RGB_CHANNELS; ++channel)
                {
                    const unsigned paper = cmyk[channel] ^ ink_flip;
                    rgb[channel] =
                        static_cast<JSAMPLE>((paper * key + K_MAX_SAMPLE / 2) / K_MAX_SAMPLE);
                }

                cmyk += K_CMYK_COMPONENTS;
                rgb += Imaging::K_RGB_CHANNELS;
            }
        }

        /** @brief Reads every row into the session's destination and finishes the stream. */
        void read_rows(Session& session)
        {
            while (session.info.output_scanline < session.info.output_height)
            {
                JSAMPLE* destination =
                    session.rows +
                    static_cast<std::size_t>(session.info.output_scanline) * session.row_bytes;
                JSAMPROW target = session.cmyk_row != nullptr ? session.cmyk_row : destination;

                if (jpeg_read_scanlines(&session.info, &target, 1) != 1)
                {
                    session.stopped_early = true;
                    return;
                }

                if (session.cmyk_row != nullptr)
                    convert_cmyk_row(session.cmyk_row, destination, session.info.output_width,
                                     session.ink_flip);
            }

            (void)jpeg_finish_decompress(&session.info);
        }

        /** @brief Destroys the decompressor and frees what the library allocated for it. */
        void close_stream(Session& session)
        {
            jpeg_destroy_decompress(&session.info);
        }

        /**
         * @brief Runs one step and reports whether it ended without a fatal library error.
         *
         * This is the only function that calls setjmp. It and the steps hold pointers and
         * integers alone, so the jump from abandon_decode unwinds libjpeg's frames and no
         * object with a destructor.
         * @return False after abandon_decode; the session's message then holds the cause.
         */
        [[nodiscard]] bool run_step(Session& session, Step step)
        {
            if (setjmp(session.resume) != 0)
                return false;

            step(session);
            return true;
        }

        /** @brief Returns the failure the session's message describes. */
        [[nodiscard]] Status failure_of(const Session& session) noexcept
        {
            const StatusCode code = session.errors.msg_code == JERR_OUT_OF_MEMORY
                                        ? StatusCode::OutOfMemory
                                        : StatusCode::InvalidFile;
            return Status::from_message(code, [&]()
            {
                return SushiLog::format_text("decode_jpeg: {}",
                                               static_cast<const char*>(session.message));
            });
        }

        /** @brief Owns a libjpeg decompressor over one file, from creation to destruction. */
        class Decompressor final
        {
            public:
                /**
                 * @brief Prepares a session over the bytes; the library is first called by run().
                 * @param bytes Must outlive the decompressor and fit an unsigned long count.
                 */
                explicit Decompressor(SushiRuntime::span<const std::uint8_t> bytes) noexcept
                {
                    session_.info.err = jpeg_std_error(&session_.errors);
                    session_.errors.error_exit = abandon_decode;
                    session_.errors.emit_message = record_warning;
                    session_.errors.output_message = discard_message;
                    session_.info.client_data = &session_;
                    session_.source = bytes.data();
                    session_.source_bytes = static_cast<unsigned long>(bytes.size());
                }

                /** @brief Destroys the library's decompressor. */
                ~Decompressor()
                {
                    (void)run_step(session_, close_stream);
                }

                /** @brief Forbids a copy; the session holds pointers into itself. */
                Decompressor(const Decompressor&) = delete;

                /** @brief Forbids copy assignment. */
                Decompressor& operator=(const Decompressor&) = delete;

                /** @brief Forbids a move; the session holds pointers into itself. */
                Decompressor(Decompressor&&) = delete;

                /** @brief Forbids move assignment. */
                Decompressor& operator=(Decompressor&&) = delete;

                /** @brief Runs one step and returns the library's failure when it has one. */
                [[nodiscard]] Status run(Step step) noexcept
                {
                    if (run_step(session_, step))
                        return Status();

                    return failure_of(session_);
                }

                /** @brief Returns the session the steps read and write. */
                [[nodiscard]] Session& session() noexcept { return session_; }

            private:
                /// @brief Holds the library's state, zeroed until the constructor fills it.
                Session session_{};
        };

        /** @brief Holds what the decoder does with one colour space a file may store. */
        struct ColorSpaceRule
        {
            /** @brief Names the colour space of the file. */
            J_COLOR_SPACE stored;

            /** @brief Names the colour space in a failure's message. */
            const char* name;

            /** @brief Reports whether the file holds one gray component. */
            bool is_gray;

            /** @brief Reports whether the library converts the file to one gray component. */
            bool converts_to_gray;

            /** @brief Names the colour space the library is asked for when RGB is wanted. */
            J_COLOR_SPACE color_output;

            /** @brief Counts the samples per pixel of color_output. */
            std::int32_t color_components;
        };

        /** @brief Holds one rule per colour space the decoder reads; a new one adds a row. */
        constexpr std::array<ColorSpaceRule, 5> K_COLOR_SPACES = {{
            {JCS_GRAYSCALE, "grayscale", true, true, JCS_EXT_RGB, Imaging::K_RGB_CHANNELS},
            {JCS_YCbCr, "YCbCr", false, true, JCS_EXT_RGB, Imaging::K_RGB_CHANNELS},
            {JCS_RGB, "RGB", false, true, JCS_EXT_RGB, Imaging::K_RGB_CHANNELS},
            {JCS_CMYK, "CMYK", false, false, JCS_CMYK, K_CMYK_COMPONENTS},
            {JCS_YCCK, "YCCK", false, false, JCS_CMYK, K_CMYK_COMPONENTS},
        }};

        /** @brief Holds the output the library is asked for and the image it becomes. */
        struct OutputPlan
        {
            /** @brief Names the colour space the library writes. */
            J_COLOR_SPACE output;

            /** @brief Counts the samples per pixel the library writes. */
            std::int32_t components;

            /** @brief Counts the channels of the decoded image. */
            std::int32_t channels;
        };

        /**
         * @brief Chooses the library output for a file's colour space and the layout asked for.
         * @return The plan, or Unsupported naming the colour space the decoder cannot convert.
         */
        [[nodiscard]] Result<OutputPlan> plan_output(J_COLOR_SPACE stored, ChannelLayout layout)
        {
            for (const ColorSpaceRule& rule : K_COLOR_SPACES)
            {
                if (rule.stored != stored)
                    continue;

                const bool wants_gray = layout == ChannelLayout::Gray ||
                                        (layout == ChannelLayout::Stored && rule.is_gray);
                if (!wants_gray)
                    return OutputPlan{rule.color_output, rule.color_components,
                                      Imaging::K_RGB_CHANNELS};

                SA_FAIL_IF(!rule.converts_to_gray, StatusCode::Unsupported,
                           "decode_jpeg: a {} image does not convert to gray", rule.name);
                return OutputPlan{JCS_GRAYSCALE, Imaging::K_GRAY_CHANNELS,
                                  Imaging::K_GRAY_CHANNELS};
            }

            return Status::from_message(StatusCode::Unsupported, [&]()
            {
                return SushiLog::format_text(
                    "decode_jpeg: colour space {} of libjpeg's J_COLOR_SPACE is not decoded",
                    static_cast<int>(stored));
            });
        }

        /** @brief Decodes a JPEG file; may throw what an allocation or Imaging::Image throws. */
        [[nodiscard]] Result<Imaging::Image> decode_checked(
            SushiRuntime::span<const std::uint8_t> bytes, const DecodeOptions& options)
        {
            SA_FAIL_IF(bytes.size() > std::numeric_limits<unsigned long>::max(),
                       StatusCode::Unsupported,
                       "decode_jpeg: {} bytes are more than libjpeg's memory source addresses",
                       bytes.size());

            Decompressor decompressor(bytes);
            Session& session = decompressor.session();
            SA_TRY(decompressor.run(open_stream));

            const JDIMENSION height = session.info.image_height;
            const JDIMENSION width = session.info.image_width;
            SA_FAIL_IF(!Imaging::is_valid_side(height) || !Imaging::is_valid_side(width),
                       StatusCode::Unsupported,
                       "decode_jpeg: the image is {}x{}; a side must be in [1, {}]", height, width,
                       Imaging::K_MAX_IMAGE_SIDE);

            SA_TRY_ASSIGN(const OutputPlan plan,
                          plan_output(session.info.jpeg_color_space, options.layout));
            session.info.out_color_space = plan.output;
            SA_TRY(decompressor.run(start_output));
            SA_FAIL_IF(session.info.output_components != plan.components ||
                           session.info.output_height != height ||
                           session.info.output_width != width,
                       StatusCode::Unsupported,
                       "decode_jpeg: libjpeg writes {}x{} with {} samples per pixel where "
                       "{}x{} with {} was asked for",
                       session.info.output_height, session.info.output_width,
                       session.info.output_components, height, width, plan.components);

            const Imaging::ImageSize size{static_cast<std::int32_t>(height),
                                          static_cast<std::int32_t>(width)};
            Imaging::Image image(size, plan.channels);
            std::vector<JSAMPLE> cmyk_row;
            if (plan.components != plan.channels)
                cmyk_row.resize(static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(plan.components));

            session.rows = image.data();
            session.row_bytes = image.row_bytes();
            session.cmyk_row = cmyk_row.empty() ? nullptr : cmyk_row.data();
            session.ink_flip = session.info.saw_Adobe_marker != 0 ? 0u : K_MAX_SAMPLE;
            SA_TRY(decompressor.run(read_rows));

            SA_FAIL_IF(session.stopped_early, StatusCode::InvalidFile,
                       "decode_jpeg: the data ended at row {} of {}",
                       session.info.output_scanline, height);
            SA_FAIL_IF(session.errors.num_warnings != 0, StatusCode::InvalidFile,
                       "decode_jpeg: the data is corrupt: {}",
                       static_cast<const char*>(session.message));

            return Result<Imaging::Image>(std::move(image));
        }
    } // namespace

    Result<Imaging::Image> decode_jpeg(SushiRuntime::span<const std::uint8_t> bytes,
                                       const DecodeOptions& options)
    {
        try
        {
            return decode_checked(bytes, options);
        }
        catch (const std::bad_alloc&)
        {
            return Status::from_message(StatusCode::OutOfMemory, []()
            {
                return std::string("decode_jpeg: the image's pixels could not be allocated");
            });
        }
        catch (const std::exception& error)
        {
            return Status::from_message(StatusCode::InvalidFile, [&]()
            {
                return SushiLog::format_text("decode_jpeg: {}", error.what());
            });
        }
    }
} // namespace SushiAI::Vision::Codecs
