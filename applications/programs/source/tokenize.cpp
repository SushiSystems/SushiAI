/**************************************************************************/
/* tokenize.cpp                                                           */
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
 * @file tokenize.cpp
 * @brief Runs `sushiai tokenize`: learns a vocabulary from a corpus, or encodes a corpus.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "tokenize.cpp".
 */

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/text/bpe_trainer.hpp>
#include <SushiAI/language/text/stream_encoder.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>

/** @brief Shortens the namespace of the tokenizer. */
namespace Text = SushiAI::Language::Text;

namespace
{
    using SushiAI::Result;
    using SushiAI::Status;
    using SushiAI::StatusCode;

    /** @brief Holds the text `--help` prints. */
    constexpr std::string_view K_USAGE =
        "sushiai tokenize - learn a byte-pair vocabulary, or encode a corpus with one.\n"
        "\n"
        "Usage: sushiai tokenize train <corpus> --vocabulary-size <N> --output <vocab.savb>\n"
        "       sushiai tokenize encode <corpus> --vocabulary <vocab.savb> --output "
        "<corpus.satk>\n"
        "\n"
        "  corpus             a text file; its bytes are read as they are\n"
        "  --vocabulary-size  the tokens of the learned vocabulary, bytes and\n"
        "                     end-of-text included\n"
        "  --vocabulary       the vocabulary file `tokenize train` wrote\n"
        "  --output           the file written; a file that is there already stops the run\n"
        "  --force            replace a file that is already there\n";

    /** @brief Holds the usage line a refused command line quotes. */
    constexpr std::string_view K_USAGE_LINE =
        "usage: sushiai tokenize train <corpus> --vocabulary-size <N> --output <file> | "
        "sushiai tokenize encode <corpus> --vocabulary <file> --output <file>";

    /** @brief Holds the flag that lets a run replace its output file. */
    constexpr std::string_view K_FORCE_FLAG = "--force";

    /** @brief Holds what the command line of one run states. */
    struct TokenizeRequest
    {
        /** @brief The verb, the first argument. */
        std::string verb;

        /** @brief The corpus file, the one argument that is no flag. */
        std::filesystem::path corpus;

        /** @brief The value of --vocabulary-size; empty when the flag is absent. */
        std::string vocabulary_size;

        /** @brief The value of --vocabulary; empty when the flag is absent. */
        std::filesystem::path vocabulary;

        /** @brief The value of --output. */
        std::filesystem::path output;

        /** @brief Whether an output file that exists is replaced. */
        bool force = false;
    };

    /** @brief Holds one flag that takes a value, and where the value goes. */
    struct ValueFlag
    {
        std::string_view flag;                             /**< The flag as it is typed. */
        void (*store)(TokenizeRequest&, std::string_view); /**< Keeps the value. */
    };

    /** @brief Keeps the value of --vocabulary-size as it was typed. */
    void store_vocabulary_size(TokenizeRequest& request, std::string_view value)
    {
        request.vocabulary_size = value;
    }

    /** @brief Keeps the value of --vocabulary. */
    void store_vocabulary(TokenizeRequest& request, std::string_view value)
    {
        request.vocabulary = value;
    }

    /** @brief Keeps the value of --output. */
    void store_output(TokenizeRequest& request, std::string_view value)
    {
        request.output = value;
    }

    /** @brief Lists the flags that take a value. */
    constexpr ValueFlag K_VALUE_FLAGS[] = {
        {"--vocabulary-size", &store_vocabulary_size},
        {"--vocabulary", &store_vocabulary},
        {"--output", &store_output},
        {"-o", &store_output},
    };

    /** @brief Returns the row of @p argument in K_VALUE_FLAGS, or null for another text. */
    [[nodiscard]] const ValueFlag* find_value_flag(std::string_view argument) noexcept
    {
        for (const ValueFlag& row : K_VALUE_FLAGS)
        {
            if (row.flag == argument)
                return &row;
        }

        return nullptr;
    }

    /**
     * @brief Reads the command line into a request.
     * @return The request; InvalidDescription quoting the usage line for no verb, a flag
     *         without a value, no corpus, two corpora or no output file.
     */
    [[nodiscard]] Result<TokenizeRequest> read_request(int argc, char** argv)
    {
        SA_FAIL_IF(argc < 2, StatusCode::InvalidDescription, "{}", K_USAGE_LINE);

        TokenizeRequest request;
        request.verb = argv[1];
        for (int at = 2; at < argc; ++at)
        {
            const std::string_view argument = argv[at];
            if (argument == K_FORCE_FLAG)
            {
                request.force = true;
                continue;
            }

            const ValueFlag* const flag = find_value_flag(argument);
            if (flag != nullptr)
            {
                SA_FAIL_IF(at + 1 >= argc, StatusCode::InvalidDescription,
                           "'{}' is followed by no value; {}", argument, K_USAGE_LINE);
                flag->store(request, argv[++at]);
                continue;
            }

            SA_FAIL_IF(!request.corpus.empty(), StatusCode::InvalidDescription,
                       "two corpus files were given, '{}' and '{}'; {}",
                       request.corpus.string(), argument, K_USAGE_LINE);
            request.corpus = argument;
        }

        SA_FAIL_IF(request.corpus.empty(), StatusCode::InvalidDescription,
                   "no corpus file was given; {}", K_USAGE_LINE);
        SA_FAIL_IF(request.output.empty(), StatusCode::InvalidDescription,
                   "no output file was given; {}", K_USAGE_LINE);
        return request;
    }

    /** @brief Returns InvalidDescription when the output exists and --force is absent. */
    [[nodiscard]] Status check_output(const TokenizeRequest& request)
    {
        std::error_code ignored;
        SA_FAIL_IF(!request.force && std::filesystem::exists(request.output, ignored),
                   StatusCode::InvalidDescription,
                   "'{}' exists already; pass {} to replace it", request.output.string(),
                   K_FORCE_FLAG);

        return Status();
    }

    /** @brief Opens the corpus for reading, or returns InvalidFile naming it. */
    [[nodiscard]] Status open_corpus(const std::filesystem::path& corpus, std::ifstream& in)
    {
        in.open(corpus, std::ios::binary);
        SA_FAIL_IF(!in.is_open(), StatusCode::InvalidFile,
                   "'{}' cannot be opened for reading", corpus.string());

        return Status();
    }

    /**
     * @brief Learns a vocabulary of --vocabulary-size tokens from the corpus and writes it.
     * @return Success, InvalidDescription for a size that is no whole number, or the failure
     *         of the read, the trainer or the writer.
     */
    [[nodiscard]] Status run_train(const TokenizeRequest& request)
    {
        Text::BPETrainerOptions options;
        const std::string& stated = request.vocabulary_size;
        const std::from_chars_result parsed =
            std::from_chars(stated.data(), stated.data() + stated.size(),
                            options.vocabulary_size);
        SA_FAIL_IF(stated.empty() || parsed.ec != std::errc() ||
                       parsed.ptr != stated.data() + stated.size(),
                   StatusCode::InvalidDescription,
                   "--vocabulary-size is '{}'; it is a whole number; {}", stated, K_USAGE_LINE);

        // The trainer reads the corpus as one text, so the whole file is held.
        std::ifstream in;
        SA_TRY(open_corpus(request.corpus, in));
        const std::string corpus((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());

        SA_TRY_ASSIGN(const Text::Vocabulary vocabulary, Text::train_bpe(corpus, options));
        SA_TRY(Text::write_vocabulary_file(request.output, vocabulary));

        std::cout << "sushiai tokenize train\n"
                  << "----------------------------------------------------------------\n"
                  << SushiLog::format_text("Corpus     : {} ({} bytes)\n",
                                             request.corpus.string(), corpus.size())
                  << SushiLog::format_text("Vocabulary : {} tokens, {} of them merges\n",
                                             vocabulary.size(), vocabulary.merges().size())
                  << SushiLog::format_text("\nwrote {}\n", request.output.string());
        return Status();
    }

    /**
     * @brief Encodes the corpus with the vocabulary of --vocabulary and writes a token file.
     * @return Success, InvalidDescription for no vocabulary file, or the failure of a read,
     *         the encoder or the writer.
     */
    [[nodiscard]] Status run_encode(const TokenizeRequest& request)
    {
        SA_FAIL_IF(request.vocabulary.empty(), StatusCode::InvalidDescription,
                   "no vocabulary file was given; {}", K_USAGE_LINE);
        SA_TRY_ASSIGN(const Text::Vocabulary vocabulary,
                      Text::read_vocabulary_file(request.vocabulary));

        std::ifstream in;
        SA_TRY(open_corpus(request.corpus, in));
        std::vector<Text::TokenID> ids;
        SA_TRY(Text::encode_stream(vocabulary, in, Text::K_STREAM_PIECE_BYTES, ids));
        SA_TRY(SushiAI::Language::Datasets::write_token_file(
            request.output, SushiRuntime::span<const Text::TokenID>(ids.data(), ids.size()),
            vocabulary.size()));

        std::cout << "sushiai tokenize encode\n"
                  << "----------------------------------------------------------------\n"
                  << SushiLog::format_text("Corpus     : {}\n", request.corpus.string())
                  << SushiLog::format_text("Vocabulary : {} ({} tokens)\n",
                                             request.vocabulary.string(), vocabulary.size())
                  << SushiLog::format_text("Tokens     : {}\n", ids.size())
                  << SushiLog::format_text("\nwrote {}\n", request.output.string());
        return Status();
    }

    /** @brief Holds one verb of the command and what runs it. */
    struct VerbEntry
    {
        std::string_view verb;                 /**< The first argument. */
        Status (*run)(const TokenizeRequest&); /**< What the verb does. */
    };

    /** @brief Lists the verbs of `sushiai tokenize`. */
    constexpr VerbEntry K_VERBS[] = {
        {"train", &run_train},
        {"encode", &run_encode},
    };

    /** @brief Runs the verb the arguments name. */
    [[nodiscard]] Status run(int argc, char** argv)
    {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
        {
            std::cout << K_USAGE;
            return Status();
        }

        SA_TRY_ASSIGN(const TokenizeRequest request, read_request(argc, argv));
        const VerbEntry* found = nullptr;
        for (const VerbEntry& entry : K_VERBS)
        {
            if (entry.verb == request.verb)
                found = &entry;
        }

        SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                   "'{}' is no verb of `sushiai tokenize`; it has 'train' and 'encode'; {}",
                   request.verb, K_USAGE_LINE);
        SA_TRY(check_output(request));
        return found->run(request);
    }
} // namespace

/** @brief Runs the tokenizer driver and returns its exit code. */
int main(int argc, char** argv)
{
    const Status status = run(argc, argv);
    if (status.is_ok())
        return EXIT_SUCCESS;

    std::cerr << "sushiai tokenize failed: " << status.message() << "\n"
              << "  kind: " << SushiAI::status_code_name(status.code()) << "\n";
    return EXIT_FAILURE;
}
