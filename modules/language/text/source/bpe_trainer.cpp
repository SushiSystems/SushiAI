/**************************************************************************/
/* bpe_trainer.cpp                                                        */
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
 * @file bpe_trainer.cpp
 * @brief Implements the trainer that learns the merges of a byte-pair vocabulary.
 * @author Mustafa Garip
 */

#include <SushiAI/language/text/bpe_trainer.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>


#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/text/token.hpp>

#include "chunks.hpp"
#include "disk.hpp"
#include "pair_key.hpp"

namespace SushiAI::Language::Text
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_UNIT = "train_bpe";

        /** @brief Returns the pair a key was made from. */
        [[nodiscard]] constexpr Merge pair_of(std::uint64_t key) noexcept
        {
            return Merge{static_cast<TokenID>(key >> 32),
                         static_cast<TokenID>(key & 0xFFFFFFFFull)};
        }

        /** @brief Holds one distinct chunk of the corpus as the tokens it is written in. */
        struct Word
        {
            std::size_t begin = 0;    /**< Where its tokens begin in the token store. */
            std::uint32_t length = 0; /**< How many tokens it is written in now. */
            std::uint64_t count = 0;  /**< How many times the chunk occurs in the corpus. */
        };

        /** @brief Holds what the learner knows of one pair of adjacent tokens. */
        struct PairState
        {
            /** @brief Counts the occurrences of the pair, each word weighted by its count. */
            std::uint64_t count = 0;

            /** @brief Lists every word the pair has occurred in; a word may be listed twice. */
            std::vector<std::uint32_t> words;
        };

        /** @brief Holds a pair and the count it had when it entered the queue. */
        struct Candidate
        {
            std::uint64_t count = 0; /**< The count of the pair at that time. */
            Merge pair;              /**< The pair. */
        };

        /** @brief Orders candidates so that the queue yields the pair to merge next. */
        struct CandidateOrder
        {
            /** @brief Reports whether @p a is merged after @p b: fewer, or equal and larger. */
            [[nodiscard]] bool operator()(const Candidate& a, const Candidate& b) const noexcept
            {
                if (a.count != b.count)
                    return a.count < b.count;
                if (a.pair.left != b.pair.left)
                    return a.pair.left > b.pair.left;

                return a.pair.right > b.pair.right;
            }
        };

        /**
         * @brief Holds the distinct chunks of a corpus and the count of every adjacent pair.
         * @see modules/language/text/README.md, section "The trainer".
         */
        class MergeLearner final
        {
            public:
                /** @brief Cuts a corpus into chunks and counts the pairs of its bytes. */
                explicit MergeLearner(std::string_view corpus);

                /** @brief Returns the pair to merge next, or nothing when none occurs enough. */
                [[nodiscard]] std::optional<Merge> take_best();

                /**
                 * @brief Rewrites every word that holds a pair and updates the counts.
                 * @param merged The token the pair becomes.
                 * @pre take_best() returned @p merge last.
                 */
                void apply(Merge merge, TokenID merged);

            private:
                /** @brief Adds the count of a word to each of its pairs and lists it there. */
                void count_word(std::uint32_t index);

                /** @brief Replaces a pair in one word, from the left, and moves its counts. */
                void rewrite_word(std::uint32_t index, Merge merge, TokenID merged);

                /** @brief Enters a pair into the queue when it occurs often enough. */
                void offer(Merge pair, std::uint64_t count);

                /**
                 * @brief Returns the state of a pair the table holds.
                 * @pre The pair has been counted; one that has not stops the process.
                 */
                [[nodiscard]] PairState& held(std::uint64_t key) noexcept;

                /// @brief Holds the tokens of every word; a word keeps the room it began with.
                std::vector<TokenID> tokens_;

                /// @brief Holds the distinct chunks in the order the corpus first shows them.
                std::vector<Word> words_;

                /// @brief Maps the key of a pair to its count and its words.
                std::unordered_map<std::uint64_t, PairState> pairs_;

                /// @brief Holds, for every pair, an entry that is not below its count.
                std::priority_queue<Candidate, std::vector<Candidate>, CandidateOrder> queue_;

                /// @brief Holds the key of each pair the merge being applied has made.
                std::vector<std::uint64_t> made_;
        };

        MergeLearner::MergeLearner(std::string_view corpus)
        {
            std::unordered_map<std::string_view, std::uint32_t> index_of;

            for (std::size_t begin = 0; begin < corpus.size();)
            {
                const std::size_t end = Detail::chunk_end(corpus, begin);
                const std::string_view chunk = corpus.substr(begin, end - begin);
                const auto entry =
                    index_of.try_emplace(chunk, static_cast<std::uint32_t>(words_.size()));

                if (entry.second)
                {
                    Word word;
                    word.begin = tokens_.size();
                    word.length = static_cast<std::uint32_t>(chunk.size());
                    words_.push_back(word);

                    for (char byte : chunk)
                        tokens_.push_back(static_cast<TokenID>(static_cast<unsigned char>(byte)));
                }

                ++words_[entry.first->second].count;
                begin = end;
            }

            for (std::uint32_t index = 0; index < words_.size(); ++index)
                count_word(index);

            for (const auto& entry : pairs_)
                offer(pair_of(entry.first), entry.second.count);
        }

        std::optional<Merge> MergeLearner::take_best()
        {
            while (!queue_.empty())
            {
                const Candidate top = queue_.top();
                queue_.pop();

                const auto found = pairs_.find(Detail::pair_key(top.pair.left, top.pair.right));
                const std::uint64_t count = found == pairs_.end() ? 0 : found->second.count;

                if (count == top.count)
                    return top.pair;

                // An entry above its pair's count is stale; the pair is entered again.
                offer(top.pair, count);
            }

            return std::nullopt;
        }

        void MergeLearner::apply(Merge merge, TokenID merged)
        {
            const std::uint64_t key = Detail::pair_key(merge.left, merge.right);
            std::vector<std::uint32_t> listed = std::move(held(key).words);

            std::sort(listed.begin(), listed.end());
            listed.erase(std::unique(listed.begin(), listed.end()), listed.end());
            made_.clear();

            for (std::uint32_t index : listed)
                rewrite_word(index, merge, merged);

            // Every word that held the pair was listed, so none holds it now.
            SA_ASSERT(held(key).count == 0);
            pairs_.erase(key);

            std::sort(made_.begin(), made_.end());
            made_.erase(std::unique(made_.begin(), made_.end()), made_.end());

            for (std::uint64_t made : made_)
                offer(pair_of(made), held(made).count);
        }

        void MergeLearner::count_word(std::uint32_t index)
        {
            const Word& word = words_[index];
            const TokenID* tokens = tokens_.data() + word.begin;

            for (std::uint32_t i = 0; i + 1 < word.length; ++i)
            {
                PairState& state = pairs_[Detail::pair_key(tokens[i], tokens[i + 1])];
                state.count += word.count;

                if (state.words.empty() || state.words.back() != index)
                    state.words.push_back(index);
            }
        }

        void MergeLearner::rewrite_word(std::uint32_t index, Merge merge, TokenID merged)
        {
            Word& word = words_[index];
            TokenID* tokens = tokens_.data() + word.begin;

            for (std::uint32_t i = 0; i + 1 < word.length; ++i)
                held(Detail::pair_key(tokens[i], tokens[i + 1])).count -= word.count;

            std::uint32_t out = 0;
            for (std::uint32_t i = 0; i < word.length;)
            {
                const bool is_pair = i + 1 < word.length && tokens[i] == merge.left &&
                                     tokens[i + 1] == merge.right;

                tokens[out++] = is_pair ? merged : tokens[i];
                i += is_pair ? 2 : 1;
            }
            word.length = out;

            for (std::uint32_t i = 0; i + 1 < word.length; ++i)
            {
                const std::uint64_t key = Detail::pair_key(tokens[i], tokens[i + 1]);
                PairState& state = pairs_[key];
                state.count += word.count;

                // A pair without the new token stood in this word before and lists it already.
                if (tokens[i] != merged && tokens[i + 1] != merged)
                    continue;
                if (state.words.empty() || state.words.back() != index)
                    state.words.push_back(index);

                made_.push_back(key);
            }
        }

        void MergeLearner::offer(Merge pair, std::uint64_t count)
        {
            if (count >= K_MIN_MERGE_COUNT)
                queue_.push(Candidate{count, pair});
        }

        PairState& MergeLearner::held(std::uint64_t key) noexcept
        {
            const auto found = pairs_.find(key);
            SA_FATAL_IF(found == pairs_.end(), "{}: the pair of {} and {} was never counted",
                        K_UNIT, pair_of(key).left, pair_of(key).right);

            return found->second;
        }

        /** @brief Learns at most @p wanted merges; may throw what a container throws. */
        [[nodiscard]] std::vector<Merge> learn_merges(std::string_view corpus,
                                                      TokenID first_merged, std::size_t wanted)
        {
            std::vector<Merge> merges;
            merges.reserve(wanted);
            MergeLearner learner(corpus);

            while (merges.size() < wanted)
            {
                const std::optional<Merge> best = learner.take_best();
                if (!best.has_value())
                    break;

                learner.apply(*best, first_merged + static_cast<TokenID>(merges.size()));
                merges.push_back(*best);
            }

            return merges;
        }
    } // namespace

    Result<Vocabulary> train_bpe(std::string_view corpus, const BPETrainerOptions& options)
    {
        return Detail::at_boundary(K_UNIT, [&]() -> Result<Vocabulary>
        {
            // The names are checked before the corpus is read, and give the first merged id.
            SA_TRY_ASSIGN(const Vocabulary base, Vocabulary::make({}, options.special_names));

            SA_FAIL_IF(options.vocabulary_size < base.size() ||
                           options.vocabulary_size > K_MAX_VOCABULARY_SIZE,
                       StatusCode::InvalidDescription,
                       "{}: a vocabulary of {} tokens was asked for; it has {} to {}", K_UNIT,
                       options.vocabulary_size, base.size(), K_MAX_VOCABULARY_SIZE);
            SA_FAIL_IF(corpus.size() > K_MAX_CORPUS_BYTES, StatusCode::Unsupported,
                       "{}: the corpus holds {} bytes; the trainer reads at most {}", K_UNIT,
                       corpus.size(), K_MAX_CORPUS_BYTES);

            const std::size_t wanted =
                static_cast<std::size_t>(options.vocabulary_size - base.size());

            return Vocabulary::make(learn_merges(corpus, base.first_merged(), wanted),
                                    options.special_names);
        });
    }
} // namespace SushiAI::Language::Text
