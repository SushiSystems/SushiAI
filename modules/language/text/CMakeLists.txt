# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# text: the token type, the byte-pair tokenizer and the vocabulary file.

sushiai_add_module(NAME text TIER language
    SOURCES
        source/bpe_trainer.cpp
        source/codec.cpp
        source/disk.cpp
        source/stream_encoder.cpp
        source/vocabulary.cpp
        source/vocabulary_file.cpp
    PUBLIC_DEPENDS core)
