# test_tokenize.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.
"""Tests end to end that `sushiai tokenize` writes a vocabulary file and a token file.

The reasoning is in cli/tests/README.md under "test_tokenize.py".
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from checkout_cli import K_DEFAULT_TREE, run_cli

from sushiai.services import discovery

pytestmark = pytest.mark.medium

# cli/tests/test_tokenize.py -> cli/tests -> cli -> <repo root>
K_REPO_ROOT = Path(__file__).resolve().parents[2]
K_BUILD_DIR = K_DEFAULT_TREE
K_TOKENIZE_TARGET = "sushiai_tokenize"

# 256 bytes, end-of-text and three merges.
K_VOCABULARY_SIZE = 260
K_CORPUS = "the cat sat on the mat\n" * 50
K_TOKEN_FILE_HEADER_BYTES = 24
K_TOKEN_ID_BYTES = 4
K_TOKEN_COUNT = re.compile(r"^Tokens\s+: (\d+)$", re.MULTILINE)


def _is_tokenize_built() -> bool:
    """Returns whether the build tree holds the tokenizer program."""
    if not K_BUILD_DIR.is_dir():
        return False
    return discovery.match_by_name(K_BUILD_DIR, K_TOKENIZE_TARGET) is not None


_needs_build = pytest.mark.skipif(
    not _is_tokenize_built(), reason="sushiai_tokenize is not built; run `sa build` first")


def _run(*args: str) -> subprocess.CompletedProcess:
    """Runs this checkout's CLI from the repository root, decoding its output as UTF-8."""
    return run_cli(*args, cwd=K_REPO_ROOT, timeout=900)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Returns the path of the corpus, written into the test's own folder."""
    path = tmp_path / "corpus.txt"
    path.write_text(K_CORPUS, encoding="utf-8", newline="\n")
    return path


def _train(corpus: Path, vocabulary: Path, *extra: str) -> subprocess.CompletedProcess:
    """Runs `tokenize train` over the corpus into the vocabulary file."""
    return _run(
        "tokenize", "train", str(corpus), "--vocabulary-size", str(K_VOCABULARY_SIZE),
        "--output", str(vocabulary), *extra)


@pytest.fixture
def trained(corpus: Path, tmp_path: Path) -> tuple[Path, subprocess.CompletedProcess]:
    """Trains the vocabulary and returns its file with the run's result."""
    vocabulary = tmp_path / "vocab.savb"
    completed = _train(corpus, vocabulary)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return vocabulary, completed


@pytest.fixture
def encoded(corpus: Path, tmp_path: Path, trained) -> tuple[Path, subprocess.CompletedProcess]:
    """Encodes the corpus with the trained vocabulary and returns the token file and the result."""
    tokens = tmp_path / "corpus.satk"
    completed = _run(
        "tokenize", "encode", str(corpus), "--vocabulary", str(trained[0]),
        "--output", str(tokens))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return tokens, completed


@_needs_build
def test_tokenize_train_writes_a_file_with_the_vocabulary_magic(trained):
    """The vocabulary file starts with the vocabulary magic."""
    assert trained[0].read_bytes().startswith(b"SAVB")


@_needs_build
def test_tokenize_train_reports_the_vocabulary_size_it_was_asked_for(trained):
    """The report states the size asked for."""
    assert f"{K_VOCABULARY_SIZE} tokens" in trained[1].stdout, trained[1].stdout


@_needs_build
def test_tokenize_encode_writes_a_file_with_the_token_magic(encoded):
    """The token file starts with the token magic."""
    assert encoded[0].read_bytes().startswith(b"SATK")


@_needs_build
def test_tokenize_encode_writes_one_id_per_token_it_reports(encoded):
    """The token file is its header and four bytes for each token the report counts."""
    counted = K_TOKEN_COUNT.search(encoded[1].stdout)
    assert counted is not None, encoded[1].stdout
    assert len(encoded[0].read_bytes()) == (
        K_TOKEN_FILE_HEADER_BYTES + K_TOKEN_ID_BYTES * int(counted.group(1)))


@_needs_build
def test_tokenize_encode_merges_the_corpus_into_fewer_tokens_than_it_has_bytes(encoded):
    """The count the report states is above zero and below the corpus length."""
    counted = K_TOKEN_COUNT.search(encoded[1].stdout)
    assert counted is not None, encoded[1].stdout
    assert 0 < int(counted.group(1)) < len(K_CORPUS)


@_needs_build
def test_tokenize_refuses_to_replace_a_file_without_force(corpus: Path, trained):
    """A second run into the same file stops and names the file that exists."""
    refused = _train(corpus, trained[0])

    assert refused.returncode != 0
    assert "exists already" in refused.stdout + refused.stderr


@_needs_build
def test_tokenize_replaces_a_file_with_force(corpus: Path, trained):
    """--force lets a second run into the same file through."""
    forced = _train(corpus, trained[0], "--force")

    assert forced.returncode == 0, forced.stdout + forced.stderr


@_needs_build
def test_train_without_a_vocabulary_size_names_the_flag(corpus: Path, tmp_path: Path):
    """The C++ diagnostic reaches the caller and names `--vocabulary-size`."""
    completed = _run(
        "tokenize", "train", str(corpus), "--output", str(tmp_path / "vocab.savb"))

    assert completed.returncode != 0
    assert "--vocabulary-size" in completed.stdout + completed.stderr


@_needs_build
def test_an_unknown_verb_lists_the_verbs_the_program_has(corpus: Path, tmp_path: Path):
    """The diagnostic names the verb given and the two the program has."""
    completed = _run(
        "tokenize", "decode", str(corpus), "--output", str(tmp_path / "none.txt"))

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "'decode'" in combined and "'train'" in combined and "'encode'" in combined, combined
