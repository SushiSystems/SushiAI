# test_tokenize.py
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests end to end that `sushiai tokenize` writes a vocabulary file and a token file.

The reasoning is in cli/tests/README.md under "test_tokenize.py".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from sushiai.services import discovery

# cli/tests/test_tokenize.py -> cli/tests -> cli -> <repo root>
K_REPO_ROOT = Path(__file__).resolve().parents[2]
K_BUILD_DIR = K_REPO_ROOT / "build"
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


_needs_cli = pytest.mark.skipif(
    shutil.which("sushiai") is None, reason="the `sushiai` CLI is not on PATH")
_needs_build = pytest.mark.skipif(
    not _is_tokenize_built(), reason="sushiai_tokenize is not built; run `sa build` first")


def _run(*args: str) -> subprocess.CompletedProcess:
    """Runs the sushiai CLI from the repository root, decoding its output as UTF-8."""
    return subprocess.run(
        ["sushiai", *args], cwd=K_REPO_ROOT, capture_output=True,
        encoding="utf-8", errors="replace", timeout=900)


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


@_needs_cli
@_needs_build
def test_tokenize_train_writes_a_vocabulary_file(corpus: Path, tmp_path: Path):
    """The file carries the vocabulary magic, and the report states the size asked for."""
    vocabulary = tmp_path / "vocab.savb"
    completed = _train(corpus, vocabulary)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert vocabulary.read_bytes().startswith(b"SAVB")
    assert f"{K_VOCABULARY_SIZE} tokens" in completed.stdout, completed.stdout


@_needs_cli
@_needs_build
def test_tokenize_encode_writes_one_id_per_token_it_reports(corpus: Path, tmp_path: Path):
    """The token file is its header and four bytes for each token the report counts."""
    vocabulary = tmp_path / "vocab.savb"
    tokens = tmp_path / "corpus.satk"
    assert _train(corpus, vocabulary).returncode == 0

    completed = _run(
        "tokenize", "encode", str(corpus), "--vocabulary", str(vocabulary),
        "--output", str(tokens))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    counted = K_TOKEN_COUNT.search(completed.stdout)
    assert counted is not None, completed.stdout
    written = tokens.read_bytes()
    assert written.startswith(b"SATK")
    assert len(written) == K_TOKEN_FILE_HEADER_BYTES + K_TOKEN_ID_BYTES * int(counted.group(1))
    assert 0 < int(counted.group(1)) < len(K_CORPUS)


@_needs_cli
@_needs_build
def test_tokenize_refuses_to_replace_a_file_without_force(corpus: Path, tmp_path: Path):
    """A second run into the same file stops, and --force lets it through."""
    vocabulary = tmp_path / "vocab.savb"
    assert _train(corpus, vocabulary).returncode == 0

    refused = _train(corpus, vocabulary)
    forced = _train(corpus, vocabulary, "--force")

    assert refused.returncode != 0
    assert "exists already" in refused.stdout + refused.stderr
    assert forced.returncode == 0, forced.stdout + forced.stderr


@_needs_cli
@_needs_build
def test_train_without_a_vocabulary_size_names_the_flag(corpus: Path, tmp_path: Path):
    """The C++ diagnostic reaches the caller and names `--vocabulary-size`."""
    completed = _run(
        "tokenize", "train", str(corpus), "--output", str(tmp_path / "vocab.savb"))

    assert completed.returncode != 0
    assert "--vocabulary-size" in completed.stdout + completed.stderr


@_needs_cli
@_needs_build
def test_an_unknown_verb_lists_the_verbs_the_program_has(corpus: Path, tmp_path: Path):
    """The diagnostic names the verb given and the two the program has."""
    completed = _run(
        "tokenize", "decode", str(corpus), "--output", str(tmp_path / "none.txt"))

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "'decode'" in combined and "'train'" in combined and "'encode'" in combined, combined
