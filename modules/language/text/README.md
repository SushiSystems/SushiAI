# text

## Owns

The token id type, the vocabulary of a byte-pair tokenizer, the trainer that learns one from a
corpus, the encoder and the decoder between text bytes and ids, and the vocabulary file. The
module owns that file format: its schema, its reader and its writer. Everything is in namespace
`SushiAI::Language::Text`.

The tokenizer works on bytes and knows no encoding. A text is any byte string, and
`decode_tokens` over what `encode_text` wrote gives it back byte for byte, invalid UTF-8
included. No unit of the module reads a tensor or an engine.

## Depends on

- `core` (`PUBLIC_DEPENDS`), for `Status` and `Result`.

## Public entry points

| Header | Declares |
| --- | --- |
| `<SushiAI/language/text/token.hpp>` | `TokenID`, `K_BYTE_TOKEN_COUNT`, `K_END_OF_TEXT`, `K_END_OF_TEXT_NAME`, the limits `K_MAX_VOCABULARY_SIZE`, `K_MAX_TOKEN_BYTES`, `K_MAX_SPECIAL_TOKENS`, `K_MAX_SPECIAL_NAME_BYTES` |
| `<SushiAI/language/text/vocabulary.hpp>` | `Merge`, `TokenKind`, `Vocabulary` |
| `<SushiAI/language/text/bpe_trainer.hpp>` | `BPETrainerOptions`, `train_bpe`, `K_MIN_MERGE_COUNT`, `K_MAX_CORPUS_BYTES` |
| `<SushiAI/language/text/codec.hpp>` | `encode_text`, `decode_tokens` |
| `<SushiAI/language/text/stream_encoder.hpp>` | `encode_stream`, `K_STREAM_PIECE_BYTES` |
| `<SushiAI/language/text/vocabulary_file.hpp>` | `write_vocabulary_file`, `read_vocabulary_file`, `parse_vocabulary_file`, the magic, the version and the header size |

Every function that can fail returns a `Status` or a `Result`. A precondition a caller broke,
such as the bytes of an id the vocabulary does not hold, stops the process.

## Token ids

`TokenID` is `std::int32_t`, the element type `NN::Embedding` gathers by.

| Ids | Kind | Stands for |
| --- | --- | --- |
| `[0, 256)` | `Byte` | Token `b` is the byte `b` |
| `[256, 256 + S)` | `Special` | No byte of a text. `256` is end-of-text, named `<end_of_text>`; the other `S - 1` names are the caller's |
| `[256 + S, 256 + S + M)` | `Merged` | Merge `i` has the id `256 + S + i` and stands for the bytes of its left token followed by those of its right |

The special tokens come before the merged ones so that end-of-text is `256` in every
vocabulary, whatever its size. A smaller merged id was learned earlier. The size handed to
`NN::Embedding` as `num_embeddings` is `Vocabulary::size()`.

`Vocabulary::make` refuses a merge that names a special token, itself or a later token, a pair
two merges name, a token above `K_MAX_TOKEN_BYTES` (256) bytes, and a vocabulary above
`K_MAX_VOCABULARY_SIZE` (1 048 576) tokens. A vocabulary has no method that changes it.

Lookups: `bytes_of(id)` from an id to its bytes, `find_merge(left, right)` from a pair to its
merged token, `find_special(name)` from a name to its special token. There is no lookup from
bytes to an id, because two merged tokens may stand for the same bytes: `(ab, c)` and `(a, bc)`
are different pairs.

## Chunks

A text is cut into chunks, and no merge is learned or applied across the edge of one. A chunk
is a run of blank bytes (space, tab, line feed, carriage return) and the run of other bytes
after it, cut at 256 bytes. `"ab ab  cd"` is the chunks `"ab"`, `" ab"` and `"  cd"`. The
trainer and the encoder share the rule (`source/chunks.hpp`), so a chunk the trainer saw
encodes to the tokens the training left it in.

The cut at 256 bytes bounds the work on a text with no blank byte, and keeps every token the
trainer learns within `K_MAX_TOKEN_BYTES`.

## The trainer

`train_bpe(corpus, options)` learns merges until the vocabulary has `options.vocabulary_size`
tokens or no pair occurs `K_MIN_MERGE_COUNT` (2) times, whichever comes first. The result may
be smaller than asked.

Each step merges the pair of adjacent tokens that occurs most often in the corpus. Of two pairs
with the same count, the one with the smaller left id is merged first, then the one with the
smaller right id. A merge replaces the occurrences of its pair in a chunk from the left, so
`a a a` under `(a, a)` becomes `aa a`. The result depends on the corpus bytes and the options
alone: no hash order, thread or clock enters it.

The algorithm:

1. Cut the corpus into chunks and keep each distinct chunk once, with the number of times it
   occurs.
2. Count every pair of adjacent bytes, each chunk weighted by its count, and list for every
   pair the chunks it occurs in.
3. Enter every pair that occurs at least twice into a priority queue ordered by count, then
   left id, then right id.
4. Take the top of the queue. An entry whose count is no longer the pair's count is stale; the
   pair is entered again with its present count and the next entry is taken.
5. Rewrite the chunks listed for the pair. For each, subtract its pairs from the counts,
   replace the pair, and add its pairs back. Pairs that hold the new token are listed and
   entered into the queue.

A pair's count only rises when the pair holds the token just made, and every such pair is
entered again, so the queue always holds, for every pair, an entry at or above its count. The
top entry that matches its pair's count is therefore the pair to merge.

Cost, with `N` corpus bytes, `L` the bytes of the distinct chunks, `M` merges and `W(m)` the
chunks listed for merge `m`:

- Steps 1 to 3 take `O(N)` expected time, for hashing each chunk once, and `O(L)` memory:
  four bytes per byte of a distinct chunk, plus the pair table.
- Merge `m` takes `O(sum of the lengths of W(m))` hash operations and at most that many queue
  entries, each `O(log Q)` for a queue of `Q` entries. A chunk has at most 256 tokens.
- The whole run is `O(N + sum over m of |W(m)| * 256 * log Q)`. No step counts the pairs of the
  corpus again. The worst case, every distinct chunk listed for every merge, is
  `O(N + M * L * log Q)`.

The time of a run on a corpus of tens of megabytes has not been measured.

The corpus is one `std::string_view` in memory, at most `K_MAX_CORPUS_BYTES` (4 GiB - 1); a
larger one is `Unsupported`. A vocabulary size below `257 + ` the caller's special tokens or
above `K_MAX_VOCABULARY_SIZE`, and a special name `Vocabulary::make` refuses, are
`InvalidDescription`, returned before the corpus is read. An allocation the run cannot get is
`OutOfMemory`.

## Encoding

`encode_text` cuts the text into chunks and, in each, applies the merges in the order they
were learned: it finds the adjacent pair whose merged token has the smallest id, replaces
every occurrence from the left, and repeats until no adjacent pair has a merge. It appends to
the ids it is given and never writes a special token; the caller appends `K_END_OF_TEXT`
between documents.

`encode_stream` reads a text from a stream `piece_bytes` at a time and appends the ids
`encode_text` gives for the whole of it. It cuts what it holds at the last place where a blank
byte follows another byte. A chunk ends at every such place, whatever came before, so the
chunks on both sides are those of the whole text. The bytes it holds stay near `piece_bytes`
on text made of words and grow by a piece each time a piece has no such place. The ids are
appended to one vector, four bytes each. `sushiai tokenize encode` calls it with
`K_STREAM_PIECE_BYTES`, one mebibyte.

A chunk of `n` tokens costs at most `n` rounds of `n` pair lookups, and `n` is at most 256, so
a text of `N` bytes costs `O(N * 256)` lookups at worst and a few lookups per byte on text made
of words.

`decode_tokens` appends the bytes of each id. A special token appends nothing. An id outside
the vocabulary is a failure that names its index.

## The vocabulary file

A binary file, by convention `*.savb`. Every integer is unsigned, 32 bits and little-endian,
written and read byte by byte whatever the host's byte order. The file stores the merges as
pairs of ids; the bytes of each token are rebuilt when the file is read.

| Offset | Bytes | Field | Rule |
| --- | --- | --- | --- |
| 0 | 4 | `magic` | The bytes `S`, `A`, `V`, `B` |
| 4 | 4 | `format_version` | `1` |
| 8 | 4 | `byte_token_count` | `256` |
| 12 | 4 | `special_count` | `S`, in `[1, 256]` |
| 16 | 4 | `merge_count` | `M`, at most `1 048 576 - 256 - S` |
| 20 | 4 | `special_bytes` | The byte count of the special section, at most `256 * 259` |
| 24 | `special_bytes` | Special section | `S` records, one per special token in id order: `name_length` (4 bytes, in `[1, 255]`) and the name's bytes. The first name is `<end_of_text>` |
| 24 + `special_bytes` | `8 * M` | Merge section | `M` records in id order: `left` (4 bytes) and `right` (4 bytes) |

The file ends at byte `24 + special_bytes + 8 * M`. The file of end-of-text and the one merge
`(97, 98)` is 49 bytes: the header, a length of 13 and the 13 bytes of the name, then
`61 00 00 00 62 00 00 00`.

The reader refuses, with `InvalidFile` and the byte offset in the message: a file that ends
before the end its header declares, bytes after that end, another magic, a version it does not
know (the message names the version of the file and the versions the build reads), a count out
of its bounds, and a special record that leaves its section. A merge `Vocabulary::make` refuses
is `InvalidDescription` and names the merge. The reader reads a file of at most 8 454 936 bytes
whole, and refuses a larger one before reading it.

| Version | Release | Change |
| --- | --- | --- |
| 1 | 0.1.0, not yet released | The first version |

## Tests

`tests/test_text_vocabulary.cpp`, `test_text_bpe_trainer.cpp`, `test_text_codec.cpp` and
`test_text_vocabulary_file.cpp`, in the suites `Unit_TextVocabulary`, `Unit_TextBpeTrainer`,
`Unit_TextCodec` and `Unit_TextVocabularyFile`. None needs a device. The files they write are
under the system's temporary directory and are removed when the test ends.
