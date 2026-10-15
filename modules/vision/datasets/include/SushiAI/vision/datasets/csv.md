# modules/vision/datasets/include/SushiAI/vision/datasets/csv.hpp


`read_csv` reads `root/<split>.csv`, or `file` when it is set. The first record that is no blank
line is the header. `targets` names the target columns and defaults to the last column;
`features` names the feature columns and defaults to every column that is no target, in file
order. A column that is neither is not read, so it may hold text. A sample is one record below
the header, and its row of `SampleList::table` holds its features and then its targets.

| Syntax | Rule |
| --- | --- |
| Records | End at a line feed, a carriage return and a line feed, or the end of the file. A line with nothing on it is skipped |
| Fields | Separated by commas. A field in double quotes may hold commas and line breaks, and `""` is one quote |
| Header names | Lose the spaces and tabs around them and must differ from each other |
| Cells | Lose the spaces and tabs around them and must spell a finite number a `float` holds, as `std::from_chars` reads a double, with one leading `+` allowed |
| Start of the file | A UTF-8 byte order mark is skipped |

| Code | When |
| --- | --- |
| `FileNotFound` | The file is absent |
| `InvalidDescription` | `features` or `targets` names a column the header lacks, with the names it has; a column is named as both; no column is left as a feature |
| `InvalidFile` | A cell of a column that is read is not a finite number, with the line, the column and its name; a record has another field count than the header, with the line; a quoted field is not closed, text follows a closing quote, or a quote stands inside a field that does not open with one, with the line and the column; a header name occurs twice; the file has no header or no record |
| `OutOfMemory` | An allocation fails |

A column counts from 1. For a cell and for a field count the line is the line the record starts
on, also when a quoted field of the record holds a line break. A failure of a quote names the
line the quote is on.
