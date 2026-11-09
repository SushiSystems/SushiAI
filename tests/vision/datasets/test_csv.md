# tests/vision/datasets/test_csv.cpp


Suites: `Small_DatasetsCsv`, `Small_DatasetsCsvCell`, `Small_DatasetsCsvQuote` and
`Medium_DatasetsCsv`.

Tests `read_csv` in `modules/vision/datasets/include/SushiAI/vision/datasets/csv.hpp`. The `Small_`
cases read `train.csv` under the folder `data` (or the file they name) from a `MemorySource`, with the default columns
unless a case names others; the `Medium_` case writes a file under the system's temporary directory.

## TheLastColumnIsTheTargetByDefault

A file with the header `a,b,y` and two records gives one file, two samples and a class count
of 0. The table has two features, one target, the columns `a`, `b`, `y`, the values 1, 2, 3, -0.5,
4, 10 and two rows; the features of row 1 end with 4 and its target is 10. The cells are short
decimals a float holds exactly, so they compare exactly.

## NamedColumnsAreTakenInTheOrderNamed

With the file `set.csv` named, the features `z`, `x` and the target `y` give the columns `z`, `x`,
`y` and the values 3, 1, 2, 6, 4, 5, so the column `id` is not read.

## AColumnNamedAsFeatureAndTargetIsRefused

The features `z`, `x` with the targets `y`, `z` fail with `InvalidDescription`, "set.csv",
"read_csv" and "the column 'z' is named as a feature and as a target".

## QuotedFieldsKeepTheirQuotesAndLineBreaks

A header with a doubled quote and a comma inside quotes, a cell of padded digits inside quotes and
a line break inside quotes read as the columns `the "x", one` and `y` with the values 1.5, 2, 3, 4.

## CarriageReturnsEndRecordsAndBlankLinesAreSkipped

A file with CRLF line ends and a blank line between records gives the columns `a`, `y` and the
values 1, 2, 3, 4.

## ALeadingByteOrderMarkIsSkipped

A file that opens with the bytes EF BB BF reads the column `a`, not a column that begins with them.

## ANonNumericCellNamesItsLineAndColumn

One row each: the cell `five` on line 3, column 2 (`b`); an empty cell on line 2, column 3 (`y`);
and `inf` on line 4, column 3 (`y`) after a first record that spans lines 2 and 3. Each fails with
`InvalidFile`, "train.csv", "read_csv" and the line, column and cell.

## ARecordOfAnotherFieldCountIsRefused

A record of two fields under a header of three, on line 4, fails with `InvalidFile`, "train.csv",
"line 4", "read_csv" and "the record has 2 fields; the header has 3".

## MalformedQuotesAreRefused

One row each: a quote left open on line 2, column 2 ("the quoted field is not closed"); text after a
closing quote on line 2, column 1 ("text follows the closing quote"); a quote inside a field on line
2, column 2 ("a quote stands inside a field"). Each fails with `InvalidFile`.

## ATargetColumnTheHeaderLacksIsRefused

The target `z` fails with `InvalidDescription`, "train.csv", "no target column 'z'" and the header
names `'a', 'b', 'y'`.

## AFeatureColumnTheHeaderLacksIsRefused

The feature `c` fails with `InvalidDescription` and "no feature column 'c'".

## AHeaderMustNameItsColumnsOnce

A header `a, b ,a` on line 2 fails with `InvalidFile`, "line 2", "column 3" and "the name 'a' is
also the name of column 1".

## AHeaderMustLeaveAFeature

A header of the single column `y` fails with `InvalidDescription` and "no column is left as a
feature".

## AnAbsentFileIsNotFound

The split "val" over an empty source fails with `FileNotFound`, "val.csv" and "read_csv".

## AFileWithoutAHeaderIsRefused

A file of two blank lines fails with `InvalidFile`, "train.csv" and "holds no header row".

## AFileWithoutARecordIsRefused

A header with no record below it fails with `InvalidFile`, "train.csv" and "holds no record below
its header".

## AFileOnDiskReadsIntoATable

The file-system form of `read_csv` reads a written `train.csv` into two samples with the values 1 to
6.
