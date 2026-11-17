# tests/core/config/test_input_normalise.cpp


Suites: `Small_InputNormalise`, `Small_InputNormaliseRefusal`.

Tests `Config::parse_input_normalise`, `Config::input_normalise_value` and the `normalise` key of a
model's `input`. Every case parses JSON text in memory and needs no device.

## AnEmptyObjectIsTheDefault

`{}` parses to a default `InputNormalise`: red first, scale 1/255, mean 0 and std 1.

## EachStatedKeyIsReadIntoItsField

`bgr`, scale 0.5, mean (0.25, 0.5, 0.75) and std (2, 4, 8) land in their fields.

## AKeyLeftOutKeepsItsDefault

With only scale 1.0 stated, the order, the mean and the std stay default.

## TheValueOfANormalisationReadsBackEqual

`input_normalise_value` of a normalisation, read by `parse_input_normalise`, is equal to it.

## AModelInputStatesTheObjectUnderNormalise

`parse_model` of an input with `normalise` fills `ModelSpec::input_normalise`.

## AModelInputWithoutTheKeyStatesNone

An input without the key leaves `ModelSpec::input_normalise` empty.

## AModelInputRefusesAnUnknownKeyOfTheObject

`{"gamma": 2.2}` fails with `InvalidDescription` naming "'input' 'normalise'" and "does not accept
'gamma'".

## TheObjectIsRefusedNamingWhy

One row each, `InvalidDescription` with the context and a phrase: an unknown key, a channel order
other than `rgb` and `bgr`, a non-string order, a scale of 0, of -0.5 and a string, a mean of two
and of four entries and a non-array, a non-number mean entry, a std of two entries, and a std
entry of 0 and of -2.

## ANormaliseThatIsNoObjectIsRefused

`[1, 2, 3]` fails with `InvalidDescription` and "must be an object".
