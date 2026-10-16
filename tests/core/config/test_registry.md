# tests/core/config/test_registry.cpp


Suites: `Small_ConfigRegistry`.

Tests `Config::find_entry`, which resolves the type a configuration file names against a
constant table. The table `K_ROWS` holds three rows of type "sgd", "adamw" and "lamb", each
with an integer value. No case needs a device.

## AKnownTypeReturnsTheTablesOwnRow

Looking up "adamw" returns a reference to `K_ROWS[1]` itself, whose value is 2.

## APrefixOfATypeIsNotAMatch

"adam", a prefix of "adamw", throws an `Error` whose message contains
"unknown optimizer 'adam'".

## AnUnknownTypeThrowsListingEveryRow

"rmsprop" throws an `Error` whose message contains "unknown optimizer 'rmsprop'" and
"this build has: 'sgd', 'adamw', 'lamb'", the rows in table order.
