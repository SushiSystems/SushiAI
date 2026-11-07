# tests/vision/imaging/compile_partner_source.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`PartnerSource` of `SushiAI/vision/imaging/partner_source.hpp` has a virtual destructor, so a source held by
a base pointer runs the destructor of its type.

The file needs no device and runs nothing.
