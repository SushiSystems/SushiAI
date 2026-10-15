# tests/vision/datasets/test_xml_element.cpp


Suites: `Small_DatasetsXmlElement`.

Tests `parse_xml` and `XMLElement` in
`modules/vision/datasets/include/SushiAI/vision/datasets/xml_element.hpp` over strings.

## ElementsTextAndLinesAreRead

A root `a` on line 1 with the children `b`, `c` and an empty `b` reads with no text of its own
and three children. The first `b` has the text "one" and the line 2, the `d` inside `c` has
the text "two", and the last `b` has the line 4 and no text. `find_child` returns the first
`b` for "b", the `c` for "c" and null for "e".

## TheFiveEntitiesAreReplaced

The text `a &amp; b &lt;c&gt; &quot;d&quot; &apos;e&apos;` reads as `a & b <c> "d" 'e'`.

## TheDeclarationCommentsAndAttributesAreSkipped

A document with a byte order mark, an XML declaration, comments before, inside and after the
root, and attributes on the root and its child reads as the root `a` on line 3 with the text
"t" and one child `b`.

## AClosingTagOfAnotherNameIsRefused

`</a>` on line 3 after `<b>` on line 2 fails with `InvalidFile`, "line 3", "parse_xml",
"'</a>' closes '<b>'" and "line 2".

## AnElementLeftOpenIsRefused

Each of these fails with `InvalidFile`: an element `b` left open ("line 3",
"the element 'b', opened on line 2, is not closed"), a comment left open ("line 2",
"a comment is not closed") and a tag without its end ("line 2", "has no '>'").

## AnUnknownEntityIsRefused

`&nbsp;` on line 3 fails with `InvalidFile`, "line 3" and "'&' opens none of". `&#65;` fails
with `InvalidFile` and "line 1".

## MarkupOutsideTheSubsetIsRefused

Each of these fails with `InvalidFile`: a CDATA section ("line 2", "CDATA"), a DOCTYPE
("line 2", "declaration") and a processing instruction inside the root ("line 3",
"processing instruction").

## AMalformedAttributeIsRefused

`k=v` without quotes fails with `InvalidFile`, "line 2", "attribute 'k'" and "not quoted".
`k` without a value fails with `InvalidFile`, "line 1" and "attribute 'k' has no '='".

## ADocumentNeedsExactlyOneRoot

Each of these fails with `InvalidFile`: two empty lines ("line 3", "holds no element"), bare
text ("line 1", "text stands where an element must open"), a second root ("line 2",
"content follows the root element 'a'") and the name `1a` ("line 1",
"a name must open with").

## NestingBeyondTheDepthIsRefused

`K_XML_MAX_DEPTH` + 1 elements opened one per line fail with `InvalidFile`, "line 65" and
"more than 64 elements are open at once".
