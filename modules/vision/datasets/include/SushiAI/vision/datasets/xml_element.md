# modules/vision/datasets/include/SushiAI/vision/datasets/xml_element.hpp


`parse_xml` reads a document into its root `XMLElement`: a name, the text, the child elements
and the line of the opening tag.

| Read | How |
| --- | --- |
| A UTF-8 byte order mark and one `<?xml ...?>` declaration at the start | Skipped |
| Comments, before, inside and after the root | Skipped |
| `<name>...</name>` and `<name/>` | An element; a name opens with an ASCII letter, `_` or `:` and goes on with those, digits, `-` and `.` |
| Attributes, `name="value"` or `name='value'` | Read and dropped |
| Character data | Appended to the text of the element it is in; the text loses the white space around it |
| `&amp;` `&lt;` `&gt;` `&quot;` `&apos;` | Replaced by their characters |

Every other thing is `InvalidFile`, and the message opens with `line <n>`: another entity,
a numeric character reference, a CDATA section, a `<!...>` declaration, a `<?...?>` instruction
other than the leading declaration, a closing tag of another name, an element or a comment left
open, an attribute without `=` or without quotes, text outside the root, a second root, and more than
`K_XML_MAX_DEPTH` (64) elements open at once. Bytes above ASCII pass through text unchanged and
are not allowed in a name. `find_child` returns the first child of a name.
