"""Read the text of `message.attributedBody`.

The blob is an NSAttributedString written by NSArchiver, the "typedstream"
format. The layout below was read from blobs made by Apple's encoder
(scripts/make_typedstream_fixtures.swift):

    04 0b "streamtyped" 81 e8 03        version, signature, system version 1000
    84 01 "@"                           type code "object"
    84                                  new object
      84 84 12 "NSAttributedString" 00  new class: name, version
      84 84 08 "NSObject" 00 85         its superclass, then nil
    92                                  type code "@" again, by reference
    84                                  new object
      84 84 08 "NSString" 01 94         new class, superclass by reference
      84 01 "+"                         type code "bytes"
      <integer> <UTF-8 bytes>           the text

Integers: a byte below 0x80 is the value. 0x81 is followed by a 2-byte
integer, 0x82 by a 4-byte integer, both in the byte order of the stream
("streamtyped" is little-endian). Apple uses the 2-byte form up to 32,767.

Errors: data that cannot be read raises TypedStreamError and nothing else.
The message of an error names an offset and what was expected there. It
never holds bytes of the blob: they are the text of a message.
"""

from __future__ import annotations

from archive.errors import TypedStreamError

__all__ = [
    "KNOWN_LAYOUTS",
    "TypedStreamError",
    "attributed_body_text",
    "decode_attributed_body",
    "parse_text",
    "read_integer",
    "read_text",
    "scan_text",
]

TAG_INTEGER_2 = 0x81
TAG_INTEGER_4 = 0x82
TAG_NEW = 0x84
TAG_NIL = 0x85
FIRST_REFERENCE = -110  # 0x92 as a signed byte
STREAM_VERSION = 4
BYTE_ORDERS = {b"streamtyped": "little", b"typedstream": "big"}
BYTES_TYPE_NEW = b"\x84\x01+"
MAX_CLASS_DEPTH = 64  # new classes in one chain. Apple's attributed strings have 3 at most.

# Everything before the text length, in the two layouts that NSArchiver writes for an attributed
# string: NSAttributedString with NSString, and the mutable classes. Both are little-endian.
# A blob that starts with one of them needs no walk through the stream. The tests check these
# bytes against the fixtures made by Apple's encoder.
_HEADER = b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84"
_NSOBJECT = b"\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84"
KNOWN_LAYOUTS: tuple[bytes, ...] = (
    _HEADER + b"\x12NSAttributedString\x00" + _NSOBJECT + b"\x08NSString\x01\x94" + BYTES_TYPE_NEW,
    _HEADER
    + b"\x19NSMutableAttributedString\x00\x84\x84\x12NSAttributedString\x00"
    + _NSOBJECT
    + b"\x0fNSMutableString\x01\x84\x84\x08NSString\x01\x95"
    + BYTES_TYPE_NEW,
)


def read_integer(data: bytes, offset: int, byteorder: str = "little") -> tuple[int, int]:
    """Read one unsigned integer. Return (value, offset after it)."""
    if offset >= len(data):
        raise TypedStreamError("integer starts after the end of the data")
    head = data[offset]
    if head < 0x80:
        return head, offset + 1
    if head == TAG_INTEGER_2:
        width = 2
    elif head == TAG_INTEGER_4:
        width = 4
    else:
        raise TypedStreamError(f"expected an integer at offset {offset}")
    end = offset + 1 + width
    if end > len(data):
        raise TypedStreamError("integer is cut short")
    return int.from_bytes(data[offset + 1 : end], byteorder), end  # type: ignore[arg-type]


class _Reader:
    """Walk a typedstream far enough to reach the first string."""

    __slots__ = ("data", "objects", "order", "pos", "strings")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.order = "little"
        self.strings: list[bytes] = []
        self.objects: list[tuple[str, ...] | None] = []

    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise TypedStreamError("unexpected end of data")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def take(self, count: int) -> bytes:
        end = self.pos + count
        if end > len(self.data):
            raise TypedStreamError("a length points past the end of the data")
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def integer(self) -> int:
        value, self.pos = read_integer(self.data, self.pos, self.order)
        return value

    def reference(self, head: int) -> int:
        """Turn a head byte that is not NEW or NIL into a table index."""
        if head == TAG_INTEGER_2:
            number = int.from_bytes(self.take(2), self.order, signed=True)  # type: ignore[arg-type]
        elif head == TAG_INTEGER_4:
            number = int.from_bytes(self.take(4), self.order, signed=True)  # type: ignore[arg-type]
        elif 0x80 <= head < 0x92:
            raise TypedStreamError(f"expected a reference at offset {self.pos - 1}")
        else:
            number = head - 256 if head >= 0x80 else head
        return number - FIRST_REFERENCE

    def header(self) -> None:
        if self.byte() != STREAM_VERSION:
            raise TypedStreamError("not a typedstream: wrong version byte")
        signature = self.take(self.integer())
        if signature not in BYTE_ORDERS:
            raise TypedStreamError("not a typedstream: wrong signature")
        self.order = BYTE_ORDERS[signature]
        self.integer()  # system version, not used

    def shared_string(self) -> bytes:
        head = self.byte()
        if head == TAG_NEW:
            value = self.take(self.integer())
            self.strings.append(value)
            return value
        if head == TAG_NIL:
            raise TypedStreamError("a string is nil where text is expected")
        index = self.reference(head)
        if not 0 <= index < len(self.strings):
            raise TypedStreamError("a string reference points nowhere")
        return self.strings[index]

    def known_chain(self, index: int) -> tuple[str, ...]:
        """The class chain that a reference points at."""
        known = self.objects[index] if 0 <= index < len(self.objects) else None
        if known is None:
            raise TypedStreamError("a class reference points nowhere")
        return known

    def class_chain(self) -> tuple[str, ...]:
        """Read a class and its superclasses. Return the names, most derived first.

        The classes are read in a loop, and no more than MAX_CLASS_DEPTH new ones. A stream made to
        hold thousands of classes inside each other is refused: it cannot use up the call stack.
        """
        names: list[str] = []
        chain: tuple[str, ...] = ()
        first_slot = len(self.objects)
        while True:
            head = self.byte()
            if head == TAG_NIL:
                break
            if head != TAG_NEW:
                chain = self.known_chain(self.reference(head))
                break
            if len(names) == MAX_CLASS_DEPTH:
                raise TypedStreamError(f"a class chain is longer than {MAX_CLASS_DEPTH} classes")
            names.append(self.shared_string().decode("ascii", "replace"))
            self.integer()  # class version, not used
            self.objects.append(None)
        # each new class took one entry of the table: the entry holds the chain from that class on
        for level in reversed(range(len(names))):
            chain = (names[level], *chain)
            self.objects[first_slot + level] = chain
        return chain

    def new_object(self) -> tuple[str, ...]:
        """Enter a new object. Return the names of its class chain."""
        if self.byte() != TAG_NEW:
            raise TypedStreamError("expected a new object")
        self.objects.append(None)
        return self.class_chain()

    def expect_type(self, code: bytes) -> None:
        start = self.pos
        if self.shared_string() != code:
            raise TypedStreamError(f"expected type code {code!r} at offset {start}")


def _as_bytes(blob: object) -> bytes:
    """The blob as bytes. A value of another type cannot be a typedstream."""
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, (bytearray, memoryview)):
        return bytes(blob)
    raise TypedStreamError(f"a blob is bytes, bytearray or memoryview, not {type(blob).__name__}")


def _utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        error = TypedStreamError("the text is not valid UTF-8")
    # raised after the handler: the codec error holds the bytes of the text and must not go with this one
    raise error


def read_text(blob: bytes | bytearray | memoryview) -> str:
    """Follow the archive structure to the string of the attributed string."""
    data = _as_bytes(blob)
    for layout in KNOWN_LAYOUTS:
        if data.startswith(layout):
            length, start = read_integer(data, len(layout))
            if start + length > len(data):
                raise TypedStreamError("a length points past the end of the data")
            return _utf8(data[start : start + length])
    return parse_text(data)


def parse_text(blob: bytes | bytearray | memoryview) -> str:
    """read_text without the shortcut for known layouts: walk the stream object by object."""
    reader = _Reader(_as_bytes(blob))
    reader.header()
    reader.expect_type(b"@")
    if "NSAttributedString" not in reader.new_object():
        raise TypedStreamError("the root object is not an attributed string")
    reader.expect_type(b"@")
    if "NSString" not in reader.new_object():
        raise TypedStreamError("the first value is not a string")
    reader.expect_type(b"+")
    return _utf8(reader.take(reader.integer()))


def scan_text(blob: bytes | bytearray | memoryview) -> str:
    """Find the first "bytes" value by its marker. For a typedstream with another outer layout.

    The data must start with a typedstream header. In other data the bytes after a marker are not
    the text of a message, so such data is refused.
    """
    data = _as_bytes(blob)
    reader = _Reader(data)
    reader.header()
    marker = data.find(BYTES_TYPE_NEW, reader.pos)
    if marker < 0:
        raise TypedStreamError("no text value in the data")
    length, start = read_integer(data, marker + len(BYTES_TYPE_NEW), reader.order)
    if start + length > len(data):
        raise TypedStreamError("a length points past the end of the data")
    return _utf8(data[start : start + length])


def decode_attributed_body(blob: bytes | bytearray | memoryview) -> str:
    """Return the plain text of an attributedBody blob, exactly as archived.

    A value that is not bytes and data that is not a typedstream raise TypedStreamError. The marker
    scan runs only after the walk through the stream failed, and only on data with a typedstream header.
    """
    data = _as_bytes(blob)
    try:
        return read_text(data)
    except TypedStreamError:
        return scan_text(data)


def attributed_body_text(blob: bytes | bytearray | memoryview | None) -> str | None:
    """Like decode_attributed_body, but None for a missing or unreadable blob and for a value that is not bytes."""
    if blob is None:
        return None
    try:
        return decode_attributed_body(blob)
    except TypedStreamError:
        return None
