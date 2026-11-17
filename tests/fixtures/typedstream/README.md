# Typedstream fixtures

Each `.hex` file is one archive, written as hexadecimal text.
Apple's encoder made each archive: `NSArchiver.archivedData(withRootObject:)`.
No file here comes from a Messages database. All the text is invented.

## Why these files exist

The exporter reads the text of a message from the `attributedBody` column of the Messages database.
That column holds archives of this type.
A test that builds its own archive can repeat the error of the decoder, and then the test passes.
An archive from Apple's encoder cannot repeat that error.

## How to make the files again

Run this command from the top folder of the repo, on macOS:

```bash
swift -suppress-warnings scripts/make_attributed_body_fixtures.swift
```

- The script writes the `.hex` files and `manifest.json` into this folder.
- The script reads each archive back with `NSUnarchiver`. It stops if the text is not the same.
- `NSArchiver` is deprecated. Without `-suppress-warnings`, Swift shows a warning for it.
- The output has no date and no machine name. A second run gives the same bytes.
- The files in this folder were made on macOS 27.2 (build 26B123) with Swift 6.4.2. If a different macOS gives different bytes, examine the difference before you commit it.

Do not change a `.hex` file by hand. Do not add an archive that Python code made.

## The files

- `manifest.json` lists each fixture: `name`, `note`, the `root` class, `archive_bytes`, `utf8_bytes` and the `sha256` of the UTF-8 text.
- `utf8_hex` is the text itself, as hexadecimal UTF-8. It is `null` for a text of more than 512 bytes.
- `<name>.hex` is the archive, with 32 bytes on each line.
- A name that starts with `mutable_` is an archive of an `NSMutableAttributedString`. The other text fixtures are archives of an `NSAttributedString`.
- Each text has one attribute, `__kIMMessagePartAttributeName`, because Messages puts that attribute on a message. The empty text of `empty` has no attribute, because an attribute needs a character.
- `not_text` is an archive of two numbers. It has no string.

## What the archives show

- The text follows the class name `NSString`, the type code `+` and a length.
- A length below 128 is one byte.
- A length from 128 to 32,767 is the byte `81` and then two bytes, low byte first.
- A length of 32,768 or more is the byte `82` and then four bytes, low byte first.
- The text is UTF-8. The encoder keeps each character, control characters included.
- The encoder refuses a string that has an unpaired surrogate. For that reason, there is no such fixture.
