import json
import sqlite3

from contact_names import contacts_cache_summary, handle_keys, load_cached_contact_names, resolve_contact_names


def test_handle_keys_normalizes_phone_numbers():
    assert handle_keys("+1 (555) 123-4567") == {"15551234567", "5551234567"}


def test_resolve_contact_names_reads_address_book_database(tmp_path):
    address_book = tmp_path / "AddressBook-v22.abcddb"
    conn = sqlite3.connect(address_book)
    conn.execute(
        """
        CREATE TABLE ZABCDRECORD (
            Z_PK INTEGER PRIMARY KEY,
            ZFIRSTNAME TEXT,
            ZLASTNAME TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ZABCDPHONENUMBER (
            ZOWNER INTEGER,
            ZFULLNUMBER TEXT
        )
        """
    )
    conn.execute("INSERT INTO ZABCDRECORD VALUES (1, 'Avery', 'Stone')")
    conn.execute("INSERT INTO ZABCDPHONENUMBER VALUES (1, '+1 (555) 123-4567')")
    conn.commit()
    conn.close()

    assert resolve_contact_names(["+15551234567"], base_dir=tmp_path, cache_path=None) == {
        "+15551234567": "Avery Stone",
    }


def test_load_cached_contact_names_matches_phone_and_email(tmp_path):
    cache_path = tmp_path / "contact_names.json"
    cache_path.write_text(json.dumps({
        "contacts": [
            {
                "name": "Avery Stone",
                "phones": ["+1 (555) 123-4567"],
                "emails": ["avery@example.com"],
            }
        ]
    }), encoding="utf-8")

    assert load_cached_contact_names(["+15551234567", "avery@example.com"], cache_path) == {
        "+15551234567": "Avery Stone",
        "avery@example.com": "Avery Stone",
    }
    assert contacts_cache_summary(cache_path)["count"] == 1
