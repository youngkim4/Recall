"""key_store.py: the key that a person gives in the app, saved in the file .env of the data folder."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

import recall_paths
from recall_services import key_store, keys

# an invented key: it has the shape of a key and opens nothing
INVENTED = "sk-invented-" + "Q7x" * 10
OTHER = "sk-another-invented-" + "Z9" * 12


@pytest.fixture
def key_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The file .env of a data folder that is not there yet. No key is in the environment."""
    path = tmp_path / "data" / ".env"
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", path)
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    return path


# --- what is told ---------------------------------------------------------------------


def test_with_no_key_the_state_says_so_and_where_a_key_goes(key_file: Path) -> None:
    assert key_store.state() == {"present": False, "source": None, "place": keys.KEY_PLACE}


def test_the_state_names_the_environment_and_never_the_key(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(keys.KEY_NAME, INVENTED)
    shown = key_store.state()
    assert shown == {"present": True, "source": "environment", "place": keys.KEY_PLACE}
    assert INVENTED[-4:] not in repr(shown)


def test_the_state_names_the_file(key_file: Path) -> None:
    key_store.save(INVENTED)
    assert key_store.state() == {"present": True, "source": "file", "place": keys.KEY_PLACE}


# --- the check of a key -----------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "sk-short",
        "pk-" + "a" * 30,
        "sk-" + "a" * 298,
        "sk-has a space-" + "a" * 20,
        "sk-has.a.dot-" + "a" * 20,
        "sk-ünicode-" + "a" * 20,
        "sk-new\nline-" + "a" * 20,
    ],
)
def test_a_value_that_is_no_key_is_refused_in_a_sentence_that_does_not_repeat_it(key_file: Path, value: str) -> None:
    with pytest.raises(key_store.KeyRefused) as refused:
        key_store.save(value)
    assert refused.value.status == 400
    sentence = str(refused.value)
    assert sentence.endswith(".") and (not value.strip() or value.strip() not in sentence)
    assert not key_file.exists()


@pytest.mark.parametrize("value", [None, 42, ["sk-" + "a" * 30], {"key": INVENTED}])
def test_a_value_that_is_not_text_is_refused(key_file: Path, value: object) -> None:
    with pytest.raises(key_store.KeyRefused) as refused:
        key_store.save(value)
    assert refused.value.status == 400 and str(refused.value) == key_store.NOT_TEXT


def test_a_key_is_trimmed_and_the_longest_and_the_shortest_pass(key_file: Path) -> None:
    shortest, longest = "sk-" + "a" * 17, "sk-" + "b" * 297
    key_store.save(f"  {shortest}\n")
    assert keys.api_key() == shortest
    key_store.save(longest)
    assert keys.api_key() == longest


# --- the file ----------------------------------------------------------------------------


def test_the_key_is_written_to_a_new_file_that_only_its_owner_reads(key_file: Path) -> None:
    assert key_store.save(INVENTED) == {"present": True, "source": "file", "place": keys.KEY_PLACE}
    assert key_file.read_text(encoding="utf-8") == f"OPENAI_API_KEY={INVENTED}\n"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert keys.api_key() == INVENTED
    assert [path.name for path in key_file.parent.iterdir()] == [".env"]


def test_every_other_line_of_the_file_is_kept_as_it_is(key_file: Path) -> None:
    key_file.parent.mkdir(parents=True)
    before = "# my settings\r\nRECALL_THEME=dusk\n\nexport OTHER_TOOL_KEY='kept as it is'\nLAST_LINE=no newline"
    key_file.write_text(before, encoding="utf-8", newline="")
    key_store.save(INVENTED)
    after = key_file.read_text(encoding="utf-8", newline="")
    assert after == before + f"\nOPENAI_API_KEY={INVENTED}\n"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_a_key_that_is_there_is_replaced_in_its_place(key_file: Path) -> None:
    key_file.parent.mkdir(parents=True)
    key_file.write_text(f"FIRST=1\nOPENAI_API_KEY={OTHER}\nLAST=2\n  export OPENAI_API_KEY = 'old'\n", encoding="utf-8")
    key_store.save(INVENTED)
    assert key_file.read_text(encoding="utf-8") == f"FIRST=1\nOPENAI_API_KEY={INVENTED}\nLAST=2\n"
    assert keys.api_key() == INVENTED


def test_a_line_that_only_starts_like_the_key_is_another_line(key_file: Path) -> None:
    key_file.parent.mkdir(parents=True)
    key_file.write_text("OPENAI_API_KEY_OLD=kept\n", encoding="utf-8")
    key_store.save(INVENTED)
    assert key_file.read_text(encoding="utf-8") == f"OPENAI_API_KEY_OLD=kept\nOPENAI_API_KEY={INVENTED}\n"


def test_bytes_that_are_not_text_on_other_lines_are_kept(key_file: Path) -> None:
    key_file.parent.mkdir(parents=True)
    key_file.write_bytes(b"BLOB=\xff\xfe\n")
    key_store.save(INVENTED)
    assert key_file.read_bytes() == b"BLOB=\xff\xfe\n" + f"OPENAI_API_KEY={INVENTED}\n".encode()


def test_delete_removes_only_the_line_of_the_key(key_file: Path) -> None:
    key_file.parent.mkdir(parents=True)
    key_file.write_text(f"FIRST=1\nOPENAI_API_KEY={OTHER}\nLAST=2\n", encoding="utf-8")
    assert key_store.remove() == {"present": False, "source": None, "place": keys.KEY_PLACE}
    assert key_file.read_text(encoding="utf-8") == "FIRST=1\nLAST=2\n"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600 and keys.api_key() == ""


def test_delete_with_no_file_writes_nothing(key_file: Path) -> None:
    assert key_store.remove()["present"] is False
    assert not key_file.exists() and not key_file.parent.exists()


def test_with_a_key_in_the_environment_a_saved_key_would_not_be_used(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(keys.KEY_NAME, OTHER)
    with pytest.raises(key_store.KeyRefused) as refused:
        key_store.save(INVENTED)
    assert (refused.value.status, str(refused.value)) == (409, key_store.ENVIRONMENT_WINS)
    assert not key_file.exists()


def test_a_file_that_can_not_be_written_is_told_in_a_sentence_with_no_key(key_file: Path) -> None:
    key_file.parent.mkdir(parents=True)
    key_file.mkdir()  # a folder in the place of the file
    with pytest.raises(key_store.KeyRefused) as refused:
        key_store.save(INVENTED)
    assert refused.value.status == 500 and str(refused.value) == key_store.NOT_WRITTEN
    assert refused.value.__cause__ is None and refused.value.__suppress_context__
    assert [path.name for path in key_file.parent.iterdir()] == [".env"]


# --- the origin of a request --------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "origin", "allowed"),
    [
        ("127.0.0.1:8765", None, True),
        ("127.0.0.1:8765", "", True),
        ("127.0.0.1:8765", "http://127.0.0.1:8765", True),
        ("localhost:8765", "http://LOCALHOST:8765", True),
        ("127.0.0.1:8765", "http://127.0.0.1:5173", False),
        ("127.0.0.1:8765", "http://localhost:8765", False),
        ("127.0.0.1:8765", "https://127.0.0.1:8765", False),
        ("127.0.0.1:8765", "http://127.0.0.1", False),
        ("127.0.0.1:8765", "null", False),
        ("127.0.0.1:8765", "http://127.0.0.1:8765/path", False),
        ("127.0.0.1:8765", "http://user@127.0.0.1:8765", False),
        ("", "http://127.0.0.1:8765", False),
    ],
)
def test_only_the_exact_origin_of_this_server_may_touch_the_key(host: str, origin: str | None, allowed: bool) -> None:
    assert key_store.same_origin(host, origin) is allowed
