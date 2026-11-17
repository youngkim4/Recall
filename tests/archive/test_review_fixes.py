"""Small fixes that the first review asked for, each with the case that showed the defect."""

from __future__ import annotations

import copy
import pickle
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from archive import import_chat_db
from archive import importer as importer_module
from archive.people import HandleRow, contact_groups, plan_people
from archive.records import SourceMessage
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.inspect_archive import query, scalar

START = utc(2024, 10, 1, 15, 0)
JUNE = "+15550100181"


def notes(builder: ChatDbBuilder, count: int) -> None:
    handle = builder.handle(JUNE)
    chat = builder.chat(JUNE, members=[handle])
    for index in range(count):
        builder.message(chat, when=START + timedelta(minutes=index), text=f"note {index}", handle=handle)
    builder.commit()


class TestTwoArchivesInOneFolder:
    def test_an_import_started_during_the_copy_of_another_archive_does_not_disturb_it(
        self, builder: ChatDbBuilder, source_path: Path, tmp_path: Path
    ) -> None:
        notes(builder, 5)
        one, two = tmp_path / "data" / "one.sqlite3", tmp_path / "data" / "two.sqlite3"
        inner: list[Any] = []

        def progress(event: dict[str, Any]) -> None:
            if event["phase"] == "copy" and not inner:
                inner.append(import_chat_db(source_path, two))

        outer = import_chat_db(source_path, one, progress=progress)

        assert (outer.status, outer.messages_added) == ("ok", 5)
        assert (inner[0].status, inner[0].messages_added) == ("ok", 5)
        assert importer_module.work_dir(one) != importer_module.work_dir(two)
        assert sorted(path.name for path in one.parent.iterdir() if "import-work" in path.name) == []


class TestImportResult:
    def test_a_result_can_be_pickled_and_copied(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        notes(builder, 3)
        result = import_chat_db(source_path, archive_path)

        for twin in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
            assert twin == result
            assert twin.as_dict() == result.as_dict()
        with pytest.raises(TypeError):
            result.counts["rows_seen"] = 0  # type: ignore[index]

    def test_a_cancelled_run_says_how_far_it_got(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes(builder, 6)
        monkeypatch.setattr(importer_module, "BATCH_ROWS", 2)
        seen: list[dict[str, Any]] = []

        def cancel() -> bool:
            return any(event["phase"] == "messages" and event["done"] >= 2 for event in seen)

        result = import_chat_db(source_path, archive_path, progress=seen.append, cancel=cancel)

        assert result.status == "cancelled"
        assert result.run_id is not None
        assert 0 < result.rows_seen < 6
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 0  # none of that work is kept


class TestBadRow:
    def test_a_row_that_cannot_be_classified_is_skipped_and_the_import_goes_on(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notes(builder, 4)
        real = importer_module.classify

        def fussy(row: SourceMessage, *, has_attachments: bool) -> Any:
            if row.text == "note 2":
                raise OverflowError("Python int too large to convert to SQLite INTEGER")
            return real(row, has_attachments=has_attachments)

        monkeypatch.setattr(importer_module, "classify", fussy)

        result = import_chat_db(source_path, archive_path)

        assert result.skipped == {"bad_row": 1}
        assert [row[0] for row in query(archive_path, "SELECT text FROM messages ORDER BY sent_ns")] == [
            "note 0",
            "note 1",
            "note 3",
        ]


class TestPeople:
    def test_at_equal_size_a_named_person_keeps_its_id_before_an_unnamed_one(self) -> None:
        # handle 1 belongs to person 10; handles 2 and 3 are their own people
        handles = [
            HandleRow(1, "p:5550100001", 10),
            HandleRow(2, "p:5550100002", 20),
            HandleRow(3, "e:x@example.com", 30),
        ]
        groups, _ = contact_groups([{"name": "Wren", "phones": ["555-010-0002"], "emails": []}])

        plans = {plan.handle_ids: plan for plan in plan_people(handles, groups)}

        assert plans[(2,)].name == "Wren" and plans[(2,)].person_id == 20
        assert (plans[(1,)].person_id, plans[(3,)].person_id) == (10, 30)

    def test_two_spellings_of_one_accented_name_are_one_name(self) -> None:
        composed = "Zo" + chr(0xEB)
        decomposed = unicodedata.normalize("NFD", composed)
        assert composed != decomposed

        made, _ = contact_groups(
            [
                {"name": composed, "phones": ["555-010-0111"], "emails": ["zoe@example.com"]},
                {"name": decomposed, "phones": [], "emails": ["zoe@example.com"]},
            ]
        )

        assert [group.name for group in made] == [composed]


class TestDirectory:
    def test_the_kind_and_the_key_of_a_stored_handle_follow_the_rules_of_this_version(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        notes(builder, 1)
        import_chat_db(source_path, archive_path)
        right = query(archive_path, "SELECT kind, match_key FROM handles")
        import sqlite3

        conn = sqlite3.connect(str(archive_path))
        conn.execute("UPDATE handles SET kind = 'other', match_key = 'o:stale'")
        conn.commit()
        conn.close()

        import_chat_db(source_path, archive_path)

        assert query(archive_path, "SELECT kind, match_key FROM handles") == right

    def test_a_folder_made_for_the_archive_is_private(
        self, builder: ChatDbBuilder, source_path: Path, tmp_path: Path
    ) -> None:
        notes(builder, 1)
        archive_path = tmp_path / "new-folder" / "archive.sqlite3"

        import_chat_db(source_path, archive_path)

        assert archive_path.parent.stat().st_mode & 0o777 == 0o700
        assert archive_path.stat().st_mode & 0o777 == 0o600


class TestArchiveOfAnEarlierBuild:
    """Schema version 1 changed while the package was built. An archive of such a build must fail in a clear way."""

    @pytest.mark.parametrize(
        "damage",
        [
            "ALTER TABLE conversation_members DROP COLUMN message_count",
            "DROP INDEX messages_by_conversation_time",
            (
                "DROP INDEX messages_by_conversation_time;"
                " CREATE INDEX messages_by_conversation_time ON messages(conversation_id, sent_ns, id, is_from_me, kind)"
            ),
        ],
    )
    def test_it_is_refused_with_a_version_error(
        self, damage: str, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import sqlite3

        from archive import Archive, ArchiveVersionError

        notes(builder, 2)
        import_chat_db(source_path, archive_path)
        conn = sqlite3.connect(str(archive_path))
        conn.executescript(damage)
        conn.commit()
        conn.close()

        with pytest.raises(ArchiveVersionError, match="earlier build"):
            Archive.open(archive_path)
        with pytest.raises(ArchiveVersionError, match="Delete the file"):
            import_chat_db(source_path, archive_path)
