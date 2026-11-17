"""A whole report through ArchiveSource on the real archive, with the fake model.

Until the join, `report_engine.archive_source.ArchiveSource` was tested with a
stand-in only. Here it reads the real `archive` package.

Most archives are pulled with `import_chat_db` from an invented chat.db: that
is the path of the app, a message id is Apple's guid, and a second pull adds
rows. One archive is loaded by the legacy bridge, which is what the app runs on
until its first pull.

The labeler is the one that the wiring must pass (`report_labeler`): a saved
name for a saved sender, the empty string for a sender nobody saved, and the
contact's name for the empty sender of a one-to-one conversation. See
`report-engine-seams.md`, section 3.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import safe_html

# The readers of the server, as they are on this branch. The module is loaded here and not
# inside a test: it keeps the standard output of the moment it loads, and a test has its own.
import ui_server
from archive import INDEXED_TEXT_CHARS, Archive, ArchiveNotFoundError
from report_engine import (
    SECTION_KEYS,
    SECTION_TITLES,
    EmptyConversationError,
    PeriodCache,
    Report,
    ReportStore,
    render_markdown,
)
from report_engine.archive_source import ArchiveSource
from report_engine.digest import MAX_QUOTE_CHARS
from report_engine.fake_llm import FakeLLM
from report_engine.transcript import ELISION_MARK
from tests.archive.builders import utc
from tests.integration.report_data import (
    GROUP_TITLE,
    HANDLE_PARTS,
    MINA_CONTACT,
    MINA_NAME,
    Pulled,
    bridged,
    estimate_on,
    export_row,
    labels_seen,
    moment,
    prompt_text,
    report_labeler,
    run_on,
)
from tests.report_engine.story_reader import headings_of
from tests.report_engine.support import LARGE, LARGE_LABELS, SMALL, SMALL_OPTIONS
from tests.report_engine.synth import GROUP, MINA, RAVI, THEO, conversation, message

TITLES = [SECTION_TITLES[key] for key in SECTION_KEYS]
MINA_LABELER = report_labeler({MINA: MINA_NAME}, MINA_NAME)
REPORT_KEYS = frozenset(
    "schema_version id conversation scope model created_at status "
    "warnings sections moments periods usage stats".split()
)
TAPBACK_SUMMARY = "Loved an invented message"  # what the builder writes into a tapback row
CURLY_OPEN, CURLY_CLOSE = chr(0x201C), chr(0x201D)
PLACEHOLDER = chr(0xFFFC)
NOBODY = "+12025550199"


@pytest.fixture
def pulled(tmp_path: Path) -> Iterator[Pulled]:
    made = Pulled(tmp_path)
    try:
        yield made
    finally:
        made.close()


@pytest.fixture
def four_months(pulled: Pulled) -> Pulled:
    """Mina's one-to-one conversation: four full months, four periods under the small options."""
    pulled.write(MINA, LARGE)
    pulled.pull([MINA_CONTACT])
    return pulled


@pytest.fixture
def cache(tmp_path: Path) -> PeriodCache:
    return PeriodCache(tmp_path / "cache" / "periods.sqlite3")


class TestOneReport:
    def test_the_report_has_one_shape(self, four_months: Pulled, cache: PeriodCache, tmp_path: Path) -> None:
        model = FakeLLM()
        with four_months.open() as archive:
            report = run_on(archive, MINA, MINA_LABELER, llm=model, cache=cache)
            stored = archive.count(conversations=[MINA])
        as_json = report.model_dump(mode="json")
        assert Report.model_validate(json.loads(json.dumps(as_json))) == report
        assert set(as_json) == REPORT_KEYS
        assert [section.key for section in report.sections] == list(SECTION_KEYS)
        assert [section.title for section in report.sections] == TITLES
        assert (report.conversation, report.status, report.warnings) == (MINA, "complete", ())
        assert [period.label for period in report.periods] == LARGE_LABELS
        assert all(period.error is None and period.cache_key for period in report.periods)
        assert (len(model.map_calls), len(model.reduce_calls), report.usage.calls) == (4, 1, 5)
        # every message of the conversation was read one time
        assert sum(period.message_count for period in report.periods) == stored == len(LARGE)
        days = [found.date for found in report.moments]
        assert len(days) == 4 and days == sorted(days)
        store = ReportStore(tmp_path / "reports")
        store.save(report)
        assert store.load_report(MINA, report.id) == report

    def test_a_short_conversation_gives_the_same_shape(self, four_months: Pulled) -> None:
        four_months.write(THEO, [{**row, "sender": "" if row["is_from_me"] else THEO} for row in SMALL])
        four_months.pull([MINA_CONTACT])
        with four_months.open() as archive:
            short = run_on(archive, THEO, report_labeler({THEO: "Theo"}, "Theo")).model_dump(mode="json")
            long = run_on(archive, MINA, MINA_LABELER).model_dump(mode="json")
        assert (len(short["periods"]), len(long["periods"])) == (1, 4)
        assert set(short) == set(long) == REPORT_KEYS
        for part in ("sections", "moments", "periods"):
            assert set(short[part][0]) == set(long[part][0])

    def test_verified_quotes_name_real_messages(self, four_months: Pulled) -> None:
        guids = four_months.source_guids()
        with four_months.open() as archive:
            report = run_on(archive, MINA, MINA_LABELER)
            assert len(report.moments) == 4
            for found in report.moments:
                assert found.quote_verified
                quoted = archive.message(found.quote_message_id)
                assert quoted is not None and quoted["conversation"] == MINA
                assert found.quote in quoted["text"]
                assert found.date == quoted["timestamp"][:10]
                assert found.quote_speaker == ("You" if quoted["is_from_me"] else MINA_NAME)
                # the id is the guid of a row of the Messages database, not a ROWID
                assert found.quote_message_id in guids
                for message_id in found.message_ids:
                    cited = archive.message(message_id)
                    assert cited is not None and cited["conversation"] == MINA

    def test_a_second_run_makes_no_period_call(self, four_months: Pulled, cache: PeriodCache) -> None:
        first_model, second_model = FakeLLM(), FakeLLM()
        with four_months.open() as archive:
            first = run_on(archive, MINA, MINA_LABELER, llm=first_model, cache=cache)
            second = run_on(archive, MINA, MINA_LABELER, llm=second_model, cache=cache)
        assert (len(first_model.map_calls), len(first_model.reduce_calls)) == (4, 1)
        assert (len(second_model.map_calls), len(second_model.reduce_calls)) == (0, 1)
        assert [period.from_cache for period in first.periods] == [False] * 4
        assert [period.from_cache for period in second.periods] == [True] * 4
        assert [p.cache_key for p in second.periods] == [p.cache_key for p in first.periods]
        assert (second.usage.calls, cache.count()) == (1, 4)
        assert second.moments == first.moments

    def test_a_run_after_appended_messages_makes_one_period_call(
        self, four_months: Pulled, cache: PeriodCache
    ) -> None:
        with four_months.open() as archive:
            first = run_on(archive, MINA, MINA_LABELER, cache=cache)
        later = [
            message(0, "2023-04-30 20:00:00", "the ferry leaves at nine tomorrow morning"),
            message(0, "2023-04-30 20:01:00", "then I will bring the green thermos", is_from_me=True),
            message(0, "2023-04-30 20:02:00", "and the map of the north shore"),
        ]
        four_months.write(MINA, later)
        pull = four_months.pull([MINA_CONTACT])
        assert (pull.messages_added, pull.messages_updated, pull.messages_removed) == (3, 0, 0)

        model = FakeLLM()
        with four_months.open() as archive:
            quote = estimate_on(archive, MINA, MINA_LABELER, cache)
            second = run_on(archive, MINA, MINA_LABELER, llm=model, cache=cache)
        assert (quote["periods"], quote["periods_cached"]) == (4, 3)
        assert (len(model.map_calls), len(model.reduce_calls), second.usage.calls) == (1, 1, 2)
        assert model.map_calls[0].first_day == "2023-04-01"
        assert "the ferry leaves at nine tomorrow morning" in model.map_calls[0].transcript
        # the three closed periods are the same text, so they keep their keys
        assert [p.cache_key for p in second.periods[:3]] == [p.cache_key for p in first.periods[:3]]
        assert [period.from_cache for period in second.periods] == [True, True, True, False]
        assert second.periods[3].cache_key != first.periods[3].cache_key
        assert second.periods[3].message_count == first.periods[3].message_count + 3
        # a row that the source still holds keeps its id across the pull
        assert second.moments[:3] == first.moments[:3]

    def test_the_estimate_agrees_with_the_run(self, four_months: Pulled, cache: PeriodCache) -> None:
        with four_months.open() as archive:
            before = estimate_on(archive, MINA, MINA_LABELER, cache)
            report = run_on(archive, MINA, MINA_LABELER, cache=cache)
            after = estimate_on(archive, MINA, MINA_LABELER, cache)
        assert (before["periods"], before["periods_cached"], before["cached_tokens"]) == (4, 0, 0)
        assert (after["periods"], after["periods_cached"]) == (len(report.periods), 4)
        assert after["cached_tokens"] > 0
        # what is left to send is the one call that writes the sections
        assert 0 < after["input_tokens"] < before["input_tokens"]


class TestTheReportFile:
    def test_the_server_reads_the_report_file(self, four_months: Pulled, tmp_path: Path) -> None:
        with four_months.open() as archive:
            text = render_markdown(run_on(archive, MINA, MINA_LABELER))
        path = tmp_path / "out" / f"analysis_{ui_server.sanitize_filename(MINA)}.md"
        path.parent.mkdir()
        path.write_text(text, encoding="utf-8")
        assert text.splitlines()[0] == f"# Conversation Analysis - {MINA}"
        assert ui_server.report_contact(path) == MINA
        assert ui_server.markdown_section(text, "Overview").splitlines()[0] == "- Scope: all dates"
        assert ui_server.report_scope(text) == (None, None)
        story = ui_server.markdown_section(text, "Summary", stop_at_next_heading=False)
        headings = [line for line in story.splitlines() if line.startswith("#")]
        assert headings == [f"### {title}" for title in TITLES]
        page = safe_html.story_html(story)
        assert headings_of(page).found == [(3, 0, title) for title in TITLES]
        assert "a" not in headings_of(page).tags and page != safe_html._plain(story)

    def test_a_report_of_some_days_states_them(self, four_months: Pulled) -> None:
        scope = {"since": "2023-02-01", "until": "2023-02-28"}
        with four_months.open() as archive:
            report = run_on(archive, MINA, MINA_LABELER, scope=scope)
            in_days = archive.count(conversations=[MINA], since=scope["since"], until=scope["until"])
        text = render_markdown(report)
        assert [period.label for period in report.periods] == ["Feb 2023"]
        assert report.periods[0].message_count == in_days == 168
        assert ui_server.markdown_section(text, "Overview").splitlines()[0] == (
            "- Scope: 2023-02-01 to 2023-02-28"
        )
        assert ui_server.report_scope(text) == ("2023-02-01", "2023-02-28")


class TestWhoSpeaks:
    def test_the_empty_sender_of_a_one_to_one_conversation_is_the_contact(self, pulled: Pulled) -> None:
        """A received message can have no handle in the source. The archive then gives the
        empty sender. With the contact's name for it, the conversation has two speakers.
        With no name for it, the engine makes a third speaker of the same person."""
        pulled.write(MINA, [*SMALL, message(0, "2024-06-10 12:00:00", "a line with no handle", sender="")])
        pulled.pull([MINA_CONTACT])
        named, unnamed = FakeLLM(), FakeLLM()
        with pulled.open() as archive:
            received = [m for m in ArchiveSource(archive).messages(MINA, None, None) if not m["is_from_me"]]
            run_on(archive, MINA, MINA_LABELER, llm=named)
            run_on(archive, MINA, report_labeler({MINA: MINA_NAME}), llm=unnamed)
        assert {row["sender"] for row in received} == {MINA, ""}
        assert labels_seen(named) == {"You", MINA_NAME}
        assert labels_seen(unnamed) == {"You", MINA_NAME, "Unsaved contact"}

    def test_an_unsaved_sender_gets_a_name_of_its_own_and_no_handle_goes_out(self, pulled: Pulled) -> None:
        pulled.chat(GROUP, members=(MINA, RAVI, THEO), title=GROUP_TITLE)
        pulled.write(GROUP, conversation("2024-06-01", days=6, per_day=6, others=(MINA, RAVI, THEO)))
        pulled.pull([MINA_CONTACT])
        model = FakeLLM()
        with pulled.open() as archive:
            report = run_on(archive, GROUP, report_labeler({MINA: MINA_NAME}), llm=model)
        assert report.conversation == GROUP
        assert labels_seen(model) == {"You", MINA_NAME, "Unsaved contact", "Unsaved contact 2"}
        sent = prompt_text(model)
        assert not [part for part in (*HANDLE_PARTS, GROUP, GROUP_TITLE) if part in sent]


class TestRowsThatAreNotMessages:
    def test_tapbacks_events_attachments_and_unsent_rows_are_in_no_transcript(self, pulled: Pulled) -> None:
        rowids = pulled.write(MINA, SMALL[:3])
        chat, mina, rows = pulled.chat(MINA), pulled.handle(MINA), pulled.builder
        rows.tapback(chat, rows.guid_of(rowids[0]), 2000, when=moment("2024-06-01 10:00:00"), handle=mina)
        rows.message(
            chat, when=moment("2024-06-01 10:01:00"), handle=mina, item_type=2, group_title="Renamed"
        )
        rows.attachment(rows.message(chat, when=moment("2024-06-01 10:02:00"), handle=mina))
        unsent = moment("2024-06-01 10:04:00")
        rows.message(chat, when=moment("2024-06-01 10:03:00"), handle=mina, retracted=unsent)
        pull = pulled.pull()
        assert (pull.messages_added, pull.reactions_added, pull.system_added) == (5, 1, 1)
        model = FakeLLM()
        with pulled.open() as archive:
            kinds = [row["kind"] for row in archive.iter_messages(MINA)]
            with_events = [row["kind"] for row in archive.iter_messages(MINA, include_system=True)]
            texts = [row["text"] for row in ArchiveSource(archive).messages(MINA, None, None)]
            report = run_on(archive, MINA, MINA_LABELER, llm=model)
        # the archive gives no tapback, and a group event only on request
        assert kinds == ["text", "text", "text", "attachment", "text"]
        assert with_events.count("system") == 1
        # a photo with no words and an unsent message come with no text: the engine leaves them out
        assert texts == [*(row["text"] for row in SMALL[:3]), "", ""]
        assert [period.message_count for period in report.periods] == [3]
        assert len(model.map_calls[0].lines) == 3
        assert TAPBACK_SUMMARY not in prompt_text(model) and "Renamed" not in prompt_text(model)

    def test_the_model_reads_the_start_of_a_long_text_and_a_quote_is_checked_against_all_of_it(
        self, pulled: Pulled
    ) -> None:
        """The archive keeps a text of any length whole. The engine cuts it for the model."""
        long_text = "harbor lantern " * 1_000
        pulled.write(MINA, [*SMALL[:3], message(0, "2024-06-02 09:00:00", long_text)])
        pulled.pull()
        model = FakeLLM()
        with pulled.open() as archive:
            report = run_on(archive, MINA, MINA_LABELER, llm=model)
            found = report.moments[0]
            quoted = archive.message(found.quote_message_id)
        assert len(long_text) > INDEXED_TEXT_CHARS
        shown = max((line.text for line in model.map_calls[0].lines), key=len)
        assert shown.endswith(ELISION_MARK) and len(shown) <= SMALL_OPTIONS.max_message_chars
        assert report.warnings == ("1 long message was cut to fit: the model did not read all of it.",)
        assert quoted is not None and quoted["text"] == long_text
        assert found.quote_verified and found.quote in long_text
        assert len(shown) > len(found.quote) and len(found.quote) <= MAX_QUOTE_CHARS


class TestDates:
    def test_a_range_includes_all_of_its_first_and_last_day(self, four_months: Pulled) -> None:
        with four_months.open() as archive:
            source = ArchiveSource(archive)
            days = [row["timestamp"] for row in source.messages(MINA, "2023-02-01", "2023-02-28")]
            stamps = [
                row["timestamp"]
                for row in source.messages(MINA, "2023-02-01 09:07:00", "2023-02-28 09:28:00")
            ]
            open_end = list(source.messages(MINA, "2023-04-30", "9999-12-31"))
        assert (days[0], days[-1], len(days)) == ("2023-02-01 09:00:00", "2023-02-28 09:35:00", 168)
        # a bound with a time is exact, and both ends are in the range
        assert (stamps[0], stamps[-1], len(stamps)) == ("2023-02-01 09:07:00", "2023-02-28 09:28:00", 166)
        assert len(open_end) == 6

    def test_the_hour_that_the_clocks_repeat_is_read_in_wall_clock_order(self, pulled: Pulled) -> None:
        """A known difference, stated here. The archive orders messages by elapsed time.
        Its time stamps are local, so in the hour that the clocks repeat a later message
        can have the earlier time stamp. The engine sorts a source that is not in order of
        its time stamps, so the transcript of that hour is in wall-clock order."""
        chat, mina = pulled.chat(MINA), pulled.handle(MINA)
        # 2024-11-03 in Los Angeles: 08:50 UTC is 01:50 before the change, 09:10 UTC is 01:10 after it
        pulled.builder.message(chat, when=utc(2024, 11, 3, 8, 50), text="the first pass", handle=mina)
        pulled.builder.message(chat, when=utc(2024, 11, 3, 9, 10), text="the second pass", handle=mina)
        pulled.pull()
        model = FakeLLM()
        with pulled.open() as archive:
            stored = [(row["timestamp"][11:16], row["text"]) for row in archive.iter_messages(MINA)]
            run_on(archive, MINA, MINA_LABELER, llm=model)
        assert stored == [("01:50", "the first pass"), ("01:10", "the second pass")]
        assert [(line.clock, line.text) for line in model.map_calls[0].lines] == [
            ("01:10", "the second pass"),
            ("01:50", "the first pass"),
        ]


class TestTheBridge:
    def test_a_report_on_a_bridge_archive_names_legacy_ids(self, tmp_path: Path) -> None:
        """The archive of the first hook-up: one export through `import_legacy_csv`."""
        loved = f"Loved {CURLY_OPEN}{SMALL[1]['text']}{CURLY_CLOSE}"
        photo = {
            **export_row(message(901, "2024-06-10 12:01:00", PLACEHOLDER), MINA),
            "attachment_types": "image/jpeg",
            "attachment_files": "~/Pictures/a.jpeg",
        }
        words = [
            *SMALL,
            message(900, "2024-06-10 12:00:00", loved),
            message(902, "2024-06-10 12:02:00", "Loved it more than the last one"),
        ]
        model = FakeLLM()
        with Archive.open(bridged(tmp_path, [*(export_row(row, MINA) for row in words), photo])) as archive:
            report = run_on(archive, MINA, MINA_LABELER, llm=model)
            assert archive.count(conversations=[MINA]) == len(SMALL) + 2
            assert report.moments
            for found in report.moments:
                assert found.quote_verified and found.quote_message_id.startswith("legacy:")
                quoted = archive.message(found.quote_message_id)
                assert quoted is not None and found.quote in quoted["text"]
                rowid = int(found.quote_message_id.removeprefix("legacy:"))
                assert archive.message_by_source_rowid(rowid) == quoted
        # the tapback phrase and the photo are in no transcript; a text that only starts like one is
        assert [period.message_count for period in report.periods] == [len(SMALL) + 1]
        assert labels_seen(model) == {"You", MINA_NAME}
        assert CURLY_OPEN not in prompt_text(model)
        assert "Loved it more than the last one" in prompt_text(model)


class TestNothingToRead:
    def test_a_missing_archive_is_the_error_of_the_archive(self, tmp_path: Path) -> None:
        with pytest.raises(ArchiveNotFoundError):
            ArchiveSource.open(tmp_path / "nowhere" / "archive.sqlite3")

    def test_an_unknown_conversation_stops_before_any_model_call(self, four_months: Pulled) -> None:
        model = FakeLLM()
        with ArchiveSource.open(four_months.archive_path) as source:
            assert list(source.messages(NOBODY, None, None)) == []
            with pytest.raises(EmptyConversationError):
                run_on(source.archive, NOBODY, MINA_LABELER, llm=model)
            quote = estimate_on(source.archive, NOBODY, MINA_LABELER, None)
        assert model.calls == []
        assert (quote["periods"], quote["input_tokens"], quote["cost_usd"]) == (0, 0, 0)
