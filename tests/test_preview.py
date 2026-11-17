"""Tests for the conversation preview the Analyze view reads."""
from datetime import datetime

import pandas as pd
import pytest

import ui_server
from conversation import activity_series, compute_stats, fill_months, progression_series
from tests.archive.builders import ChatDbBuilder
from tests.export_kit import (  # noqa: F401 - archives and contacts_cache are fixtures
    AVERY,
    BLAKE,
    GROUP,
    MODEL,
    NAMES,
    PLACEHOLDER,
    UNSAVED,
    archives,
    contacts_cache,
    loaded,
    pulled,
    row,
    save_names,
)

pytestmark = pytest.mark.usefixtures("archives", "contacts_cache")


@pytest.fixture
def saved_names():
    save_names(NAMES)


def preview(csv, contact):
    return ui_server.preview_payload(csv, contact, MODEL)


def test_recent_messages_say_who_sent_them(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "are you up"),
        row(AVERY, "2025-01-01 09:01:00", "barely", from_me=1),
    ])

    recent = preview(csv, AVERY)["recentMessages"]

    assert [message["text"] for message in recent] == ["are you up", "barely"]
    assert [message["isFromMe"] for message in recent] == [False, True]
    # in a conversation with one person the name is already on the page
    assert [message["senderName"] for message in recent] == ["", ""]


def test_group_messages_carry_the_name_of_who_wrote_them(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "who is driving", sender=BLAKE),
        row(GROUP, "2025-01-01 09:01:00", "not me", sender=UNSAVED),
        row(GROUP, "2025-01-01 09:02:00", "I can", from_me=1, sender=""),
    ])

    recent = preview(csv, GROUP)["recentMessages"]

    assert [message["senderName"] for message in recent] == ["Blake Reyes", "", ""]
    # an unsaved number is passed along so the interface can label it its own way
    assert [message["sender"] for message in recent] == [BLAKE, UNSAVED, ""]


def test_recent_messages_never_show_the_attachment_placeholder(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", PLACEHOLDER, attachments="image/heic"),
        row(AVERY, "2025-01-01 09:01:00", f"{PLACEHOLDER} from the lake", attachments="image/heic|image/heic"),
        row(AVERY, "2025-01-01 09:02:00", "no picture here"),
    ])

    recent = preview(csv, AVERY)["recentMessages"]

    assert [message["text"] for message in recent] == ["", "from the lake", "no picture here"]
    assert [message["attachment"] for message in recent] == ["Photo", "2 photos", ""]


def test_a_long_recent_message_is_marked_as_cut(tmp_path, saved_names):
    text = " ".join(["word"] * 200)
    csv = loaded(tmp_path / "messages.csv", [row(AVERY, "2025-01-01 09:00:00", text)])

    (message,) = preview(csv, AVERY)["recentMessages"]

    assert message["text"].endswith("word…")
    assert len(message["text"]) <= ui_server.RECENT_MESSAGE_LIMIT + 1


def test_dynamics_do_not_count_silence_up_to_today(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2020-01-01 09:00:00", "hello"),
        row(AVERY, "2020-01-02 09:00:00", "hi", from_me=1),
    ])

    # how long a conversation has been quiet depends on where the archive ends,
    # which the interface knows; a count of days up to today would be wrong
    assert "quietDays" not in preview(csv, AVERY)["dynamics"]


def test_speakers_are_the_people_who_wrote_in_a_group(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "one", sender=BLAKE),
        row(GROUP, "2025-01-01 09:01:00", "two", sender=BLAKE),
        row(GROUP, "2025-01-01 09:02:00", "three", sender=UNSAVED),
        # a row you sent can carry the handle it went to; that is not a speaker
        row(GROUP, "2025-01-01 09:03:00", "four", from_me=1, sender=AVERY),
    ])

    dynamics = preview(csv, GROUP)["dynamics"]

    assert dynamics["speakerCount"] == 2
    assert dynamics["speakers"] == [
        {"handle": BLAKE, "name": "Blake Reyes", "count": 2, "share": 0.667},
        {"handle": UNSAVED, "name": "", "count": 1, "share": 0.333},
    ]


def test_every_person_in_a_group_is_listed_not_only_the_busiest(tmp_path, saved_names):
    handles = [f"+1555000{index:04d}" for index in range(12)]
    csv = loaded(tmp_path / "messages.csv", [
        row(GROUP, f"2025-01-01 09:{index:02d}:00", "here", sender=handle)
        for index, handle in enumerate(handles)
    ])

    dynamics = preview(csv, GROUP)["dynamics"]

    # a place in the list is only true when the whole list is there to count from
    assert dynamics["speakerCount"] == 12
    assert len(dynamics["speakers"]) == 12


def test_a_conversation_with_one_person_has_no_speaker_list(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "hello"),
        row(AVERY, "2025-01-01 09:01:00", "hi", from_me=1),
    ])

    assert "speakers" not in preview(csv, AVERY)["dynamics"]


def test_the_last_day_of_a_range_is_part_of_it(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-30 09:00:00", "before"),
        row(AVERY, "2025-01-31 23:30:00", "late on the last day"),
        row(AVERY, "2025-02-01 00:10:00", "the day after"),
    ])

    payload = ui_server.preview_payload(csv, AVERY, MODEL, since="2025-01-01", until="2025-01-31")

    assert payload["stats"]["totalMessages"] == 2


def test_the_months_of_a_long_conversation_are_all_there(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, f"{2012 + index // 12}-{index % 12 + 1:02d}-15 09:00:00", "hi")
        for index in range(100)
    ])

    monthly = preview(csv, AVERY)["monthly"]

    # a chart that drops the early years would not match the totals beside it
    assert len(monthly) == 100
    assert monthly[0]["month"] == "2012-01"


def test_a_silent_month_keeps_its_place_on_the_chart(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-05 09:00:00", "one"),
        row(AVERY, "2025-01-10 09:00:00", "two", from_me=1),
    ])

    monthly = preview(csv, AVERY)["monthly"]

    assert [month["month"] for month in monthly] == ["2024-11", "2024-12", "2025-01"]
    assert monthly[1] == {"month": "2024-12", "total": 0, "sent": 0, "received": 0, "sentRatio": 0.0}


def test_the_days_are_the_whole_conversation_whatever_dates_are_asked_for(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-12-30 09:00:00", "one"),
        row(AVERY, "2024-12-30 23:59:00", "two", from_me=1),
        row(AVERY, "2025-01-02 00:00:00", "three"),
    ])

    payload = ui_server.preview_payload(csv, AVERY, MODEL, since="2025-01-01", until="2025-01-31")

    assert [month["month"] for month in payload["monthly"]] == ["2025-01"]
    # the dates are chosen from this calendar, so it does not narrow with them;
    # a day with nothing in it still has its place
    assert payload["daily"] == {"start": "2024-12-30", "counts": [2, 0, 0, 1]}


def test_attachments_are_counted_within_the_dates(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-05 09:00:00", PLACEHOLDER, attachments="image/heic"),
        row(AVERY, "2025-01-10 09:00:00", f"{PLACEHOLDER} from the lake", from_me=1, attachments="image/jpeg|video/mp4"),
    ])

    whole = preview(csv, AVERY)["stats"]["attachments"]
    january = ui_server.preview_payload(csv, AVERY, MODEL, since="2025-01-01", until="2025-01-31")["stats"]["attachments"]

    assert (whole["total"], whole["photos"], whole["videos"]) == (3, 2, 1)
    assert (whole["sent_attachments"], whole["received_attachments"]) == (2, 1)
    # the figures beside a range of dates are for those dates
    assert (january["total"], january["photos"], january["videos"]) == (2, 1, 1)


def local(year, month, day, hour=9, minute=0):
    """A time on the clock of this machine, as Messages keeps it."""
    return datetime(year, month, day, hour, minute).astimezone()


def test_reactions_are_counted_within_the_dates_of_their_messages(tmp_path, saved_names):
    # an export holds no reactions: this archive is pulled from an invented Messages database
    builder = ChatDbBuilder(tmp_path / "chat.db")
    avery = builder.handle(AVERY)
    chat = builder.chat(AVERY, members=[avery])
    first = builder.message(chat, when=local(2024, 11, 5), text="one", handle=avery)
    second = builder.message(chat, when=local(2025, 1, 10), text="two", handle=avery)
    builder.tapback(chat, builder.guid_of(first), 2000, when=local(2024, 11, 5, 9, 1), from_me=True)
    builder.tapback(chat, builder.guid_of(second), 2001, when=local(2025, 1, 10, 9, 1), from_me=True)
    builder.close()
    archive = pulled(tmp_path / "archive.sqlite3", builder.path, NAMES)

    whole = preview(archive, AVERY)["stats"]
    january = ui_server.preview_payload(archive, AVERY, MODEL, since="2025-01-01", until="2025-01-31")["stats"]

    # a tapback is no message
    assert whole["totalMessages"] == 2
    assert (whole["reactions"]["total"], whole["reactions"]["loves"], whole["reactions"]["likes"]) == (2, 1, 1)
    assert whole["reactions"]["sent_reactions"] == 2
    assert (january["reactions"]["total"], january["reactions"]["likes"]) == (1, 1)


def test_the_chart_is_by_day_for_a_short_range_and_by_month_for_a_long_one(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-05 09:00:00", "one"),
        row(AVERY, "2025-01-10 09:00:00", "two", from_me=1),
        row(AVERY, "2025-01-12 09:00:00", "three"),
    ])

    whole = preview(csv, AVERY)["activity"]
    january = ui_server.preview_payload(csv, AVERY, MODEL, since="2025-01-01", until="2025-01-31")["activity"]

    assert whole["unit"] == "month"
    assert [point["key"] for point in whole["points"]] == ["2024-11", "2024-12", "2025-01"]
    assert january["unit"] == "day"
    assert [point["key"] for point in january["points"]] == ["2025-01-10", "2025-01-11", "2025-01-12"]


# ---- a row that does not say who sent it: an archive holds none, the figures on a frame still read one


def frame(rows):
    """Messages as the figures read them, with a direction that can be missing."""
    made = pd.DataFrame(rows, columns=["timestamp", "chat_id", "sender", "is_from_me"])
    return made.assign(timestamp=pd.to_datetime(made["timestamp"]), is_from_me=made["is_from_me"].astype("object"))


def test_a_row_that_does_not_say_who_sent_it_names_no_speaker(saved_names):
    conv = frame([
        ("2025-01-01 09:00:00", GROUP, BLAKE, 0),
        ("2025-01-01 09:01:00", GROUP, UNSAVED, 0),
        # a sent row can carry the handle it went to; with no direction it could be one of those
        ("2025-01-01 09:02:00", GROUP, AVERY, ""),
    ])

    dynamics = ui_server.compute_dynamics(conv)

    assert dynamics["speakerCount"] == 2
    assert {speaker["handle"] for speaker in dynamics["speakers"]} == {BLAKE, UNSAVED}


def test_the_chart_and_the_figures_agree_when_a_row_does_not_say_who_sent_it():
    conv = frame([
        ("2025-01-01 09:00:00", AVERY, AVERY, 0),
        ("2025-01-01 09:01:00", AVERY, AVERY, 1),
        ("2025-01-02 09:00:00", AVERY, AVERY, ""),
    ])

    stats = compute_stats(conv)
    monthly = fill_months(progression_series(conv)).to_dict(orient="records")

    assert (stats.sent_count, stats.received_count, stats.unknown_direction_count) == (1, 1, 1)
    # the bar over the chart reads the figures; a picked day or month reads the series
    for series in (monthly, activity_series(conv)["points"]):
        assert sum(point["sent"] for point in series) == stats.sent_count
        assert sum(point["received"] for point in series) == stats.received_count
        assert sum(point["total"] for point in series) == stats.total_messages


def test_people_and_the_received_figure_count_the_same_rows(saved_names):
    conv = frame([
        ("2025-01-01 09:00:00", GROUP, BLAKE, 0),
        ("2025-01-01 09:01:00", GROUP, UNSAVED, 0),
        ("2025-01-01 09:02:00", GROUP, AVERY, ""),
        ("2025-01-01 09:03:00", GROUP, "", 1),
    ])

    written_by_people = sum(speaker["count"] for speaker in ui_server.compute_dynamics(conv)["speakers"])

    assert written_by_people == compute_stats(conv).received_count == 2


def test_the_figures_of_a_preview_have_two_directions_and_no_third(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "one", sender=BLAKE),
        row(GROUP, "2025-01-01 09:01:00", "two", sender=UNSAVED),
        row(GROUP, "2025-01-02 09:03:00", "three", from_me=1, sender=""),
    ])

    payload = preview(csv, GROUP)
    stats = payload["stats"]

    assert (stats["sentCount"], stats["receivedCount"], stats["unknownDirectionCount"]) == (1, 2, 0)
    assert sum(speaker["count"] for speaker in payload["dynamics"]["speakers"]) == stats["receivedCount"]
    for series in (payload["monthly"], payload["activity"]["points"]):
        assert sum(point["sent"] for point in series) == stats["sentCount"]
        assert sum(point["received"] for point in series) == stats["receivedCount"]
        assert sum(point["total"] for point in series) == stats["totalMessages"]


def test_a_preview_is_never_of_a_conversation_that_only_holds_the_id_asked_for(tmp_path, saved_names):
    csv = loaded(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "are you up"),
        row(AVERY, "2025-01-01 09:01:00", "barely", from_me=1),
    ])

    # the start of Avery's number is not Avery, and is nobody in this export
    with pytest.raises(ValueError, match="No messages found"):
        preview(csv, AVERY[:-1])
    assert preview(csv, AVERY)["stats"]["totalMessages"] == 2


def test_a_large_archive_is_read_with_no_warning_about_its_columns(tmp_path):
    """A column that is empty for many rows and then holds text was read in
    parts, and pandas wrote a warning that went to the lines of the job."""
    import warnings

    from conversation import load_messages

    rows = ["message_id,timestamp,sender,text,chat_id,is_from_me,service,attachment_types"]
    rows += [f"{place},2024-01-01 10:00:00,+15550100,invented text {place},+15550100,1,iMessage," for place in range(1, 260_001)]
    rows += ["260001,2024-01-02 10:00:00,+15550100,with a photo,+15550100,1,iMessage,image/jpeg"]
    path = tmp_path / "messages.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        frame = load_messages(str(path))

    assert len(frame) == 260_001
    assert [str(warning.message)[:40] for warning in raised] == []

