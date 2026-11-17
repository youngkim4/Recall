"""Tests for the conversation preview the Analyze view reads."""
import pandas as pd
import pytest

import ui_server

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")

COLUMNS = ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me", "attachment_types"]
PLACEHOLDER = "￼"
AVERY = "+15550000001"
BLAKE = "+15550000002"
UNSAVED = "+15550000009"
GROUP = "chat77"
NAMES = {AVERY: "Avery Stone", BLAKE: "Blake Reyes"}


def write_export(path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["message_id"] = range(1, len(frame) + 1)
    frame.to_csv(path, index=False)
    return path


def row(chat_id, timestamp, text, from_me=0, sender=None, attachments=""):
    return (0, timestamp, sender if sender is not None else chat_id, text, chat_id, "", from_me, attachments)


@pytest.fixture
def saved_names(monkeypatch):
    monkeypatch.setattr(
        ui_server,
        "resolve_contact_names",
        lambda handles, **_: {handle: NAMES[handle] for handle in handles if handle in NAMES},
    )


def preview(csv, contact):
    return ui_server.preview_payload(csv, contact, "gpt-5.4-mini")


def test_recent_messages_say_who_sent_them(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "are you up"),
        row(AVERY, "2025-01-01 09:01:00", "barely", from_me=1),
    ])

    recent = preview(csv, AVERY)["recentMessages"]

    assert [message["text"] for message in recent] == ["are you up", "barely"]
    assert [message["isFromMe"] for message in recent] == [False, True]
    # in a conversation with one person the name is already on the page
    assert [message["senderName"] for message in recent] == ["", ""]


def test_group_messages_carry_the_name_of_who_wrote_them(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "who is driving", sender=BLAKE),
        row(GROUP, "2025-01-01 09:01:00", "not me", sender=UNSAVED),
        row(GROUP, "2025-01-01 09:02:00", "I can", from_me=1, sender=""),
    ])

    recent = preview(csv, GROUP)["recentMessages"]

    assert [message["senderName"] for message in recent] == ["Blake Reyes", "", ""]
    # an unsaved number is passed along so the interface can label it its own way
    assert [message["sender"] for message in recent] == [BLAKE, UNSAVED, ""]


def test_recent_messages_never_show_the_attachment_placeholder(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", PLACEHOLDER, attachments="image/heic"),
        row(AVERY, "2025-01-01 09:01:00", f"{PLACEHOLDER} from the lake", attachments="image/heic|image/heic"),
        row(AVERY, "2025-01-01 09:02:00", "no picture here"),
    ])

    recent = preview(csv, AVERY)["recentMessages"]

    assert [message["text"] for message in recent] == ["", "from the lake", "no picture here"]
    assert [message["attachment"] for message in recent] == ["Photo", "2 photos", ""]


def test_a_long_recent_message_is_marked_as_cut(tmp_path, saved_names):
    text = " ".join(["word"] * 200)
    csv = write_export(tmp_path / "messages.csv", [row(AVERY, "2025-01-01 09:00:00", text)])

    (message,) = preview(csv, AVERY)["recentMessages"]

    assert message["text"].endswith("word…")
    assert len(message["text"]) <= ui_server.RECENT_MESSAGE_LIMIT + 1


def test_dynamics_do_not_count_silence_up_to_today(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2020-01-01 09:00:00", "hello"),
        row(AVERY, "2020-01-02 09:00:00", "hi", from_me=1),
    ])

    # how long a conversation has been quiet depends on where the archive ends,
    # which the interface knows; a count of days up to today would be wrong
    assert "quietDays" not in preview(csv, AVERY)["dynamics"]


def test_speakers_are_the_people_who_wrote_in_a_group(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
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
    csv = write_export(tmp_path / "messages.csv", [
        row(GROUP, f"2025-01-01 09:{index:02d}:00", "here", sender=handle)
        for index, handle in enumerate(handles)
    ])

    dynamics = preview(csv, GROUP)["dynamics"]

    # a place in the list is only true when the whole list is there to count from
    assert dynamics["speakerCount"] == 12
    assert len(dynamics["speakers"]) == 12


def test_a_conversation_with_one_person_has_no_speaker_list(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "hello"),
        row(AVERY, "2025-01-01 09:01:00", "hi", from_me=1),
    ])

    assert "speakers" not in preview(csv, AVERY)["dynamics"]


def test_the_last_day_of_a_range_is_part_of_it(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-30 09:00:00", "before"),
        row(AVERY, "2025-01-31 23:30:00", "late on the last day"),
        row(AVERY, "2025-02-01 00:10:00", "the day after"),
    ])

    payload = ui_server.preview_payload(csv, AVERY, "gpt-5.4-mini", since="2025-01-01", until="2025-01-31")

    assert payload["stats"]["totalMessages"] == 2


def test_the_months_of_a_long_conversation_are_all_there(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, f"{2012 + index // 12}-{index % 12 + 1:02d}-15 09:00:00", "hi")
        for index in range(100)
    ])

    monthly = preview(csv, AVERY)["monthly"]

    # a chart that drops the early years would not match the totals beside it
    assert len(monthly) == 100
    assert monthly[0]["month"] == "2012-01"


def test_a_silent_month_keeps_its_place_on_the_chart(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-05 09:00:00", "one"),
        row(AVERY, "2025-01-10 09:00:00", "two", from_me=1),
    ])

    monthly = preview(csv, AVERY)["monthly"]

    assert [month["month"] for month in monthly] == ["2024-11", "2024-12", "2025-01"]
    assert monthly[1] == {"month": "2024-12", "total": 0, "sent": 0, "received": 0, "sentRatio": 0.0}


def test_the_days_are_the_whole_conversation_whatever_dates_are_asked_for(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-12-30 09:00:00", "one"),
        row(AVERY, "2024-12-30 23:59:00", "two", from_me=1),
        row(AVERY, "2025-01-02 00:00:00", "three"),
    ])

    payload = ui_server.preview_payload(csv, AVERY, "gpt-5.4-mini", since="2025-01-01", until="2025-01-31")

    assert [month["month"] for month in payload["monthly"]] == ["2025-01"]
    # the dates are chosen from this calendar, so it does not narrow with them;
    # a day with nothing in it still has its place
    assert payload["daily"] == {"start": "2024-12-30", "counts": [2, 0, 0, 1]}


def test_attachments_and_reactions_are_counted_within_the_dates(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-05 09:00:00", "one"),
        row(AVERY, "2025-01-10 09:00:00", "two"),
    ])
    pd.DataFrame({
        "message_id": [1, 2],
        "chat_id": [AVERY, AVERY],
        "category": ["photo", "photo"],
        "is_from_me": [0, 0],
        "timestamp": ["2024-11-05 09:00:00", "2025-01-10 09:00:00"],
    }).to_csv(tmp_path / "messages_attachments.csv", index=False)
    pd.DataFrame({
        "reaction_id": [1, 2],
        "chat_id": [AVERY, AVERY],
        "reaction_type": ["love", "love"],
        "is_add": [True, True],
        "is_from_me": [1, 1],
        "timestamp": ["2024-11-05 09:01:00", "2025-01-10 09:01:00"],
    }).to_csv(tmp_path / "messages_reactions.csv", index=False)

    whole = preview(csv, AVERY)["stats"]
    january = ui_server.preview_payload(csv, AVERY, "gpt-5.4-mini", since="2025-01-01", until="2025-01-31")["stats"]

    assert whole["attachments"]["photos"] == 2
    assert whole["reactions"]["total"] == 2
    # the figures beside a range of dates are for those dates
    assert january["attachments"]["photos"] == 1
    assert january["reactions"]["total"] == 1


def test_the_chart_is_by_day_for_a_short_range_and_by_month_for_a_long_one(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-05 09:00:00", "one"),
        row(AVERY, "2025-01-10 09:00:00", "two", from_me=1),
        row(AVERY, "2025-01-12 09:00:00", "three"),
    ])

    whole = preview(csv, AVERY)["activity"]
    january = ui_server.preview_payload(csv, AVERY, "gpt-5.4-mini", since="2025-01-01", until="2025-01-31")["activity"]

    assert whole["unit"] == "month"
    assert [point["key"] for point in whole["points"]] == ["2024-11", "2024-12", "2025-01"]
    assert january["unit"] == "day"
    assert [point["key"] for point in january["points"]] == ["2025-01-10", "2025-01-11", "2025-01-12"]


def test_a_row_that_does_not_say_who_sent_it_names_no_speaker(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "one", sender=BLAKE),
        row(GROUP, "2025-01-01 09:01:00", "two", sender=UNSAVED),
        # a sent row can carry the handle it went to; with no direction it could be one of those
        row(GROUP, "2025-01-01 09:02:00", "three", from_me="", sender=AVERY),
    ])

    dynamics = preview(csv, GROUP)["dynamics"]

    assert dynamics["speakerCount"] == 2
    assert {speaker["handle"] for speaker in dynamics["speakers"]} == {BLAKE, UNSAVED}


def test_the_chart_and_the_figures_agree_when_a_row_does_not_say_who_sent_it(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(AVERY, "2025-01-01 09:00:00", "one"),
        row(AVERY, "2025-01-01 09:01:00", "two", from_me=1),
        row(AVERY, "2025-01-02 09:00:00", "three", from_me=""),
    ])

    payload = preview(csv, AVERY)
    stats = payload["stats"]

    assert (stats["sentCount"], stats["receivedCount"], stats["unknownDirectionCount"]) == (1, 1, 1)
    # the bar over the chart reads the figures; a picked day or month reads the series
    for series in (payload["monthly"], payload["activity"]["points"]):
        assert sum(point["sent"] for point in series) == stats["sentCount"]
        assert sum(point["received"] for point in series) == stats["receivedCount"]
        assert sum(point["total"] for point in series) == stats["totalMessages"]


def test_people_and_the_received_figure_count_the_same_rows(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "one", sender=BLAKE),
        row(GROUP, "2025-01-01 09:01:00", "two", sender=UNSAVED),
        row(GROUP, "2025-01-01 09:02:00", "three", from_me="", sender=AVERY),
        row(GROUP, "2025-01-01 09:03:00", "four", from_me=1, sender=""),
    ])

    payload = preview(csv, GROUP)

    written_by_people = sum(speaker["count"] for speaker in payload["dynamics"]["speakers"])
    assert written_by_people == payload["stats"]["receivedCount"] == 2


def test_a_preview_is_never_of_a_conversation_that_only_holds_the_id_asked_for(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
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

