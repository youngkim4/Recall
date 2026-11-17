import json
import os
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from analysis import (
    sanitize_filename,
    load_messages,
    load_attachments,
    load_reactions,
    filter_conversation,
    compute_stats,
    compute_attachment_stats,
    compute_reaction_stats,
    progression_series,
    format_all_messages,
    estimate_tokens,
    truncate_to_tokens,
    chunk_by_year,
    get_token_budget,
    _parse_json_events,
    write_markdown_report,
    TOKEN_BUDGET,
    SECONDS_PER_DAY,
)
from conversation import find_conversation


class TestSanitizeFilename:
    def test_phone_number(self):
        assert sanitize_filename("+12165551234") == "+12165551234"

    def test_strips_special_chars(self):
        assert sanitize_filename("hello world!@#") == "helloworld"

    def test_empty_returns_conversation(self):
        assert sanitize_filename("!@#$%") == "conversation"

    def test_preserves_hyphens_underscores(self):
        assert sanitize_filename("my-chat_2024") == "my-chat_2024"


class TestLoadMessages:
    def test_loads_csv(self, sample_messages_csv):
        df = load_messages(sample_messages_csv)
        assert len(df) == 5
        assert "timestamp" in df.columns

    def test_timestamp_parsed(self, sample_messages_csv):
        df = load_messages(sample_messages_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_since_filter(self, sample_messages_csv):
        from datetime import datetime
        df = load_messages(sample_messages_csv, since=datetime(2024, 2, 1))
        assert len(df) == 3  # Feb + 2x Mar

    def test_until_filter(self, sample_messages_csv):
        from datetime import datetime
        df = load_messages(sample_messages_csv, until=datetime(2024, 1, 31))
        assert len(df) == 2  # 2x Jan

    def test_until_includes_the_whole_of_that_day(self, sample_messages_csv):
        from datetime import datetime
        # February's message was sent at 14:30 on the 20th
        df = load_messages(sample_messages_csv, until=datetime(2024, 2, 20))
        assert len(df) == 3  # 2x Jan + Feb

    def test_missing_timestamp_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["hi"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_messages(str(csv))

    def test_adds_missing_columns(self, tmp_path):
        csv = tmp_path / "minimal.csv"
        pd.DataFrame({
            "timestamp": ["2024-01-01 10:00"],
            "text": ["hi"],
        }).to_csv(csv, index=False)
        df = load_messages(str(csv))
        assert "is_from_me" in df.columns
        assert "service" in df.columns


class TestLoadAttachments:
    def test_loads_existing(self, tmp_path, sample_attachments_df):
        csv = tmp_path / "messages.csv"
        csv.touch()
        att_csv = tmp_path / "messages_attachments.csv"
        sample_attachments_df.to_csv(att_csv, index=False)
        df = load_attachments(str(csv))
        assert len(df) == 3

    def test_returns_empty_when_missing(self, tmp_path):
        csv = tmp_path / "messages.csv"
        csv.touch()
        df = load_attachments(str(csv))
        assert df.empty


class TestLoadReactions:
    def test_loads_existing(self, tmp_path, sample_reactions_df):
        csv = tmp_path / "messages.csv"
        csv.touch()
        react_csv = tmp_path / "messages_reactions.csv"
        sample_reactions_df.to_csv(react_csv, index=False)
        df = load_reactions(str(csv))
        assert len(df) == 3

    def test_returns_empty_when_missing(self, tmp_path):
        csv = tmp_path / "messages.csv"
        csv.touch()
        df = load_reactions(str(csv))
        assert df.empty


class TestFilterConversation:
    def test_exact_match(self, sample_messages_df):
        result = filter_conversation(sample_messages_df, "+1234")
        assert len(result) == 5

    def test_part_of_an_id_is_no_conversation(self, sample_messages_df):
        # text that an id only holds would show one conversation as another
        assert filter_conversation(sample_messages_df, "1234").empty
        assert filter_conversation(sample_messages_df, "+123").empty

    def test_messages_sent_at_the_same_time_keep_their_order(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-15 10:00:00"] * 40),
            "text": [f"m{index}" for index in range(40)],
            "chat_id": ["+1234"] * 40,
        })
        assert filter_conversation(df, "+1234")["text"].tolist() == [f"m{index}" for index in range(40)]

    def test_no_match(self, sample_messages_df):
        result = filter_conversation(sample_messages_df, "+9999")
        assert result.empty

    def test_sorted_by_timestamp(self, sample_messages_df):
        result = filter_conversation(sample_messages_df, "+1234")
        timestamps = result["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_missing_chat_id_raises(self):
        df = pd.DataFrame({"text": ["hi"]})
        with pytest.raises(ValueError, match="chat_id"):
            filter_conversation(df, "+1234")


class TestFindConversation:
    IDS = pd.Series(["+15550000001", "+15550000001", "+1555", "sam99@example.com", None, "chat4471"], dtype="string")

    def test_a_whole_id_is_that_conversation(self):
        assert find_conversation(self.IDS, "+15550000001") == "+15550000001"

    def test_a_whole_id_wins_over_the_ids_that_hold_it(self):
        assert find_conversation(self.IDS, "+1555") == "+1555"

    def test_text_that_one_id_holds_is_that_conversation(self):
        # a number typed with no country code, or the start of an address
        assert find_conversation(self.IDS, "5550000001") == "+15550000001"
        assert find_conversation(self.IDS, " sam99 ") == "sam99@example.com"

    def test_text_that_several_ids_hold_is_refused(self):
        with pytest.raises(ValueError, match="2 conversations"):
            find_conversation(self.IDS, "555")

    @pytest.mark.parametrize("wanted", ["+19990000000", "", "   ", None])
    def test_text_that_no_id_holds_is_refused(self, wanted):
        with pytest.raises(ValueError):
            find_conversation(self.IDS, wanted)


class TestComputeStats:
    def test_basic_stats(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.total_messages == 5
        assert stats.sent_count == 2
        assert stats.received_count == 3

    def test_empty_df(self):
        empty = pd.DataFrame(columns=["timestamp", "text", "chat_id", "is_from_me"])
        stats = compute_stats(empty)
        assert stats.total_messages == 0
        assert stats.active_days == 0

    def test_active_days(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.active_days == 3  # Jan 15, Feb 20, Mar 10

    def test_busiest_day(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.busiest_day_count == 2  # Jan 15 or Mar 10

    def test_longest_gap(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.longest_gap_days > 0

    def test_with_attachments(self, sample_messages_df, sample_attachments_df):
        stats = compute_stats(sample_messages_df, sample_attachments_df)
        assert stats.attachments.total == 3
        assert stats.attachments.photos == 1

    def test_with_reactions(self, sample_messages_df, sample_reactions_df):
        stats = compute_stats(sample_messages_df, reactions_df=sample_reactions_df)
        assert stats.reactions.total == 3


class TestComputeAttachmentStats:
    def test_basic(self, sample_attachments_df):
        stats = compute_attachment_stats(sample_attachments_df, "+1234")
        assert stats.total == 3
        assert stats.photos == 1
        assert stats.videos == 1
        assert stats.documents == 1

    def test_sent_received(self, sample_attachments_df):
        stats = compute_attachment_stats(sample_attachments_df, "+1234")
        assert stats.sent_attachments == 2
        assert stats.received_attachments == 1

    def test_empty_df(self):
        stats = compute_attachment_stats(pd.DataFrame(), "+1234")
        assert stats.total == 0

    def test_no_match(self, sample_attachments_df):
        stats = compute_attachment_stats(sample_attachments_df, "+9999")
        assert stats.total == 0


class TestComputeReactionStats:
    def test_basic(self, sample_reactions_df):
        stats = compute_reaction_stats(sample_reactions_df, "+1234")
        assert stats.total == 3
        assert stats.loves == 1
        assert stats.likes == 1
        assert stats.laughs == 1

    def test_sent_received(self, sample_reactions_df):
        stats = compute_reaction_stats(sample_reactions_df, "+1234")
        assert stats.sent_reactions == 2
        assert stats.received_reactions == 1

    def test_empty_df(self):
        stats = compute_reaction_stats(pd.DataFrame(), "+1234")
        assert stats.total == 0


class TestProgressionSeries:
    def test_returns_monthly(self, sample_messages_df):
        result = progression_series(sample_messages_df)
        assert "month" in result.columns
        assert "total" in result.columns
        assert "sent" in result.columns
        assert "received" in result.columns
        assert "sent_ratio" in result.columns

    def test_month_count(self, sample_messages_df):
        result = progression_series(sample_messages_df)
        assert len(result) == 3  # Jan, Feb, Mar


class TestFormatAllMessages:
    def test_format(self, sample_messages_df):
        result = format_all_messages(sample_messages_df)
        assert "ME:" in result
        assert "THEM:" in result
        assert "hello" in result

    def test_missing_is_from_me(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
            "text": ["hi"],
        })
        result = format_all_messages(df)
        assert "??:" in result


class TestEstimateTokens:
    def test_basic(self):
        tokens = estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        result = truncate_to_tokens("Hello", 100)
        assert result == "Hello"

    def test_truncates_long_text(self):
        long_text = "word " * 1000
        result = truncate_to_tokens(long_text, 10)
        assert estimate_tokens(result) <= 10

    def test_empty_returns_empty(self):
        assert truncate_to_tokens("", 100) == ""

    def test_zero_tokens(self):
        assert truncate_to_tokens("Hello", 0) == ""


class TestChunkByYear:
    def test_single_year(self, sample_messages_df):
        chunks = chunk_by_year(sample_messages_df)
        assert len(chunks) == 1
        label, df, text = chunks[0]
        assert label == "2024"
        assert len(df) == 5

    def test_multi_year_packs_small_years_together(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2023-06-01 10:00",
                "2024-01-15 10:00",
                "2024-06-15 10:00",
            ]),
            "text": ["a", "b", "c"],
            "is_from_me": [0, 1, 0],
            "chat_id": ["+1234"] * 3,
        })
        # default budget: consecutive small years pack into one chunk so the
        # model sees the cross-year arc in a single pass
        chunks = chunk_by_year(df)
        assert [c[0] for c in chunks] == ["2023 – 2024"]
        assert sum(len(cdf) for _, cdf, _ in chunks) == 3

        # budget that fits each year alone but not both: years stay separate
        year_tokens = [
            estimate_tokens(format_all_messages(df[df["timestamp"].dt.year == y]))
            for y in (2023, 2024)
        ]
        split = chunk_by_year(df, max_tokens=max(year_tokens))
        assert [c[0] for c in split] == ["2023", "2024"]

    def test_all_messages_included(self, sample_messages_df):
        chunks = chunk_by_year(sample_messages_df)
        total = sum(len(df) for _, df, _ in chunks)
        assert total == 5


class TestParseJsonEvents:
    def test_valid_json_object(self):
        content = json.dumps({"events": [{"date": "2024-01-15", "title": "Test"}]})
        result = _parse_json_events(content)
        assert len(result) == 1

    def test_valid_json_array(self):
        content = json.dumps([{"date": "2024-01-15", "title": "Test"}])
        result = _parse_json_events(content)
        assert len(result) == 1

    def test_markdown_fenced(self):
        content = "```json\n" + json.dumps({"events": [{"date": "2024-01-15"}]}) + "\n```"
        result = _parse_json_events(content)
        assert len(result) == 1

    def test_invalid_json(self):
        result = _parse_json_events("not json at all")
        assert result == []

    def test_missing_events_key(self):
        content = json.dumps({"data": []})
        result = _parse_json_events(content)
        assert result == []


class TestWriteMarkdownReport:
    def test_creates_file(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        path = write_markdown_report(str(tmp_path), "+1234", stats, monthly, "Test summary")
        assert os.path.exists(path)

    def test_content_includes_stats(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        path = write_markdown_report(str(tmp_path), "+1234", stats, monthly, "Test summary")
        content = Path(path).read_text()
        assert "+1234" in content
        assert "Test summary" in content
        assert "5" in content  # total messages

    def test_with_events(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        events = pd.DataFrame({
            "date": ["2024-01-15"],
            "title": ["First contact"],
            "detail": ["They said hello"],
            "category": ["milestone"],
            "score": [0.9],
            "quote": ["hello"],
        })
        path = write_markdown_report(str(tmp_path), "+1234", stats, monthly, "Summary", events)
        content = Path(path).read_text()
        assert "First contact" in content
        assert "Key Events" in content


class TestConstants:
    def test_token_budget(self):
        assert TOKEN_BUDGET == 170_000

    def test_latest_model_budget_uses_long_context(self):
        assert get_token_budget("gpt-5.5") > TOKEN_BUDGET

    def test_mini_budget_does_not_match_larger_family(self):
        assert get_token_budget("gpt-5.4-mini") < get_token_budget("gpt-5.4")

    def test_seconds_per_day(self):
        assert SECONDS_PER_DAY == 86_400.0


class TestMonthsOnACalendar:
    def test_a_month_with_no_messages_is_filled_in(self):
        from conversation import fill_months, progression_series

        conv = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-15 10:00", "2024-03-10 09:00", "2024-03-11 09:00"]),
            "text": ["a", "b", "c"],
            "is_from_me": [1, 0, 1],
        })

        filled = fill_months(progression_series(conv))

        assert [str(month.date()) for month in filled["month"]] == ["2024-01-01", "2024-02-01", "2024-03-01"]
        assert filled["total"].tolist() == [1, 0, 2]
        assert filled["sent"].tolist() == [1, 0, 1]
        assert filled["received"].tolist() == [0, 0, 1]

    def test_day_totals_run_from_the_first_day_to_the_last(self):
        from conversation import day_totals

        conv = pd.DataFrame({"timestamp": pd.to_datetime(["2023-12-30 23:59", "2024-01-01 00:00", "2024-01-01 08:00", None])})

        assert day_totals(conv) == {"start": "2023-12-30", "counts": [1, 0, 2]}

    def test_no_messages_means_no_days(self):
        from conversation import day_totals

        assert day_totals(pd.DataFrame({"timestamp": pd.to_datetime([])})) == {"start": "", "counts": []}


class TestAttachmentsAndReactionsFollowTheDates:
    def test_attachments_are_narrowed_to_the_days_asked_for(self, tmp_path, sample_attachments_csv):
        from datetime import datetime
        from conversation import load_attachments

        loaded = load_attachments(str(tmp_path / "messages.csv"), since=datetime(2024, 2, 1))

        assert loaded["filename"].tolist() == ["doc.pdf"]

    def test_reactions_include_the_whole_of_the_last_day(self, tmp_path, sample_reactions_csv):
        from datetime import datetime
        from conversation import load_reactions

        loaded = load_reactions(str(tmp_path / "messages.csv"), until=datetime(2024, 1, 15))

        assert loaded["reaction_type"].tolist() == ["love", "like"]

    def test_with_no_dates_everything_is_loaded(self, tmp_path, sample_attachments_csv, sample_reactions_csv):
        from conversation import load_attachments, load_reactions

        assert len(load_attachments(str(tmp_path / "messages.csv"))) == 3
        assert len(load_reactions(str(tmp_path / "messages.csv"))) == 3

    def test_a_phone_number_keeps_its_plus_sign(self, tmp_path):
        from conversation import load_attachments

        pd.DataFrame({"chat_id": ["+15550000001"], "category": ["photo"]}).to_csv(
            tmp_path / "messages_attachments.csv", index=False
        )

        assert load_attachments(str(tmp_path / "messages.csv"))["chat_id"].tolist() == ["+15550000001"]


class TestActivitySeries:
    def test_a_short_stretch_is_counted_by_day(self):
        from conversation import activity_series

        conv = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-30 10:00", "2024-01-30 11:00", "2024-02-01 09:00"]),
            "text": ["a", "b", "c"],
            "is_from_me": [1, 0, 1],
        })

        assert activity_series(conv) == {
            "unit": "day",
            "points": [
                {"key": "2024-01-30", "total": 2, "sent": 1, "received": 1},
                {"key": "2024-01-31", "total": 0, "sent": 0, "received": 0},
                {"key": "2024-02-01", "total": 1, "sent": 1, "received": 0},
            ],
        }

    def test_a_long_stretch_is_counted_by_month(self):
        from conversation import activity_series

        conv = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-15 10:00", "2024-04-10 09:00"]),
            "text": ["a", "b"],
            "is_from_me": [1, 0],
        })

        series = activity_series(conv)

        assert series["unit"] == "month"
        assert [point["key"] for point in series["points"]] == ["2024-01", "2024-02", "2024-03", "2024-04"]
        assert series["points"][0] == {"key": "2024-01", "total": 1, "sent": 1, "received": 0}

    def test_the_longest_stretch_counted_by_day_is_two_months(self):
        from conversation import ACTIVITY_DAYS_LIMIT, activity_series

        def spanning(days):
            return pd.DataFrame({
                "timestamp": [pd.Timestamp("2024-01-01 09:00"), pd.Timestamp("2024-01-01 09:00") + pd.Timedelta(days=days - 1)],
                "text": ["a", "b"],
                "is_from_me": [1, 0],
            })

        assert activity_series(spanning(ACTIVITY_DAYS_LIMIT))["unit"] == "day"
        assert activity_series(spanning(ACTIVITY_DAYS_LIMIT + 1))["unit"] == "month"

    def test_no_messages_is_an_empty_series(self):
        from conversation import activity_series

        empty = pd.DataFrame({"timestamp": pd.to_datetime([]), "text": [], "is_from_me": []})

        assert activity_series(empty) == {"unit": "month", "points": []}


class TestDirectionMasks:
    VALUES = [1, 0, 1.0, 0.0, "1", "0", "1.0", "0.0", "true", "false", "T", "F", "yes", "no", "y", "n", None, "", "maybe"]

    def test_the_masks_agree_with_the_rule_for_one_value(self):
        from conversation import parse_boolish, received_mask, sent_mask

        series = pd.Series(self.VALUES, dtype="object")
        parsed = [parse_boolish(value) for value in self.VALUES]

        assert sent_mask(series).tolist() == [not pd.isna(value) and value == 1 for value in parsed]
        assert received_mask(series).tolist() == [not pd.isna(value) and value == 0 for value in parsed]

    def test_a_value_that_does_not_say_is_neither_sent_nor_received(self):
        from conversation import received_mask, sent_mask

        series = pd.Series([None, "", "maybe"], dtype="object")

        assert not sent_mask(series).any()
        assert not received_mask(series).any()


class TestARowThatDoesNotSayWhoSentIt:
    """A row with no direction adds to the total and to neither voice, in every count."""

    @staticmethod
    def conversation(stamps):
        # each stamp holds one sent, one received and one row that does not say
        rows = [(stamp, direction) for stamp in stamps for direction in (1, 0, None)]
        return pd.DataFrame({
            "timestamp": pd.to_datetime([stamp for stamp, _ in rows]),
            "text": ["x"] * len(rows),
            "chat_id": ["+1234"] * len(rows),
            "is_from_me": pd.Series([direction for _, direction in rows], dtype="object"),
        })

    def test_months_count_it_in_the_total_only(self):
        monthly = progression_series(self.conversation(["2024-01-15 10:00", "2024-03-02 09:00"]))

        assert monthly["total"].tolist() == [3, 3]
        assert monthly["sent"].tolist() == [1, 1]
        assert monthly["received"].tolist() == [1, 1]

    def test_month_points_count_it_in_the_total_only(self):
        from conversation import month_points

        points = month_points(self.conversation(["2024-01-15 10:00", "2024-03-02 09:00"]))

        assert points == [
            {"key": "2024-01", "total": 3, "sent": 1, "received": 1},
            {"key": "2024-02", "total": 0, "sent": 0, "received": 0},
            {"key": "2024-03", "total": 3, "sent": 1, "received": 1},
        ]

    def test_day_points_count_it_in_the_total_only(self):
        from conversation import activity_series

        series = activity_series(self.conversation(["2024-01-30 10:00", "2024-02-01 09:00"]))

        assert series == {
            "unit": "day",
            "points": [
                {"key": "2024-01-30", "total": 3, "sent": 1, "received": 1},
                {"key": "2024-01-31", "total": 0, "sent": 0, "received": 0},
                {"key": "2024-02-01", "total": 3, "sent": 1, "received": 1},
            ],
        }

    def test_the_series_add_up_to_the_figures_beside_them(self):
        from conversation import activity_series, month_points

        conv = self.conversation(["2024-01-15 10:00", "2024-01-16 10:00", "2024-06-02 09:00"])
        stats = compute_stats(conv)

        assert (stats.sent_count, stats.received_count, stats.unknown_direction_count) == (3, 3, 3)
        for points in (month_points(conv), activity_series(conv)["points"]):
            assert sum(point["sent"] for point in points) == stats.sent_count
            assert sum(point["received"] for point in points) == stats.received_count
            assert sum(point["total"] for point in points) == stats.total_messages

    def test_attachments_count_it_in_the_total_only(self):
        attachments = pd.DataFrame({
            "chat_id": ["+1234"] * 4,
            "category": ["photo"] * 4,
            "is_from_me": pd.Series([1, 0, 0, None], dtype="object"),
        })

        stats = compute_attachment_stats(attachments, "+1234")

        assert (stats.total, stats.sent_attachments, stats.received_attachments) == (4, 1, 2)

    def test_reactions_count_it_in_the_total_only(self):
        reactions = pd.DataFrame({
            "chat_id": ["+1234"] * 4,
            "reaction_type": ["love"] * 4,
            "is_add": [True] * 4,
            "is_from_me": pd.Series([1, 1, 0, None], dtype="object"),
        })

        stats = compute_reaction_stats(reactions, "+1234")

        assert (stats.total, stats.sent_reactions, stats.received_reactions) == (4, 2, 1)

    def test_a_blank_read_back_from_a_file_is_a_row_that_does_not_say(self, tmp_path):
        csv = tmp_path / "messages.csv"
        csv.write_text(
            "message_id,timestamp,sender,text,chat_id,is_from_me\n"
            "1,2024-01-15 10:00:00,+1234,a,+1234,1\n"
            "2,2024-01-15 10:01:00,+1234,b,+1234,0\n"
            "3,2024-01-15 10:02:00,+1234,c,+1234,\n",
            encoding="utf-8",
        )

        monthly = progression_series(load_messages(str(csv)))

        assert monthly[["total", "sent", "received"]].values.tolist() == [[3, 1, 1]]
