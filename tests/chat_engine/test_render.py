from chat_engine.registry import ConversationAliases, EvidenceRegistry
from chat_engine.render import (
    ConversationBook,
    Limits,
    ResultWriter,
    clip_text,
    estimate_tokens,
    message_body,
    render_block,
)
from tests.chat_engine.tiny import GROUP, MAYA_PHONE, UNSAVED_PHONE, tiny_archive, tiny_labeler


def _book() -> ConversationBook:
    return ConversationBook(tiny_archive(), tiny_labeler(), ConversationAliases())


def _by_id(message_id: str) -> dict:
    return tiny_archive().context(message_id, radius=0)[0]


def test_estimate_tokens_is_conservative() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdef") == 2
    assert estimate_tokens("ótimo") >= 2


def test_clip_text_keeps_short_text_whole() -> None:
    assert clip_text("a  b\nc", 100) == "a b c"


def test_clip_text_says_when_it_cuts() -> None:
    text = "x" * 50 + " END"
    shown = clip_text(text, 20)
    assert shown.startswith("x" * 20)
    assert "[cut: 34 more characters]" in shown
    assert "END" not in shown


def test_clip_text_can_centre_on_a_match() -> None:
    text = "a" * 300 + " needle " + "b" * 300
    shown = clip_text(text, 100, focus=301)
    assert "needle" in shown
    assert shown.startswith("[cut: ")
    assert "more characters]" in shown


def test_message_body_for_attachments_system_and_unsent() -> None:
    limits = Limits()
    attachment = _by_id("g5")
    assert message_body(attachment, limits) == "[attachment: image]"
    system = {**_by_id("g1"), "kind": "system", "text": "Maya named the conversation"}
    assert message_body(system, limits) == "[system] Maya named the conversation"
    unsent = {**_by_id("g1"), "unsent": True, "text": ""}
    assert message_body(unsent, limits) == "[unsent message]"
    edited = {**_by_id("g1"), "edited": True}
    assert message_body(edited, limits).endswith("(edited)")
    both = {**_by_id("g1"), "attachments": ["a.mov", "b.jpg", "c.png", "notes.xyz"]}
    assert message_body(both, limits).endswith("[attachments: video, image x2, file]")
    mimes = {**_by_id("g1"), "attachments": ["video/quicktime", "text/vcard", "application/pdf", "audio/amr"]}
    assert message_body(mimes, limits).endswith("[attachments: video, contact card, pdf, audio]")
    unknown = {**_by_id("g1"), "attachments": ["application/octet-stream", "+17205550199.vcf"]}
    assert message_body(unknown, limits).endswith("[attachments: file, contact card]")


def test_book_gives_alias_label_and_title() -> None:
    book = _book()
    assert book.title(MAYA_PHONE) == "c1 · Maya Okafor · direct"
    assert book.title(GROUP) == "c2 · Lisbon crew · group"
    assert book.title(UNSAVED_PHONE) == "c3 · a number ending in 0199 · direct"
    assert book.title("gone") == "c4 · an unsaved contact · direct"
    assert book.alias(MAYA_PHONE) == "c1"


def test_book_reads_each_conversation_once() -> None:
    archive = tiny_archive()
    book = ConversationBook(archive, tiny_labeler(), ConversationAliases())
    book.label(MAYA_PHONE)
    book.title(MAYA_PHONE)
    assert [name for name, _ in archive.calls] == ["conversation"]


def test_render_block_registers_every_line_and_marks_hits() -> None:
    archive = tiny_archive()
    registry = EvidenceRegistry()
    book = ConversationBook(archive, tiny_labeler(), ConversationAliases())
    lines = render_block(
        archive.context("a2", radius=1), book=book, registry=registry, limits=Limits(), marks={"a2": "match"}
    )
    assert lines == [
        "== c1 · Maya Okafor · direct ==",
        "2021-03-04",
        "m1 18:20 You: are we still on for tonight",
        "m2 18:22 Maya Okafor: yes!! the lease got signed btw  <<< match",
        "m3 18:23 You: omg congrats",
    ]
    assert registry.get("m2")["id"] == "a2"


def test_render_block_names_group_senders_and_never_shows_handles() -> None:
    archive = tiny_archive()
    registry = EvidenceRegistry()
    book = ConversationBook(archive, tiny_labeler(), ConversationAliases())
    lines = render_block(archive.context("g3", radius=2), book=book, registry=registry, limits=Limits(), marks={})
    text = "\n".join(lines)
    assert "Jonas Weber: window seat for me please" in text
    assert "a number ending in 0199: can someone forward the lease" in text
    assert "You: I call the aisle" in text
    assert "7205550199" not in text and "+1720" not in text


def test_render_block_starts_a_new_date_line_when_the_day_changes() -> None:
    archive = tiny_archive()
    book = ConversationBook(archive, tiny_labeler(), ConversationAliases())
    lines = render_block(
        archive.context("g4", radius=1), book=book, registry=EvidenceRegistry(), limits=Limits(), marks={}
    )
    assert lines[1] == "2019-06-01"
    assert "2019-06-02" in lines
    assert lines[-1].endswith("[attachment: image]")


def test_result_writer_stops_at_the_budget() -> None:
    writer = ResultWriter(budget_tokens=10, count_tokens=estimate_tokens)
    assert writer.fits("a" * 12)
    writer.add("a" * 12)
    assert not writer.fits("b" * 30)
    writer.add("[cut]", force=True)
    assert writer.text() == "a" * 12 + "\n[cut]"
    assert writer.tokens >= 4
