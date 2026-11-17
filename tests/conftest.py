import atexit
import os
import shutil
import socket
import sqlite3
import tempfile
import threading

# These three lines come before any import of the project.
# - The key is not a key, whatever the shell or a .env file holds.
# - "~" is an empty folder: setup_status and contact_names work out where
#   Messages and Contacts are from it when they are imported.
_HOME = tempfile.mkdtemp(prefix="recall-tests-home-")
atexit.register(shutil.rmtree, _HOME, ignore_errors=True)
os.environ.update({"OPENAI_API_KEY": "test-key-not-real", "HOME": _HOME})

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

THIS_MACHINE = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def stay_on_this_machine(monkeypatch):
    """No test reaches another host: not an AI provider, not anything else."""
    connect = socket.socket.connect
    look_up = socket.getaddrinfo

    def refuse(host):
        raise ConnectionRefusedError(f"a test tried to reach {host}; tests stay on this machine")

    def guarded_connect(sock, address):
        # an address that is not a pair is a socket file on this machine
        if isinstance(address, tuple) and address[0] not in THIS_MACHINE:
            refuse(address[0])
        return connect(sock, address)

    def guarded_look_up(host, *args, **kwargs):
        if host is not None and host not in THIS_MACHINE:
            refuse(host)
        return look_up(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_look_up)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def no_saved_names(tmp_path, monkeypatch):
    """No name comes from the Contacts cache of this Mac."""
    import ui_server

    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", tmp_path / "contact_names.json")
    monkeypatch.setattr(ui_server, "resolve_contact_names", lambda handles, **_: {})


@pytest.fixture
def messages_home(tmp_path, monkeypatch):
    """A folder that stands in for ~/Library/Messages. The database in it is `chat.db`."""
    import setup_status
    import ui_server

    home = tmp_path / "Messages"
    home.mkdir()
    home = home.resolve()
    monkeypatch.setattr(setup_status, "LIVE_MESSAGES_DIR", home)
    monkeypatch.setattr(setup_status, "LIVE_CHAT_DB", home / "chat.db")
    monkeypatch.setattr(ui_server, "LIVE_CHAT_DB", home / "chat.db")
    return home


@pytest.fixture
def archive(tmp_path):
    """An archive that is already there: its three files and what is in them."""
    folder = tmp_path / "archive"
    folder.mkdir()
    files = {
        folder / "messages.csv": (
            "message_id,timestamp,sender,text,chat_id,is_from_me,service\n"
            "1,2020-01-01 00:00:00,me,kept,+15550100,1,iMessage\n"
        ),
        folder / "messages_attachments.csv": "message_id,filename\n1,kept.jpg\n",
        folder / "messages_reactions.csv": "reaction_id,reaction_type\n1,kept\n",
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    return files


@pytest.fixture
def marker(tmp_path, monkeypatch):
    """The file that records how far setup got."""
    import ui_server

    saves = tmp_path / "saves"
    monkeypatch.setattr(ui_server, "SAVES_DIR", saves)
    monkeypatch.setattr(ui_server, "SETUP_MARKER_PATH", saves / "setup.json")
    return saves / "setup.json"


@pytest.fixture
def no_jobs():
    """The list of jobs is empty before a test and after it."""
    import ui_server

    def clear():
        with ui_server.JOBS_LOCK:
            ui_server.JOBS.clear()

    clear()
    yield
    clear()


@pytest.fixture
def fresh_bundle_cache():
    """Empty the parsed-export cache around a test, so each one reads its own CSV."""
    import ui_server

    blank = {"sig": None, "df": None, "catalog": None, "names": None}
    with ui_server._ASK_CACHE_LOCK:
        ui_server._ASK_CACHE.update(blank)
    yield
    with ui_server._ASK_CACHE_LOCK:
        ui_server._ASK_CACHE.update(blank)


@pytest.fixture
def server():
    """The app's own request handler on a loopback port, for a test of a route."""
    import ui_server

    try:
        httpd = ui_server.LoopbackHTTPServer(("127.0.0.1", 0), ui_server.RecallHandler)
    except OSError as exc:
        pytest.skip(f"no loopback socket here: {exc}")
    # a short wait between looks, so the server stops soon after a test ends
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture
def sample_messages_df():
    """Minimal messages DataFrame for testing."""
    return pd.DataFrame({
        "message_id": [1, 2, 3, 4, 5],
        "timestamp": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-01-15 10:05:00",
            "2024-02-20 14:30:00",
            "2024-03-10 09:00:00",
            "2024-03-10 09:01:00",
        ]),
        "sender": ["them", "them", "them", "them", "them"],
        "text": ["hello", "how are you", "long time no see", "let's meet", "ok cool"],
        "chat_id": ["+1234", "+1234", "+1234", "+1234", "+1234"],
        "is_from_me": [0, 1, 0, 1, 0],
        "service": ["iMessage"] * 5,
    })


@pytest.fixture
def sample_messages_csv(tmp_path, sample_messages_df):
    """Write sample messages to a CSV and return the path."""
    csv_path = tmp_path / "messages.csv"
    sample_messages_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_attachments_df():
    return pd.DataFrame({
        "message_id": [1, 2, 3],
        "chat_id": ["+1234", "+1234", "+1234"],
        "filename": ["photo.jpg", "video.mp4", "doc.pdf"],
        "mime_type": ["image/jpeg", "video/mp4", "application/pdf"],
        "total_bytes": [1024, 2048, 512],
        "is_from_me": [1, 0, 1],
        "timestamp": ["2024-01-15 10:00", "2024-01-15 10:05", "2024-02-20 14:30"],
        "category": ["photo", "video", "document"],
    })


@pytest.fixture
def sample_attachments_csv(tmp_path, sample_attachments_df):
    csv_path = tmp_path / "messages_attachments.csv"
    sample_attachments_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_reactions_df():
    return pd.DataFrame({
        "reaction_id": [1, 2, 3],
        "associated_message_guid": ["g1", "g2", "g3"],
        "associated_message_type": [2000, 2001, 2003],
        "is_from_me": [1, 0, 1],
        "chat_id": ["+1234", "+1234", "+1234"],
        "timestamp": ["2024-01-15 10:00", "2024-01-15 10:05", "2024-02-20 14:30"],
        "reaction_type": ["love", "like", "laugh"],
        "is_add": [True, True, True],
    })


@pytest.fixture
def sample_reactions_csv(tmp_path, sample_reactions_df):
    csv_path = tmp_path / "messages_reactions.csv"
    sample_reactions_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_chat_db(tmp_path):
    """Create a minimal in-memory-like SQLite DB mimicking chat.db schema."""
    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("""CREATE TABLE handle (
        ROWID INTEGER PRIMARY KEY,
        id TEXT
    )""")
    c.execute("""CREATE TABLE chat (
        ROWID INTEGER PRIMARY KEY,
        chat_identifier TEXT
    )""")
    c.execute("""CREATE TABLE message (
        ROWID INTEGER PRIMARY KEY,
        date INTEGER,
        handle_id INTEGER,
        text TEXT,
        attributedBody BLOB,
        is_from_me INTEGER,
        service TEXT,
        associated_message_guid TEXT,
        associated_message_type INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE chat_message_join (
        chat_id INTEGER,
        message_id INTEGER
    )""")
    c.execute("""CREATE TABLE attachment (
        ROWID INTEGER PRIMARY KEY,
        filename TEXT,
        mime_type TEXT,
        total_bytes INTEGER
    )""")
    c.execute("""CREATE TABLE message_attachment_join (
        message_id INTEGER,
        attachment_id INTEGER
    )""")

    # Insert test data
    # Apple epoch: 2001-01-01. We use nanoseconds.
    # 2024-01-15 10:00:00 UTC = 726915600 seconds since 2001-01-01
    base_ns = 726915600 * 1_000_000_000

    c.execute("INSERT INTO handle VALUES (1, '+15551234567')")
    c.execute("INSERT INTO chat VALUES (1, '+15551234567')")

    c.execute("INSERT INTO message VALUES (1, ?, 1, 'hello there', NULL, 0, 'iMessage', NULL, 0)",
              (base_ns,))
    c.execute("INSERT INTO message VALUES (2, ?, 1, 'hey!', NULL, 1, 'iMessage', NULL, 0)",
              (base_ns + 60_000_000_000,))
    c.execute("INSERT INTO message VALUES (3, ?, 1, NULL, NULL, 0, 'iMessage', 'guid1', 2000)",
              (base_ns + 120_000_000_000,))

    c.execute("INSERT INTO chat_message_join VALUES (1, 1)")
    c.execute("INSERT INTO chat_message_join VALUES (1, 2)")
    c.execute("INSERT INTO chat_message_join VALUES (1, 3)")

    c.execute("INSERT INTO attachment VALUES (1, 'photo.jpg', 'image/jpeg', 1024)")
    c.execute("INSERT INTO message_attachment_join VALUES (1, 1)")

    conn.commit()
    conn.close()
    return str(db_path)
