# PyInstaller spec for the bundled Recall server (onedir, nested later into
# Recall.app/Contents/Helpers/RecallServer.app by scripts/build_release_app.sh).
#
# Build: venv/bin/pyinstaller packaging/recall_server.spec --noconfirm

from pathlib import Path

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent

# the frozen server serves these ROOT-relative, read-only assets
BUNDLE_DATAS = [
    (str(ROOT / "app" / "dist"), "app/dist"),
    (str(ROOT / "ui"), "ui"),
    # the facts of the offered models: recall_services/models.py reads this file from beside itself
    (str(ROOT / "recall_services" / "models.json"), "recall_services"),
    # the facts of the embedding models: meaning_index/model_facts.py reads this file from beside itself
    (str(ROOT / "meaning_index" / "models.json"), "meaning_index"),
]


def fetch_vocabularies(folder: Path) -> Path:
    """The vocabularies of tiktoken that the server uses, fetched at build time into a new folder.
    The app ships them (server_entry.use_bundled_vocabulary): with no network, a first start works."""
    import os
    import shutil

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(folder)
    import tiktoken

    # conversation.TOKENIZER_NAMES and meaning_index.tokens.ENCODING
    for name in ("o200k_base", "cl100k_base"):
        try:
            tiktoken.get_encoding(name)
        except Exception as exc:
            raise SystemExit(f"could not fetch the tiktoken vocabulary {name}: {type(exc).__name__}") from exc
    return folder


BUNDLE_DATAS = [*BUNDLE_DATAS, (str(fetch_vocabularies(ROOT / "build" / "tiktoken-cache")), "tiktoken_cache")]

# tripwire: the repo root holds the founder's personal iMessage archive in
# dev mode -- a careless add-data here would ship it to every downloader
FORBIDDEN_PARTS = (
    "messages.csv",
    "messages_attachments",
    "messages_reactions",
    "memories.csv",
    "chat.db",
    "/saves",
    "/out",
    ".env",
)
for src, _dest in BUNDLE_DATAS:
    lowered = src.lower()
    for part in FORBIDDEN_PARTS:
        if part in lowered:
            raise SystemExit(f"privacy tripwire: refusing to bundle {src}")

a = Analysis(
    [str(SPEC_DIR / "server_entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=BUNDLE_DATAS,
    hiddenimports=[
        # tiktoken registers encodings via a namespace plugin PyInstaller misses;
        # without these the frozen build dies with "Unknown encoding cl100k_base"
        "tiktoken_ext",
        "tiktoken_ext.openai_public",
    ],
    excludes=["tkinter", "test", "unittest", "pydoc_data"],
    noarchive=False,
)

# second tripwire: nothing personal may slip in via hooks either
for entry in list(a.datas):
    dest = str(entry[0]).lower()
    src = str(entry[1]).lower()
    for part in FORBIDDEN_PARTS:
        if part in dest or part in src:
            raise SystemExit(f"privacy tripwire: collected data entry {entry[:2]}")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RecallServer",
    console=True,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="RecallServer",
)

app = BUNDLE(
    coll,
    name="RecallServer.app",
    bundle_identifier="app.recall.mac.server",
    info_plist={
        "LSUIElement": True,
        "LSMinimumSystemVersion": "13.0",
        "NSHumanReadableCopyright": "Recall",
    },
)
