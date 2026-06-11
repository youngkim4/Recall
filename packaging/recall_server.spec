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
]

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
