#!/usr/bin/env bash
# Build the self-contained, distributable Recall.app into dist/.
#
# Unlike scripts/build_macos_app.sh (dev shell pointing at the repo checkout),
# this bundles the Python server with PyInstaller so a stranger's Mac needs
# no repo, no venv, and no Xcode tools. Signing: ad-hoc by default; set
# RECALL_SIGN_IDENTITY="Developer ID Application: ..." for a real identity.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
APP_DIR="$DIST/Recall.app"
CONTENTS="$APP_DIR/Contents"
MACOS="$CONTENTS/MacOS"
HELPERS="$CONTENTS/Helpers"
RESOURCES="$CONTENTS/Resources"
PYI_DIST="$ROOT/build/pyinstaller-dist"
SIGN_IDENTITY="${RECALL_SIGN_IDENTITY:--}"

# 1. frontend
if [[ "${RECALL_SKIP_FRONTEND_BUILD:-0}" != "1" ]]; then
  NPM_BIN="${NPM_BIN:-}"
  if [[ -z "$NPM_BIN" ]] && command -v npm >/dev/null 2>&1; then
    NPM_BIN="$(command -v npm)"
  fi
  if [[ -z "$NPM_BIN" && -x "/opt/homebrew/Cellar/node/25.4.0/bin/npm" ]]; then
    NPM_BIN="/opt/homebrew/Cellar/node/25.4.0/bin/npm"
  fi
  if [[ -z "$NPM_BIN" ]]; then
    echo "Could not find npm. Set NPM_BIN or RECALL_SKIP_FRONTEND_BUILD=1." >&2
    exit 1
  fi
  NPM_DIR="$(cd "$(dirname "$NPM_BIN")" && pwd)"
  (cd "$ROOT/app" && PATH="$NPM_DIR:$PATH" "$NPM_BIN" run build)
fi

# 2. frozen server (collects app/dist + ui into its own bundle)
"$ROOT/venv/bin/pyinstaller" "$ROOT/packaging/recall_server.spec" \
  --noconfirm \
  --distpath "$PYI_DIST" \
  --workpath "$ROOT/build/pyinstaller-work"

# 3. shell binary
rm -rf "$APP_DIR"
mkdir -p "$MACOS" "$HELPERS" "$RESOURCES"
swiftc \
  -O \
  -parse-as-library \
  -framework Cocoa \
  -framework Foundation \
  -framework WebKit \
  "$ROOT/macos/Recall/RecallApp.swift" \
  -o "$MACOS/Recall"
chmod +x "$MACOS/Recall"
cp "$ROOT/macos/Recall/Info.plist" "$CONTENTS/Info.plist"

# 4. nest the frozen server (complete .app in a sanctioned nesting site)
cp -R "$PYI_DIST/RecallServer.app" "$HELPERS/RecallServer.app"

# 5. contacts exporter helper
EXPORTER_APP="$HELPERS/Recall Contacts Exporter.app"
mkdir -p "$EXPORTER_APP/Contents/MacOS"
cp "$ROOT/scripts/export_contacts_Info.plist" "$EXPORTER_APP/Contents/Info.plist"
swiftc -O "$ROOT/scripts/export_contacts.swift" \
  -o "$EXPORTER_APP/Contents/MacOS/RecallContactsExporter"
chmod +x "$EXPORTER_APP/Contents/MacOS/RecallContactsExporter"

# 6. icon
ICONSET="$RESOURCES/RecallIcon.iconset"
ICON_PNG="$RESOURCES/RecallIcon-1024.png"
swift "$ROOT/scripts/render_macos_icon.swift" "$ICON_PNG" >/dev/null
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$RESOURCES/RecallIcon.icns"
rm -rf "$ICONSET" "$ICON_PNG"

# 7. privacy tripwire: a release bundle must never contain personal data
if find "$APP_DIR" \( -name "messages*.csv" -o -name "chat.db" -o -name "memories.csv" -o -name "*.env" \) | grep -q .; then
  echo "privacy tripwire: personal data found inside $APP_DIR -- aborting" >&2
  exit 1
fi

# 8. sign inside-out (ad-hoc unless RECALL_SIGN_IDENTITY is set)
xattr -cr "$APP_DIR"
codesign --force --sign "$SIGN_IDENTITY" "$EXPORTER_APP"
codesign --force --deep --sign "$SIGN_IDENTITY" "$HELPERS/RecallServer.app"
codesign --force --sign "$SIGN_IDENTITY" "$APP_DIR"
codesign --verify --strict "$APP_DIR"

echo "Built $APP_DIR ($(du -sh "$APP_DIR" | cut -f1))"