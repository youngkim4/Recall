#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${1:-"$ROOT/saves/Recall.app"}"
CONTENTS="$APP_DIR/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
ICONSET="$RESOURCES/RecallIcon.iconset"
ICON_PNG="$RESOURCES/RecallIcon-1024.png"

if [[ -f "$ROOT/app/package.json" && "${RECALL_SKIP_FRONTEND_BUILD:-0}" != "1" ]]; then
  NPM_BIN="${NPM_BIN:-}"
  if [[ -z "$NPM_BIN" ]] && command -v npm >/dev/null 2>&1; then
    NPM_BIN="$(command -v npm)"
  fi
  if [[ -z "$NPM_BIN" && -x "/opt/homebrew/Cellar/node/25.4.0/bin/npm" ]]; then
    NPM_BIN="/opt/homebrew/Cellar/node/25.4.0/bin/npm"
  fi
  if [[ -z "$NPM_BIN" ]]; then
    echo "Could not find npm to build the React UI. Set NPM_BIN=/path/to/npm or RECALL_SKIP_FRONTEND_BUILD=1." >&2
    exit 1
  fi
  NPM_DIR="$(cd "$(dirname "$NPM_BIN")" && pwd)"
  (cd "$ROOT/app" && PATH="$NPM_DIR:$PATH" "$NPM_BIN" run build)
fi

mkdir -p "$MACOS" "$RESOURCES"
rm -f "$MACOS/Recall"

swiftc \
  -O \
  -parse-as-library \
  -framework Cocoa \
  -framework Foundation \
  -framework WebKit \
  "$ROOT/macos/Recall/RecallApp.swift" \
  -o "$MACOS/Recall"

cp "$ROOT/macos/Recall/Info.plist" "$CONTENTS/Info.plist"
printf '%s\n' "$ROOT" > "$RESOURCES/RecallRoot.path"

# pre-build the Contacts exporter so customer machines never need swiftc
# (Contents/Helpers is a sanctioned code-nesting site; Resources is not)
HELPERS="$CONTENTS/Helpers"
mkdir -p "$HELPERS"
EXPORTER_APP="$HELPERS/Recall Contacts Exporter.app"
EXPORTER_MACOS="$EXPORTER_APP/Contents/MacOS"
rm -rf "$EXPORTER_APP" "$RESOURCES/Recall Contacts Exporter.app"
mkdir -p "$EXPORTER_MACOS"
cp "$ROOT/scripts/export_contacts_Info.plist" "$EXPORTER_APP/Contents/Info.plist"
swiftc \
  -O \
  "$ROOT/scripts/export_contacts.swift" \
  -o "$EXPORTER_MACOS/RecallContactsExporter"
chmod +x "$EXPORTER_MACOS/RecallContactsExporter"
codesign --force --deep --sign - "$EXPORTER_APP" 2>/dev/null || true

rm -rf "$ICONSET" "$ICON_PNG" "$RESOURCES/recall-mark.svg.png" "$RESOURCES/RecallIcon.icns"
swift "$ROOT/scripts/render_macos_icon.swift" "$ICON_PNG" >/dev/null
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$ICON_PNG" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$RESOURCES/RecallIcon.icns"
rm -rf "$ICONSET" "$ICON_PNG"

chmod +x "$MACOS/Recall"

echo "Built $APP_DIR"
