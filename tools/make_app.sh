#!/bin/sh
# Crea ~/Applications/Crystal.app: doppio clic (o Spotlight → "Crystal") = pannello di controllo.
# L'app va fuori dal progetto: sul disco esterno (exFAT) macOS non riesce a firmarla.
set -e
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HOME/Applications}/Crystal.app"
TMP="$(mktemp -d)"
sed "s|__PROJECT__|$PROJECT|" "$PROJECT/tools/Crystal.applescript" > "$TMP/Crystal.applescript"
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
osacompile -o "$DEST" "$TMP/Crystal.applescript"
rm -rf "$TMP"
echo "✅ $DEST  →  progetto: $PROJECT"
