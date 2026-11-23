#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -x "$SCRIPT_DIR/bin/amie" ]]; then
    AMIE_ROOT="$SCRIPT_DIR"
else
    AMIE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
fi

ADDON_BASE="${XDG_DATA_HOME:-$HOME/.local/share}/amie/addOn"
ADDON_DIR="$ADDON_BASE/atoi4onnx"
REPO_URL="https://github.com/samging/atoi4onnx.git"

run() {
    if ! command -v git >/dev/null 2>&1; then
        echo "git is not installed" >&2
        exit 1
    fi

    echo "git found at: $(command -v git)"
    mkdir -p "$ADDON_BASE"

    if [[ -d "$ADDON_DIR/.git" ]]; then
        git -C "$ADDON_DIR" pull --ff-only
    else
        git clone "$REPO_URL" "$ADDON_DIR"
    fi

    bash "$ADDON_DIR/dummy.sh" "$AMIE_ROOT"
    #cat /tmp/aoti\-setup.log
}

run
