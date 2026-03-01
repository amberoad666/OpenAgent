#!/usr/bin/env bash
# Install OpenAgent and add to PATH

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Add to PATH via shell profile
SHELL_RC=""
if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "$(which zsh 2>/dev/null)" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ] || [ "$SHELL" = "$(which bash 2>/dev/null)" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    EXPORT_LINE="export PATH=\"$SCRIPT_DIR:\$PATH\""
    if ! grep -qF "$SCRIPT_DIR" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# OpenAgent" >> "$SHELL_RC"
        echo "$EXPORT_LINE" >> "$SHELL_RC"
        echo "Added $SCRIPT_DIR to PATH in $SHELL_RC"
        echo "Run: source $SHELL_RC  (or restart terminal)"
    else
        echo "Already in PATH ($SHELL_RC)"
    fi
else
    echo "Could not detect shell profile. Add this to your shell config:"
    echo "  export PATH=\"$SCRIPT_DIR:\$PATH\""
fi

echo ""
echo "Done! Run: open-agent  or  oa"
