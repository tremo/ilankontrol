#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
DEPS_STAMP="$VENV_DIR/.deps_installed"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

MISSING_MODULES=0
for mod in streamlit langchain langchain_openai langchain_anthropic langchain_google_genai; do
  if ! python3 -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)"; then
    MISSING_MODULES=1
    break
  fi
done

if [ ! -f "$DEPS_STAMP" ] || [ "$SCRIPT_DIR/requirements.txt" -nt "$DEPS_STAMP" ] || [ "$MISSING_MODULES" -eq 1 ]; then
  echo "Installing dependencies..."
  python3 -m pip install --upgrade pip
  python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"
  touch "$DEPS_STAMP"
fi

echo "Starting simulation UI..."
streamlit run "$SCRIPT_DIR/streamlit_app.py"
