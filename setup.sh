#!/usr/bin/env bash
# Setup script for the Draft Picks as Options pipeline.
# Creates the draft_options virtual environment and installs dependencies.
# Usage: ./setup.sh

set -e  # exit on any error

VENV_NAME="draft_options"

# Check python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Install Python 3 first."
    exit 1
fi

# Create the venv if it doesn't already exist
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
else
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
"$VENV_NAME/bin/pip" install --upgrade pip
if ! "$VENV_NAME/bin/pip" install -r requirements.txt; then
    echo "Standard install failed (often an 'externally managed environment' issue)."
    echo "Retrying with --break-system-packages..."
    "$VENV_NAME/bin/pip" install -r requirements.txt --break-system-packages
fi

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $VENV_NAME/bin/activate        (Mac/Linux)"
echo "  $VENV_NAME\\Scripts\\activate           (Windows)"
echo ""
echo "Then run the pipeline with:"
echo "  python run_pipeline_v7.3.py"
