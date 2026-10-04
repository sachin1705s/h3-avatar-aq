#!/bin/zsh
set -euo pipefail

cd /Users/sachin/h3-avatar-aq

echo "Starting TA2.0 dataset download into Modal volume..."
echo "Project: /Users/sachin/h3-avatar-aq"
echo "Command: modal run modal_h3_avatar.py::download_dataset"
echo

modal run modal_h3_avatar.py::download_dataset

echo
echo "Dataset download command finished. Checking asset status..."
modal run modal_h3_avatar.py::asset_status

echo
echo "Done. You can close this Terminal window."
read -r "?Press Enter to close..."
