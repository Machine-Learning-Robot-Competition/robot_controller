#!/bin/bash

# Prompt for base directory or use the provided argument
read -p "Enter your ROS workspace root directory: " input_dir
BASE_DIR=${input_dir:-$HOME}  # Use $HOME if no input is provided

# Directory name to search for
DIR_NAME="robot_controller"

# Find the first matching directory path
FOLDER_PATH=$(find "$BASE_DIR" -type d -name "$DIR_NAME" 2>/dev/null | head -n 1)

if [ -z "$FOLDER_PATH" ]; then
    echo "Directory named '$DIR_NAME' not found under $BASE_DIR."
    exit 1
fi

# Specify the subdirectory to be added to PYTHONPATH (e.g., a subfolder within 'src')
SUB_DIR="src"
SUB_FOLDER_PATH="$FOLDER_PATH/$SUB_DIR"

# Check if the subdirectory exists
if [ ! -d "$SUB_FOLDER_PATH" ]; then
    echo "Directory '$SUB_DIR' not found under $FOLDER_PATH."
    exit 1
fi

# Print the found subdirectory and ask for confirmation
echo "Found directory: $SUB_FOLDER_PATH"
read -p "Do you want to add this to your PYTHONPATH in .bashrc? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
    echo "Operation cancelled by user."
    exit 0
fi

# Check if the line is already in .bashrc
if ! grep -q "export PYTHONPATH=\"\$PYTHONPATH:$SUB_FOLDER_PATH\"" ~/.bashrc; then
    # Add the line to .bashrc
    echo "export PYTHONPATH=\"\$PYTHONPATH:$SUB_FOLDER_PATH\"" >> ~/.bashrc
    echo "Added $SUB_FOLDER_PATH to PYTHONPATH in .bashrc"
else
    echo "$SUB_FOLDER_PATH is already in PYTHONPATH"
fi

# Source .bashrc to apply changes immediately (optional)
source ~/.bashrc