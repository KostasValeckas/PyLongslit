#!/bin/bash

# a bash script for copying a list of files to a directory
# useful in preparing reductions
# courtesy of ChatGPT

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <file_list> <output_directory>"
  exit 1
fi

# Input parameters
file_list=$1
output_directory=$2

# Check if the file list exists
if [ ! -f "$file_list" ]; then
  echo "Error: File list '$file_list' does not exist."
  exit 1
fi

# Check if the output directory exists, if not create it
if [ ! -d "$output_directory" ]; then
  echo "Output directory '$output_directory' does not exist. Creating it."
  mkdir -p "$output_directory"
fi

# Loop through each line in the file list
while IFS= read -r file; do
  # Check if the file exists
  if [ -f "$file" ]; then
    # Copy the file to the output directory
    cp "$file" "$output_directory"
    echo "Copied '$file' to '$output_directory'"
  else
    echo "Warning: '$file' does not exist. Skipping."
  fi
done < "$file_list"

echo "All files copied."
