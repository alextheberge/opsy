#!/bin/bash

# Find all Go files in the project
find . -name "*.go" -type f | while read -r file; do
  # Replace all occurrences of the original repository path with the fork's repository path
  sed -i '' 's|github.com/datolabs-io/opsy|github.com/alextheberge/opsy|g' "$file"
done

echo "Import paths updated successfully."
