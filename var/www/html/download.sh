#!/bin/bash

INPUT_FILE="all_audio_paths2.txt"
OUTPUT_DIR="/Volumes/My Passport"

# Loop through each line in the input file
while IFS= read -r line; do
    if [[ "$line" == gs://* ]]; then
        echo "⬇️  Downloading: $line"

        # Extract just the filename
        filename=$(basename "$line")

        # Use quotes around the output path to handle spaces
        gsutil cp "$line" "$OUTPUT_DIR/$filename"
    else
        echo "⚠️  Skipping invalid line: $line"
    fi
done < "$INPUT_FILE"

echo "✅ All downloads completed to: $OUTPUT_DIR"
