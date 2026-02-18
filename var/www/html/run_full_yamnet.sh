#!/bin/bash
# Run YAMNet on FULL vocal training manifest

INPUT="./vocal_training_manifest_with_alternates.json"
OUTPUT="./vocal_training_manifest_yamnet_labeled_FULL.json"

echo "Running YAMNet on full dataset..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo ""

# Check input exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input not found: $INPUT"
    exit 1
fi

# Get total count
TOTAL=$(python3 -c "import json; print(len(json.load(open('$INPUT'))))")
echo "Total entries: $TOTAL"
echo "Estimated time: ~10-11 hours"
echo ""

read -p "Start full YAMNet labeling? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Run YAMNet (NO --max_files flag!)
python yamnet_labeling.py \
    --input_manifest "$INPUT" \
    --output_manifest "$OUTPUT" \
    --skip_existing \
    --create_report

echo ""
echo "Done! Full YAMNet labeling complete."
echo "Output: $OUTPUT"
