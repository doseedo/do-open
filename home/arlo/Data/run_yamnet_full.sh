#!/bin/bash
# Run YAMNet labeling on full vocal training manifest
# This will take approximately 10-11 hours for 32,016 files

set -e

INPUT_MANIFEST="./vocal_training_manifest_with_alternates.json"
OUTPUT_MANIFEST="./vocal_training_manifest_yamnet_labeled.json"
FILTERED_MANIFEST="./vocal_training_manifest_filtered_clean.json"

echo "=========================================="
echo "YAMNet Full Dataset Labeling"
echo "=========================================="
echo ""
echo "Input:  $INPUT_MANIFEST"
echo "Output: $OUTPUT_MANIFEST"
echo ""
echo "Estimated time: 10-11 hours for ~32,000 files"
echo ""

# Check if input exists
if [ ! -f "$INPUT_MANIFEST" ]; then
    echo "Error: Input manifest not found: $INPUT_MANIFEST"
    exit 1
fi

# Step 1: Run YAMNet labeling
echo "Step 1: Running YAMNet labeling..."
echo "  (Use Ctrl+C to stop, resume with --skip_existing)"
echo ""

python yamnet_labeling.py \
    --input_manifest "$INPUT_MANIFEST" \
    --output_manifest "$OUTPUT_MANIFEST" \
    --skip_existing \
    --create_report

echo ""
echo "=========================================="
echo "Step 1 Complete: YAMNet Labeling"
echo "=========================================="
echo ""
echo "Output saved to:"
echo "  - Labels: $OUTPUT_MANIFEST"
echo "  - Review: ${OUTPUT_MANIFEST%.json}_review.txt"
echo ""

# Step 2: Analyze results
echo "Step 2: Analyzing results..."
echo ""

python filter_by_yamnet_labels.py \
    --analyze_only \
    --input "$OUTPUT_MANIFEST"

echo ""
echo "=========================================="
echo "Step 2 Complete: Analysis"
echo "=========================================="
echo ""
echo "Review the report file:"
echo "  cat ${OUTPUT_MANIFEST%.json}_review.txt | less"
echo ""

# Step 3: Ask about filtering
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "To filter the manifest, run:"
echo ""
echo "  # Conservative (clean vocals only):"
echo "  python filter_by_yamnet_labels.py \\"
echo "    --input $OUTPUT_MANIFEST \\"
echo "    --output $FILTERED_MANIFEST \\"
echo "    --exclude_warnings 'music,instrument,noise,static' \\"
echo "    --min_confidence 0.5 \\"
echo "    --require_vocal"
echo ""
echo "  # Moderate (allow some background):"
echo "  python filter_by_yamnet_labels.py \\"
echo "    --input $OUTPUT_MANIFEST \\"
echo "    --output $FILTERED_MANIFEST \\"
echo "    --exclude_warnings 'noise,static,distortion' \\"
echo "    --min_confidence 0.3 \\"
echo "    --require_vocal"
echo ""
echo "  # Interactive (choose options):"
echo "  python filter_by_yamnet_labels.py --interactive"
echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
