#!/bin/bash
# Process manifest in batches for better control and monitoring

INPUT_MANIFEST="${1:-./vocal_training_manifest_with_alternates.json}"
OUTPUT_MANIFEST="${2:-./vocal_training_manifest_yamnet_labeled.json}"
BATCH_SIZE="${3:-1000}"

echo "=========================================="
echo "YAMNet Batch Processor"
echo "=========================================="
echo ""
echo "Input:      $INPUT_MANIFEST"
echo "Output:     $OUTPUT_MANIFEST"
echo "Batch size: $BATCH_SIZE files"
echo ""

# Get total files
TOTAL=$(python3 -c "import json; print(len(json.load(open('$INPUT_MANIFEST'))))")
echo "Total files: $TOTAL"
echo ""

# Calculate number of batches
NUM_BATCHES=$(( ($TOTAL + $BATCH_SIZE - 1) / $BATCH_SIZE ))
echo "Number of batches: $NUM_BATCHES"
echo ""

# Ask for confirmation
read -p "Start processing? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Process in batches
for ((batch=1; batch<=NUM_BATCHES; batch++)); do
    START=$(( ($batch - 1) * $BATCH_SIZE ))
    END=$(( $batch * $BATCH_SIZE ))

    if [ $END -gt $TOTAL ]; then
        END=$TOTAL
    fi

    echo ""
    echo "=========================================="
    echo "Batch $batch/$NUM_BATCHES: Processing files $START-$END"
    echo "=========================================="
    echo ""

    # Run YAMNet on this batch
    python yamnet_labeling.py \
        --input_manifest "$INPUT_MANIFEST" \
        --output_manifest "$OUTPUT_MANIFEST" \
        --skip_existing \
        --max_files $END

    # Show progress
    PROCESSED=$(python3 -c "
import json
data = json.load(open('$OUTPUT_MANIFEST'))
processed = sum(1 for e in data if e.get('yamnet_labels', {}).get('status'))
print(processed)
")

    PERCENT=$(( $PROCESSED * 100 / $TOTAL ))

    echo ""
    echo "Progress: $PROCESSED/$TOTAL ($PERCENT%)"
    echo ""

    # Brief pause between batches
    sleep 1
done

echo ""
echo "=========================================="
echo "All batches complete!"
echo "=========================================="
echo ""
echo "Final results:"
python filter_by_yamnet_labels.py --analyze_only --input "$OUTPUT_MANIFEST"
