#!/bin/bash
# Full corpus emergent hierarchy discovery
# This script runs the complete iterative discovery pipeline on the entire MIDI corpus
# with optimized CPU settings for 55-core parallelism

set -e  # Exit on error

# Configuration
CORPUS_PATH="/home/arlo/do-repo/midi_generator/midi_corpus/big_band"
OUTPUT_DIR="./full_corpus_discovery_results"
MAX_ITERATIONS=10
MAX_ERROR=0.03
MIN_COMPOSITION_FREQ=5
MAX_COMPOSITIONS_PER_ITER=100
SCALES="64 128 256"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/discovery_full_corpus.log"

# Create output directory
mkdir -p "${RUN_DIR}"

echo "=========================================================================="
echo "FULL CORPUS EMERGENT HIERARCHY DISCOVERY"
echo "=========================================================================="
echo ""
echo "Corpus path:              ${CORPUS_PATH}"
echo "Output directory:         ${RUN_DIR}"
echo "Max iterations:           ${MAX_ITERATIONS}"
echo "Max error threshold:      ${MAX_ERROR}"
echo "Min composition freq:     ${MIN_COMPOSITION_FREQ}"
echo "Max compositions/iter:    ${MAX_COMPOSITIONS_PER_ITER}"
echo "Scales:                   ${SCALES} timesteps"
echo "CPU cores available:      $(nproc)"
echo ""
echo "Starting discovery at: $(date)"
echo ""
echo "=========================================================================="
echo ""

# Count MIDI files in corpus
MIDI_COUNT=$(find "${CORPUS_PATH}" -type f \( -name "*.mid" -o -name "*.midi" \) | wc -l)
echo "Found ${MIDI_COUNT} MIDI files in corpus"
echo ""

# Run discovery with unbuffered output (limit to 50 cores)
python -u scripts/run_emergent_discovery.py \
    --corpus-path "${CORPUS_PATH}" \
    --scales ${SCALES} \
    --max-error ${MAX_ERROR} \
    --max-iterations ${MAX_ITERATIONS} \
    --min-composition-frequency ${MIN_COMPOSITION_FREQ} \
    --max-compositions-per-iteration ${MAX_COMPOSITIONS_PER_ITER} \
    --num-workers 50 \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?

echo ""
echo "=========================================================================="
echo "DISCOVERY COMPLETE"
echo "=========================================================================="
echo ""
echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Results saved to: ${RUN_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# Generate summary statistics
if [ -f "${RUN_DIR}/final_results.json" ]; then
    echo "Final Results Summary:"
    python -c "
import json
with open('${RUN_DIR}/final_results.json') as f:
    results = json.load(f)
    print(f\"  Total files processed: {results['num_files']}\")
    print(f\"  Total iterations: {results['total_iterations']}\")
    print(f\"  Total time: {results['total_time_seconds']/60:.1f} minutes ({results['total_time_seconds']/3600:.1f} hours)\")
    print(f\"  Final transform count: {results['final_transform_count']}\")
    print(f\"  Primitives: {len([t for t in results['final_transforms'] if 'type' not in t])}\")
    print(f\"  Compositions: {len([t for t in results['final_transforms'] if t.get('type') == 'composition'])}\")
    print()
    print('Iteration History:')
    for iter_data in results['iteration_history']:
        print(f\"  Iteration {iter_data['iteration']}: {iter_data['new_compositions']} new compositions, {iter_data['total_derivations']} derivations ({iter_data['time_seconds']/60:.1f} min)\")
" 2>/dev/null || echo "  Could not parse final results"
fi

echo ""
echo "=========================================================================="

exit ${EXIT_CODE}
