#!/bin/bash
# Mute Translator Pipeline
# Run each step sequentially, testing between steps

set -e  # Exit on error

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
EVAL_DIR="${SCRIPT_DIR}/eval_output"
SYNTHETIC_DIR="${SCRIPT_DIR}/synthetic_data"
STUDENT_DIR="${SCRIPT_DIR}/student_checkpoints"

# Config
MANIFEST="/home/arlo/Data.backup/final_training_manifest_brass_only.json"
DCAE_CHECKPOINT="/home/arlo/Data/ACE-Step/checkpoints"

mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR" "$SYNTHETIC_DIR" "$STUDENT_DIR"

echo "=============================================="
echo "MUTE TRANSLATOR PIPELINE"
echo "=============================================="
echo ""
echo "Steps:"
echo "  1. Train latent translator (dry → muted)"
echo "  2. Evaluate translator"
echo "  3. Generate synthetic muted data"
echo "  4. Train student model"
echo ""

# Step 1: Train translator
step1_train() {
    echo "=============================================="
    echo "STEP 1: Training Latent Translator"
    echo "=============================================="

    python "${SCRIPT_DIR}/train_translator.py" \
        --manifest "$MANIFEST" \
        --output_dir "$CHECKPOINT_DIR" \
        --model_type small \
        --batch_size 16 \
        --learning_rate 1e-4 \
        --num_epochs 100 \
        --window_frames 128 \
        --samples_per_epoch 5000 \
        --device cuda

    echo ""
    echo "Translator training complete!"
    echo "Checkpoint: ${CHECKPOINT_DIR}/best.pt"
    echo ""
}

# Step 2: Evaluate translator
step2_evaluate() {
    echo "=============================================="
    echo "STEP 2: Evaluating Translator"
    echo "=============================================="

    python "${SCRIPT_DIR}/evaluate_translator.py" \
        --checkpoint "${CHECKPOINT_DIR}/best.pt" \
        --dcae_checkpoint "$DCAE_CHECKPOINT" \
        --manifest "$MANIFEST" \
        --output_dir "$EVAL_DIR" \
        --num_samples 10 \
        --device cuda

    echo ""
    echo "Evaluation complete!"
    echo "Results: ${EVAL_DIR}/evaluation_results.json"
    echo "Audio samples: ${EVAL_DIR}/"
    echo ""
    echo ">>> LISTEN TO THE SAMPLES BEFORE PROCEEDING <<<"
    echo ""
}

# Step 3: Generate synthetic data
step3_generate() {
    echo "=============================================="
    echo "STEP 3: Generating Synthetic Muted Data"
    echo "=============================================="

    python "${SCRIPT_DIR}/generate_synthetic.py" \
        --checkpoint "${CHECKPOINT_DIR}/best.pt" \
        --dcae_checkpoint "$DCAE_CHECKPOINT" \
        --manifest "$MANIFEST" \
        --output_dir "$SYNTHETIC_DIR" \
        --save_latents \
        --skip_existing \
        --device cuda

    echo ""
    echo "Synthetic data generation complete!"
    echo "Manifest: ${SYNTHETIC_DIR}/synthetic_manifest.json"
    echo ""
}

# Step 4: Train student
step4_train_student() {
    echo "=============================================="
    echo "STEP 4: Training Student Model"
    echo "=============================================="

    python "${SCRIPT_DIR}/train_student.py" \
        --synthetic_manifest "${SYNTHETIC_DIR}/synthetic_manifest.json" \
        --output_dir "$STUDENT_DIR" \
        --model_type mel \
        --batch_size 16 \
        --learning_rate 1e-4 \
        --num_epochs 100 \
        --device cuda

    echo ""
    echo "Student training complete!"
    echo "Checkpoint: ${STUDENT_DIR}/best.pt"
    echo ""
}

# Test inference
test_inference() {
    echo "=============================================="
    echo "Testing Inference"
    echo "=============================================="

    # Test teacher mode
    echo "Testing teacher mode..."
    python "${SCRIPT_DIR}/inference.py" \
        --mode teacher \
        --translator_checkpoint "${CHECKPOINT_DIR}/best.pt" \
        --dcae_checkpoint "$DCAE_CHECKPOINT" \
        --input "$1" \
        --output "${SCRIPT_DIR}/test_teacher_output.wav" \
        --device cuda

    # Test student mode (if available)
    if [ -f "${STUDENT_DIR}/best.pt" ]; then
        echo "Testing student mode..."
        python "${SCRIPT_DIR}/inference.py" \
            --mode student \
            --student_checkpoint "${STUDENT_DIR}/best.pt" \
            --input "$1" \
            --output "${SCRIPT_DIR}/test_student_output.wav" \
            --device cuda
    fi

    echo ""
    echo "Test complete!"
}

# Main
case "${1:-}" in
    "1"|"train")
        step1_train
        ;;
    "2"|"evaluate")
        step2_evaluate
        ;;
    "3"|"generate")
        step3_generate
        ;;
    "4"|"student")
        step4_train_student
        ;;
    "test")
        if [ -z "$2" ]; then
            echo "Usage: $0 test <input_audio.wav>"
            exit 1
        fi
        test_inference "$2"
        ;;
    "all")
        step1_train
        step2_evaluate
        echo ""
        echo ">>> Review eval results, then run: $0 3 <<<"
        ;;
    *)
        echo "Usage: $0 <step>"
        echo ""
        echo "Steps:"
        echo "  1 or train    - Train latent translator"
        echo "  2 or evaluate - Evaluate translator (generates audio samples)"
        echo "  3 or generate - Generate synthetic muted data"
        echo "  4 or student  - Train student model"
        echo "  test <file>   - Test inference on a file"
        echo "  all           - Run steps 1-2, then pause for review"
        echo ""
        echo "Recommended workflow:"
        echo "  1. Run: $0 1       # Train translator"
        echo "  2. Run: $0 2       # Evaluate (listen to outputs)"
        echo "  3. If satisfied, run: $0 3  # Generate synthetic data"
        echo "  4. Run: $0 4       # Train student"
        echo "  5. Run: $0 test input.wav  # Test final model"
        ;;
esac
