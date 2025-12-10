#!/bin/bash

SESSION=logs

# Kill any stray Celery workers
echo "Killing all Celery workers..."

sudo pkill tmux

sudo pkill -f 'celery.*worker'

echo "Stopping all Celery workers for user $USER..."
pgrep -u "$USER" -f 'celery.*worker' | xargs -r kill -9

sleep 1
echo "Remaining Celery processes:"
ps aux | grep '[c]elery.*worker' || echo "✅ No residual Celery workers"

# If session already exists, just attach

# if tmux has-session -t $SESSION 2>/dev/null; then
#   echo "Session already running. Attaching..."
#   tmux attach-session -t $SESSION
#   exit 0
# fi




echo "Launching log monitor..."

sudo pkill uvicorn

# Initialize ACE-Step model before starting server
echo "Initializing ACE-Step model..."
cd /home/arlo/Data

# Set model paths (update these if needed)
export ACE_CHECKPOINT="/mnt/models/epoch=102-step=60000.ckpt"
export ACE_CHECKPOINT_DIR="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"
export ACE_MANIFEST="/home/arlo/Data/final_training_manifest_final.json"

sudo pkill -f 'celery.*worker'

echo "Starting new tmux session: $SESSION"
cd ~/ScoreAI || exit

echo "Restarting Docker containers..."
docker-compose down
docker-compose up -d

echo "Clearing old logs..."
sudo journalctl --rotate
sudo journalctl --vacuum-time=1s

sudo pkill -f 'celery.*worker'

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start ACE-Step FastAPI server with run_fastapi_server.py (pane 0)
tmux new-session -d -s $SESSION -n logs \
  "eval \"\$(conda shell.bash hook)\" && conda activate ace_step && cd /home/arlo/Data && while true; do \
    python3 /home/arlo/Data/run_fastapi_server.py \
      --checkpoint \"$ACE_CHECKPOINT\" \
      --checkpoint_dir \"$ACE_CHECKPOINT_DIR\" \
      --manifest \"$ACE_MANIFEST\" \
      --port 8070 || { echo '❌ FastAPI crashed. Restarting in 3s...'; sleep 3; }; \
  done"

# Split horizontally and start Celery worker (pane 1)
tmux split-window -h -t $SESSION:0 \
  "eval \"\$(conda shell.bash hook)\" && conda activate ace_step && cd /home/arlo/Data && \
  export ACE_CHECKPOINT=\"$ACE_CHECKPOINT\" && \
  export ACE_CHECKPOINT_DIR=\"$ACE_CHECKPOINT_DIR\" && \
  export ACE_MANIFEST=\"$ACE_MANIFEST\" && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6 && \
  export CUDA_LAUNCH_BLOCKING=0 && \
  export PYTORCH_NO_CUDA_MEMORY_CACHING=0 && \
  export OMP_NUM_THREADS=4 && \
  export MKL_NUM_THREADS=4 && \
  echo '🚀 Starting Celery with L4-optimized settings...' && \
  echo '   Model precision: bfloat16' && \
  echo '   DCAE: on GPU (bfloat16)' && \
  echo '   ACE_STEP_PIPELINE: cpu_offload=True' && \
  echo '   Checkpoint loading: mmap=True (low RAM usage)' && \
  echo '   Swap: 10GB available' && \
  CUDA_VISIBLE_DEVICES=0 celery -A genfrominterface.celery_app worker --loglevel=INFO --heartbeat-interval=30 -P solo --concurrency=1 -n worker1@%h || { echo 'Celery crashed. Press Enter.'; read; }"

# Split pane 1 vertically and start Docker logs (pane 2)
tmux split-window -v -t $SESSION:0.1 \
  'cd ~/ScoreAI && docker-compose logs -f || { echo "Docker logs failed. Press Enter."; read; }'

# Split pane 0 vertically and start Auth service (pane 3)
tmux split-window -v -t $SESSION:0.0 \
  "eval \"\$(conda shell.bash hook)\" && conda activate ace_step && cd /home/arlo && while true; do \
    python3 /home/arlo/acauth.py || { echo '❌ Auth service crashed. Restarting in 3s...'; sleep 3; }; \
  done"

# Split pane 3 vertically and start Chatbot service (pane 4)
tmux split-window -v -t $SESSION:0.3 \
  "eval \"\$(conda shell.bash hook)\" && conda activate ace_step && cd /var/www/html && \
  export \$(cat /home/arlo/.env | xargs) && while true; do \
    python3 /var/www/html/chatbot_service.py || { echo '❌ Chatbot service crashed. Restarting in 3s...'; sleep 3; }; \
  done"

# Label the panes
tmux select-pane -t $SESSION:0.0 -T "ACE-Step"
tmux select-pane -t $SESSION:0.1 -T "Celery"
tmux select-pane -t $SESSION:0.2 -T "Docker"
tmux select-pane -t $SESSION:0.3 -T "Auth"
tmux select-pane -t $SESSION:0.4 -T "Chatbot"

# Set tiled layout (equal size panes)
tmux select-layout -t $SESSION:0 tiled
tmux set-option -g mouse on

tmux attach-session -t $SESSION
