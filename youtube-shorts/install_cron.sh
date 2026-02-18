#!/bin/bash
# Install/update the cron job for YouTube Shorts auto-uploader

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

# Parse upload_time from config.yaml
UPLOAD_TIME=$(grep 'upload_time:' "$CONFIG" | sed 's/.*: *"\?\([0-9]*:[0-9]*\)"\?.*/\1/')
HOUR=$(echo "$UPLOAD_TIME" | cut -d: -f1)
MINUTE=$(echo "$UPLOAD_TIME" | cut -d: -f2)

CRON_CMD="$MINUTE $HOUR * * * cd $SCRIPT_DIR && /usr/bin/python3 $SCRIPT_DIR/uploader.py >> $SCRIPT_DIR/upload.log 2>&1"
CRON_TAG="# youtube-shorts-uploader"

# Remove existing entry if present, then add new one
(crontab -l 2>/dev/null | grep -v "$CRON_TAG") | { cat; echo "$CRON_CMD $CRON_TAG"; } | crontab -

echo "Cron job installed: $CRON_CMD"
echo "Uploads will run daily at $UPLOAD_TIME UTC."
