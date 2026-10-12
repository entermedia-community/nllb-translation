#!/bin/bash

# Uvicorn Translation launcher script for Docker container
# All configurations are loaded from .env file in the same directory as this script or in /root/.env

# Kill any process running on port 8600
lsof -ti :8600 | xargs -r kill -9

#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES=0

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi


LOGFILE=/root/logs/uvicorn
mkdir -p "$LOGFILE"

uvicorn main:app \
  --port 8600 \
  --host 0.0.0.0 \
  --workers 1 \
  2>&1 | multilog t s5000000 n3 "$LOGFILE" &

deactivate