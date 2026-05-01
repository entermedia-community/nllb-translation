#!/bin/bash
lsof -ti :8600 | xargs -r kill -9

export CUDA_VISIBLE_DEVICES=1
uvicorn main:app \
  --port 8600 \
  --host 0.0.0.0 \
  --workers 1 > /dev/null 2>&1 &