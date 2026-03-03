export CUDA_VISIBLE_DEVICES="1"
export HF_HOME=~/models
uvicorn main:app --host 0.0.0.0 --port 8600 --workers 1 > /dev/null 2>&1 &