export CUDA_VISIBLE_DEVICES="1"
export HF_HOME=~/models
uvicorn main:app --port 8600 > /dev/null 2>&1 &