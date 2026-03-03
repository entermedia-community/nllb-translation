export CUDA_VISIBLE_DEVICES="1"
export HF_HOME=~/models
uvicorn main:app --port 7600 > /dev/null 2>&1 &