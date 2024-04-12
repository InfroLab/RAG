# Preparing conda
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/Anaconda.sh
bash ~/Anaconda.sh -b -p
conda init

# Installing package dependencies
conda env create --name rag_ss --file environment.yml
pip install fastapi==0.110.1
pip install peft==0.10.0
pip install tensorrt-llm==0.8.0 --extra-index-url https://pypi.nvidia.com

# Preparing apts and downloading necessary files
sudo apt-get install git
sudo apt-get install tar 
git clone --branch v0.8.0 https://github.com/NVIDIA/TensorRT-LLM.git:TRT-LLM
cd ./model
git clone --remote=git@hf.co:Athroniaeth/mistral-7b-v0.2-trtllm-int4 
export TOKENIZERS_PARALLELISM=false
trtllm-build --checkpoint_dir ./mistral-7b-v0.2-trtllm-int4/1-gpu --output_dir ../engine --gemm_plugin float16 --max_input_len 32256
cd ..
python -m run.py --max_output_len=50 --tokenizer_dir ./model/mistral-7b-v0.2-trtllm-int4/Mistral-7B-Instruct-v0.2 --engine_dir=./engine --max_attention_window_size=4096 --run_profiling