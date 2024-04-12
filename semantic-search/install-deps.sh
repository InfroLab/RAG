# Preparing conda
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/Anaconda.sh
bash ~/Anaconda.sh -b -p
conda init

# Installing package dependencies
conda env create --name rag_llm --file environment.yml
pip install peft==0.10.0
pip install tensorrt-llm==0.8.0 --extra-index-url https://pypi.nvidia.com


# Preparing apts and downloading necessary files
sudo apt-get install unzip
cd ./data/raw
kaggle datasets download -d everydaycodings/global-news-dataset
wget https://data.world/crawlfeeds/cnbc-news-dataset
unzip global-news-dataset.zip
rm global-news-dataset.zip
unzip cnbc-news-dataset
rm cnbc-news-dataset