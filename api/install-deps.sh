# Preparing conda
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/Anaconda.sh
bash ~/Anaconda.sh -b -p
conda init

# Installing package dependencies
conda env create --name rag_ss --file environment.yml