# MobileNet 
Simple pytorch implementation of mobilenet trained on Visual Wake Words dataset.

## Installations
conda create -p ./venv python=3.10 -y
conda activate ./venv
pip install -r requirements.txt

wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
tar -xvf vw_coco2014_96.tar.gz
