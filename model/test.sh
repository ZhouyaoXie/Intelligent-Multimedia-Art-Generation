#!/bin/bash
# bash script to set up musemorphose and run test script

# install dependencies
pip3 install -r requirements.txt

# download musemorphose pretrained weights
wget -O musemorphose_pretrained_weights.pt https://zenodo.org/record/5119525/files/musemorphose_pretrained_weights.pt?download=1

# download musemorphose REMI-pop-1.7K dataset
wget -O remi_dataset.tar.gz https://zenodo.org/record/4782721/files/remi_dataset.tar.gz?download=1
tar xzvf remi_dataset.tar.gz
rm remi_dataset.tar.gz

# compute attribute classes
python3 musemorphose/attributes.py

# run python test script
CURRENTDATE=`date +"%Y_%m_%d_%T"`
python3 model/music_encoder_test.py >> "test_log_${CURRENTDATE}"