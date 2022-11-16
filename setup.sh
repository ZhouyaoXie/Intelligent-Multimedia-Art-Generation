#!/bin/bash
# bash script to set up musemorphose and run test script

echo "install dependencies"
pip3 install -r requirements.txt

echo "download musemorphose pretrained weights"
wget -O musemorphose_pretrained_weights.pt https://zenodo.org/record/5119525/files/musemorphose_pretrained_weights.pt?download=1

echo "download musemorphose converted weights"
# gdown --fuzzy https://drive.google.com/file/d/15v5tRmOueYUg0pxrdX1i29KrxrzjjA4v/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1TbahcwKvj3i-_GRvRoZSs4aGTd_mIItg/view?usp=sharing  # updated weights
mv converted_pretrained_weights_updated.pt converted_pretrained_weights.pt  # rename to original name

echo "setup dataloader"
sh dataloader/setup.sh

# echo "compute attribute classes"
# python3 musemorphose/attributes.py

# run python test script
# python3 model/music_encoder_test.py
