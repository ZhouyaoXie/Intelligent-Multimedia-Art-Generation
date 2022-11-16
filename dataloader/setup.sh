
#!/bin/bash
# bash script to set up dataset and dataloader 

# install dependencies
pip3 install -r requirements.txt

# download the musemorphose-main folder
gdown --fuzzy https://drive.google.com/file/d/1kvjs-XDdCOpyTFZ0UUi5SNVNldeStTna/view?usp=sharing
unzip MuseMorphose-main.zip
rm MuseMorphose-main.zip

# download the processed dataset
# current dataset: remi_dataset_0926_updated.tar.gz
gdown --fuzzy https://drive.google.com/file/d/1lMOY5ocnYyaW9ypgIQ3_obpZPfpP5VMM/view?usp=sharing
tar xzvf remi_dataset_0926_updated.tar.gz
mv content/remi_dataset/ MuseMorphose-main/
rm -r content/
rm remi_dataset_0926_updated.tar.gz

# remove original remi_vocab.pkl and replace with ours
rm MuseMorphose-main/pickles/remi_vocab.pkl
# gdown --fuzzy https://drive.google.com/file/d/15GIjOpW4biMXanUDcE8W7ZHQ0Jcyph7E/view?usp=sharing  # original remi_vocab with 404 tokens
# mv remi_vocab.pkl MuseMorphose-main/pickles/
gdown --fuzzy https://drive.google.com/file/d/1b5bOUgo5_0QOWwNGm2iKB0H1KK02VMCX/view?usp=share_link  # updated on 1116 with 406 tokens
mv remi_vocab_updated.pkl MuseMorphose-main/pickles/remi_vocab.pkl

# download no_events_fn.txt where we store files that we do not consider 
# because they have no events between some consecutive bars
gdown --fuzzy https://drive.google.com/file/d/1Fc1kxMTH-A4XIAh8lLvIf7MKuDMFd8Gp/view?usp=sharing

# download the splitted datasets with labels
gdown --folder https://drive.google.com/drive/folders/13UbMp_Q0PvnPsVNs7Aw4RWeiwT5cRatl?usp=sharing
