# Intelligent-Multimedia-Art-Generation

Music generation with CLIP-like text-music alignment model and Transformer music decoder.

## Documents
| Document             | Version | Link                                                                                                                                |
|----------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------|
| Vision               | v1      | [Google Doc](https://docs.google.com/document/d/1nXXPTrBMunDMKFK5zi0diH6auGFCNYr10UBuZXhbqMc/edit?usp=sharing)                                    |
| Requirement          | v1      | [Google Doc](https://docs.google.com/document/d/1OR6C8o-StwKZijQPvilHhEX7YncZJfTh/edit?usp=sharing&ouid=100645612073317945557&rtpof=true&sd=true) |
| Design               | v1      | [Google Doc](https://docs.google.com/document/d/1PWTMj7yC1GmBwJa2xMFi4Q0-vxIFWnJ5su-Z1_BGdII/edit?usp=sharing)                                    |
| Plan                 | v1      | [Google Doc](https://docs.google.com/document/d/1d4pKB81OoADSUBac-hzbuBLVFNRn6Rae5ga186-hsaI/edit?usp=sharing)                                    |
| Midterm Presentation | v1      | [Google Slide](https://docs.google.com/presentation/d/1eq4siGh2KAKda78kX-bInrw0dw_KqGriUX0oTFDo8-Y/edit?usp=sharing)                                |

## Directory Structure
root  
~/config  
  /default.yaml  
~/clip  // CLIP model  
  /__init__.py  
  /clip.py // load CLIP model from checkpoint  
  /model.py // define model architecture  
  /simple_tokenizer.py  
~/musemorphose // MuseMorphose model  
  /model  
    /musemorphose.py // define decoder & MuseMorphose model  
    /transformer_encoder.py // define encoder  
    /transformer_helpers.py // define positionEncoding & tokenEncoding layer  
   /__init__.py  
  /attributes.py  
  /dataloader.py  
  /extract_encoder.py  
  /generate.py  
  /remi2midi.py  
  /train.py  
  /utils.py  
~/model  
  /model.py // define MusicCLIP model  
  /load_model.py // load CLIP and MuseMorphose & create MusicCLIP instance  
    
    