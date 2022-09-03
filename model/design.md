# model.py

### class MusicCLIP

- **__init__**:
    - ~~TODO: define text and music encoder architecture~~
    - ~~TODO: test initialize model locally~~
- **initialize_params**: ~~TODO~~
- **encode_text**: 
    - ~~TODO: modify CLIP function~~
    - ~~TODO: test locally~~
    - TODO: CUDA test
- **encode_music**: 
    - TODO: modify MuseMorphose forward function
    - TODO: CUDA test 
- **forward**: 
    - TODO: modify CLIP function, add in music part
    - TODO: CUDA test

# load_model.py

- **load_CLIP**: 
    - ~~TODO: load CLIP model from pretrained weights~~
- **load_MuseMorphose_encoder**:
    - ~~TODO: load MuseMorphose model and return encoder, TokenEmbedding layer, and FF layer~~
- **load**: 
    - ~~TODO: initialize MusicCLIP model using load_CLIP() and load_MuseMorphose_encoder~~
    - TODO: CUDA test
