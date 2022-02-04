# DAIR for VQA 

### Table of Contents

   * [Setup and Dependencies](#setup-and-dependencies)
   * [Preprocessing](#preprocessing)
   * [Usage](#usage)
   * [Acknowledgments](#acknowledgments)

### Setup and Dependencies

Run the following command to download and extract all the data files

```
sh setup.sh
```
#### Python 3 dependencies (tested on Python 3.6.2)

- torch
- torchvision
- h5py
- tqdm
- ipdb

### Preprocessing
```
python preprocess-images.py
python preprocess-vocab.py
```

### Usage
- Train the model with:
```
python train.py
```
Set the lambda and gamma values in the config.py file

### Acknowledgments

1. pytorch-vqa [repository](https://github.com/Cyanogenoid/pytorch-vqa)
2. Invariant VQA code [repository](https://github.com/AgarwalVedika/pytorch-vqa)




