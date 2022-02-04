#!/usr/bin/env bash

mkdir -p data
mkdir -p data/mscoco
mkdir -p data/edit_mscoco
mkdir -p data/preprocessed_data
mkdir -p data/mini_datasets_qa
mkdir -p data/mini_datasets_qa/0.1_0.1
mkdir -p data/mini_datasets_qa/0.1_0.1/match

cd data
wget http://images.cocodataset.org/zips/train2014.zip 
wget http://images.cocodataset.org/zips/val2014.zip 
wget http://images.cocodataset.org/zips/test2015.zip 

unzip train2014.zip
unzip val2014.zip
unzip test2015.zip

gdown 'https://drive.google.com/uc?export=download&id=1YQCE0joZIWl8OofQVfNsdbPJVG0u9i5t' -O mini_datasets_qa/0.1_0.1/match/v2_mscoco_train2014_annotations.json
gdown 'https://drive.google.com/uc?export=download&id=1M-mFwAUFh_sMIG4yUhIQCUPtJMzuw1Mi' -O mini_datasets_qa/0.1_0.1/match/v2_OpenEnded_mscoco_train2014_questions.json
gdown 'https://drive.google.com/uc?export=download&id=18uK05KBjJqgPXbNm46KVBKLyFoKB3cMO' -O mini_datasets_qa/0.1_0.1/match/v2_mscoco_val2014_annotations.json
gdown 'https://drive.google.com/uc?export=download&id=1KjK4D0GS07Lc1O0mhuk8so-jox4dxvii' -O mini_datasets_qa/0.1_0.1/match/v2_OpenEnded_mscoco_val2014_questions.json

wget https://datasets.d2.mpi-inf.mpg.de/rakshith/causalvqa/rawdata/train2014_edited.tar.gz
wget https://datasets.d2.mpi-inf.mpg.de/rakshith/causalvqa/rawdata/val2014_edited.tar.gz

wget https://datasets.d2.mpi-inf.mpg.de/rakshith/causalvqa/rawdata/CV_VQA_train2014.tar.gz
wget https://datasets.d2.mpi-inf.mpg.de/rakshith/causalvqa/rawdata/CV_VQA_val2014.tar.gz

tar -xf val2014_edited.tar.gz
tar -xf train2014_edited.tar.gz

tar -xf CV_VQA_train2014.tar.gz
tar -xf CV_VQA_val2014.tar.gz

mv BS/vedika2/nobackup/thesis/IMAGES_counting_del1_edited_VQA_v2 .

mv BS/vedika2/nobackup/thesis/final_edited_VQA_v2/Images/train2014 edit_mscoco
mv BS/vedika2/nobackup/thesis/final_edited_VQA_v2/Images/val2014 edit_mscoco

rm -rf BS