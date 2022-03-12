mkdir -p data/mini_datasets_qa/0.1_0.1/all
cd data
gdown 'https://drive.google.com/uc?export=download&id=1IKz3tUCmVZShfoZA5Hi0eRAISKuAf6eq' -O mini_datasets_qa/0.1_0.1/all/v2_OpenEnded_mscoco_train2014_questions.json
gdown 'https://drive.google.com/uc?export=download&id=1cg8dbw7CNa14RaAm1gEh7WPNM_V1WhXr' -O mini_datasets_qa/0.1_0.1/all/v2_OpenEnded_mscoco_val2014_questions.json
gdown 'https://drive.google.com/uc?export=download&id=14IUtIo9d-9a7QRHH7wArl9wNwAEjUS9m' -O mini_datasets_qa/0.1_0.1/all/v2_mscoco_train2014_annotations.json
gdown 'https://drive.google.com/uc?export=download&id=1sYBV-rYVi8BpEAr0c8PXmklYtmkA3k0z' -O mini_datasets_qa/0.1_0.1/all/v2_mscoco_val2014_annotations.json