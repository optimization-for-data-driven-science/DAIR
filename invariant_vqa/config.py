# preprocess config
preprocess_batch_size = 96
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping
preprocessed_path = './data/preprocessed_data/orig_edit_del1_VQAset.h5'  ##'./preprocessed_data/orig_edit_VQAset.h5'
train_path = './data/mscoco/train2014'  # directory of training images
val_path = './data/mscoco/val2014'  # directory of validation images
test_path = './data/mscoco/test2015'  # directory of test images
edit_train_path =  './data/edit_mscoco/train2014'
edit_val_path =  './data/edit_mscoco/val2014'
del1_path_train = './data/IMAGES_counting_del1_edited_VQA_v2/train2014'
del1_path_val =  './data/IMAGES_counting_del1_edited_VQA_v2/val2014'
# del_1_QA_folder = /BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0

# training config
epochs = 50
batch_size = 96   ## originally 128 for learning
initial_lr = 1e-3 #1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000

#train.path
edit_loader_type= None
orig_edit_equal_batch = 0          #### only when you use different data aug styles (2/3) #(setting 4 & 5) orig_edit_equal => means you are getting orig with its corresponding edits
orig_edit_diff_ratio_naive = 0
orig_edit_diff_ratio_naive_no_edit_ids_repeat = 0
regulate_old_loss = 0
load_only_orig_ids = 0
enforce_consistency = 1
model_path_no_attn = './models/no_attn_biasTrue.pth'     # biasTrue better than biasFalse by ~0.3 points!
model_path_show_ask_attend_answer = './models/data_aug_SAAA_data_aug3_get_edits_origamt_0.5_CE_0.3_KL_0_MSE_1/0.1_0.1/ermepoch_25.pth'   ### the one initially with everything
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to
dset = 'v2'   ## change to v1 if you want to evaluate on VQA v1- TODO changes to be made in the preprocess-vocab then- to hndle......
task = 'OpenEnded'
dataset = 'mscoco'

# # make sure train.py and data.py: edit_orig_combine is true
ques_type = '0.1_0.1'  # ['how many', 'is this a', 'is there a', 'what color is the', 'counting']
data_split = ''  # ['orig_10', 'orig_all', 'orig_10_edit_10','orig_all_edit_10',  'orig_all_edit_all']


#
# # # ###SETTING_5 enforcinng consistency- keeping regulating old loss as 0 here:ou got to change line 109 if stmt in train.py
old_CE = 0
orig_amt = 0.5
edit_loader_type = 'get_edits'
load_only_orig_ids = 1

gamma = 0.5
lambda_ = 1
regulate_old_loss = 0
lam_edit_loss = 0.5
air = 1
enforce_consistency = 1
lam_CE = 0
lam_KL = 0
lam_MSE = 1
model_type =  'data_aug_SAAA_data_aug3_{}_origamt_{}_CE_{}_KL_{}_MSE_{}'.format(edit_loader_type, orig_amt, lam_CE, lam_KL, lam_MSE)           #'finetuning_CNN_LSTM'  #finetuning_CNN_LSTM_data_aug2
orig_edit_equal_batch = 1
orig_edit_diff_ratio_naive = 0
orig_edit_diff_ratio_naive_no_edit_ids_repeat = 0
trained_model_save_folder = './models/' + model_type  + '/' + ques_type  + '/' + data_split # os.path.join('./models', ques_type)
qa_path = './data/mini_datasets_qa' + '/'+ ques_type + '/' + data_split  # directory containing the question and annotation jsons


ques_type = '0.1_0.0'   #0.1_0.0'   # '0.1_0.1 # ['how many', 'is this a', 'is there a', 'what color is the', 'counting']
test_data_split = 'edit_10' #['orig_90_10',  'orig_90_all', 'edit_10', 'edit_all', 'del_1', 'orig_all' in case of 0.1_0.0 and 0.1_0.1 ] #'orig_10_10' was for validation for all 50 epoch models

results_with_attn_pth = './logs/train_with_attn_.pth'                          ### change this!! train/val/test!
results_with_attn_pkl = './logs/train_with_attn_.pickle'
results_no_attn_pth = './logs/train_no_attn_.pth'
results_no_attn_pkl = './logs/train_no_attn_.pickle'

vis_attention =0
