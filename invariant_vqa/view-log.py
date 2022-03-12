import sys
import torch
import matplotlib; matplotlib #.use('agg')
import matplotlib.pyplot as plt
import config
import ipdb
import numpy as np
import os
def main():

    ques_types= ['how many', 'is this a', 'is there a', 'what color is the', 'counting']
    data_splits = ['orig_10', 'orig_all', 'orig_10_edit_10','orig_all_edit_10',  'orig_all_edit_all']
    exp = 'finetuning_CNN_LSTM'

    plots_dir = os.path.join('../pytorch-vqa/plots', exp)
    os.makedirs(plots_dir, exist_ok=True)
    for ques_type in ques_types:

        for data_split in data_splits:
            path = '../pytorch-vqa/models/'+ exp + '/' + ques_type + '/' + data_split + '/epoch_49.pth'
            results = torch.load(path)
            #
            # ipdb > results.keys()
            # dict_keys(['name', 'tracker', 'config', 'weights', 'eval', 'vocab'])

            train_loss = torch.FloatTensor(results['tracker']['train_loss'])   ### train_loss, train_acc, val_loss, val_acc
            train_loss = train_loss.mean(dim=1).numpy()

            val_loss = torch.FloatTensor(results['tracker']['val_loss'])   ### train_loss, train_acc, val_loss, val_acc
            val_loss = val_loss.mean(dim=1).numpy()


            train_acc = torch.FloatTensor(results['tracker']['train_acc'])   ### train_loss, train_acc, val_loss, val_acc
            train_acc = train_acc.mean(dim=1).numpy()

            val_acc = torch.FloatTensor(results['tracker']['val_acc'])   ### train_loss, train_acc, val_loss, val_acc
            val_acc = val_acc.mean(dim=1).numpy()


            str_put = 'maximum val acc is {} at epoch {}'.format(np.sort(val_acc)[-1], np.argsort(val_acc)[-1]) + \
            '\n maximum train acc is {} at epoch {}'.format(np.sort(train_acc)[-1], np.argsort(train_acc)[-1])

            plt.figure(figsize=(10,10))
            plt.plot(train_acc, color='blue', label='train_acc')
            plt.plot(val_acc, color='green', label='val_acc')
            plt.plot(train_loss, color='red', label='train_loss')
            plt.plot(val_loss, color='cyan',label='val_loss')
            plt.xlabel('number of epochs' + '\n' + str_put)
            plt.title(exp + '_' + ques_type + '_' + data_split + 'train_val_acc_loss.png')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.savefig( plots_dir + '/' +  ques_type + '_' + data_split + 'train_val_acc_loss.png')


if __name__ == '__main__':
    main()

#
# ipdb> import numpy as np    Show, Ask, Attend, Answer model
# ipdb> np.argsort(val_acc)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 12, 14, 15, 16,
#        18, 17, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 29, 30, 33, 31, 32,
#        34, 35, 37, 36, 38, 40, 39, 41, 43, 42, 45, 44, 46, 48, 49, 47])
# ipdb> np.sort(val_acc)
# array([0.4824009 , 0.51662946, 0.5344958 , 0.5452822 , 0.555804  ,
#        0.56044877, 0.566875  , 0.57034075, 0.572756  , 0.5775462 ,
#        0.58055615, 0.58244365, 0.5847884 , 0.5848229 , 0.58689755,
#        0.5874984 , 0.5894494 , 0.59056854, 0.5911736 , 0.5914908 ,
#        0.59241176, 0.5925824 , 0.59333783, 0.59477365, 0.5953596 ,
#        0.5959344 , 0.5959605 , 0.59653765, 0.5977818 , 0.59793806,
#        0.59800565, 0.5980756 , 0.59808034, 0.5984064 , 0.59883094,
#        0.59891266, 0.59916866, 0.5992872 , 0.6002374 , 0.6002566 ,
#        0.6002604 , 0.60065734, 0.6006643 , 0.60078   , 0.6011724 ,
#        0.60119426, 0.6012218 , 0.60132676, 0.6014821 , 0.6014947 ],
#       dtype=float32)
# ipdb> np.argsort(train_acc)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
# ipdb> np.sort(train_acc)
# array([0.41020718, 0.4885804 , 0.5173783 , 0.537218  , 0.55178577,
#        0.56434745, 0.5742161 , 0.58298755, 0.5917035 , 0.5976437 ,
#        0.6032905 , 0.6095887 , 0.6148498 , 0.61892503, 0.62323827,
#        0.6280237 , 0.63096017, 0.63504225, 0.6376792 , 0.64091665,
#        0.64309883, 0.6465949 , 0.64803445, 0.65151364, 0.6541523 ,
#        0.65553087, 0.6585514 , 0.6590608 , 0.6614735 , 0.6628266 ,
#        0.665018  , 0.66681844, 0.6675081 , 0.66960114, 0.67103016,
#        0.6716329 , 0.67311853, 0.6742379 , 0.67530197, 0.677334  ,
#        0.678118  , 0.6793243 , 0.6799203 , 0.68022466, 0.6821093 ,
#        0.68220884, 0.68290883, 0.684336  , 0.68478334, 0.6861863 ],
#       dtype=float32)
# ipdb> np.argsort(train_loss)
# array([49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
#        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
#        15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
# ipdb> np.argsort(val_loss)
# array([32, 40, 38, 36, 31, 37, 34, 47, 45, 41, 46, 27, 28, 24, 33, 43, 48,
#        39, 29, 22, 44, 30, 26, 49, 42, 25, 35, 20, 23, 19, 21, 18, 17, 16,
#        15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0])
