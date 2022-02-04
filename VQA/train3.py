import sys
import os.path
import os
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
import config
import data2 as data
import model
import model2   ## modeifed net to have no attention
import utils
import time
import argparse
from pathlib import Path
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_type', default= 'finetuning_CNN_LSTM_data_aug2_{}_origamt_{}'.format(config.edit_loader_type, config.orig_amt), type=str)  ## 'with_attn' , 'no_attn', 'finetuning_SAAA'  'finetuning_CNN_LSTM_data_aug2'

print(config.model_type)

def update_learning_rate(args, optimizer, iteration):
    lr = args.initial_lr * 0.5**(float(iteration) / args.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0

log_softmax = nn.LogSoftmax(dim=1).cuda()  ### nn.LogSoftmax().cuda()
just_softmax = nn.Softmax(dim=1).cuda()
consistency_criterion_CE = nn.CrossEntropyLoss().cuda()
def consistency_criterion_MSE(x, y):  # ((softmax_edit-softmax_orig)**2).sum(dim=1).mean() BETTER IMPLEMENT THIS
    return ((x - y) ** 2).sum(dim=1).mean()
# ((consistency_criterion_MSE = nn.MSELoss().cuda() # this si wrong- divided by sum_square_loss(3000*edit_batch_size)
consistency_criterion_MSE_avg_false = nn.MSELoss(reduction='sum').cuda()  # == nn.MSELoss(size_average=False).cuda()   ==((softmax_edit-softmax_orig)**2).sum(dim=1).sum()
consistency_criterion_KL = nn.KLDivLoss(reduction='batchmean').cuda()


def run(args, net, loader, optimizer, tracker, train=False,test=False, prefix='', epoch=0, dataset=None):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        # answ = []   # edit_vedika FT101
        # idxs = []
        # accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    pos2neg = 0
    neg2pos = 0
    neg2neg = 0
    total_edits = 0
    for batch in tq:                                            #for v, q, a, idx, img_id, ques_id, q_len in tq:
        v, q, a, idx, img_id, ques_id, q_len = batch
        var_params = {
            'requires_grad': False,
        }
        if (train and config.orig_edit_equal_batch) or (train and config.orig_edit_diff_ratio_naive) or (train and config.orig_edit_diff_ratio_naive_no_edit_ids_repeat):
            
            edit_batch = data.get_edit_train_batch(dataset=dataset, ques_id_batch=ques_id, item_ids = idx)
            if edit_batch is not None:
                ## stack both orig and edit together
                v, q, a, idx,  ques_id, q_len =  [torch.cat((batch[i], edit_batch[i]), dim=0) for i in [0,1,2,3,5,6]]
                img_id = img_id + edit_batch[4]  # edit_img_id = edit_batch[4]
                q_len_new = [int(i) for i in q_len]
                sorting_order = np.argsort(q_len_new)[::-1]     #as q_len has to be sorted!
                #sorted_batch  = [[dat[i] for dat in [v, q, a, idx, img_id, ques_id, q_len]] for i in sorting_order]   # shape 102
                sorted_batch2 = [[dat[i] for i in sorting_order] for dat in [v, q, a, idx, img_id, ques_id, q_len]]
                v, q, a, idx, ques_id, q_len = [torch.stack(sorted_batch2[i], dim=0) for i in [0, 1, 2, 3, 5, 6]]
                img_id = sorted_batch2[4]
                # v, q, a, idx, img_id, ques_id, q_len
        if(test and config.orig_edit_equal_batch):
            edit_batch_v = data.get_edit_train_batch(dataset=dataset, ques_id_batch=ques_id, item_ids = idx)
            if edit_batch_v is not None:
                v_e, q_e, a_e, idx_e, img_id_e, ques_id_e, q_len_e, is_edit_batch = edit_batch_v
                v_e = Variable(v_e.cuda(), **var_params)
                q_e = Variable(q_e.cuda(), **var_params)
                a_e = Variable(a_e.cuda(), **var_params)
                q_len_e = Variable(q_len_e.cuda(), **var_params)

        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)

        if train:
            out = net(v, q, q_len)
            nll = -log_softmax(out)  ## taking softmax here     ## calculating  -log(p_pred)
            loss_1 = (nll * a / 10).sum(dim=1)     ### SO THIS COMPLETES CROSS ENTROPY : -p_true* log(p_pred) as  'a/10' does the role of being p_true  - ans has avlue 10 where its true
            loss = (nll * a / 10).sum(dim=1).mean()    ## mean of te batch #TODO- divide it into true and edit loss- which are resp being divided by the resp sizes
            

            flag = 1
            if ('data_aug2' in config.model_type or 'data_aug3' in config.model_type) and config.edit_loader_type == 'get_edits':
                loc2 = {} ## mapping ques_ids to location indices
                for key in set(list(ques_id)):
                    loc2[int(key)] = [idx for idx, i in enumerate(list(ques_id)) if i == key]
                loc3 = [key for key in loc2.keys() if len(loc2[key]) > 1]   ## those keys only whihc has two samples: edit and orig in our case

                all_true_ids = [idx for idx,i in enumerate(img_id) if len(i)==12]
                true_batch_order = [loc2[key][0] if len(img_id[loc2[key][0]])==12 else loc2[key][1] for key in loc3]
                edit_batch_order = [loc2[key][0] if len(img_id[loc2[key][0]])==25 else loc2[key][1] for key in loc3]    ## TODO only handles one edit case
                #print('Tr:',len(true_batch_order))
                #print('Edit:', len(edit_batch_order))
                if len(true_batch_order) == 0 or len(edit_batch_order) == 0:
                    flag = 0
                    print('skipping consistency loss')
                ### just checks to make sure correspondence between true_batch and edit_batch
                img_id_true = [img_id[i] for i in true_batch_order]   #TODO need not save it, can be done on the fly in lines 122-124
                img_id_edit = [img_id[i] for i in edit_batch_order]
                for i in range(len(true_batch_order)):
                    assert len(img_id_true[i]) < len(img_id_edit[i])
                    assert img_id_true[i] == img_id_edit[i][0:12]

                assert sorted(set(true_batch_order) &  set(all_true_ids)) == sorted(set(true_batch_order))

                if config.regulate_old_loss:   ## TODO right now only for data_aug2 and data_aug3 techniques; for naive ratio mixing- need to do it separately
                    ## divide the loss .... loss = (loss_real/n_real) + (loss_fake/n_fake)
                    #loss_1_orig = torch.stack([loss_1[i] for i in all_true_ids])
                    #loss_1_orig_has_edit = torch.stack([loss_1[i] for i in true_batch_order])
                    #loss_1_edit = torch.stack([loss_1[i] for i in edit_batch_order])
                    loss_old = loss
                    loss_orig = torch.stack([loss_1[i] for i in all_true_ids]).mean()
                    loss_edit = torch.stack([loss_1[i] for i in edit_batch_order]).mean()    ## ==loss_1_edit.sum()/num_edit_samples
                    #loss_e = loss_orig + (len(edit_batch_order)/len(all_true_ids))*loss_edit                #TODO tuning parameter now is num_edit_samples/num_true_samples
                    #loss_5 = loss_orig + 0.5 * loss_edit
                    # bcd = np.array(torch.Tensor.cpu(loss_1_orig.detach()))                          #TODO loss_track implementation to save both loss_old, loss_orig, loss_edit

                if flag and args.enforce_consistency:    # ony for data_aug3 strategy
                    softmax_out= just_softmax(out)
                    softmax_out = Variable(softmax_out.cuda(), **var_params)
                    softmax_orig = torch.stack([softmax_out[i] for i in true_batch_order])   # for orig softmax out
                    softmax_edit = torch.stack([softmax_out[i] for i in edit_batch_order])     ### for edit take- -log_softmax
                    nll_orig = torch.stack([nll[i] for i in true_batch_order])
                    nll_edit = torch.stack([nll[i] for i in edit_batch_order])

                    # consistency_loss_CE =  consistency_criterion_CE(softmax_edit, softmax_orig)  #RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target'
                    consistency_loss_edit_orig = (nll_edit * softmax_orig).sum(dim=1).mean()   # softmax(orig)* log_softmax(edit)
                    consistency_loss_orig_edit = (nll_orig * softmax_edit).sum(dim=1).mean()   # softmax(edit)* log_softmax(orig)
                    if config.old_CE :
                        consistency_loss_CE = (consistency_loss_edit_orig + consistency_loss_orig_edit)/2
                    else:
                        consistency_loss_CE = consistency_loss_edit_orig
                    consistency_loss_KL = consistency_criterion_KL(softmax_orig, softmax_edit)
                    consistency_loss_MSE = consistency_criterion_MSE(softmax_orig, softmax_edit)

                    if config.regulate_old_loss:
                        loss = loss_orig + (config.lam_edit_loss*loss_edit) + (config.lam_CE*consistency_loss_CE) + (config.lam_KL*abs(consistency_loss_KL)) + (config.lam_MSE*consistency_loss_MSE)
                    else:
                        loss_vqa = loss
                        loss = loss_vqa + (config.lam_CE*consistency_loss_CE) +  (config.lam_KL*abs(consistency_loss_KL)) + (config.lam_MSE*consistency_loss_MSE)

                    

            acc = utils.batch_accuracy(out.data, a.data).cpu()
            global total_iterations
            update_learning_rate(args,optimizer, total_iterations)

            #ipdb.set_trace()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iterations += 1

        elif test:
            with torch.no_grad():
                out = net(v, q, q_len)
                out2 = net(v_e, q_e, q_len_e)
                nll = -log_softmax(out)  ## taking softmax here
                nll2 = -log_softmax(out2)
                loss = (nll * a / 10).sum(dim=1).mean()
                acc = utils.batch_accuracy(out.data, a.data).cpu()
                acc2 = utils.batch_accuracy(out2.data, a_e.data).cpu()   ### taking care of volatile=True for val
                _, out_index = out.max(dim=1, keepdim=True)
                _, out2_index = out2.max(dim=1, keepdim=True)
            
        else:
            with torch.no_grad():
                out = net(v, q, q_len)    
                nll = -log_softmax(out)  ## taking softmax here    
                loss = (nll * a / 10).sum(dim=1).mean()
                acc = utils.batch_accuracy(out.data, a.data).cpu()
        
        if test:
            pos2neg += (torch.tensor(is_edit_batch).view(-1,1)*((acc == 1.0)*(acc2 != 1.0))).sum().item()
            neg2pos += (torch.tensor(is_edit_batch).view(-1,1)*((acc != 1.0)*(acc2 == 1.0))).sum().item()
            neg2neg += (torch.tensor(is_edit_batch).view(-1,1)*((acc != 1.0)*(acc2 != 1.0)*(~torch.eq(out_index,out2_index).cpu()))).sum().item()
            total_edits += sum(is_edit_batch)

        loss_tracker.append(loss.item())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
    if test:    
        preds_flipped = pos2neg + neg2pos + neg2neg
        print(prefix,' pos2neg :', fmt(pos2neg/total_edits), 'neg2pos :', fmt(neg2pos/total_edits), 'neg2neg :', fmt(neg2neg/total_edits), 'preds. flipped:', fmt(preds_flipped/total_edits))

def main(args):
    start_time = time.time()
    print('saving to : ', args.trained_model_save_folder)
    cudnn.benchmark = True

    train_dataset, train_loader = data.get_loader(train=True, prefix= args.prefix)
    val_dataset, val_loader = data.get_loader(val=True)
    test_dataset, test_loader = data.get_loader(test=True)


    if config.model_type == 'no_attn':
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(args.trained_model_save_folder)

    elif config.model_type == 'with_attn':
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(args.trained_model_save_folder)

    elif 'finetuning_CNN_LSTM' in config.model_type:
        #ipdb.set_trace()
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_no_attn)
        net.load_state_dict(torch.load(model_path)["weights"])   ## SO LOAD  THE MODEL HERE- WE WANT TO START FINETUNING FROM THE BEST WE HAVE
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif 'data_aug_CNN_LSTM' in config.model_type:
        #ipdb.set_trace()
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif 'data_aug_SAAA' in config.model_type:
        #ipdb.set_trace()
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif 'finetuning_SAAA' in config.model_type:
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_show_ask_attend_answer)
        net.load_state_dict(torch.load(model_path)["weights"])   ## SO LOAD  THE MODEL HERE- WE WANT TO START FINETUNING FROM THE BEST WE HAVE
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

        # os.makedirs(target_name, exist_ok=True)
    print('will save to {}'.format(target_name))

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        _ = run(args, net, train_loader, optimizer, tracker, train=True, test= False,prefix='train', epoch=i, dataset=train_dataset)   
        _ = run(args, net, val_loader, optimizer, tracker, train=False, test=False, prefix='val', epoch=i, dataset= val_dataset)   
        _ = run(args, net, test_loader, optimizer, tracker, train=False, test = True, prefix='test', epoch=i, dataset= test_dataset)

        results = {
            'tracker': tracker.to_dict(),   ## tracker saves acc/loss for all 50 epochs- since it appends the values ( lines 91..)
            'config': config_as_dict,
            'weights': net.state_dict(),
            'vocab': train_loader.dataset.vocab,
        }
        saving_target_name = 'epoch_{}.pth'.format(i)   ## you want to have all finetuned models- so save every model at everye epoch
        torch.save(results, os.path.join(target_name, saving_target_name))   ## keys:  "name", "tracker", "config", "weights", "eval", "vocab"

    

    print('time_taken:', time.time() - start_time)
    print(config.model_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQA')
    parser.add_argument('--trained_model_save_folder', default='./models/vqa_MSE/', type = str)
    parser.add_argument('--enforce_consistency', default = 1, type=int)
    parser.add_argument('--prefix', default='real_iv', type = str)
    parser.add_argument('--initial_lr',default = 1e-3, type = float)
    parser.add_argument('--lr_halflife',default = 50000, type = float)
    args = parser.parse_args()
    Path(args.trained_model_save_folder).mkdir(parents=True, exist_ok=True)
    main(args)
