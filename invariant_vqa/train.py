import os.path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import config
import data
import model
import model2   ## modified net to have no attention
import utils
import time
import argparse
from pathlib import Path

print(config.model_type)

def update_learning_rate(optimizer, iteration):
    lr = args.initial_lr * 0.5**(float(iteration) / args.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0

log_softmax = nn.LogSoftmax(dim=1).cuda()  ### nn.LogSoftmax().cuda()
just_softmax = nn.Softmax(dim=1).cuda()
consistency_criterion_CE = nn.CrossEntropyLoss().cuda()

def run(args,net, loader, optimizer, tracker, train=False, prefix='', epoch=0, dataset=None):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    pos2neg = 0
    neg2pos = 0
    neg2neg = 0
    total_edits = 0
    for batch in tq:                                            #for v, q, a, idx, img_id, ques_id, q_len in tq:
        v, q, a, idx, img_id, ques_id, q_len = batch
        if (config.orig_edit_equal_batch) or (config.orig_edit_diff_ratio_naive) or (config.orig_edit_diff_ratio_naive_no_edit_ids_repeat):
            
            edit_batch = data.get_edit_train_batch(dataset=dataset, ques_id_batch=ques_id, item_ids = idx)
            if edit_batch is not None:
                v_e, q_e, a_e, idx_e, img_id_e, ques_id_e, q_len_e, is_edit_batch = edit_batch

        var_params = {
            'requires_grad': False,
        }
        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)

        if train:
            v_e = Variable(v_e.cuda(), **var_params)
            q_e = Variable(q_e.cuda(), **var_params)
            a_e = Variable(a_e.cuda(), **var_params)
            q_len_e = Variable(q_len_e.cuda(), **var_params)            
            out = net(v, q, q_len)
            out2 = net(v_e, q_e, q_len_e)
            nll = -log_softmax(out)
            nll2 = -log_softmax(out2)
            _, out_index = out.max(dim=1, keepdim=True)
            _, out2_index = out2.max(dim=1, keepdim=True)
            loss_1 = (nll * a / 10).sum(dim=1)     ### SO THIS COMPLETES CROSS ENTROPY : -p_true* log(p_pred) as  'a/10' does the role of being p_true  - ans has avlue 10 where its true
            loss_2 = (nll2 * a_e / 10).sum(dim=1)
            
            loss = (nll * a / 10).sum(dim=1).mean()

            if ('data_aug2' in config.model_type or 'data_aug3' in config.model_type) and config.edit_loader_type == 'get_edits':
            
                loss_1 += 1e-7
                loss_2 += 1e-7
                loss = args.gamma*loss_1.mean() + (1-args.gamma)*loss_2.mean() + args._lambda*(loss_1.pow(0.5) - loss_2.pow(0.5)).pow(2).mean()

            acc = utils.batch_accuracy(out.data, a.data).cpu()
            acc2 = utils.batch_accuracy(out2.data, a.data).cpu()
            global total_iterations
            update_learning_rate(optimizer, total_iterations)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iterations += 1

        else:
            with torch.no_grad():
                out = net(v, q, q_len)
                out2 = net(v_e, q_e, q_len_e)
                nll = -log_softmax(out)  ## taking softmax here
                nll2 = -log_softmax(out2)
                loss = (nll * a / 10).sum(dim=1).mean()
                acc = utils.batch_accuracy(out.data, a.data).cpu()
                acc2 = utils.batch_accuracy(out2.data, a.data).cpu()   ### taking care of volatile=True for val
                _, out_index = out.max(dim=1, keepdim=True)
                _, out2_index = out2.max(dim=1, keepdim=True)


        pos2neg += (torch.tensor(is_edit_batch).view(-1,1)*((acc == 1.0)*(acc2 != 1.0))).sum().item()
        neg2pos += (torch.tensor(is_edit_batch).view(-1,1)*((acc != 1.0)*(acc2 == 1.0))).sum().item()
        neg2neg += (torch.tensor(is_edit_batch).view(-1,1)*((acc != 1.0)*(acc2 != 1.0)*(~torch.eq(out_index,out2_index).cpu()))).sum().item()
        total_edits += sum(is_edit_batch)
        loss_tracker.append(loss.item())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
    print(total_edits)
    preds_flipped = pos2neg + neg2pos + neg2neg
    print(prefix,' pos2neg :', fmt(pos2neg/total_edits), 'neg2pos :', fmt(neg2pos/total_edits), 'neg2neg :', fmt(neg2neg/total_edits), 'preds. flipped:', fmt(preds_flipped/total_edits))

def main(args):
    start_time = time.time()

    cudnn.benchmark = True

    train_dataset, train_loader = data.get_loader(train=True, prefix = 'del1')
    val_dataset, val_loader = data.get_loader(val=True)
    #test_loader = data.get_loader(test=True)

    print("Done with data loading")
    if config.model_type == 'no_attn':
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(config.model_path_no_attn)

    elif config.model_type == 'with_attn':
        net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(config.model_path_show_ask_attend_answer)

    elif 'finetuning_CNN_LSTM' in config.model_type:
        
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        model_path = os.path.join(config.model_path_no_attn)
        net.load_state_dict(torch.load(model_path)["weights"])   ## SO LOAD  THE MODEL HERE- WE WANT TO START FINETUNING FROM THE BEST WE HAVE
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif 'data_aug_CNN_LSTM' in config.model_type:
        
        net = nn.DataParallel(model2.Net(train_loader.dataset.num_tokens)).cuda()
        target_name = os.path.join(args.trained_model_save_folder)    # so this will store the models
        os.makedirs(target_name, exist_ok=True)

    elif 'data_aug_SAAA' in config.model_type:
        
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
        _ = run(args, net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i, dataset=train_dataset)   ## prefix needed as ths is passed to tracker- which stroes then train_acc/loss
        _ = run(args, net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i, dataset= val_dataset)    ## prefix needed as ths is passed to tracker- which stroes then val acc/loss

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
    parser.add_argument('--gamma',default = 0.5, type = float)
    parser.add_argument('--_lambda',default = 1, type = float)
    parser.add_argument('--initial_lr',default = 1e-3, type = float)
    parser.add_argument('--lr_halflife',default = 50000, type = float)
    parser.add_argument('--trained_model_save_folder', default='./models/lambda_1_gamma_half/', type = str)
    args = parser.parse_args()
    Path(args.trained_model_save_folder).mkdir(parents=True, exist_ok=True)
    main(args)
