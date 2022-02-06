from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

# import submitit
import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

from model import *

import numpy as np


from dataprocess import *

import os
import copy

torch.manual_seed(42)
np.random.seed(42)


class Arg:
	  def __init__(self):
        return

def main(args):
    
    loader_train1, loader_train2, loader_test = loadData(args)

    device = torch.device('cuda')
    
    model = train(args, loader_train1, loader_train2, loader_test, device)


    fname = "model/MNIST.pth"
    torch.save(model, fname)
    # model = torch.load(fname)

    print("Training done, model save to %s :)" % fname)


def test(model, loader_test, device):

    num_correct = 0
    num_samples = 0
    model.eval()
    for batch in loader_test:

        X = batch["org"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            logits = model(X)
        preds = (logits > 0).long()

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    accuracy = float(num_correct) / num_samples * 100
    print('\nAccuracy(org) = %.2f%%' % accuracy)

    num_correct = 0
    num_samples = 0

    for batch in loader_test:

        X = batch["rnd"].to(device)
        y = batch["label"].to(device)

        with torch.no_grad():
            logits = model(X)
        preds = (logits > 0).long()

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    accuracy = float(num_correct) / num_samples * 100
    print('\nAccuracy(rnd) = %.2f%%' % accuracy)

    model.train()

def train(args, loader_train1, loader_train2, loader_test, device):

    model = LeNet5()
    model = model.to(device)
    model.train()
        
    loss_f = nn.CrossEntropyLoss(reduction="none")

    SCHEDULE_EPOCHS = [20, 20] 
    SCHEDULE_LRS = [0.005, 0.0005]
    
    for num_epochs, learning_rate in zip(SCHEDULE_EPOCHS, SCHEDULE_LRS):
        
        print('\nTraining %d epochs with learning rate %.4f' % (num_epochs, learning_rate))
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            
            print('\nTraining epoch %d / %d ...\n' % (epoch + 1, num_epochs))
            
            for i, batch in enumerate(zip(loader_train1, loader_train2)):

                X = batch[0]["org"].to(device)

                if args.aug == "org":
                	X2 = batch[1]["org"].to(device)

                elif args.aug == "rnd":
                	X2 = batch[0]["rnd"].to(device)

                else:
                	raise ValueError("Unknown augmentation")


                y = batch[0]["label"].to(device)


                preds1 = model(X)
                preds2 = model(X2)

                eps = 1e-7 

                loss1 = F.binary_cross_entropy_with_logits(preds1, y, reduction='none') + eps
                loss2 = F.binary_cross_entropy_with_logits(preds2, y, reduction='none') + eps

                # loss = loss1.mean() + args.lambda_ * (loss1.pow(0.5) - loss2.pow(0.5)).pow(2).mean()

                loss = 0.5 * (loss1.mean() + loss2.mean()) + args.lambda_ * (loss1.pow(0.5) - loss2.pow(0.5)).pow(2).mean()        
                
                if (i + 1) % args.print_every == 0:
                    print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))
            
            test(model, loader_test, device)

    return model

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--lambda_', default=1.0, type=float,
                        help='lambda')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='size of each batch of cifar-10 training images')
    parser.add_argument('--print-every', default=50, type=int,
                        help='number of iterations to wait before printing')
    parser.add_argument('--aug', default="aug", type=str,
                        help='source of data augmentation')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

    # # lambdas = [0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    # lambdas = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    # for lambda_ in lambdas:
    #     args = Arg()
    #     args.data_dir = "./dataset" + str(int(lambda_)) + "org"
    #     args.print_every = 50
    #     args.lambda_ = lambda_
    #     args.batch_size = 64
    #     args.aug = "aug"


    #     executor = submitit.AutoExecutor(folder="org")
    #     num_gpus_per_node = 1
    #     nodes = 1
    #     executor.update_parameters(
    #             gpus_per_node=num_gpus_per_node,
    #             tasks_per_node=num_gpus_per_node,  # one task per GPU
    #             cpus_per_task=4, # 10 cpus per gpu is generally good
    #             nodes=nodes,
    #             # Below are cluster dependent parameters
    #             slurm_partition="a100",
    #             timeout_min=999999
    #         )

    #     job = executor.submit(main, args)  # will compute add(5, 7)
    #     print(job.job_id)  # ID of your job


