from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict


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
from torchvision import utils

from model import *
# from model2 import FullyConnectedNet

import numpy as np

import matplotlib.pyplot as plt

from dataprocess import *

import os
import copy

torch.manual_seed(42)
np.random.seed(42)


def main(args):
	
	loader_train, loader_test = loadData(args)
	dtype = torch.FloatTensor
	
	model = train(args, loader_train, loader_test, dtype)

	fname = "model/MNIST.pth"
	# torch.save(model, fname)

	print("Training done, model save to %s :)" % fname)


def test(model, loader_test, dtype):
	num_correct = 0
	num_samples = 0
	model.eval()
	for batch in loader_test:

		X = batch["org"].type(dtype)
		y = batch["label"].type(dtype).long()

		# grid = utils.make_grid(X_[:64])
		# print(type(grid))
		# utils.save_image(grid, 'aa.png')
		# exit(0)

		with torch.no_grad():
			logits = model(X)
		_, preds = logits.max(1)

		num_correct += (preds == y).sum()
		num_samples += preds.size(0)

	accuracy = float(num_correct) / num_samples * 100
	print('\nAccuracy(org) = %.2f%%' % accuracy)

	num_correct = 0
	num_samples = 0

	for batch in loader_test:

		X = batch["90"].type(dtype)
		y = batch["label"].type(dtype).long()

		with torch.no_grad():
			logits = model(X)
		_, preds = logits.max(1)

		num_correct += (preds == y).sum()
		num_samples += preds.size(0)

	accuracy = float(num_correct) / num_samples * 100
	print('\nAccuracy(rnd) = %.2f%%' % accuracy)

	model.train()

def train(args, loader_train, loader_test, dtype):

	model = LeNet5()
	model = model.type(dtype)
	model.train()
		
	loss_f = nn.CrossEntropyLoss(reduction="none")

	SCHEDULE_EPOCHS = [20, 20] 
	SCHEDULE_LRS = [0.005, 0.0005]
	
	for num_epochs, learning_rate in zip(SCHEDULE_EPOCHS, SCHEDULE_LRS):
		
		print('\nTraining %d epochs with learning rate %.4f' % (num_epochs, learning_rate))
		
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(num_epochs):
			
			print('\nTraining epoch %d / %d ...\n' % (epoch + 1, num_epochs))
			
			for i, batch in enumerate(loader_train):

				X_ = batch["org"].type(dtype)
				X_2 = batch["rnd"].type(dtype)
				y_ = batch["label"].type(dtype).long()

				preds1 = model(X_)
				preds2 = model(X_2)


				loss1 = loss_f(preds1, y_) + 1e-7
				loss2 = loss_f(preds2, y_) + 1e-7

				gamma = 0.5

				loss = gamma * loss1.mean() + (1 - gamma) * loss2.mean() + args.lambda_ * (loss1.pow(0.5) - loss2.pow(0.5)).pow(2).mean()
				
				if (i + 1) % args.print_every == 0:
					print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))
			
			test(model, loader_test, dtype)

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

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	main(args)

