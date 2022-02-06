from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn
import torchvision.datasets.utils as dataset_utils

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

from PIL import Image

import os

import numpy as np
import time

torch.manual_seed(42)
np.random.seed(int(time.time()))



def color_grayscale_arr(arr):
    """Converts grayscale image to either red or green"""

    assert arr.ndim == 2
    dtype = arr.dtype
    (h, w) = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    
    red = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)],
                             axis=2)
    
    green = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr,
                             np.zeros((h, w, 1), dtype=dtype)], axis=2)


    c1 = (np.random.rand() * arr).astype(arr.dtype)
    c2 = (np.random.rand() * arr).astype(arr.dtype)
    c3 = (np.random.rand() * arr).astype(arr.dtype)

    rnd = np.concatenate([c1, c2, c3], axis=2) 
    return red, green, rnd

class ColoredMNIST(dset.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./dataset', env='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img1, img2, rnd, target = self.data_label_tuples[index]

    if self.transform is not None:
      img1 = self.transform(img1)
      img2 = self.transform(img2)
      rnd = self.transform(rnd)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return {"org": img1,
    		"alt": img2,
    		"rnd": rnd,
    		"label": target,

    }

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = dset.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # # Flip the color with a probability e that depends on the environment
      # if idx < 20000:
      #   # 20% in the first training environment
      #   if np.random.uniform() < 0.2:
      #     color_red = not color_red
      # elif idx < 40000:
      #   # 10% in the first training environment
      #   if np.random.uniform() < 0.1:
      #     color_red = not color_red
      # else:
      #   # 90% in the test environment
      #   if np.random.uniform() < 0.9:
      #     color_red = not color_red

      red_arr, green_arr, rnd = color_grayscale_arr(im_array)

      if idx < 20000:

        # print(color_red)
        # print(binary_label)

        if np.random.uniform() < 0.2:
          # print("?")
          color_red = not (binary_label == 0)
        # print(color_red)

        if color_red:
       	  train1_set.append((Image.fromarray(red_arr), Image.fromarray(green_arr), Image.fromarray(rnd), binary_label))
        else:
          train1_set.append((Image.fromarray(green_arr), Image.fromarray(red_arr), Image.fromarray(rnd), binary_label))

      if idx < 20000:

        if np.random.uniform() < 0.1:
          # print("!")
          color_red = not (binary_label == 0)
        # print(color_red)
        # print()


        if color_red:
          train2_set.append((Image.fromarray(red_arr), Image.fromarray(green_arr), Image.fromarray(rnd), binary_label))
        else:
          train2_set.append((Image.fromarray(green_arr), Image.fromarray(red_arr), Image.fromarray(rnd), binary_label))

      if idx >= 40000:
      # if idx >= 20000 and idx < 40000:

        if np.random.uniform() < 0.9:
          color_red = not (binary_label == 0)
        if color_red:
          test_set.append((Image.fromarray(red_arr), Image.fromarray(green_arr), Image.fromarray(rnd), binary_label))
        else:
          test_set.append((Image.fromarray(green_arr), Image.fromarray(red_arr), Image.fromarray(rnd), binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    # dataset_utils.makedir_exist_ok(colored_mnist_dir)

    if not os.path.exists(colored_mnist_dir):
    	os.makedirs(colored_mnist_dir)

    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))
    
    def __len__(self):
        return self.num_samples

def loadData(args):


    # transform_augment_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor()])
    transform = T.Compose([T.ToTensor()])

    # MNIST_train = dset.MNIST('./dataset', train=True, transform=T.ToTensor(), download=True)

    # MNIST_test = dset.MNIST('./dataset', train=False, transform=T.ToTensor(), download=True)

    MNIST_train1 = ColoredMNIST(root=args.data_dir, env='train1', transform=T.ToTensor())
    MNIST_train2 = ColoredMNIST(root=args.data_dir, env='train2', transform=T.ToTensor())
    MNIST_test = ColoredMNIST(root=args.data_dir, env='test', transform=T.ToTensor())

    # MNIST_train = ColoredMNIST(root='./dataset', env='train1')
    # MNIST_test = ColoredMNIST(root='./dataset', env='test')

    loader_train1 = DataLoader(MNIST_train1, batch_size=args.batch_size)
    loader_train2 = DataLoader(MNIST_train2, batch_size=args.batch_size)
    
    loader_test = DataLoader(MNIST_test, batch_size=args.batch_size * 10)

    return loader_train1, loader_train2, loader_test