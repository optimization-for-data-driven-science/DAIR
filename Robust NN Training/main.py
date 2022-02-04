'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack, GradientSignAttack, LinfBasicIterativeAttack

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--gamma',default = 0.5, type = float)
parser.add_argument('--_lambda',default = 2.15, type = float)
parser.add_argument('--train_batch_size', default=200, type=int)
parser.add_argument('--test_batch_size', default=1000, type=int)
parser.add_argument('--train_attack',default="fgsm", type=str)
parser.add_argument('--test_attack',default="fgsm", type=str)
parser.add_argument('--tr_iter',default=7,type=int)
parser.add_argument('--test_iter',default=20,type=int)
parser.add_argument('--epochs',default = 120, type = int)
parser.add_argument('--device',default="cuda:0", type=str)
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


if args.train_attack == "fgsm":
    adversary_train =  GradientSignAttack(net,loss_fn =  nn.CrossEntropyLoss(reduction="sum"), eps=8/255)
elif args.train_attack == "pgd":
    adversary_train =  LinfPGDAttack( \
    net, loss_fn=nn.CrossEntropyLoss(), eps=8/255, \
    nb_iter=args.tr_iter, eps_iter=2/255, rand_init=True, clip_min=0.0, \
    clip_max=1.0, targeted=False)
elif args.train_attack == "ifgsm":
    adversary_train =LinfBasicIterativeAttack(net, \
    loss_fn=nn.CrossEntropyLoss(), nb_iter = args.tr_iter, eps=8/255)

if args.test_attack == "fgsm":
    adversary_test =  GradientSignAttack(net,loss_fn =  nn.CrossEntropyLoss(reduction = "sum"), eps=8/255)
elif args.test_attack == "pgd":
    adversary_test =  LinfPGDAttack( \
    net, loss_fn=nn.CrossEntropyLoss(), eps=8/255, \
    nb_iter=args.test_iter, eps_iter=2/255, rand_init=True, clip_min=0.0, \
    clip_max=1.0, targeted=False)
elif args.test_attack == "ifgsm":
    adversary_test =LinfBasicIterativeAttack(net, \
    loss_fn=nn.CrossEntropyLoss(), nb_iter = args.test_iter,eps=8/255)
elif args.test_attack == "all":
            adversary_test = list()
            adversary_test = [GradientSignAttack(net,loss_fn =  nn.CrossEntropyLoss(),eps=8/255),
            LinfPGDAttack( \
            net, loss_fn=nn.CrossEntropyLoss(), eps=8/255, \
            nb_iter=7, eps_iter=2/255, rand_init=True, clip_min=0.0, \
            clip_max=1.0, targeted=False),
            LinfPGDAttack( \
            net, loss_fn=nn.CrossEntropyLoss(), eps=8/255, \
            nb_iter=20, eps_iter=2/255, rand_init=True, clip_min=0.0, \
            clip_max=1.0, targeted=False),
            #LinfBasicIterativeAttack(net, \
            #loss_fn=nn.CrossEntropyLoss(), nb_iter = 10, eps=8/255),
            ]
mapper = ["fgsm", "pgd7", "pgd20" #, "ifgsm"
]
loss_func = nn.CrossEntropyLoss(reduction = "none")
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with ctx_noparamgrad_and_eval(net):
            data = adversary_train.perturb(inputs, targets)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs_adv = net(data)
        
        loss1 = loss_func(outputs, targets)
        loss2 = loss_func(outputs_adv, targets)
        loss1 += 1e-7
        loss2 += 1e-7
        loss = args.gamma*loss1.mean() + (1-args.gamma)*loss2.mean() + args._lambda*(loss1.pow(0.5) - loss2.pow(0.5)).pow(2).mean()
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, adv_predicted = outputs_adv.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        adv_correct += adv_predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% | Adv Acc: %.3f%% '
                     % (train_loss/(batch_idx+1), 100.*correct/total, 100.*adv_correct/total))


def test(epoch):
    global best_acc
    net.eval()
    if isinstance(adversary_test,list):
        print('\n-----------------------------\n')
        for idx, adversary in enumerate(adversary_test):
            test_loss = 0
            correct = 0
            total = 0
            adv_test_loss = 0
            adv_correct = 0            
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                advdata = adversary.perturb(inputs, targets)
                with torch.no_grad():
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    
                    adv_outputs = net(advdata)
                    adv_loss = criterion(adv_outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)

                    adv_test_loss += adv_loss.item()
                    _, adv_predicted = adv_outputs.max(1)

                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    adv_correct += adv_predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader), mapper[idx] + ' Acc: %.3f%% | Adv Acc: %.3f%% '
                                % ( 100.*correct/total,  100.*adv_correct/total))
    else:
        test_loss = 0
        correct = 0
        total = 0
        adv_test_loss = 0
        adv_correct = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            advdata = adversary_test.perturb(inputs, targets)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                
                adv_outputs = net(advdata)
                adv_loss = criterion(adv_outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)

                adv_test_loss += adv_loss.item()
                _, adv_predicted = adv_outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                adv_correct += adv_predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), ' Acc: %.3f%% | Adv Acc: %.3f%% '
                                % ( 100.*correct/total,  100.*adv_correct/total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
