from scipy.io import loadmat
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from statistics import stdev, mean
loss_rmse_train = []
loss_mse_train = []
loss_rmse_test = []
loss_mse_test = []
for i in range(5):
    x = loadmat('qsar.mat')
    x_train = x['X_train']
    x_test = x['X_test']

    y_train = x['y_train']
    y_test = x['y_test']

    np.random.seed(i)
    perm_train = np.random.permutation(len(x_train))
    np.random.seed(i)
    perm_test = np.random.permutation(len(x_test))

    x_train = torch.from_numpy(x_train[perm_train]).float()
    y_train = torch.from_numpy(y_train[perm_train]).float()

    x_test = torch.from_numpy(x_test[perm_test]).float()
    y_test = torch.from_numpy(y_test[perm_test]).float()

    x_train_org = torch.cat([x_train, torch.empty(3085, 1)], dim=1)
    x_train_alt = torch.cat([x_train, (torch.rand((3085, 1)) > 0).float()], dim=1) #augmented set
    x_test = torch.cat([x_test, torch.empty(1000, 1)], dim=1)

    bad_ratio = 0.2 # this means 20% labels are corrupted
    y_train[:int(len(y_train)*bad_ratio)] = torch.from_numpy(np.random.normal(5, 5, int(len(y_train) * bad_ratio))).unsqueeze(1)

    y_median_train = torch.median(y_train)
    y_median_test = torch.median(y_test)


    for i in range(3085):
        if y_train[i][0] > y_median_train:
            x_train_org[i][-1] = 1
        else:
            x_train_org[i][-1] = 0


    for i in range(1000):
        if y_test[i][0] > y_median_test:
            x_test[i][-1] = 0
        else:
            x_test[i][-1] = 1

    print('----------')
    
    W = torch.randn(411, 1) * 1e-2
    b = torch.randn(1) * 1e-2
    W.requires_grad = True
    b.requires_grad = True

    optimizer = optim.SGD([W, b], lr=1e-2, momentum=0.9)
    t = -2
    gamma, lambda_ = 0.5, 10  # hyperparameters for DAIR; for ERM, set gamma = 1, lambda = 0; for DA-ERM set gamma = 0.5, lambda = 0
    huber = nn.SmoothL1Loss()

    objective = 'term'

    for i in tqdm(range(10000),total= 10000):

        optimizer.zero_grad()
        y_pred1 = x_train_org @ W + b
        y_pred2 = x_train_alt @ W + b
        if objective == 'l2':
            loss1 = (y_pred1 - y_train).pow(2)
            loss2 = (y_pred2 - y_train).pow(2)
            loss1 += 1e-7
            loss2 += 1e-7

        elif objective == 'huber':
            loss1 = huber(y_pred1, y_train)
            loss2 = huber(y_pred2, y_train)

        elif objective == 'term':
            loss1 = (1/t)*torch.log(torch.exp(t * ((y_pred1 - y_train) ** 2)).mean())
            loss2 = (1/t)*torch.log(torch.exp(t * ((y_pred2 - y_train) ** 2)).mean())
        
        loss = gamma * loss1.mean() + (1-gamma)*loss2.mean() + lambda_ * (loss1.pow(0.5) - loss2.pow(0.5)).pow(2).mean()

        loss.backward()
        optimizer.step()
        

    with torch.no_grad():
        y_pred = x_train_org @ W + b
    loss = (y_pred - y_train).pow(2).mean()
    loss_rmse_train.append(loss.pow(0.5).item())
    loss_mse_train.append(loss.item())
    ## Test
    with torch.no_grad():
        y_pred = x_test @ W + b
    loss = (y_pred - y_test).pow(2).mean()
    loss_rmse_test.append(loss.pow(0.5).item())
    loss_mse_test.append(loss.item())

print("train RMSE : {:.2f} +/- {:.2f}".format(mean(loss_rmse_train), stdev(loss_rmse_train)))
print("train MSE : {:.2f} +/- {:.2f}".format(mean(loss_mse_train), stdev(loss_mse_train)))
print("test RMSE : {:.2f} +/- {:.2f}".format(mean(loss_rmse_test), stdev(loss_rmse_test)))
print("test MSE : {:.2f} +/- {:.2f}".format(mean(loss_mse_test), stdev(loss_mse_test)))

#prints Standard Dev not S.E.





