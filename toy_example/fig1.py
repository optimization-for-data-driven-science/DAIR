import torch
import torch.optim as optim


################## DA-ERM ################## 


X = torch.randn(10000, 1)
eps = torch.randn(10000, 1) * 0.5

y = X + eps
a = 0.5
n = torch.randn(10000, 1) * (0.1 ** 0.5)

X_train = torch.cat([X, y], dim=1)
X_aug = torch.cat([X, a * y + n], dim=1)


## w init
w = torch.randn(2, 1) * 0.001
w.requires_grad = True

## SGD also works and has similar results
optimizer = optim.Adam([w], lr=1e-3)
max_iter = 10000

for i in range(max_iter):
    optimizer.zero_grad()
    y_pred1 = X_train @ w
    y_pred2 = X_aug @ w 
    loss1 = (y - y_pred1).pow(2).mean()
    loss2 = (y - y_pred2).pow(2).mean()
    loss = (loss1 + loss2) / 2
    loss.backward()
    optimizer.step()

print("#" * 10 + " DA-ERM " + "#" * 10)
print(w.detach().numpy())
print()



################## DAIR ################## 

X = torch.randn(10000, 1)
eps = torch.randn(10000, 1) * 0.5

y = X + eps
a = 0.5
n = torch.randn(10000, 1) * (0.1 ** 0.5)

X_train = torch.cat([X, y], dim=1)
X_aug = torch.cat([X, a * y + n], dim=1)


## w init
w = torch.randn(2, 1) * 0.001
w.requires_grad = True

## SGD also works and has similar results
optimizer = optim.Adam([w], lr=1e-3)
max_iter = 10000

for i in range(max_iter):
    optimizer.zero_grad()
    y_pred1 = X_train @ w
    y_pred2 = X_aug @ w 
    loss1 = (y - y_pred1).pow(2).mean()
    loss2 = (y - y_pred2).pow(2).mean()
    loss = (loss1.mean() + loss2.mean()) / 2
    loss = loss + 5000 * (loss1.pow(0.5) - loss2.pow(0.5)).pow(2).mean()
    loss.backward()
    optimizer.step()

print("#" * 10 + " DAIR " + "#" * 10)
print(w.detach().numpy())
print()


################## ERM ################## 


X = torch.randn(10000, 1)
eps = torch.randn(10000, 1) * 0.5

y = X + eps
a = 0.5
n = torch.randn(10000, 1) * (0.1 ** 0.5)

X_train = torch.cat([X, y], dim=1)
X_aug = torch.cat([X, a * y + n], dim=1)


## w init
w = torch.randn(2, 1) * 0.001
w.requires_grad = True

## SGD also works and has similar results
optimizer = optim.Adam([w], lr=1e-3)
max_iter = 10000

for i in range(max_iter):
    optimizer.zero_grad()
    y_pred1 = X_train @ w
    loss = (y - y_pred1).pow(2).mean()
    loss.backward()
    optimizer.step()

print("#" * 10 + " ERM " + "#" * 10)
print(w.detach().numpy())
print()







