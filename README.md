# DAIR: Data Augmented Invariant Regularization

While deep learning through empirical risk minimization (ERM) has succeeded at achieving human-level performance at a variety of complex tasks, ERM generalizes poorly to distribution shift. Synthetic data augmentation followed by empirical risk minimization (DA-ERM) is a simple and widely used solution to remedy this problem. In this paper, we propose data augmented invariant regularization (DAIR), a simple regularization that is applied directly on the loss function, making it widely applicable regardless of network architecture or problem setup. We apply DAIR to multiple real-world learning problems, namely robust regression, visual question answering, robust deep neural network training, and neural task-oriented dialog modeling. Our experiments show that DAIR consistently outperforms ERM and DA-ERM with little marginal cost and sets new state-of-the-art results in several benchmarks.

This repository contains the data, code, and experiments to reproduce our empirical results. 
## Getting started

### Dependencies

The following dependencies are needed. (The latest versions will work)
* python3
* sklearn
* numpy
* matplotlib
* colorsys
* seaborn
* scipy
* cvxpy (optional)

## How to run the code for different applications

**1. Toy Example** 

```
cd DAIR/toy_example
python fig1.py
```

**2. Colored MNIST**

* To run the adversarial augmentation scheme: 
```
cd DAIR/CMNIST
python main.py --scheme adversarial
``` 
* To run the random augmentation scheme: 
```
cd DAIR/CMNIST
python main.py --scheme random
``` 
    
**3. Rotated MNIST**


* To run the weak augmentation scheme: 
```
cd DAIR/RMNIST
python main.py --scheme weak
``` 
* To run the random strong scheme: 
```
cd DAIR/RMNIST
python main.py --scheme strong
```

**4. Robust Regression**

```
cd DAIR/robust_regression
python main.py
```

**5. Invariant Visual Question Answering**

```
cd DAIR/invariant_vqa
python train4.py --trained_model_save_folder <unique path> --_lambda <some value> --prefix real_iv
``` 
  
**6. Training Robust Neural Networks**

* To run DAIR: 
```
cd DAIR/robust_nn/DAIR
python main.py
```
* To run TRADES (built upon [TRADES](https://arxiv.org/abs/1901.08573)):
```
cd DAIR/robust_nn/TRADES
python train_trades_cifar10.py
```

* To run TRADES + ATTA (built upon [TRADES + ATTA](https://arxiv.org/abs/1912.11969)):
```
cd DAIR/robust_nn/ATTA
python train_atta_cifar.py
```

* To run APART (built upon [APART](https://arxiv.org/abs/2010.08034)):
```
cd DAIR/robust_nn/APART
python train.py
```

**7. Neural Task-oriented Dialog Modeling**

```
```
    
    

 
    
    
   
