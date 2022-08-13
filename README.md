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

Our code in this repository works with ParlAI commit a9c40b78d368704315fcb2917eb2cafbdc430334. Please copy all files under parlai folder to your parlai installation folder.
 

* To run normal training
```
parlai train_model --model bart --task multiwoz_v22 --seed_np 42 --batchsize 8 --model-file path_to_save --fp16 false --optimizer adamw --learningrate 1e-5  --save-every-n-secs 600 --truncate 448 --entity multiwoz --num-epochs 4 --datatype train:ordered
```

* To run DAIR training
```
parlai train_model --model bart --task multiwoz_v22 --seed_np 42 --comp_scramble true --scramble_mode create_gibberish_entity --batchsize 6 --model-file path_to_save --fp16 false --optimizer adamw --learningrate 1e-5  --save-every-n-secs 600 --truncate 448 --entity multiwoz --num-epochs 4 --bart_loss_fn loss1 --datatype train:ordered --comp_train True --reg_type sqrt --back_prop_replaced_entity_loss True --lambda_ your_lambda
```

* To run KL training
```
parlai train_model --model bart --task multiwoz_v22 --seed_np 42 --comp_scramble true --scramble_mode create_gibberish_entity --batchsize 6 --model-file path_to_save --fp16 false --optimizer adamw --learningrate 1e-5  --save-every-n-secs 600 --truncate 448 --entity multiwoz --num-epochs 4 --bart_loss_fn loss1 --datatype train:ordered --comp_train True --reg_type kl --back_prop_replaced_entity_loss True --lambda_ your_lambda
```
    
* To run normal testing
```
parlai eval_model --task multiwoz_v22 --seed_np 42 --model-file path_to_save --datatype test --entity multiwoz --batchsize 32
```
 
* To run testing with SGD entities
```
parlai eval_model --task multiwoz_v22 --seed_np 42 --model-file path_to_save --datatype test --entity g_sgd --batchsize 32
```

* To obtain Consistency Metric (CM) with SGD entities
```
parlai eval_model --task multiwoz_v22 --seed_np 42 --model-file path_to_save  --datatype test --entity1 multiwoz --entity2 g_sgd --new_metric True
```

 
    
    
   
