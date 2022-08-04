# Distribution-Aware Participant Selection for Federated Learning

> Pytorch implementation for **Fedcism**, a participant selection framework based on Class Imbalance using Similarity Measurement. 
> By adaptively blending the distribution similarity factor into the utility function that can determine the sampling priority of participant selection, our method can tolerate various levels of data imbalance while penalizing straggler with acceptable computation overhead in aggregator.

Correspondence to: 
  - Yu-Hsuan Yeh @yhyeh (yhyeh.cs09@nycu.edu.tw)
  - Kate C.J. Lin (katelin@cs.nycu.edu.tw)
  
## Paper

[**Distribution-Aware Participant Selection for Federated Learning**](https://)<br>
[Yu-Hsuan Yeh](https://www.linkedin.com/in/yhyeh/), [Kate C.J. Lin](https://people.cs.nctu.edu.tw/~katelin/)<br>

If you find this repository useful, please cite our paper.

## Installation

First check that the requirements are satisfied:</br>
Python 3.6</br>
torch 1.2.0</br>
torchvision 0.4.0</br>
numpy 1.18.1</br>
sklearn 0.20.0</br>
matplotlib 3.1.2</br>
Pillow 4.1.1</br>

The next step is to clone the repository:
```bash
git clone https://github.com/
```

## Data

We run FedAvg, Oort and Fedcism experiments on CIFAR-10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html)). See our paper for a description how we process and partition the data for federated learning experiments.

## Training Reproduction
Results can be reproduced running the following:

### Fedcism (algo3), Fedcism w/ constant-gamma(algo1), Oort (algo0)
#### CIFAR10 
Please refer to example script [scripts/algo3_gbalan.sh](scripts/algo3_gbalan.sh).

### FedAvg
#### MNIST
> python main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save run1

#### CIFAR10 
> python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --results_save run1

## Figure Plotting
The figures are plotted using pyplot on jupyter notebook, please refer to folder [result_processing](../result_processing/).
# Acknowledgements
This codebase was adapted from [LG-FedAvg](https://github.com/pliang279/LG-FedAvg).
