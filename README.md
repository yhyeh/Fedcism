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
- Clone environment using conda [Doc](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
  - Makesure you already have conda installed [Doc](https://docs.anaconda.com/anaconda/install/)
```bash
conda env create --name fedcism_env -f conda_environment.yml
```
- Install required python packages [Doc](https://pip.pypa.io/en/stable/cli/pip_install/#)
```bash
pip install -r pip_requirements.txt
```
- Clone the repository
```bash
git clone https://github.com/yhyeh/Fedcism.git
```

## Data
We run FedAvg, Oort and Fedcism experiments on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). See our paper for how we process and partition the data for federated learning experiments.

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
The figures are plotted using pyplot on jupyter notebook, please refer to folder [result_processing](result_processing/).

# Acknowledgements
This codebase was adapted from [LG-FedAvg](https://github.com/pliang279/LG-FedAvg).
