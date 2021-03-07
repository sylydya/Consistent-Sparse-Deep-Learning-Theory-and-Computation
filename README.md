Consistent Sparse Deep Learning:  Theory and Computation
===============================================================
We propose a frequentist-like method for learning sparse DNNs and justify its consistency under the Bayesian framework. The  structure  of  the sparse  DNN  can  be  consistently  determined  using  a  Laplace  approximation-based  marginal posterior  inclusion  probability  approach on a trained Bayesian neural network with mixture of normal prior. 
### Related Publication

Yan Sun <sup> * </sup>, Qifan Song <sup> * </sup>, Faming Liang, [Consistent Sparse Deep Learning: Theory and Computation.](https://arxiv.org/pdf/2102.13229.pdf), JASA, in press.
### Simulation:

Generate Data:
```{python}
python Generate_Data.py
```
##### On the First Dataset 
Regression:
```{python}
python Simulation_Regression.py --data_index 1 --activation 'tanh'
python Simulation_Regression.py --data_index 1 --activation 'relu'
```
Regression Baseline:
```{python}
python Dropout_Regression.py --data_index 1 --activation 'tanh'
python Dropout_Regression.py --data_index 1 --activation 'relu'
python Spinn_Regression.py --data_index 1 --activation 'tanh'
python Spinn_Regression.py --data_index 1 --activation 'relu'
python DPF_Regression.py --data_index 1 --activation 'tanh'
python DPF_Regression.py --data_index 1 --activation 'relu'
```

Classification
```{python}
python Simulation_Classification.py --data_index 1 --activation 'tanh'
python Simulation_Classification.py --data_index 1 --activation 'relu'
```
Classification Baseline:
```{python}
python Dropout_Classification.py --data_index 1 --activation 'tanh'
python Dropout_Classification.py --data_index 1 --activation 'relu'
python Spinn_Classification.py --data_index 1 --activation 'tanh'
python Spinn_Classification.py --data_index 1 --activation 'relu'
python DPF_Classification.py --data_index 1 --activation 'tanh'
python DPF_Classification.py --data_index 1 --activation 'relu'
```

Structure Selection
```{python}
python Simulation_Structure.py --data_index 1
```
Structure Selection Baseline:
```{python}
python Spinn_structure.py --data_index 1
```


### Real Data:
CIFAR ResNet Compression
```{python}
python cifar_run.py --model_path 'resnet_32_10percent/' --sigma0 0.00002 --lambdan 0.00001 -depth 32 --seed 1
python cifar_run_vb.py --model_path 'resnet_32_10percent_vb/' --sigma0 0.00002 --lambdan 0.00001 --prune_ratio 0.1 -depth 32 --seed 1
python cifar_run.py --model_path 'resnet_32_5percent/' --sigma0 0.00006 --lambdan 0.0000001 -depth 32 --seed 1
python cifar_run_vb.py --model_path 'resnet_32_5percent_vb/' --sigma0 0.00006 --lambdan 0.0000001 --prune_ratio 0.05 -depth 32 --seed 1
python cifar_run.py --model_path 'resnet_20_10percent/' --sigma0 0.00004 --lambdan 0.000001 -depth 20 --seed 1
python cifar_run_vb.py --model_path 'resnet_20_10percent_vb/' --sigma0 0.00004 --lambdan 0.000001 --prune_ratio 0.1 -depth 20 --seed 1
python cifar_run.py --model_path 'resnet_20_20percent/' --sigma0 0.000006 --lambdan 0.000001 -depth 20 --seed 1
python cifar_run_vb.py --model_path 'resnet_20_20percent_vb/' --sigma0 0.000006 --lambdan 0.000001 --prune_ratio 0.2 -depth 20 --seed 1
```

