# RusTReach Learning Repository

This is the codebase for learning model weights for RusTReach experiments and testing the MFNLC approach.

Fork of Code Repository of IROS 22' paper **Model-free Neural Lyapunov Control for Safe Robot Navigation**

[ArXiv](https://arxiv.org/abs/2203.01190) 


## Project Structure

```
├── README.md
├── setup.py
└── shrl
    ├── config.py       # config file, including data path, default devices, ect. 
    ├── envs            # simulation environments
    ├── evaluation      # evaluation utils
    ├── exps            # experiment scripts
    ├── learn           # low-level controller and neural Lyapunov function learning algorithms
    ├── monitor         # high-level monitor
    ├── plan            # high-level planner, RRT & RRT*
    └── tests           # test cases
```

## Install
**Note** A gpu is required to train and run these experiments.

1. Install necessary dependencies.

```commandline
pip install pip==21.0.1
pip install -e .
```

## Learning RusTReach Model Weights
 
```shell
python exps/train/no_obstacle/lyapunov_td3/bicycle.py
```

```shell
python exps/train/no_obstacle/lyapunov_td3/quadcopter.py
```


## Convert Model to ONNX

```shell
python evaluation/convert_to_onnx.py --robot_name ["Bicycle", "Quadcopter"]
```

## Evaluate MFNLC

### Corridor

```shell
python exps/corr_exp/lyapunov/bicycle.py
```

```shell
python exps/corr_exp/lyapunov/quadcopter.py
```

### Neighborhood

```shell
python exps/nbd_exp/lyapunov/bicycle.py
```

```shell
python exps/nbd_exp/lyapunov/quadcopter.py
```