# Direct Advantage Estimation
Official repository for "Direct Advantage Estimation"

## Requirements

Install requirements:
```setup
pip install -r requirements.txt
```

## Training

To reproduce the results, run the following command:
```setup
python train.py --algo {algo} --hparam_file {hyperparameter_file} --envs {env} 
```

`--algo`: Either `PPO` or `CustomPPO`

`--hparam_file`: See `./params/` for the hyperparameters used in the paper, the files are named by `{algo}_{network}.yml`

`--envs`: The environment to traing. (e.g., `Pong`, `Breakout`...) For MinAtar environments, please add the suffix `-MinAtar-v0`. (e.g., `Breakout-MinAtar-v0`)
